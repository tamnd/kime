//! `kime-train`: fine tunes a compat checkpoint on training lines and writes a checkpoint kime
//! serves. `kime train` runs this binary.

#![forbid(unsafe_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use burn::module::AutodiffModule;
use kime_model::Model;
use kime_train::data::{Example, Renderer, lines};
use kime_train::model::Compat;
use kime_train::rng::Rng;
use kime_train::train::{Config, Event, evaluate, fit, render_all};
use kime_train::{Train, export};
use serde_json::json;

const USAGE: &str =
    "usage: kime-train --base <checkpoint dir> --data <file or dir>[,...] --out <dir>
    [--only source,...] [--max-lines N] [--eval-frac 0.02] [--eval-max 5000]
    [--epochs 1] [--steps N] [--lr 3e-5] [--warmup 0.02] [--tokens 8192] [--accum 1]
    [--seed 13] [--no-shuffle] [--eval-every N] [--train-layers N] [--log run.jsonl] [--dump <dir>]

--data takes .jsonl or .jsonl.zst files, or folders of them, such as the shards/ folder
tools/data/convert.py writes. --only keeps the shards whose names start with the given sources.
--train-layers trains only the top N encoder layers and the heads, which needs far less memory.
--dump writes the laid out questions of the first epoch and the held out lines to <dir>/train.jsonl
and <dir>/eval.jsonl and stops, for tools/ref/train_ref.py to train on the same questions.";

struct Args {
    base: PathBuf,
    data: Vec<PathBuf>,
    out: PathBuf,
    only: Vec<String>,
    max_lines: Option<usize>,
    eval_frac: f64,
    eval_max: usize,
    log: Option<PathBuf>,
    train_layers: Option<usize>,
    dump: Option<PathBuf>,
    cfg: Config,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut base, mut data, mut out) = (None, Vec::new(), None);
    let mut a = Args {
        base: PathBuf::new(),
        data: Vec::new(),
        out: PathBuf::new(),
        only: Vec::new(),
        max_lines: None,
        eval_frac: 0.02,
        eval_max: 5000,
        log: None,
        train_layers: None,
        dump: None,
        cfg: Config::default(),
    };
    while let Some(flag) = it.next() {
        if flag == "--no-shuffle" {
            a.cfg.shuffle_options = false;
            continue;
        }
        if flag == "-h" || flag == "--help" {
            return Err(String::new());
        }
        let v = it.next().ok_or(format!("{flag} needs a value"))?;
        let num = |v: &str| v.parse::<f64>().map_err(|_| format!("{flag}: not a number: {v}"));
        let int = |v: &str| v.parse::<usize>().map_err(|_| format!("{flag}: not a count: {v}"));
        match flag.as_str() {
            "--base" => base = Some(PathBuf::from(v)),
            "--data" => data.extend(v.split(',').map(PathBuf::from)),
            "--out" => out = Some(PathBuf::from(v)),
            "--only" => a.only.extend(v.split(',').map(str::to_string)),
            "--max-lines" => a.max_lines = Some(int(&v)?),
            "--eval-frac" => a.eval_frac = num(&v)?,
            "--eval-max" => a.eval_max = int(&v)?,
            "--epochs" => a.cfg.epochs = int(&v)?,
            "--steps" => a.cfg.max_steps = Some(int(&v)?),
            "--lr" => a.cfg.lr = num(&v)?,
            "--warmup" => a.cfg.warmup = num(&v)?,
            "--tokens" => a.cfg.tokens = int(&v)?,
            "--accum" => a.cfg.accum = int(&v)?,
            "--seed" => a.cfg.seed = int(&v)? as u64,
            "--eval-every" => a.cfg.eval_every = int(&v)?,
            "--log" => a.log = Some(PathBuf::from(v)),
            "--train-layers" => a.train_layers = Some(int(&v)?),
            "--dump" => a.dump = Some(PathBuf::from(v)),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    a.base = base.ok_or("--base is required")?;
    a.out = out.ok_or("--out is required")?;
    if data.is_empty() {
        return Err("--data is required".into());
    }
    a.data = data;
    Ok(a)
}

/// The data files, with folders expanded and `--only` applied, in name order.
fn files(a: &Args) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for p in &a.data {
        if p.is_dir() {
            let rd = std::fs::read_dir(p).map_err(|e| format!("{}: {e}", p.display()))?;
            for e in rd {
                let f = e.map_err(|e| e.to_string())?.path();
                let name = f.file_name().unwrap_or_default().to_string_lossy().to_string();
                if name.ends_with(".jsonl") || name.ends_with(".jsonl.zst") {
                    out.push(f);
                }
            }
        } else {
            out.push(p.clone());
        }
    }
    let keep = |f: &Path| {
        let name = f.file_name().unwrap_or_default().to_string_lossy();
        a.only.is_empty()
            || a.only
                .iter()
                .any(|s| name.starts_with(&format!("{s}-")) || name.starts_with(&format!("{s}.")))
    };
    out.retain(|f| keep(f));
    out.sort();
    Ok(out)
}

fn run(a: &Args) -> Result<(), String> {
    let base = Model::open(&a.base).map_err(|e| format!("{}: {e}", a.base.display()))?;
    let renderer = Renderer::new(&base)?;
    let files = files(a)?;
    if files.is_empty() {
        return Err("no data files".into());
    }
    let t0 = Instant::now();
    let mut all = Vec::new();
    for f in &files {
        for l in lines(f).map_err(|e| format!("{}: {e}", f.display()))? {
            all.push(l.map_err(|e| format!("{}: {e}", f.display()))?);
        }
    }
    let mut rng = Rng::new(a.cfg.seed ^ 0x6576_616c);
    rng.shuffle(&mut all);
    let n_eval =
        ((all.len() as f64 * a.eval_frac).round() as usize).min(a.eval_max).min(all.len() / 2);
    let mut train = all.split_off(n_eval);
    let held = all;
    if let Some(m) = a.max_lines {
        train.truncate(m);
    }
    let eval = render_all(&renderer, &held, None);
    eprintln!(
        "{} files, {} training lines, {} held out lines with {} questions, read in {:.1}s",
        files.len(),
        train.len(),
        held.len(),
        eval.len(),
        t0.elapsed().as_secs_f64()
    );

    if let Some(dir) = &a.dump {
        // The first epoch of `fit` renders with a generator seeded like this one.
        let mut rng = Rng::new(a.cfg.seed);
        let first = render_all(&renderer, &train, a.cfg.shuffle_options.then_some(&mut rng));
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (name, examples) in [("train.jsonl", &first), ("eval.jsonl", &eval)] {
            let p = dir.join(name);
            dump(&p, examples).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        eprintln!(
            "wrote {} training and {} held out questions to {}",
            first.len(),
            eval.len(),
            dir.display()
        );
        return Ok(());
    }

    let dev = Default::default();
    let mut model = Compat::<Train>::load(&base.spec, &base.tensors, &dev);
    let depth = model.depth();
    if let Some(n) = a.train_layers {
        model = model.freeze_below(n);
    }
    let mut log = match &a.log {
        Some(p) => Some(std::fs::File::create(p).map_err(|e| format!("{}: {e}", p.display()))?),
        None => None,
    };
    let mut write = |v: serde_json::Value| {
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{v}");
        }
    };
    let t0 = Instant::now();
    let before = evaluate(&model.valid(), &eval, &a.cfg, &dev);
    eprintln!(
        "before: loss {:.4} accuracy {:.4} nll {:.4} on {} questions ({:.0}s)",
        before.loss,
        before.accuracy,
        before.nll,
        before.questions,
        t0.elapsed().as_secs_f64()
    );
    write(json!({"eval": 0, "loss": before.loss, "accuracy": before.accuracy, "nll": before.nll}));
    let t0 = Instant::now();
    let mut last = None;
    let every = 10;
    let model = fit(model, &renderer, &train, &eval, &a.cfg, &dev, |e| match e {
        Event::Step { step, total, lr, loss, grad_norm, tokens_per_s, questions_per_s } => {
            write(
                json!({"step": step, "lr": lr, "loss": loss, "grad_norm": grad_norm, "tokens_per_s": tokens_per_s}),
            );
            if step % every == 0 || step == total || *step == 1 {
                eprintln!(
                    "step {step}/{total} lr {lr:.2e} loss {loss:.4} grad norm {grad_norm:.3} {tokens_per_s:.0} tokens/s {questions_per_s:.1} questions/s"
                );
            }
        }
        Event::Eval { step, metrics } => {
            write(
                json!({"eval": step, "loss": metrics.loss, "accuracy": metrics.accuracy, "nll": metrics.nll}),
            );
            eprintln!(
                "eval at step {step}: loss {:.4} accuracy {:.4} nll {:.4}",
                metrics.loss, metrics.accuracy, metrics.nll
            );
            last = Some(*metrics);
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    let after = last.unwrap_or_default();
    let note = json!({
        "base": a.base.display().to_string(),
        "data": files.iter().map(|f| f.file_name().unwrap_or_default().to_string_lossy().to_string()).collect::<Vec<_>>(),
        "training_lines": train.len(),
        "held_out_lines": held.len(),
        "epochs": a.cfg.epochs,
        "max_steps": a.cfg.max_steps,
        "lr": a.cfg.lr,
        "tokens": a.cfg.tokens,
        "accum": a.cfg.accum,
        "seed": a.cfg.seed,
        "shuffle_options": a.cfg.shuffle_options,
        "train_layers": a.train_layers.unwrap_or(depth).min(depth),
        "seconds": secs.round(),
        "held_out_before": {"loss": before.loss, "accuracy": before.accuracy, "nll": before.nll},
        "held_out_after": {"loss": after.loss, "accuracy": after.accuracy, "nll": after.nll},
    });
    std::fs::create_dir_all(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    export::save(&model.valid(), &base, &a.out, note)
        .map_err(|e| format!("{}: {e}", a.out.display()))?;
    eprintln!("trained in {secs:.0}s, wrote {}", a.out.display());
    Ok(())
}

/// Writes examples as JSON lines, one question each.
fn dump(path: &Path, examples: &[Example]) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for e in examples {
        let v = json!({
            "ids": e.q.ids,
            "markers": e.q.markers,
            "qtype": e.q.qtype,
            "probs": e.probs,
            "hard": e.hard,
            "weight": e.weight,
        });
        writeln!(f, "{v}")?;
    }
    f.flush()
}

fn main() -> ExitCode {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("kime-train: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&a) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime-train: {e}");
            ExitCode::FAILURE
        }
    }
}

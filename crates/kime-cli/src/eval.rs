//! `kime eval`: run quality suites through kime, or score answers another engine wrote, and
//! write the report.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use kime::{Device, Kime, Precision};
use kime_eval::contam::manifest_conflicts;
use kime_eval::report::{Summary, markdown, rows_parquet, rows_tsv, summarize};
use kime_eval::suite::{Case, Scored, load, probs, probs_json, score};
use serde_json::{Map, Value, json};

use crate::gate::{self, Ran};
use crate::predict::{device, precision};

const USAGE: &str = "usage: kime eval <suite.jsonl or directory>... [--model laya] [--device auto|cpu|cuda[:N]|metal]
       [--precision f16|f32|int8] [--answers <dir>] [--out <dir>] [--title <text>]
       [--data-manifest <file>] [--suite quality|order|agent|all]... [--gate [--baseline <results.json>]]
Runs every suite through the model and writes <out>/report.md, results.json, rows.tsv,
rows.parquet and one <suite>.answers.jsonl per suite. With --answers, the answers are read from
<dir>/<suite>.answers.jsonl,
one {\"id\": ..., \"answers\": {...}} line per case in Laya's JSON shape, and no model is loaded.
--data-manifest names the training data manifest of the model under test. Its blake3 goes into
results.json, and nothing is scored if it lists a test split or the split a suite is drawn from,
as the manifest.json next to the suites records it.
--suite keeps only the suites of a group: quality is every suite of spec/13-benchmarks.md,
order the order.* copies with the options shuffled, and agent the mind2web.* browser steps.
--gate then runs the model behaviour tests of spec/15-testing.md on the same suites, writes
<out>/gate.md and gate.json, and fails if any test misses its threshold. --baseline is the
results.json of the previous release, for the calibration regression test.";

/// Requests handed to the engine at once.
pub(crate) const CHUNK: usize = 256;

struct Opts {
    suites: Vec<PathBuf>,
    model: String,
    device: Device,
    precision: Precision,
    answers: Option<PathBuf>,
    out: PathBuf,
    title: Option<String>,
    data_manifest: Option<PathBuf>,
    groups: Vec<String>,
    gate: bool,
    baseline: Option<PathBuf>,
}

/// The groups `--suite` takes.
const GROUPS: [&str; 4] = ["quality", "order", "agent", "all"];

/// The group a suite belongs to, by its name.
fn group(suite: &str) -> &'static str {
    if suite.starts_with("order.") {
        "order"
    } else if suite.starts_with("mind2web.") {
        "agent"
    } else {
        "quality"
    }
}

fn opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        suites: Vec::new(),
        model: "laya".into(),
        device: Device::Auto,
        precision: Precision::F16,
        answers: None,
        out: PathBuf::from("eval-out"),
        title: None,
        data_manifest: None,
        groups: Vec::new(),
        gate: false,
        baseline: None,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--model" => o.model = val()?,
            "--device" => o.device = device(&val()?)?,
            "--precision" => o.precision = precision(&val()?)?,
            "--answers" => o.answers = Some(val()?.into()),
            "--out" => o.out = val()?.into(),
            "--title" => o.title = Some(val()?),
            "--data-manifest" => o.data_manifest = Some(val()?.into()),
            "--gate" => o.gate = true,
            "--baseline" => o.baseline = Some(val()?.into()),
            "--suite" => {
                let g = val()?;
                if !GROUPS.contains(&g.as_str()) {
                    return Err(format!(
                        "no suite group {g:?}, the groups are {}",
                        GROUPS.join(", ")
                    ));
                }
                o.groups.push(g);
            }
            "--help" | "-h" => return Err(USAGE.into()),
            p if !p.starts_with('-') => {
                let p = PathBuf::from(p);
                if p.is_dir() {
                    let mut files: Vec<PathBuf> = std::fs::read_dir(&p)
                        .map_err(|e| format!("{}: {e}", p.display()))?
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|f| {
                            f.extension().is_some_and(|x| x == "jsonl")
                                && !name(f).ends_with(".answers")
                        })
                        .collect();
                    files.sort();
                    o.suites.extend(files);
                } else {
                    o.suites.push(p);
                }
            }
            other => return Err(format!("unknown option {other:?}\n{USAGE}")),
        }
    }
    if !o.groups.is_empty() && !o.groups.iter().any(|g| g == "all") {
        o.suites.retain(|p| o.groups.iter().any(|g| g == group(&name(p))));
        if o.suites.is_empty() {
            return Err(format!("no suite is in {}", o.groups.join(", ")));
        }
    }
    if o.suites.is_empty() {
        return Err(USAGE.into());
    }
    if o.gate && o.answers.is_some() {
        return Err("--gate asks the model again, so it needs a model, not --answers".into());
    }
    if o.baseline.is_some() && !o.gate {
        return Err("--baseline is only read by --gate".into());
    }
    Ok(o)
}

/// A suite's name: its file name without `.jsonl`.
fn name(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

pub(crate) fn run(args: &[String]) -> ExitCode {
    match eval(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime eval: {e}");
            ExitCode::FAILURE
        }
    }
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Reads the data manifest, refuses it if it lists a split a suite is drawn from, and returns
/// what results.json records about it.
fn data_manifest(o: &Opts) -> Result<Value, String> {
    let Some(path) = &o.data_manifest else {
        return Ok(Value::Null);
    };
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let data: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut conflicts = Vec::new();
    let mut dirs: Vec<&Path> = o.suites.iter().filter_map(|p| p.parent()).collect();
    dirs.dedup();
    for dir in dirs {
        let m = dir.join("manifest.json");
        let tests = match std::fs::read_to_string(&m) {
            Ok(t) => serde_json::from_str(&t).map_err(|e| format!("{}: {e}", m.display()))?,
            Err(_) => Value::Null,
        };
        let used: Map<String, Value> = tests
            .as_object()
            .map(|t| {
                t.iter()
                    .filter(|(k, _)| o.suites.iter().any(|p| name(p) == **k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();
        conflicts.extend(manifest_conflicts(&data, &Value::Object(used)));
    }
    let blank = json!({});
    conflicts.extend(manifest_conflicts(&data, &blank));
    conflicts.sort();
    conflicts.dedup();
    if !conflicts.is_empty() {
        return Err(format!(
            "{} lists data the suites test on, so no result is written:\n  {}",
            path.display(),
            conflicts.join("\n  ")
        ));
    }
    Ok(
        json!({"file": path.display().to_string(), "blake3": blake3::hash(&bytes).to_hex().to_string()}),
    )
}

fn eval(args: &[String]) -> Result<(), String> {
    let o = opts(args)?;
    let manifest = data_manifest(&o)?;
    let baseline = match &o.baseline {
        Some(p) => {
            let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            Some(serde_json::from_str::<Value>(&t).map_err(|e| format!("{}: {e}", p.display()))?)
        }
        None => None,
    };
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;
    let kime = match o.answers {
        Some(_) => None,
        None => {
            let t = Instant::now();
            let k = Kime::builder()
                .model(&o.model)
                .device(o.device)
                .precision(o.precision)
                .build()
                .map_err(|e| e.to_string())?;
            eprintln!("{} on {}, loaded in {:.2?}", k.model_id(), k.device(), t.elapsed());
            Some(k)
        }
    };
    let (mut summaries, mut results, mut tsv) = (Vec::new(), Map::new(), String::new());
    let (mut rows, mut loaded) = (Vec::new(), Vec::new());
    for path in &o.suites {
        let suite = name(path);
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let cases = load(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let t = Instant::now();
        let (scored, dropped) = match (&kime, &o.answers) {
            (Some(k), _) => run_suite(k, &cases, &o.out.join(format!("{suite}.answers.jsonl")))?,
            (None, Some(dir)) => read_answers(&cases, &dir.join(format!("{suite}.answers.jsonl")))?,
            (None, None) => unreachable!("a model is loaded when there are no answers"),
        };
        let s = summarize(&suite, &scored, dropped);
        eprintln!(
            "{suite}: {} questions, {} dropped, accuracy {:.4}, ECE {:.4}, in {:.1?}",
            s.all.n,
            s.dropped,
            s.all.accuracy,
            s.all.ece,
            t.elapsed()
        );
        tsv.push_str(&rows_tsv(&suite, &scored, tsv.is_empty()));
        if let Value::Object(m) = s.to_json() {
            results.extend(m);
        }
        summaries.push(s);
        rows.push((suite, scored));
        loaded.push(cases);
    }
    let title = o.title.clone().unwrap_or_else(|| match &kime {
        Some(k) => format!("kime eval, {} on {}, {:?}", k.model_id(), k.device(), o.precision),
        None => "kime eval".into(),
    });
    let meta = json!({"title": title, "kime": env!("CARGO_PKG_VERSION"),
        "model": kime.as_ref().map(Kime::model_id), "data_manifest": manifest});
    let all = json!({"meta": meta, "suites": results});
    write(
        &o.out.join("results.json"),
        &(serde_json::to_string_pretty(&all).unwrap_or_default() + "\n"),
    )?;
    let mut md = markdown(&title, &summaries);
    md.push_str(&match manifest.get("blake3") {
        Some(h) => format!("\nData manifest blake3 {}, no test split listed.\n", h.as_str().unwrap_or_default()),
        None => "\nNo data manifest was given, so the training data of the model was not checked against the suites.\n".into(),
    });
    write(&o.out.join("report.md"), &md)?;
    write(&o.out.join("rows.tsv"), &tsv)?;
    let pq = o.out.join("rows.parquet");
    std::fs::write(&pq, rows_parquet(&rows)).map_err(|e| format!("{}: {e}", pq.display()))?;
    print!("{md}");
    match (&kime, o.gate) {
        (Some(k), true) => {
            run_gate(k, &title, &rows, &loaded, &summaries, baseline.as_ref(), &o.out)
        }
        _ => Ok(()),
    }
}

/// Runs the behaviour tests, writes gate.md and gate.json, and fails when a test fails.
fn run_gate(
    kime: &Kime,
    title: &str,
    rows: &[(String, Vec<Scored>)],
    loaded: &[Vec<Case>],
    summaries: &[Summary],
    baseline: Option<&Value>,
    out: &Path,
) -> Result<(), String> {
    let t = Instant::now();
    let ran: Vec<Ran<'_>> = rows
        .iter()
        .zip(loaded)
        .zip(summaries)
        .map(|(((name, scored), cases), summary)| Ran { name, cases, scored, summary })
        .collect();
    let checks = gate::checks(kime, &ran, baseline)?;
    let md = kime_eval::gate::markdown(&format!("Release gate, {title}"), &checks);
    write(&out.join("gate.md"), &md)?;
    let json = serde_json::to_string_pretty(&kime_eval::gate::to_json(&checks)).unwrap_or_default();
    write(&out.join("gate.json"), &(json + "\n"))?;
    print!("\n{md}");
    eprintln!("gate in {:.1?}", t.elapsed());
    if kime_eval::gate::passed(&checks) {
        Ok(())
    } else {
        Err("the checkpoint does not pass the gate, see gate.md".into())
    }
}

/// Answers every valid request of a suite with kime, writes the answers, and scores them.
fn run_suite(kime: &Kime, cases: &[Case], out: &Path) -> Result<(Vec<Scored>, usize), String> {
    let valid: Vec<&Case> = cases.iter().filter(|c| c.request.is_ok()).collect();
    let mut answers = HashMap::new();
    for chunk in valid.chunks(CHUNK) {
        let reqs: Vec<_> = chunk.iter().filter_map(|c| c.request.as_ref().ok().cloned()).collect();
        let res = kime.decide_batch(&reqs).map_err(|e| e.to_string())?;
        for (c, r) in chunk.iter().zip(res) {
            answers.insert(c.id.clone(), r);
        }
    }
    let mut lines = String::new();
    let (mut scored, mut dropped) = (Vec::new(), 0);
    for c in cases {
        let res = answers.get(&c.id);
        if let Some(r) = res {
            let mut line = Map::new();
            line.insert("id".into(), json!(c.id));
            if let Value::Object(m) = r.to_json() {
                line.extend(m);
            }
            lines.push_str(&Value::Object(line).to_string());
            lines.push('\n');
        }
        let (s, d) = score(c, |q| res.and_then(|r| r.get(&q.id)).map(probs));
        scored.extend(s);
        dropped += d;
    }
    write(out, &lines)?;
    Ok((scored, dropped))
}

/// Scores the answers another engine wrote for a suite.
fn read_answers(cases: &[Case], path: &Path) -> Result<(Vec<Scored>, usize), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut by_id = HashMap::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)
            .map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))?;
        let id = v.get("id").map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_string));
        by_id.insert(id.unwrap_or_else(|| (i + 1).to_string()), v);
    }
    let (mut scored, mut dropped) = (Vec::new(), 0);
    for c in cases {
        let answers = by_id.get(&c.id).and_then(|v| v.get("answers"));
        let (s, d) =
            score(c, |q| answers.and_then(|a| a.get(&q.id)).and_then(|a| probs_json(q, a)));
        scored.extend(s);
        dropped += d;
    }
    Ok((scored, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn suite_groups() {
        let files =
            ["en.sst2.jsonl", "order.xnli.en.jsonl", "mind2web.task.jsonl", "app.phishing.jsonl"];
        let pick = |extra: &[&str]| {
            let mut a = args(&files);
            a.extend(args(extra));
            opts(&a).map(|o| o.suites.iter().map(|p| name(p)).collect::<Vec<_>>())
        };
        assert_eq!(pick(&[]).unwrap().len(), 4);
        assert_eq!(pick(&["--suite", "quality"]).unwrap(), ["en.sst2", "app.phishing"]);
        assert_eq!(
            pick(&["--suite", "agent", "--suite", "order"]).unwrap(),
            ["order.xnli.en", "mind2web.task"]
        );
        assert_eq!(pick(&["--suite", "all"]).unwrap().len(), 4);
        assert!(pick(&["--suite", "speed"]).is_err());
    }
}

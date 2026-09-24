//! `kime predict`: answer questions from the command line, one request or a JSONL stream of them.

use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::time::Instant;

use kime::answer::{Answer, Response};
use kime::request::{Limits, Request, parse};
use kime::{Device, Kime, Precision};
use serde_json::{Value, json};

const USAGE: &str = "usage: kime predict [--model laya] --state <text or @file> --questions <@file>
       kime predict [--model laya] --request <@file>
       kime predict [--model laya] --batch < requests.jsonl > responses.jsonl
options: --format json|table  --device auto|cpu|cuda[:N]  --threads N  --precision f16|f32|int8
A value starting with @ is read from that file, and @- from stdin.";

/// Requests handed to the engine at once in `--batch` mode.
const CHUNK: usize = 512;

struct Opts {
    model: String,
    state: Option<String>,
    questions: Option<String>,
    request: Option<String>,
    batch: bool,
    table: bool,
    device: Device,
    precision: Precision,
}

fn opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        model: "laya".into(),
        state: None,
        questions: None,
        request: None,
        batch: false,
        table: false,
        device: Device::Auto,
        precision: Precision::F16,
    };
    let mut threads = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--model" => o.model = val()?,
            "--state" => o.state = Some(val()?),
            "--questions" => o.questions = Some(val()?),
            "--request" => o.request = Some(val()?),
            "--batch" => o.batch = true,
            "--format" => {
                o.table = match val()?.as_str() {
                    "table" => true,
                    "json" => false,
                    f => return Err(format!("unknown format {f:?}")),
                }
            }
            "--device" => o.device = device(&val()?)?,
            "--threads" => {
                threads = Some(val()?.parse::<usize>().map_err(|e| format!("--threads: {e}"))?);
            }
            "--precision" => o.precision = precision(&val()?)?,
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    if let Some(t) = threads {
        match o.device {
            Device::Auto | Device::Cpu { .. } => o.device = Device::Cpu { threads: t },
            _ => return Err("--threads is for the CPU".into()),
        }
    }
    Ok(o)
}

/// A `--device` value.
pub(crate) fn device(d: &str) -> Result<Device, String> {
    Ok(match d {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu { threads: 0 },
        "cuda" => Device::Cuda(0),
        d => match d.strip_prefix("cuda:").and_then(|n| n.parse().ok()) {
            Some(n) => Device::Cuda(n),
            None => return Err(format!("unknown device {d:?}")),
        },
    })
}

/// A `--precision` value.
pub(crate) fn precision(p: &str) -> Result<Precision, String> {
    Ok(match p {
        "f16" | "fp16" => Precision::F16,
        "f32" | "fp32" => Precision::F32,
        "int8" => Precision::Int8,
        p => return Err(format!("unknown precision {p:?}")),
    })
}

/// The text of `@file`, `@-` for stdin, or the argument itself.
fn read(arg: &str) -> Result<String, String> {
    match arg.strip_prefix('@') {
        Some("-") => std::io::read_to_string(std::io::stdin()).map_err(|e| format!("stdin: {e}")),
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}")),
        None => Ok(arg.to_string()),
    }
}

fn json(text: &str, what: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|e| format!("{what} is not JSON: {e}"))
}

/// The one request the options describe.
fn request(o: &Opts) -> Result<Request, String> {
    let body = match (&o.request, &o.state, &o.questions) {
        (Some(r), None, None) => json(&read(r)?, "the request")?,
        (None, Some(s), Some(q)) => {
            // A state read from a file is JSON when it parses as JSON, and text otherwise.
            let text = read(s)?;
            let state = match s.starts_with('@') {
                true => serde_json::from_str(&text).unwrap_or(Value::String(text)),
                false => Value::String(text),
            };
            json!({"state": state, "questions": json(&read(q)?, "the questions")?})
        }
        _ => return Err(USAGE.into()),
    };
    parse(&body, &Limits::LAYA).map_err(|p| problems(&p))
}

fn problems(p: &[kime::request::Problem]) -> String {
    let p: Vec<String> = p.iter().map(|p| p.to_json().to_string()).collect();
    format!("invalid request: {}", p.join(", "))
}

pub(crate) fn run(args: &[String]) -> ExitCode {
    match predict(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime predict: {e}");
            ExitCode::FAILURE
        }
    }
}

fn predict(args: &[String]) -> Result<(), String> {
    let o = opts(args)?;
    let single = if o.batch { None } else { Some(request(&o)?) };
    let t = Instant::now();
    let kime = Kime::builder()
        .model(&o.model)
        .device(o.device)
        .precision(o.precision)
        .build()
        .map_err(|e| e.to_string())?;
    eprintln!("{} on {}, loaded in {:.2?}", kime.model_id(), kime.device(), t.elapsed());
    let mut out = std::io::stdout().lock();
    if let Some(req) = single {
        let t = Instant::now();
        let res = kime.decide(&req).map_err(|e| e.to_string())?;
        eprintln!("answered in {:.2?}", t.elapsed());
        let text = if o.table {
            table(&res)
        } else {
            serde_json::to_string_pretty(&res.to_json()).unwrap_or_default() + "\n"
        };
        return out.write_all(text.as_bytes()).map_err(|e| e.to_string());
    }
    batch(&kime, &mut out)
}

/// JSONL in, JSONL out, in order. A line that does not parse or validate gets
/// `{"error": ...}` on its output line and the rest go on.
fn batch(kime: &Kime, out: &mut impl Write) -> Result<(), String> {
    let t = Instant::now();
    let (mut lines, mut n, mut tokens) = (std::io::stdin().lock().lines(), 0, 0);
    loop {
        let mut chunk: Vec<Result<Request, String>> = Vec::with_capacity(CHUNK);
        for line in lines.by_ref() {
            let line = line.map_err(|e| format!("stdin: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            chunk.push(
                json(&line, "the line")
                    .and_then(|b| parse(&b, &Limits::LAYA).map_err(|p| problems(&p))),
            );
            if chunk.len() == CHUNK {
                break;
            }
        }
        if chunk.is_empty() {
            break;
        }
        let good: Vec<Request> = chunk.iter().filter_map(|r| r.as_ref().ok().cloned()).collect();
        let mut answers = kime.decide_batch(&good).map_err(|e| e.to_string())?.into_iter();
        for r in &chunk {
            let v = match r {
                Ok(_) => {
                    let res =
                        answers.next().unwrap_or_else(|| unreachable!("one response per request"));
                    tokens += res.input_tokens;
                    res.to_json()
                }
                Err(e) => json!({"error": e}),
            };
            writeln!(out, "{v}").map_err(|e| e.to_string())?;
        }
        n += chunk.len();
    }
    let s = t.elapsed().as_secs_f64();
    eprintln!("{n} requests, {tokens} tokens in {s:.2} s, {:.1} requests/s", n as f64 / s);
    Ok(())
}

fn table(res: &Response) -> String {
    let mut rows =
        vec![["question".to_string(), "answer".into(), "confidence".into(), "act".into()]];
    for (id, a) in &res.answers {
        let (answer, conf, act) = match a {
            Answer::Choice { choice, confidence, act_probability, .. } => {
                (choice.clone(), *confidence, *act_probability)
            }
            Answer::Score { score, confidence, act_probability, .. } => {
                (format!("{score:.4}"), *confidence, *act_probability)
            }
            Answer::Noul { noul, confidence, act_probability } => {
                (format!("{noul:.4}"), *confidence, *act_probability)
            }
        };
        rows.push([id.clone(), answer, format!("{conf:.4}"), format!("{act:.4}")]);
    }
    let w: Vec<usize> =
        (0..4).map(|i| rows.iter().map(|r| r[i].chars().count()).max().unwrap_or(0)).collect();
    let mut s = String::new();
    for r in rows {
        let line = format!(
            "{:<a$}  {:<b$}  {:>c$}  {:>d$}",
            r[0],
            r[1],
            r[2],
            r[3],
            a = w[0],
            b = w[1],
            c = w[2],
            d = w[3]
        );
        s.push_str(line.trim_end());
        s.push('\n');
    }
    s
}

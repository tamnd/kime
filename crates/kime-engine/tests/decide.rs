//! The whole path against Laya: every parity case goes through [`Kime::decide_batch`] on the CPU,
//! all of them in shared batches, and each response is compared with the one `Agent.system_one`
//! returned. Answers are rounded to 4 places, so a logit a few ulps away from Laya's can move the
//! last digit. The test counts those and fails on anything bigger or on any other difference.
//!
//! It needs the Laya checkpoints in the Hugging Face cache (`kime pull laya`) and passes with a note
//! when they are missing unless `KIME_REQUIRE_WEIGHTS` is set.

use kime_core::request::{Limits, Request, parse};
use kime_engine::{Device, Kime};
use serde_json::Value;

fn lines(name: &str) -> Vec<Value> {
    let path = format!("{}/../kime-eval/fixtures/parity/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// The largest difference between two answers of the same shape, or None when the shapes or any
/// non number differ.
fn diff(a: &Value, b: &Value) -> Option<f64> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Some((x.as_f64()? - y.as_f64()?).abs()),
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            x.iter().zip(y).try_fold(0.0, |m: f64, (x, y)| Some(m.max(diff(x, y)?)))
        }
        (Value::Object(x), Value::Object(y)) if x.len() == y.len() => {
            x.iter().try_fold(0.0, |m: f64, (k, x)| Some(m.max(diff(x, y.get(k)?)?)))
        }
        _ => (a == b).then_some(0.0),
    }
}

fn check(model: &str) {
    let threads = std::env::var("KIME_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(0);
    let kime = match Kime::builder().model(model).device(Device::Cpu { threads }).build() {
        Ok(k) => k,
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping {model}: {e}");
            return;
        }
    };
    let (cases, dumps) = (lines("cases.jsonl"), lines(&format!("{model}.jsonl")));
    let mut reqs: Vec<Request> = Vec::new();
    let mut wants = Vec::new();
    for (case, dump) in cases.iter().zip(&dumps) {
        let Some(want) = dump.get("answer").filter(|a| a.is_object()) else { continue };
        reqs.push(parse(case, &Limits::LAYA).unwrap());
        wants.push((case["id"].clone(), want.clone()));
    }
    let t = std::time::Instant::now();
    let got = kime.decide_batch(&reqs).unwrap();
    let took = t.elapsed();
    let (mut same, mut last_digit, mut bad) = (0, 0, Vec::new());
    for (res, (id, want)) in got.iter().zip(&wants) {
        let res = res.to_json();
        match diff(&res, want) {
            Some(0.0) => same += 1,
            Some(d) if d <= 1.5e-4 => last_digit += 1,
            _ => bad.push(format!("{id}\n  got  {res}\n  want {want}")),
        }
    }
    eprintln!(
        "{model} on {}: {} requests in {took:?}, {same} equal, {last_digit} off in the 4th place, {} wrong",
        kime.device(),
        reqs.len(),
        bad.len()
    );
    for b in bad.iter().take(5) {
        eprintln!("{b}");
    }
    assert!(bad.is_empty());
    // One at a time gives the same bits as in the big batch.
    for (r, g) in reqs.iter().zip(&got).take(20) {
        assert_eq!(&kime.decide(r).unwrap(), g);
    }
}

#[test]
fn laya_english() {
    check("laya");
}

#[test]
fn laya_multilingual() {
    check("laya-multilingual");
}

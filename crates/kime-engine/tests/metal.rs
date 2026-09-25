//! The laya checkpoint on the Apple GPU gives the CPU's answers: FP32 to 1e-4 and FP16 to 1e-2
//! on every probability, and `Device::Auto` picks the GPU on a Mac.
//!
//! It needs the laya checkpoint, under `$KIME_MODELS/laya` or in the Hugging Face cache, and
//! passes with a note when it is missing unless `KIME_REQUIRE_WEIGHTS` is set.
#![cfg(target_os = "macos")]

use kime_core::request::{Limits, parse};
use kime_engine::{Device, Kime, Precision};
use serde_json::{Value, json};

fn open(device: Device, precision: Precision) -> Option<Kime> {
    let model =
        std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |m| format!("{m}/laya"));
    match Kime::builder().model(model).device(device).precision(precision).build() {
        Ok(k) => Some(k),
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// Every number in the answers, in order.
fn numbers(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Number(n) => out.push(n.as_f64().unwrap()),
        Value::Array(a) => a.iter().for_each(|x| numbers(x, out)),
        Value::Object(o) => o.values().for_each(|x| numbers(x, out)),
        _ => {}
    }
}

fn answers(k: &Kime) -> Vec<f64> {
    let q = json!({
        "topic": {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"cancel": "", "refund": "", "upgrade": ""}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
            "criteria": ["low", "medium", "high"]}});
    let states = [
        "I was charged twice for my subscription, please refund me",
        "Where is my parcel? It is three weeks late.",
        "Thanks, the new plan works great, can I add two more seats?",
    ];
    let reqs: Vec<_> = states
        .iter()
        .map(|s| parse(&json!({"state": s, "questions": q}), &Limits::JEV).unwrap())
        .collect();
    let mut out = Vec::new();
    for r in k.decide_batch(&reqs).unwrap() {
        numbers(&r.to_json()["answers"], &mut out);
    }
    out
}

#[test]
fn metal_matches_the_cpu() {
    let Some(cpu) = open(Device::Cpu { threads: 0 }, Precision::F32) else { return };
    let want = answers(&cpu);
    for (p, tol) in [(Precision::F32, 1e-4), (Precision::F16, 1e-2)] {
        let k = open(Device::Metal, p).unwrap();
        assert!(k.device().starts_with("metal"), "{}", k.device());
        let got = answers(&k);
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst <= tol, "{p:?}: off by {worst}");
    }
    let auto = open(Device::Auto, Precision::F16).unwrap();
    assert!(auto.device().starts_with("metal"), "{}", auto.device());
}

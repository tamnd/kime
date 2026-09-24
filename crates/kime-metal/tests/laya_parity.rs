//! The Metal compat graph against Laya's own outputs on the 200 parity cases, at both precisions.
//!
//! In FP32 the GPU is held to the CPU's bounds: argmax agrees on every question, the logits are
//! within 1e-3 and the probabilities within 1e-4.
//!
//! In FP16 it gets the CUDA backend's bounds: probabilities within 6e-3, logits within 1.5e-1, and
//! argmax has to agree wherever Laya's top two logits are at least 1e-2 apart.
//!
//! The test needs macOS and the weights under `$KIME_MODELS/laya`, and passes with a note when
//! either is missing unless `KIME_REQUIRE_WEIGHTS` is set.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use kime_metal::{Precision, executor};
use kime_model::Model;
use kime_tensor::{BatchBuf, Outputs};
use serde_json::Value;

struct Question {
    ids: Vec<u32>,
    markers: Vec<u32>,
    qtype: u8,
    logits: Vec<f32>,
}

fn questions(name: &str) -> Vec<Question> {
    let path = format!("{}/../kime-eval/fixtures/parity/{name}.jsonl", env!("CARGO_MANIFEST_DIR"));
    let nums = |v: &Value| -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    };
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        for q in case["questions"].as_array().into_iter().flatten() {
            out.push(Question {
                ids: nums(&q["ids"]).iter().map(|&x| x as u32).collect(),
                markers: nums(&q["markers"]).iter().map(|&x| x as u32).collect(),
                qtype: q["qtype"].as_u64().unwrap() as u8,
                logits: nums(&q["logits"]).iter().map(|&x| x as f32).collect(),
            });
        }
    }
    out
}

fn softmax(l: &[f32]) -> Vec<f64> {
    let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = l.iter().map(|&x| f64::from(x - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn argmax(l: &[f32]) -> Option<usize> {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a)))
}

fn check(name: &str, sub: &str, precision: Precision) {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya").join(sub));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights for {name}");
        eprintln!("skipping {name}: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let model = Model::open(&dir).unwrap();
    let mut exec = match executor(&model, precision) {
        Ok(e) => e,
        Err(e) if std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none() => {
            eprintln!("skipping {name}: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    let qs = questions(name);
    let (mut logit_err, mut prob_err, mut agree, mut ties) = (0f64, 0f64, 0, 0);
    let (mut sum_err, mut n_logits) = (0f64, 0usize);
    for chunk in qs.chunks(16) {
        buf.clear();
        for q in chunk {
            buf.push(&q.ids, &q.markers, q.qtype);
        }
        exec.run(&buf.batch(), &mut out).unwrap();
        let mut at = 0;
        for q in chunk {
            let got = &out.logits[at..at + q.logits.len()];
            at += q.logits.len();
            for (a, b) in got.iter().zip(&q.logits) {
                logit_err = logit_err.max(f64::from((a - b).abs()));
                sum_err += f64::from((a - b).abs());
                n_logits += 1;
            }
            for (a, b) in softmax(got).iter().zip(softmax(&q.logits)) {
                prob_err = prob_err.max((a - b).abs());
            }
            let mut r = q.logits.clone();
            r.sort_by(|a, b| b.total_cmp(a));
            let margin = r.first().copied().unwrap_or(0.0) - r.get(1).copied().unwrap_or(0.0);
            if argmax(got) == argmax(&q.logits) {
                agree += 1;
            } else if precision == Precision::F16 && margin < 1e-2 {
                ties += 1;
                eprintln!("{name}: argmax differs on a near tie, margin {margin:.2e}");
            } else {
                eprintln!("{name}: argmax differs, {got:?} against {:?}", q.logits);
            }
        }
        assert_eq!(at, out.logits.len());
    }
    eprintln!(
        "{name} {precision:?}: {} questions, argmax agrees on {agree}, max logit error {logit_err:.2e} mean {:.2e}, max probability error {prob_err:.2e}",
        qs.len(),
        sum_err / n_logits.max(1) as f64
    );
    assert_eq!(agree + ties, qs.len(), "{name}: argmax");
    let (logit, prob) = match precision {
        Precision::F32 => (1e-3, 1e-4),
        Precision::F16 => (1.5e-1, 6e-3),
    };
    assert!(prob_err < prob, "{name}: probability error {prob_err}");
    assert!(logit_err < logit, "{name}: logit error {logit_err}");
}

#[test]
fn laya_english_f32() {
    check("laya", "", Precision::F32);
}

#[test]
fn laya_multilingual_f32() {
    check("laya-multilingual", "multilingual", Precision::F32);
}

#[test]
fn laya_english_f16() {
    check("laya", "", Precision::F16);
}

#[test]
fn laya_multilingual_f16() {
    check("laya-multilingual", "multilingual", Precision::F16);
}

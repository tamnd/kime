//! The determinism test from spec/15-testing.md on the Laya weights: every parity question alone,
//! inside 20 random batches of other questions, and alone again right after an unrelated batch,
//! has to give the same logits and act logits bit for bit. When one does not, the test runs the
//! question again alone and in its batch with every step's output kept, and names the first step
//! whose rows for that question differ.
//!
//! It reads the weights from `$KIME_MODELS/laya` like `laya_parity` and passes with a note when
//! they are missing unless `KIME_REQUIRE_WEIGHTS` is set. Run it with
//! `cargo test --release -p kime-cpu --test determinism`.

mod common;

use std::path::PathBuf;

use common::Rng;
use kime_cpu::{CpuBackend, executor, par};
use kime_model::Model;
use kime_tensor::{BatchBuf, Executor, Outputs, Rows};
use serde_json::Value;

type Q = (Vec<u32>, Vec<u32>, u8);

fn questions() -> Vec<Q> {
    let path = format!("{}/../kime-eval/fixtures/parity/laya.jsonl", env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        for q in case["questions"].as_array().into_iter().flatten() {
            let ids = |k: &str| -> Vec<u32> {
                q[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
            };
            out.push((ids("ids"), ids("markers"), q["qtype"].as_u64().unwrap() as u8));
        }
    }
    out
}

/// Each question's logits and act logits as bits, in batch order.
fn run(exec: &mut Executor<CpuBackend>, qs: &[&Q]) -> Vec<(Vec<u32>, [u32; 2])> {
    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    for q in qs {
        buf.push(&q.0, &q.1, q.2);
    }
    exec.run(&buf.batch(), &mut out).unwrap();
    let mut at = 0;
    qs.iter()
        .zip(&out.act)
        .map(|(q, a)| {
            let l = out.logits[at..at + q.1.len()].iter().map(|x| x.to_bits()).collect();
            at += q.1.len();
            (l, a.map(f32::to_bits))
        })
        .collect()
}

/// Every step's rows for question `at` of `qs`, from one run with dumps on.
fn steps(exec: &mut Executor<CpuBackend>, qs: &[&Q], at: usize) -> Vec<(&'static str, Vec<f32>)> {
    for p in exec.plans_mut() {
        p.dump();
    }
    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    for q in qs {
        buf.push(&q.0, &q.1, q.2);
    }
    let bucket = exec.run(&buf.batch(), &mut out).unwrap();
    let plan = exec.plans_mut().find(|p| p.bucket() == bucket).unwrap();
    let tok: usize = qs[..at].iter().map(|q| q.0.len()).sum();
    let mark: usize = qs[..at].iter().map(|q| q.1.len()).sum();
    let (q, mut kept) = (qs[at], Vec::new());
    for d in plan.dumps() {
        let (lo, n) = match d.rows {
            Rows::Tokens => (tok, q.0.len()),
            Rows::Seqs => (at, 1),
            Rows::Markers => (mark, q.1.len()),
        };
        kept.push((d.name, d.data[lo * d.width..(lo + n) * d.width].to_vec()));
    }
    kept
}

fn explain(exec: &mut Executor<CpuBackend>, batch: &[&Q], at: usize) -> String {
    let alone = steps(exec, &batch[at..=at], 0);
    let inside = steps(exec, batch, at);
    for (i, (a, b)) in alone.iter().zip(&inside).enumerate() {
        let diff = a.1.iter().zip(&b.1).position(|(x, y)| x.to_bits() != y.to_bits());
        if let Some(j) = diff {
            return format!(
                "step {i} ({}) first differs at value {j}: {} alone, {} in the batch",
                a.0, a.1[j], b.1[j]
            );
        }
    }
    "no step differs when run again, so the difference is not repeatable".into()
}

#[test]
fn same_bits_alone_in_batches_and_after_other_work() {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya"));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no Laya weights");
        eprintln!("skipping: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let model = Model::open(&dir).unwrap();
    let mut exec = executor(&model, par::available()).unwrap();
    let qs = questions();
    let refs: Vec<&Q> = qs.iter().collect();
    let alone: Vec<_> = refs.iter().map(|q| run(&mut exec, &[q]).remove(0)).collect();

    let mut rng = Rng(16);
    let mut order: Vec<usize> = (0..qs.len()).collect();
    for round in 0..20 {
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut start = 0;
        while start < order.len() {
            let end = (start + 1 + rng.below(32)).min(order.len());
            let batch: Vec<&Q> = order[start..end].iter().map(|&i| refs[i]).collect();
            for (k, got) in run(&mut exec, &batch).iter().enumerate() {
                let i = order[start + k];
                if *got != alone[i] {
                    let why = explain(&mut exec, &batch, k);
                    panic!(
                        "round {round}: question {i} differs in a batch of {}: {why}",
                        batch.len()
                    );
                }
            }
            start = end;
        }
    }

    // After an unrelated batch has run in the same plan, alone again.
    for i in 0..qs.len() {
        let other: Vec<&Q> = (0..1 + rng.below(16)).map(|_| refs[rng.below(qs.len())]).collect();
        run(&mut exec, &other);
        let got = run(&mut exec, &[refs[i]]).remove(0);
        assert!(got == alone[i], "question {i} differs after other work");
    }
}

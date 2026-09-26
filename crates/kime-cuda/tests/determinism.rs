//! The determinism test from spec/15-testing.md on the GPU, in FP32 and FP16: every Laya parity
//! question alone, inside 20 random batches of other questions, and alone again right after an
//! unrelated batch, has to give the same logits and act logits bit for bit. It is the CPU's
//! `determinism` test without the per step dumps, which the CUDA plans do not keep, so a failure
//! names the question, the batch and the largest logit difference.
//!
//! It needs a GPU and the weights under `$KIME_MODELS/laya`, and passes with a note when either is
//! missing unless `KIME_REQUIRE_WEIGHTS` is set. Run it with
//! `cargo test --release -p kime-cuda --test determinism`.

mod common;

use std::path::PathBuf;

use common::Rng;
use kime_cuda::{CudaBackend, Precision, executor};
use kime_model::Model;
use kime_tensor::{BatchBuf, Executor, Outputs};
use serde_json::Value;

type Q = (Vec<u32>, Vec<u32>, u8);
type Bits = (Vec<u32>, [u32; 2]);

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
fn run(exec: &mut Executor<CudaBackend>, qs: &[&Q]) -> Vec<Bits> {
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

fn gap(a: &Bits, b: &Bits) -> f32 {
    a.0.iter()
        .chain(&a.1)
        .zip(b.0.iter().chain(&b.1))
        .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
        .fold(0.0, f32::max)
}

fn check(precision: Precision) {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya"));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no Laya weights");
        eprintln!("skipping: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let model = Model::open(&dir).unwrap();
    let mut exec = match executor(&model, 0, precision) {
        Ok(e) => e,
        Err(e) if std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none() => {
            eprintln!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let qs = questions();
    let refs: Vec<&Q> = qs.iter().collect();
    let alone: Vec<_> = refs.iter().map(|q| run(&mut exec, &[q]).remove(0)).collect();

    let mut rng = Rng(16);
    let mut order: Vec<usize> = (0..qs.len()).collect();
    let mut batches = 0;
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
                assert!(
                    *got == alone[i],
                    "{precision:?} round {round}: question {i} differs in a batch of {}, by up to {:.2e}",
                    batch.len(),
                    gap(got, &alone[i])
                );
            }
            batches += 1;
            start = end;
        }
    }

    // After an unrelated batch has run in the same plan, alone again.
    for i in 0..qs.len() {
        let other: Vec<&Q> = (0..1 + rng.below(16)).map(|_| refs[rng.below(qs.len())]).collect();
        run(&mut exec, &other);
        let got = run(&mut exec, &[refs[i]]).remove(0);
        assert!(
            got == alone[i],
            "{precision:?}: question {i} differs after other work, by up to {:.2e}",
            gap(&got, &alone[i])
        );
    }
    eprintln!(
        "{precision:?}: {} questions, the same bits alone, in {batches} random batches and after other work",
        qs.len()
    );
}

#[test]
fn same_bits_alone_in_batches_and_after_other_work_f32() {
    check(Precision::F32);
}

#[test]
fn same_bits_alone_in_batches_and_after_other_work_f16() {
    check(Precision::F16);
}

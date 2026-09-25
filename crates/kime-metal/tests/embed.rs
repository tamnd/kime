//! The pooled embedding graph on the GPU against the CPU, on the token ids of the first 64 parity
//! questions, in batches of 16. In FP32 every row has a cosine of at least 0.99999 to the CPU's,
//! and in FP16 at least 0.999.
//!
//! The test needs macOS and the weights under `$KIME_MODELS/laya`, and passes with a note when
//! either is missing unless `KIME_REQUIRE_WEIGHTS` is set.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use kime_metal::{Precision, executor};
use kime_model::Model;
use kime_tensor::{BatchBuf, Buckets, Outputs};
use serde_json::Value;

fn sequences() -> Vec<Vec<u32>> {
    let path = format!("{}/../kime-eval/fixtures/parity/laya.jsonl", env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        for q in case["questions"].as_array().into_iter().flatten() {
            out.push(
                q["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
            );
        }
    }
    out.truncate(64);
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
    let nb: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    dot / (na * nb).sqrt()
}

#[test]
fn matches_the_cpu() {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya"));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights");
        eprintln!("skipping: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let model = Model::open(&dir).unwrap();
    let graph = || model.graph.embed_plan(&model.spec);
    let buckets = Buckets::default();
    let mut cpu = kime_cpu::executor(&model, 0).unwrap();
    let cpu_lane = cpu.add_graph(graph(), &buckets, "state");
    let seqs = sequences();
    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    let mut want = Vec::new();
    for chunk in seqs.chunks(16) {
        buf.clear();
        for s in chunk {
            buf.push(s, &[], 0);
        }
        cpu.run_lane(cpu_lane, &buf.batch(), &mut out).unwrap();
        want.extend_from_slice(&out.pooled);
    }
    let d = model.spec.encoder.d;
    for (precision, bound) in [(Precision::F32, 0.99999), (Precision::F16, 0.999)] {
        let mut gpu = match executor(&model, precision) {
            Ok(e) => e,
            Err(e) if std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none() => {
                eprintln!("skipping: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        let lane = gpu.add_graph(graph(), &buckets, "state");
        let mut got = Vec::new();
        for chunk in seqs.chunks(16) {
            buf.clear();
            for s in chunk {
                buf.push(s, &[], 0);
            }
            gpu.run_lane(lane, &buf.batch(), &mut out).unwrap();
            assert_eq!(out.pooled.len(), chunk.len() * d);
            got.extend_from_slice(&out.pooled);
        }
        let worst =
            got.chunks(d).zip(want.chunks(d)).map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);
        eprintln!("{precision:?}: {} sequences, worst cosine to the CPU {worst:.7}", seqs.len());
        assert!(worst > bound, "{precision:?}: worst cosine {worst}");
    }
}

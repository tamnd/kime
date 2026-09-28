//! The Burn model against Laya's own logits on the parity cases, with Laya's weights loaded.
//!
//! This is what makes training start from the model kime serves: the same weights through the
//! Burn graph have to give Laya's logits, up to how far Laya is from itself (9.4e-5 in the logits
//! between two thread counts, see kime-cpu's test of the same name) plus what a GPU's sums add.
//! The weights are not in CI, so the test reads them from `$KIME_MODELS/laya` and passes with a
//! note when they are missing unless `KIME_REQUIRE_WEIGHTS` is set. Run it with
//! `cargo test --release -p kime-train --no-default-features --features metal --test laya_parity`. The
//! ndarray backend runs it on one core and takes over half an hour.

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::{Tensor, TensorData};
use kime_cpu::compat::act_features;
use kime_model::Model;
use kime_train::Base;
use kime_train::model::{Batch, Compat, Question};
use serde_json::Value;

struct Case {
    name: String,
    q: Question,
    logits: Vec<f32>,
    act: [f32; 2],
}

fn cases(name: &str) -> Vec<Case> {
    let path = format!("{}/../kime-eval/fixtures/parity/{name}.jsonl", env!("CARGO_MANIFEST_DIR"));
    let nums = |v: &Value| -> Vec<f64> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    };
    let mut out = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        for q in case["questions"].as_array().into_iter().flatten() {
            let act = nums(&q["act_probs"]);
            out.push(Case {
                name: format!("{}/{}", case["id"].as_str().unwrap(), q["qid"].as_str().unwrap()),
                q: Question {
                    ids: nums(&q["ids"]).iter().map(|&x| x as u32).collect(),
                    markers: nums(&q["markers"]).iter().map(|&x| x as u32).collect(),
                    qtype: q["qtype"].as_u64().unwrap() as usize,
                },
                logits: nums(&q["logits"]).iter().map(|&x| x as f32).collect(),
                act: [act[0] as f32, act[1] as f32],
            });
        }
    }
    out
}

fn argmax(l: &[f32]) -> Option<usize> {
    (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a)))
}

fn softmax(l: &[f32]) -> Vec<f64> {
    let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = l.iter().map(|&x| f64::from(x - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn check(name: &str, sub: &str) {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya").join(sub));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights for {name}");
        eprintln!("skipping {name}: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let model = Model::open(&dir).unwrap();
    let dev = Default::default();
    let net = Compat::<Base>::load(&model.spec, &model.tensors, &dev);
    let cs = cases(name);
    let (mut logit_err, mut prob_err, mut act_err, mut agree) = (0f64, 0f64, 0f64, 0);
    let start = Instant::now();
    for chunk in cs.chunks(16) {
        let qs: Vec<Question> = chunk.iter().map(|c| c.q.clone()).collect();
        let batch = Batch::<Base>::new(&qs, net.shape(), &dev);
        let (z, h) = net.forward(&batch);
        let z = z.into_data().to_vec::<f32>().unwrap();
        let mut feats = Vec::new();
        let mut at = 0;
        let mut got = Vec::new();
        for c in chunk {
            let l = z[at..at + c.logits.len()].to_vec();
            at += c.logits.len();
            feats.extend_from_slice(&act_features(&l));
            got.push(l);
        }
        let feats = Tensor::from_data(TensorData::new(feats, [chunk.len(), 4]), &dev);
        let act = net.act(&batch, h, feats).into_data().to_vec::<f32>().unwrap();
        for (i, (c, l)) in chunk.iter().zip(&got).enumerate() {
            for (a, b) in l.iter().zip(&c.logits) {
                logit_err = logit_err.max(f64::from((a - b).abs()));
            }
            for (a, b) in softmax(l).iter().zip(softmax(&c.logits)) {
                prob_err = prob_err.max((a - b).abs());
            }
            for (a, b) in act[2 * i..2 * i + 2].iter().zip(&c.act) {
                act_err = act_err.max(f64::from((a - b).abs() / b.abs().max(1.0)));
            }
            if argmax(l) == argmax(&c.logits) {
                agree += 1;
            } else {
                eprintln!("{name} {}: argmax differs, {l:?} against {:?}", c.name, c.logits);
            }
        }
    }
    eprintln!(
        "{name}: {} questions in {:.1}s, argmax agrees on {agree}, max logit error {logit_err:.2e}, max probability error {prob_err:.2e}, max act error {act_err:.2e} relative",
        cs.len(),
        start.elapsed().as_secs_f64()
    );
    assert_eq!(agree, cs.len());
    assert!(logit_err < 1e-3, "{name}: logit error {logit_err}");
    assert!(prob_err < 1e-4, "{name}: probability error {prob_err}");
    assert!(act_err < 1e-3, "{name}: act error {act_err}");
}

#[test]
fn laya_english() {
    check("laya", "");
}

#[test]
fn laya_multilingual() {
    check("laya-multilingual", "multilingual");
}

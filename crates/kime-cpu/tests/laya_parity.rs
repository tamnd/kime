//! The FP32 reference forward pass against Laya's own on the 200 parity cases.
//!
//! The fixtures hold the token ids, marker positions, option logits and act logits that Laya
//! 0.3.7 produced on the CPU in FP32 (`tools/ref/laya_ref.py`). The weights are 1.5 GB and not in
//! CI, so the test reads them from `$KIME_MODELS/laya`, and passes with a note when they are
//! missing unless `KIME_REQUIRE_WEIGHTS` is set. It is slow without optimizations, so run it with
//! `cargo test --release -p kime-cpu --test laya_parity`.
//!
//! The bounds come from how far Laya is from itself. The same Laya run with 8 threads instead of
//! the default 32 on an i9-13900K moves the English logits by up to 9.4e-5 and the probabilities
//! by up to 1.2e-5, because MKL splits the sums differently. kime cannot be held closer to Laya
//! than Laya is to itself, so the bounds sit a little above that, and argmax has to agree on every
//! question. The plan executor has to match the reference bit for bit on every chunk.

use std::path::PathBuf;

use kime_cpu::{Compat, Input, executor, par};
use kime_model::Model;
use kime_tensor::{BatchBuf, Outputs};
use serde_json::Value;

struct Question {
    case: String,
    ids: Vec<u32>,
    markers: Vec<u32>,
    qtype: usize,
    logits: Vec<f32>,
    act: [f32; 2],
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
            let act = nums(&q["act_probs"]);
            out.push(Question {
                case: format!("{}/{}", case["id"].as_str().unwrap(), q["qid"].as_str().unwrap()),
                ids: nums(&q["ids"]).iter().map(|&x| x as u32).collect(),
                markers: nums(&q["markers"]).iter().map(|&x| x as u32).collect(),
                qtype: q["qtype"].as_u64().unwrap() as usize,
                logits: nums(&q["logits"]).iter().map(|&x| x as f32).collect(),
                act: [act[0] as f32, act[1] as f32],
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

fn check(name: &str, sub: &str) {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya").join(sub));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights for {name}");
        eprintln!("skipping {name}: set KIME_MODELS to a folder holding laya/ with its weights");
        return;
    };
    let model = Model::open(&dir).unwrap();
    let mut compat = Compat::new(&model, par::available());
    let mut exec = executor(&model, par::available()).unwrap();
    let (mut buf, mut plan) = (BatchBuf::default(), Outputs::default());
    let qs = questions(name);
    assert!(qs.len() >= 600, "{name}: only {} questions", qs.len());

    let (mut logit_err, mut prob_err, mut act_err) = (0f64, 0f64, 0f64);
    let (mut sum_err, mut n_logits) = (0f64, 0usize);
    let mut agree = 0;
    for chunk in qs.chunks(16) {
        let inputs: Vec<Input<'_>> = chunk
            .iter()
            .map(|q| Input { ids: &q.ids, markers: &q.markers, qtype: q.qtype })
            .collect();
        let outs = compat.forward(&inputs);
        // The plan executor runs the same kernels, so it has to give the same bits.
        buf.clear();
        for q in chunk {
            buf.push(&q.ids, &q.markers, q.qtype as u8);
        }
        exec.run(&buf.batch(), &mut plan).unwrap();
        let flat: Vec<u32> =
            outs.iter().flat_map(|o| o.logits.iter().map(|x| x.to_bits())).collect();
        assert_eq!(
            flat,
            plan.logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "plan logits"
        );
        for (o, a) in outs.iter().zip(&plan.act) {
            assert_eq!(o.act.map(f32::to_bits), a.map(f32::to_bits), "plan act");
        }
        for (q, o) in chunk.iter().zip(&outs) {
            assert_eq!(o.logits.len(), q.logits.len(), "{}", q.case);
            for (a, b) in o.logits.iter().zip(&q.logits) {
                logit_err = logit_err.max(f64::from((a - b).abs()));
                sum_err += f64::from((a - b).abs());
                n_logits += 1;
            }
            for (a, b) in softmax(&o.logits).iter().zip(softmax(&q.logits)) {
                prob_err = prob_err.max((a - b).abs());
            }
            // The act logits run to the thousands, so their error is relative.
            for (a, b) in o.act.iter().zip(&q.act) {
                act_err = act_err.max(f64::from((a - b).abs() / b.abs().max(1.0)));
            }
            if argmax(&o.logits) == argmax(&q.logits) {
                agree += 1;
            } else {
                eprintln!(
                    "{name} {}: argmax differs, {:?} against {:?}",
                    q.case, o.logits, q.logits
                );
            }
        }
    }
    eprintln!(
        "{name}: {} questions, argmax agrees on {agree}, logit error max {logit_err:.2e} mean {:.2e}, max probability error {prob_err:.2e}, max act error {act_err:.2e} relative",
        qs.len(),
        sum_err / n_logits.max(1) as f64
    );
    assert_eq!(agree, qs.len());

    // The same bits on one thread as on all of them.
    let few: Vec<Input<'_>> = qs[..6]
        .iter()
        .map(|q| Input { ids: &q.ids, markers: &q.markers, qtype: q.qtype })
        .collect();
    let many = compat.forward(&few);
    compat.set_threads(1);
    let one = compat.forward(&few);
    for (a, b) in many.iter().zip(&one) {
        assert!(a.logits.iter().zip(&b.logits).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(a.act.iter().zip(&b.act).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
    assert!(logit_err < 1.5e-4, "{name}: logit error {logit_err}");
    assert!(prob_err < 2e-5, "{name}: probability error {prob_err}");
    assert!(act_err < 5e-5, "{name}: act error {act_err}");
}

#[test]
fn laya_english() {
    check("laya", "");
}

#[test]
fn laya_multilingual() {
    check("laya-multilingual", "multilingual");
}

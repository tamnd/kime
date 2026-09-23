//! Every CUDA kernel against the FP32 reference forward pass, on a small random checkpoint with
//! two heads, a 16 token window and one global layer. Batches mix up to 12 questions of 0 to 70
//! tokens, so attention tiles cross sequence ends and window edges, and the rounds span several
//! buckets. Skipped, with a note, on machines without an NVIDIA GPU.

mod common;

use common::{Rng, fill, question, tiny};
use kime_cpu::{Compat, Input};
use kime_cuda::{CudaBackend, Precision};
use kime_tensor::{BatchBuf, Buckets, Executor, HostTensor, Outputs};

fn check(precision: Precision, logit_bound: f32) {
    let (spec, graph, tensors) = tiny();
    let backend = match CudaBackend::new(0, precision) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipped, no GPU: {e}");
            return;
        }
    };
    let host: Vec<HostTensor<'_>> = (0..tensors.entries().len())
        .map(|i| {
            let v = tensors.view(i);
            HostTensor { dtype: v.dtype, shape: v.shape, bytes: v.bytes }
        })
        .collect();
    let plan = graph.plan(&spec);
    let mut exec =
        Executor::new(backend, &host, plan, &Buckets::default(), "compat", spec.encoder.vocab, 3)
            .unwrap();
    let compat = Compat::from_parts(&spec, &graph, &tensors, 2);
    let mut rng = Rng(5);
    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    let (mut worst_logit, mut worst_act) = (0f32, 0f32);
    for round in 0..16 {
        let n = 1 + rng.below(12);
        let qs: Vec<_> = (0..n).map(|_| question(&mut rng, 70)).collect();
        fill(&mut buf, &qs);
        let bucket = exec.run(&buf.batch(), &mut out).unwrap();
        let inputs: Vec<Input<'_>> = qs
            .iter()
            .map(|q| Input { ids: &q.0, markers: &q.1, qtype: usize::from(q.2) })
            .collect();
        let want = compat.forward(&inputs);
        let mut at = 0;
        for (s, w) in want.iter().enumerate() {
            let got = &out.logits[at..at + w.logits.len()];
            at += w.logits.len();
            for (a, b) in got.iter().zip(&w.logits) {
                worst_logit = worst_logit.max((a - b).abs());
                assert!(
                    (a - b).abs() <= logit_bound,
                    "round {round} seq {s} in {bucket}: {got:?} against {:?}",
                    w.logits
                );
            }
            for (a, b) in out.act[s].iter().zip(&w.act) {
                worst_act = worst_act.max((a - b).abs());
                assert!((a - b).abs() <= logit_bound, "round {round} seq {s} act");
            }
        }
        assert_eq!(at, out.logits.len());
    }
    assert!(exec.warm().count() >= 2, "the rounds should span more than one bucket");
    eprintln!("{precision:?}: max logit error {worst_logit:.2e}, max act error {worst_act:.2e}");
}

#[test]
fn kernels_match_the_reference_in_f32() {
    check(Precision::F32, 1e-4);
}

#[test]
fn kernels_match_the_reference_in_f16() {
    check(Precision::F16, 2e-2);
}

//! The plan executor against the reference forward pass on a small random checkpoint: the same
//! bits for every batch, whatever else is in the batch and however many threads run it.

mod common;

use common::{Rng, fill, question, tiny};
use kime_cpu::{Compat, Input, executor_from};
use kime_tensor::{BatchBuf, Outputs};

#[test]
fn plan_matches_the_reference_bit_for_bit() {
    let (spec, graph, tensors) = tiny();
    let compat = Compat::from_parts(&spec, &graph, &tensors, 3);
    let mut exec = executor_from(&spec, &graph, &tensors, 4).unwrap();
    let mut one = executor_from(&spec, &graph, &tensors, 1).unwrap();
    let mut rng = Rng(3);
    let (mut buf, mut out, mut alone) =
        (BatchBuf::default(), Outputs::default(), Outputs::default());
    for round in 0..12 {
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
            let same = got.iter().zip(&w.logits).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "round {round} seq {s} in {bucket}: {got:?} against {:?}", w.logits);
            assert_eq!(
                out.act[s].map(f32::to_bits),
                w.act.map(f32::to_bits),
                "round {round} seq {s}"
            );
        }
        assert_eq!(at, out.logits.len());

        // Each question alone on one thread gives the bits it gave inside the batch.
        let mut at = 0;
        for (s, q) in qs.iter().enumerate() {
            fill(&mut buf, std::slice::from_ref(q));
            one.run(&buf.batch(), &mut alone).unwrap();
            let k = q.1.len();
            assert_eq!(alone.logits, out.logits[at..at + k], "round {round} seq {s} alone");
            assert_eq!(alone.act[0].map(f32::to_bits), out.act[s].map(f32::to_bits));
            at += k;
        }
    }
    assert!(exec.warm().count() >= 2, "the rounds should span more than one bucket");
}

#[test]
fn bad_batches_are_refused() {
    let (spec, graph, tensors) = tiny();
    let mut exec = executor_from(&spec, &graph, &tensors, 2).unwrap();
    let mut out = Outputs::default();
    let mut buf = BatchBuf::default();
    buf.push(&[1, 2, 99], &[0], 0);
    assert!(exec.run(&buf.batch(), &mut out).unwrap_err().to_string().contains("vocabulary"));
    buf.clear();
    buf.push(&[1, 2], &[2], 0);
    assert!(exec.run(&buf.batch(), &mut out).unwrap_err().to_string().contains("marker 2"));
    buf.clear();
    buf.push(&[1, 2], &[], 3);
    assert!(exec.run(&buf.batch(), &mut out).is_err());
    buf.clear();
    let long = vec![1u32; 20000];
    buf.push(&long, &[], 0);
    assert!(exec.run(&buf.batch(), &mut out).unwrap_err().to_string().contains("no compat bucket"));
    // An empty batch is fine and produces nothing.
    buf.clear();
    exec.run(&buf.batch(), &mut out).unwrap();
    assert!(out.logits.is_empty() && out.act.is_empty());
}

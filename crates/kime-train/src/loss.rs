//! The proper score loss of spec/12-training.md.
//!
//! For each question, with `p` the softmax of its option logits and `t` its target:
//!
//! ```text
//! loss = -( sum_i t_i log max(p_i, 1e-4) + 0.5 * (sum_i t_i p_i) / ||p||_2 )
//!        + RankedProbabilityScore(p, t)   for score questions
//! ```
//!
//! averaged over questions with their weights. Questions have different numbers of options, so
//! the logits are laid out `[questions, most options]` with padding that gets no probability.

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::data::Example;

/// The log score floor, Laya's `log_floor`.
pub const LOG_FLOOR: f64 = 1e-4;
/// The weight of the spherical score.
pub const W_SPHERICAL: f64 = 0.5;
/// The weight of the ranked probability score on score questions.
pub const W_ORDINAL: f64 = 1.0;

/// The targets of a batch, laid out to match its logits.
#[derive(Debug, Clone)]
pub struct Targets<B: Backend> {
    index: Tensor<B, 1, Int>,
    pad: Tensor<B, 2>,
    t: Tensor<B, 2>,
    weight: Tensor<B, 1>,
    ordinal: Tensor<B, 1>,
    upper: Tensor<B, 2>,
    /// Options per question.
    pub options: Vec<usize>,
}

impl<B: Backend> Targets<B> {
    /// The targets of `examples`, whose logits come flat and in order.
    ///
    /// # Panics
    ///
    /// If `examples` is empty.
    #[must_use]
    pub fn new(examples: &[&Example], dev: &B::Device) -> Self {
        assert!(!examples.is_empty(), "no examples");
        let n = examples.len();
        let k = examples.iter().map(|e| e.probs.len()).max().unwrap_or(1).max(1);
        let (mut index, mut pad, mut t) = (vec![0i64; n * k], vec![0f32; n * k], vec![0f32; n * k]);
        let mut ordinal = vec![0f32; n];
        let mut at = 0i64;
        for (s, e) in examples.iter().enumerate() {
            let m = e.probs.len();
            for j in 0..k {
                if j < m {
                    index[s * k + j] = at + j as i64;
                    t[s * k + j] = e.probs[j];
                } else {
                    pad[s * k + j] = -1e9;
                }
            }
            if e.q.qtype == 1 && m > 1 {
                ordinal[s] = (W_ORDINAL / (m - 1) as f64) as f32;
            }
            at += m as i64;
        }
        let total: f32 = examples.iter().map(|e| e.weight).sum();
        let weight: Vec<f32> = examples.iter().map(|e| e.weight / total.max(1e-12)).collect();
        let upper: Vec<f32> = (0..k * k).map(|x| if x / k <= x % k { 1.0 } else { 0.0 }).collect();
        Self {
            index: Tensor::from_data(TensorData::new(index, [n * k]), dev),
            pad: Tensor::from_data(TensorData::new(pad, [n, k]), dev),
            t: Tensor::from_data(TensorData::new(t, [n, k]), dev),
            weight: Tensor::from_data(TensorData::new(weight, [n]), dev),
            ordinal: Tensor::from_data(TensorData::new(ordinal, [n]), dev),
            upper: Tensor::from_data(TensorData::new(upper, [k, k]), dev),
            options: examples.iter().map(|e| e.probs.len()).collect(),
        }
    }
}

/// The weighted mean loss of a batch, `[1]`, and each question's probabilities, `[n, k]`, with
/// zeros past its options.
#[must_use]
pub fn proper_score<B: Backend>(
    logits: Tensor<B, 1>,
    t: &Targets<B>,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let [n, k] = t.pad.dims();
    let z = logits.select(0, t.index.clone()).reshape([n, k]) + t.pad.clone();
    let p = softmax(z, 1);
    let log = (t.t.clone() * p.clone().clamp_min(LOG_FLOOR).log()).sum_dim(1).reshape([n]);
    let norm = p.clone().powi_scalar(2).sum_dim(1).sqrt().reshape([n]);
    let sph = (t.t.clone() * p.clone()).sum_dim(1).reshape([n]) / norm;
    let cdf = (p.clone() - t.t.clone()).matmul(t.upper.clone());
    let rps = cdf.powi_scalar(2).sum_dim(1).reshape([n]) * t.ordinal.clone();
    let per = -(log + sph * W_SPHERICAL) + rps;
    ((per * t.weight.clone()).sum(), p)
}

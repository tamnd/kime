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
//!
//! Two other objectives are here for the ablations of #37: plain cross entropy against the soft
//! target, and `rlcd_pg`, the Gaussian policy gradient Laya's notebook trains with, which only
//! estimates the gradient of the proper score where [`proper_score`] has it exactly.

use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::data::Example;
use crate::rng::Rng;

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
    real: Vec<bool>,
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
        let mut real = vec![false; n * k];
        let mut at = 0i64;
        for (s, e) in examples.iter().enumerate() {
            let m = e.probs.len();
            for j in 0..k {
                if j < m {
                    index[s * k + j] = at + j as i64;
                    t[s * k + j] = e.probs[j];
                    real[s * k + j] = true;
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
            real,
            options: examples.iter().map(|e| e.probs.len()).collect(),
        }
    }
}

/// Which objective training minimizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Objective {
    /// The proper score, the default.
    Proper,
    /// Cross entropy against the soft target, with no floor and no spherical term.
    CrossEntropy,
    /// The policy gradient estimate of the proper score, with this many samples a question.
    Pg {
        /// Perturbations per question, G.
        samples: usize,
    },
}

impl Objective {
    /// The name the command line and the run notes use.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Proper => "proper".into(),
            Self::CrossEntropy => "ce".into(),
            Self::Pg { samples } => format!("pg{samples}"),
        }
    }
}

/// The option logits `[n, k]`, padding pushed to -1e9.
fn laid_out<B: Backend>(logits: Tensor<B, 1>, t: &Targets<B>) -> Tensor<B, 2> {
    let [n, k] = t.pad.dims();
    logits.select(0, t.index.clone()).reshape([n, k]) + t.pad.clone()
}

/// Each question's proper score loss, `[n]`, from its probabilities.
fn per_question<B: Backend>(p: Tensor<B, 2>, t: &Targets<B>) -> Tensor<B, 1> {
    let [n, _] = t.pad.dims();
    let log = (t.t.clone() * p.clone().clamp_min(LOG_FLOOR).log()).sum_dim(1).reshape([n]);
    let norm = p.clone().powi_scalar(2).sum_dim(1).sqrt().reshape([n]);
    let sph = (t.t.clone() * p.clone()).sum_dim(1).reshape([n]) / norm;
    let cdf = (p - t.t.clone()).matmul(t.upper.clone());
    let rps = cdf.powi_scalar(2).sum_dim(1).reshape([n]) * t.ordinal.clone();
    -(log + sph * W_SPHERICAL) + rps
}

/// The weighted mean loss of a batch, `[1]`, and each question's probabilities, `[n, k]`, with
/// zeros past its options.
#[must_use]
pub fn proper_score<B: Backend>(
    logits: Tensor<B, 1>,
    t: &Targets<B>,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let p = softmax(laid_out(logits, t), 1);
    ((per_question(p.clone(), t) * t.weight.clone()).sum(), p)
}

/// Cross entropy against the soft target, `-sum_i t_i log p_i`, weighted like [`proper_score`].
#[must_use]
pub fn cross_entropy<B: Backend>(
    logits: Tensor<B, 1>,
    t: &Targets<B>,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let [n, _] = t.pad.dims();
    let z = laid_out(logits, t);
    let p = softmax(z.clone(), 1);
    let per = -(t.t.clone() * log_softmax(z, 1)).sum_dim(1).reshape([n]);
    ((per * t.weight.clone()).sum(), p)
}

/// `rlcd_pg`: perturbs the logits `samples` times with Gaussian noise of width `sigma`, scores
/// each perturbation with the proper score, normalizes the rewards within each question's group,
/// and gives the surrogate whose gradient moves the logits towards the better perturbations,
/// `mean_g A_g ||z_g - z||^2 / (2 sigma^2)` with `z_g` held fixed. Its value is not a loss to
/// compare, only its gradient is used.
#[must_use]
pub fn rlcd_pg<B: Backend>(
    logits: Tensor<B, 1>,
    t: &Targets<B>,
    samples: usize,
    sigma: f64,
    rng: &mut Rng,
) -> (Tensor<B, 1>, Tensor<B, 2>) {
    let [n, k] = t.pad.dims();
    let dev = t.pad.device();
    let z = laid_out(logits, t);
    let p = softmax(z.clone(), 1);
    let fixed = z.clone().detach();
    let g = samples.max(2);
    let mut moved = Vec::with_capacity(g);
    let mut rewards = Vec::with_capacity(g);
    for _ in 0..g {
        let noise: Vec<f32> =
            t.real.iter().map(|&r| if r { (rng.normal() * sigma) as f32 } else { 0.0 }).collect();
        let zg = fixed.clone() + Tensor::from_data(TensorData::new(noise, [n, k]), &dev);
        rewards.push(-per_question(softmax(zg.clone(), 1), t).reshape([1, n]));
        moved.push(zg);
    }
    let r = Tensor::cat(rewards, 0);
    let mean = r.clone().mean_dim(0);
    let std = (r.clone() - mean.clone()).powi_scalar(2).mean_dim(0).sqrt();
    let adv = ((r - mean) / (std + 1e-6)).detach();
    let mut per: Option<Tensor<B, 1>> = None;
    for (i, zg) in moved.into_iter().enumerate() {
        let a = adv.clone().slice([i..i + 1, 0..n]).reshape([n]);
        let d = (zg - z.clone()).powi_scalar(2).sum_dim(1).reshape([n]);
        let term = a * d / (2.0 * sigma * sigma * g as f64);
        per = Some(match per {
            Some(s) => s + term,
            None => term,
        });
    }
    let per = per.unwrap_or_else(|| Tensor::zeros([n], &dev));
    ((per * t.weight.clone()).sum(), p)
}

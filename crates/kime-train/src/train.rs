//! The training loop: batches bucketed by length, the proper score loss, AdamW with warmup and
//! cosine decay, and clipping of the global gradient norm, as spec/12-training.md sets them.

use std::time::Instant;

use burn::module::{AutodiffModule, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor};

use crate::data::{Example, Renderer};
use crate::loss::{Objective, Targets, cross_entropy, proper_score, rlcd_pg};
use crate::model::{Batch, Compat, Question};
use crate::rng::Rng;

/// How a run trains.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Peak learning rate, 3e-5 for initialized weights.
    pub lr: f64,
    /// Where the cosine ends.
    pub min_lr: f64,
    /// Share of the steps spent warming up.
    pub warmup: f64,
    /// AdamW's decoupled weight decay.
    pub weight_decay: f64,
    /// AdamW's betas.
    pub betas: (f64, f64),
    /// AdamW's epsilon.
    pub eps: f64,
    /// The largest global gradient norm, 0 for none.
    pub clip: f64,
    /// Padded tokens per micro batch.
    pub tokens: usize,
    /// Micro batches per optimizer step.
    pub accum: usize,
    /// Sequences are padded to a multiple of this, so batches share shapes.
    pub bucket: usize,
    /// Passes over the data.
    pub epochs: usize,
    /// Stop after this many optimizer steps, when set.
    pub max_steps: Option<usize>,
    /// Seed of the batch order and the option shuffles.
    pub seed: u64,
    /// Shuffle the options of choice questions every epoch.
    pub shuffle_options: bool,
    /// Evaluate every this many steps, 0 for only at the end.
    pub eval_every: usize,
    /// What training minimizes. Evaluation always reports the proper score.
    pub objective: Objective,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lr: 3e-5,
            min_lr: 1e-6,
            warmup: 0.02,
            weight_decay: 0.01,
            betas: (0.9, 0.98),
            eps: 1e-8,
            clip: 1.0,
            tokens: 8192,
            accum: 1,
            bucket: 64,
            epochs: 1,
            max_steps: None,
            seed: 13,
            shuffle_options: true,
            eval_every: 0,
            objective: Objective::Proper,
        }
    }
}

/// The learning rate at `step` of `total`: linear warmup from 0, then cosine down to `min_lr`.
#[must_use]
pub fn lr_at(cfg: &Config, step: usize, total: usize) -> f64 {
    let warm = ((cfg.warmup * total as f64).round() as usize).max(1);
    if step < warm {
        return cfg.lr * (step + 1) as f64 / warm as f64;
    }
    let done = (step - warm) as f64 / (total - warm).max(1) as f64;
    cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + (std::f64::consts::PI * done.min(1.0)).cos())
}

/// The noise width of `rlcd_pg` at `step` of `total`, 1.0 falling linearly to 0.3 as spec/12-training.md
/// sets it.
#[must_use]
pub fn sigma_at(step: usize, total: usize) -> f64 {
    let done = step as f64 / total.saturating_sub(1).max(1) as f64;
    1.0 - 0.7 * done.min(1.0)
}

/// Micro batches for one epoch: each is a padded length and the examples in it. Examples of the
/// same padded length go together, as many as fit in `cfg.tokens`, and the batches come in a
/// random order.
#[must_use]
pub fn plan(lens: &[usize], cfg: &Config, rng: &mut Rng) -> Vec<(usize, Vec<usize>)> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    rng.shuffle(&mut order);
    let mut by_len: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in order {
        by_len.entry(lens[i].next_multiple_of(cfg.bucket.max(1))).or_default().push(i);
    }
    let mut out = Vec::new();
    for (t, idx) in by_len {
        let n = (cfg.tokens / t).max(1);
        out.extend(idx.chunks(n).map(|c| (t, c.to_vec())));
    }
    rng.shuffle(&mut out);
    out
}

/// Sums the squares of every gradient.
struct SquaredNorm<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    sum: Option<Tensor<B::InnerBackend, 1>>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for SquaredNorm<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let s = g.powi_scalar(2).sum();
            self.sum = Some(match self.sum.take() {
                Some(t) => t + s,
                None => s,
            });
        }
    }
}

/// Multiplies every gradient by a factor.
struct Scale<'a> {
    grads: &'a mut GradientsParams,
    by: f64,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads.register::<B::InnerBackend, D>(param.id, g * self.by);
        }
    }
}

/// Scales the gradients so their global norm is at most `max`, as PyTorch's `clip_grad_norm_`
/// does, and gives the norm before clipping.
pub fn clip_global<B: AutodiffBackend, M: AutodiffModule<B>>(
    model: &M,
    grads: &mut GradientsParams,
    max: f64,
) -> f64 {
    let mut sq = SquaredNorm::<B> { grads, sum: None };
    model.visit(&mut sq);
    let norm = sq.sum.map_or(0.0, |s| s.into_scalar().elem::<f64>().sqrt());
    if max > 0.0 && norm > max {
        model.visit(&mut Scale { grads, by: max / (norm + 1e-6) });
    }
    norm
}

/// Loss and accuracy over a set of examples.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Metrics {
    /// Weighted mean proper score loss.
    pub loss: f64,
    /// Share of questions with a gold option whose argmax is the gold.
    pub accuracy: f64,
    /// Mean negative log probability of the gold option, floored at 1e-4.
    pub nll: f64,
    /// Questions counted.
    pub questions: usize,
}

/// Scores `examples` without gradients, in the same buckets training uses.
pub fn evaluate<B: Backend>(
    model: &Compat<B>,
    examples: &[Example],
    cfg: &Config,
    dev: &B::Device,
) -> Metrics {
    if examples.is_empty() {
        return Metrics::default();
    }
    let lens: Vec<usize> = examples.iter().map(|e| e.q.ids.len()).collect();
    let (mut loss, mut weight, mut right, mut gold, mut nll) = (0f64, 0f64, 0usize, 0usize, 0f64);
    for (t, idx) in plan(&lens, cfg, &mut Rng::new(0)) {
        let refs: Vec<&Example> = idx.iter().map(|&i| &examples[i]).collect();
        let qs: Vec<Question> = refs.iter().map(|e| e.q.clone()).collect();
        let batch = Batch::<B>::padded(&qs, t, model.shape(), dev);
        let (z, _) = model.forward(&batch);
        let targets = Targets::new(&refs, dev);
        let (l, p) = proper_score(z, &targets);
        let w: f64 = refs.iter().map(|e| f64::from(e.weight)).sum();
        loss += l.into_scalar().elem::<f64>() * w;
        weight += w;
        let k = p.dims()[1];
        let p = p.into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default();
        for (s, e) in refs.iter().enumerate() {
            let Some(h) = e.hard else { continue };
            let row = &p[s * k..s * k + e.probs.len()];
            let best = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)));
            right += usize::from(best == Some(h));
            gold += 1;
            nll -= f64::from(row[h].max(1e-4)).ln();
        }
    }
    Metrics {
        loss: loss / weight.max(1e-12),
        accuracy: right as f64 / gold.max(1) as f64,
        nll: nll / gold.max(1) as f64,
        questions: examples.len(),
    }
}

/// What the loop reports as it goes.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// An optimizer step finished.
    Step {
        /// Steps done.
        step: usize,
        /// Steps in the run.
        total: usize,
        /// Its learning rate.
        lr: f64,
        /// Mean loss over its micro batches.
        loss: f64,
        /// Global gradient norm before clipping.
        grad_norm: f64,
        /// Real tokens, not padding, per second since the start.
        tokens_per_s: f64,
        /// Questions per second since the start.
        questions_per_s: f64,
    },
    /// An evaluation finished.
    Eval {
        /// Steps done when it ran.
        step: usize,
        /// Its result.
        metrics: Metrics,
    },
}

/// Renders every line, with the options of choice questions shuffled when a generator is given.
#[must_use]
pub fn render_all(r: &Renderer, lines: &[String], mut rng: Option<&mut Rng>) -> Vec<Example> {
    let mut out = Vec::with_capacity(lines.len());
    for l in lines {
        out.extend(r.render_with(l, rng.as_deref_mut()).0);
    }
    out
}

/// Trains `model` on `lines` and gives it back, evaluating on `eval` as `cfg` asks and at the end.
pub fn fit<B: AutodiffBackend>(
    mut model: Compat<B>,
    renderer: &Renderer,
    lines: &[String],
    eval: &[Example],
    cfg: &Config,
    dev: &B::Device,
    mut report: impl FnMut(&Event),
) -> Compat<B> {
    let mut rng = Rng::new(cfg.seed);
    // The policy gradient's noise has its own generator, so every objective sees the same batches.
    let mut noise = Rng::new(cfg.seed ^ 0x7267_5f70);
    let mut optim = AdamWConfig::new()
        .with_beta_1(cfg.betas.0 as f32)
        .with_beta_2(cfg.betas.1 as f32)
        .with_epsilon(cfg.eps as f32)
        .with_weight_decay(cfg.weight_decay as f32)
        .init::<B, Compat<B>>();
    let accum = cfg.accum.max(1);
    let mut total = 0;
    let (mut step, mut tokens, mut questions) = (0usize, 0usize, 0usize);
    let start = Instant::now();
    'epochs: for epoch in 0..cfg.epochs {
        let examples = if cfg.shuffle_options {
            render_all(renderer, lines, Some(&mut rng))
        } else {
            render_all(renderer, lines, None)
        };
        let lens: Vec<usize> = examples.iter().map(|e| e.q.ids.len()).collect();
        let batches = plan(&lens, cfg, &mut rng);
        if epoch == 0 {
            total = (batches.len().div_ceil(accum) * cfg.epochs)
                .min(cfg.max_steps.unwrap_or(usize::MAX));
        }
        for group in batches.chunks(accum) {
            if step >= total {
                break 'epochs;
            }
            let lr = lr_at(cfg, step, total);
            let mut acc = GradientsAccumulator::<Compat<B>>::new();
            let mut loss_sum = 0.0;
            for (t, idx) in group {
                let refs: Vec<&Example> = idx.iter().map(|&i| &examples[i]).collect();
                let qs: Vec<Question> = refs.iter().map(|e| e.q.clone()).collect();
                let batch = Batch::<B>::padded(&qs, *t, model.shape(), dev);
                let (z, _) = model.forward(&batch);
                let targets = Targets::new(&refs, dev);
                let (loss, _) = match cfg.objective {
                    Objective::Proper => proper_score(z, &targets),
                    Objective::CrossEntropy => cross_entropy(z, &targets),
                    Objective::Pg { samples } => {
                        rlcd_pg(z, &targets, samples, sigma_at(step, total), &mut noise)
                    }
                };
                let loss = loss / group.len() as f64;
                loss_sum += loss.clone().into_scalar().elem::<f64>();
                acc.accumulate(&model, GradientsParams::from_grads(loss.backward(), &model));
                tokens += qs.iter().map(|q| q.ids.len()).sum::<usize>();
                questions += qs.len();
            }
            let mut grads = acc.grads();
            let grad_norm = clip_global(&model, &mut grads, cfg.clip);
            model = optim.step(lr, model, grads);
            step += 1;
            let secs = start.elapsed().as_secs_f64();
            report(&Event::Step {
                step,
                total,
                lr,
                loss: loss_sum,
                grad_norm,
                tokens_per_s: tokens as f64 / secs,
                questions_per_s: questions as f64 / secs,
            });
            if cfg.eval_every > 0 && step % cfg.eval_every == 0 && step < total {
                let metrics = evaluate(&model.valid(), eval, cfg, dev);
                report(&Event::Eval { step, metrics });
            }
        }
    }
    let metrics = evaluate(&model.valid(), eval, cfg, dev);
    report(&Event::Eval { step, metrics });
    model
}

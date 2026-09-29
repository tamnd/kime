//! The loss, the schedule, clipping and export, on the CPU backend.

use std::path::PathBuf;

use burn::backend::{Autodiff, NdArray};
use burn::module::Param;
use burn::optim::GradientsParams;
use burn::tensor::{Tensor, TensorData};
use kime_model::Model;
use kime_train::data::{Example, Renderer};
use kime_train::loss::{Targets, cross_entropy, proper_score, rlcd_pg};
use kime_train::model::{Compat, Question};
use kime_train::rng::Rng;
use kime_train::train::{Config, clip_global, lr_at, plan};
use serde_json::json;

type B = NdArray;

fn example(k: usize, qtype: usize, probs: &[f32], weight: f32) -> Example {
    Example {
        q: Question { ids: vec![1; 8], markers: (0..k as u32).collect(), qtype },
        probs: probs.to_vec(),
        hard: None,
        weight,
    }
}

fn reference(z: &[f64], t: &[f64], score: bool) -> f64 {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    let p: Vec<f64> = e.iter().map(|x| x / s).collect();
    let log: f64 = p.iter().zip(t).map(|(p, t)| t * p.max(1e-4).ln()).sum();
    let norm = p.iter().map(|p| p * p).sum::<f64>().sqrt();
    let sph = p.iter().zip(t).map(|(p, t)| p * t).sum::<f64>() / norm;
    let mut rps = 0.0;
    if score {
        let (mut cp, mut ct) = (0.0, 0.0);
        for (p, t) in p.iter().zip(t) {
            cp += p;
            ct += t;
            rps += (cp - ct) * (cp - ct);
        }
        rps /= (z.len() - 1) as f64;
    }
    -(log + 0.5 * sph) + rps
}

#[test]
fn proper_score_matches_the_formula() {
    let dev = Default::default();
    let a = example(3, 0, &[0.0, 0.0, 1.0], 1.0);
    let b = example(4, 1, &[0.7, 0.2, 0.1, 0.0], 3.0);
    let c = example(2, 2, &[0.00001, 0.99999], 1.0);
    let z = [0.0, 1.0, 2.0, 0.5, -1.0, 0.0, 2.0, -9.0, 3.0];
    let want = (reference(&[0.0, 1.0, 2.0], &[0.0, 0.0, 1.0], false)
        + 3.0 * reference(&[0.5, -1.0, 0.0, 2.0], &[0.7, 0.2, 0.1, 0.0], true)
        + reference(&[-9.0, 3.0], &[0.00001, 0.99999], false))
        / 5.0;
    let logits = Tensor::<B, 1>::from_data(TensorData::new(z.to_vec(), [9]), &dev);
    let (loss, p) = proper_score(logits, &Targets::new(&[&a, &b, &c], &dev));
    let got = f64::from(loss.into_scalar());
    assert!((got - want).abs() < 1e-5, "{got} against {want}");
    let p = p.into_data().to_vec::<f32>().unwrap();
    assert_eq!(p.len(), 12);
    assert!(p[3].abs() < 1e-12, "padding gets no probability");
}

#[test]
fn schedule_warms_up_and_decays() {
    let cfg = Config { lr: 1e-4, min_lr: 1e-6, warmup: 0.1, ..Config::default() };
    assert!((lr_at(&cfg, 0, 100) - 1e-5).abs() < 1e-12);
    assert!((lr_at(&cfg, 9, 100) - 1e-4).abs() < 1e-12);
    assert!((lr_at(&cfg, 10, 100) - 1e-4).abs() < 1e-12);
    assert!((lr_at(&cfg, 100, 100) - 1e-6).abs() < 1e-12);
    let mid = lr_at(&cfg, 55, 100);
    assert!((mid - (1e-6 + 0.5 * (1e-4 - 1e-6))).abs() < 1e-9, "{mid}");
}

#[test]
fn plan_buckets_and_covers_every_example() {
    let lens: Vec<usize> = (0..1000).map(|i| 10 + (i * 37) % 500).collect();
    let cfg = Config { tokens: 4096, bucket: 64, ..Config::default() };
    let batches = plan(&lens, &cfg, &mut Rng::new(1));
    let mut seen = vec![0; lens.len()];
    for (t, idx) in &batches {
        assert_eq!(t % 64, 0);
        assert!(idx.len() * t <= 4096 || idx.len() == 1);
        for &i in idx {
            assert!(lens[i] <= *t && lens[i] + 64 > *t);
            seen[i] += 1;
        }
    }
    assert!(seen.iter().all(|&s| s == 1));
}

// Burn's derive only finds the parameters when the backend parameter is called B.
#[derive(burn::module::Module, Debug)]
struct Two<B: burn::tensor::backend::Backend> {
    a: Param<Tensor<B, 1>>,
    b: Param<Tensor<B, 2>>,
}

#[test]
fn clipping_bounds_the_global_norm() {
    type A = Autodiff<B>;
    let dev = Default::default();
    let m = Two::<A> {
        a: Param::from_tensor(Tensor::from_data(TensorData::new(vec![1f32, 2.0], [2]), &dev)),
        b: Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![1f32, 1.0, 1.0, 1.0], [2, 2]),
            &dev,
        )),
    };
    // Gradients 3a and 2b: [3, 6] and four 2s, a norm of sqrt(9 + 36 + 16) = sqrt(61).
    let loss = m.a.val().powi_scalar(2).sum() * 1.5 + m.b.val().powi_scalar(2).sum();
    let mut g = GradientsParams::from_grads::<A, _>(loss.backward(), &m);
    let norm = clip_global::<A, _>(&m, &mut g, 1.0);
    assert!((norm - 61f64.sqrt()).abs() < 1e-5, "{norm}");
    let ga = g.get::<B, 1>(m.a.id).unwrap().into_data().to_vec::<f32>().unwrap();
    let gb = g.get::<B, 2>(m.b.id).unwrap().into_data().to_vec::<f32>().unwrap();
    let after: f32 = ga.iter().chain(&gb).map(|x| x * x).sum::<f32>().sqrt();
    assert!((after - 1.0).abs() < 1e-4, "{after}");
}

fn laya() -> Option<Model> {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya"));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights for laya");
        eprintln!("skipping: set KIME_MODELS to a folder holding laya/ with its weights");
        return None;
    };
    Some(Model::open(&dir).unwrap())
}

#[test]
fn export_gives_the_base_back() {
    let Some(base) = laya() else { return };
    let dev = Default::default();
    let model = Compat::<B>::load(&base.spec, &base.tensors, &dev);
    // Every weight but the temperatures is a parameter the optimizer sees.
    let trained: usize =
        base.tensors.entries().iter().filter(|e| e.name != "temperature").map(|e| e.numel()).sum();
    assert_eq!(burn::module::Module::num_params(&model), trained);
    let out = std::env::temp_dir().join(format!("kime-train-export-{}", std::process::id()));
    kime_train::export::save(&model, &base, &out, json!({"test": true})).unwrap();
    let back = Model::open(&out).unwrap();
    assert_eq!(back.tensors.entries().len(), base.tensors.entries().len());
    for e in base.tensors.entries() {
        let (a, b) = (base.tensors.get(&e.name).unwrap(), back.tensors.get(&e.name).unwrap());
        assert_eq!((a.dtype, a.shape), (b.dtype, b.shape), "{}", e.name);
        assert!(a.bytes == b.bytes, "{} changed", e.name);
    }
    let cfg: serde_json::Value =
        serde_json::from_slice(back.file("rl_agent_config.json").unwrap()).unwrap();
    assert_eq!(cfg["kime_train"]["test"], true);
    assert_eq!(back.file("tokenizer/tokenizer.json"), base.file("tokenizer/tokenizer.json"));
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn shuffled_options_carry_their_targets() {
    let Some(base) = laya() else { return };
    let r = Renderer::new(&base).unwrap();
    let line = json!({
        "state": {"text": "where is my card"},
        "questions": {"intent": {"type": "choice", "instructions": "Which intent?",
            "criteria": {"card_arrival": "card not here", "lost_card": "card lost", "top_up": "top up", "refund": "a refund"}}},
        "targets": {"intent": {"probs": [0.1, 0.6, 0.0, 0.3], "hard": 1}}
    })
    .to_string();
    let (plain, _) = r.render(&line);
    let mut rng = Rng::new(5);
    let mut orders = std::collections::HashSet::new();
    for _ in 0..20 {
        let (ex, skipped) = r.render_with(&line, Some(&mut rng));
        assert!(skipped.is_empty());
        let e = &ex[0];
        // The gold is still 0.6, and the same multiset of targets.
        assert!((e.probs[e.hard.unwrap()] - 0.6).abs() < 1e-6);
        let mut s = e.probs.clone();
        s.sort_by(f32::total_cmp);
        assert_eq!(s, vec![0.0, 0.1, 0.3, 0.6]);
        assert_eq!(e.q.ids.len(), plain[0].q.ids.len());
        orders.insert(format!("{:?}", e.probs));
    }
    assert!(orders.len() > 5, "only {} orders", orders.len());
}

/// Counts the weights that still take gradients.
struct Trainable(usize);

impl<A: burn::tensor::backend::Backend> burn::module::ModuleVisitor<A> for Trainable {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<A, D>>) {
        if param.val().is_require_grad() {
            self.0 += param.val().shape().num_elements();
        }
    }
}

#[test]
fn freezing_leaves_only_the_top_layers() {
    use burn::module::Module;
    let Some(base) = laya() else { return };
    let dev = Default::default();
    let count = |n: usize| {
        let m = Compat::<Autodiff<B>>::load(&base.spec, &base.tensors, &dev).freeze_below(n);
        let mut t = Trainable(0);
        m.visit(&mut t);
        (t.0, m.num_params(), m.depth(), m.shape().d, m.shape().inter)
    };
    let (two, total, depth, d, inter) = count(2);
    let (three, ..) = count(3);
    let (all, ..) = count(depth);
    assert_eq!(all, total, "training every layer freezes nothing");
    // One more layer is its attention norm, the fused qkv, the output, the mlp norm and the mlp.
    assert_eq!(three - two, d + 3 * d * d + d * d + d + d * 2 * inter + inter * d);
    assert!(two < total / 4, "{two} of {total} still train with two layers");
}

#[test]
fn cross_entropy_matches_the_formula() {
    let dev = Default::default();
    let a = example(3, 0, &[0.2, 0.0, 0.8], 1.0);
    let b = example(2, 0, &[1.0, 0.0], 1.0);
    let z = [0.0f32, 1.0, 2.0, -1.0, 1.0];
    let ce = |z: &[f64], t: &[f64]| {
        let s: f64 = z.iter().map(|x| x.exp()).sum();
        -z.iter().zip(t).map(|(z, t)| t * (z - s.ln())).sum::<f64>()
    };
    let want = (ce(&[0.0, 1.0, 2.0], &[0.2, 0.0, 0.8]) + ce(&[-1.0, 1.0], &[1.0, 0.0])) / 2.0;
    let logits = Tensor::<B, 1>::from_data(TensorData::new(z.to_vec(), [5]), &dev);
    let (loss, _) = cross_entropy(logits, &Targets::new(&[&a, &b], &dev));
    let got = f64::from(loss.into_scalar());
    assert!((got - want).abs() < 1e-5, "{got} against {want}");
}

/// The gradient of `loss` with respect to the logits.
fn logit_grad(
    z: &[f32],
    mut f: impl FnMut(Tensor<Autodiff<B>, 1>) -> Tensor<Autodiff<B>, 1>,
) -> Vec<f32> {
    let dev = Default::default();
    let x = Tensor::<Autodiff<B>, 1>::from_data(TensorData::new(z.to_vec(), [z.len()]), &dev)
        .require_grad();
    let g = f(x.clone()).backward();
    x.grad(&g).unwrap().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn policy_gradient_points_where_the_exact_gradient_does() {
    let dev = Default::default();
    let a = example(4, 0, &[0.1, 0.6, 0.2, 0.1], 1.0);
    let b = example(3, 1, &[0.0, 0.3, 0.7], 1.0);
    let z = [1.0f32, -0.5, 0.3, 0.0, 0.8, 0.1, -0.4];
    let exact = logit_grad(&z, |x| proper_score(x, &Targets::new(&[&a, &b], &dev)).0);
    let mut rng = Rng::new(7);
    let pg = logit_grad(&z, |x| rlcd_pg(x, &Targets::new(&[&a, &b], &dev), 4096, 0.3, &mut rng).0);
    let dot: f32 = exact.iter().zip(&pg).map(|(a, b)| a * b).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (norm(&exact) * norm(&pg));
    assert!(cos > 0.9, "cosine {cos} between {exact:?} and {pg:?}");
    // Eight samples, as training uses, are noisy but still point the right way on average.
    let mut sum = vec![0f32; z.len()];
    for _ in 0..64 {
        let g = logit_grad(&z, |x| rlcd_pg(x, &Targets::new(&[&a, &b], &dev), 8, 0.3, &mut rng).0);
        sum.iter_mut().zip(g).for_each(|(s, g)| *s += g);
    }
    let dot: f32 = exact.iter().zip(&sum).map(|(a, b)| a * b).sum();
    assert!(dot / (norm(&exact) * norm(&sum)) > 0.8);
}

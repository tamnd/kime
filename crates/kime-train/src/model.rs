//! The laya compat graph as a Burn module, so it can be trained.
//!
//! It is the graph `kime_model::laya` describes and `kime_cpu::Compat` runs, written with Burn
//! tensors: the ModernBERT encoder with RoPE and the sliding window, the type embedding, the
//! PyTorch style decision head, the option scorer and the act head. Weights load by name from a
//! checkpoint kime serves, so a model starts from Laya's weights or any other compat checkpoint.
//!
//! Sequences are padded to the longest in the batch. Padding keys get a large negative bias
//! before the softmax, which makes their weight exactly zero, so a question gets the same logits
//! alone as in any batch up to float rounding.
//!
//! Linear weights are stored `[in, out]`, the transpose of the checkpoint, because that is what
//! Burn's `linear` takes.

use burn::module::{Module, Param};
use burn::tensor::activation::{gelu, relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::module::{embedding, linear};
use burn::tensor::{Int, Tensor, TensorData};
use kime_model::Tensors;
use kime_model::laya::{ACT_HIDDEN, HEAD_DIM, LayaSpec, TORCH_EPS};

/// Added to the score of a key a query must not see. Finite so gradients stay finite, and far
/// enough below any real score that its weight rounds to zero.
const MASKED: f32 = -1e9;

/// One encoder layer.
#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    attn_norm: Option<Param<Tensor<B, 1>>>,
    wqkv: Param<Tensor<B, 2>>,
    wo: Param<Tensor<B, 2>>,
    mlp_norm: Param<Tensor<B, 1>>,
    wi: Param<Tensor<B, 2>>,
    mlp_wo: Param<Tensor<B, 2>>,
    #[module(skip)]
    global: bool,
}

/// One decision head layer, a PyTorch `TransformerEncoderLayer` with `norm_first` and ReLU.
#[derive(Module, Debug)]
pub struct HeadLayer<B: Backend> {
    in_proj: Affine<B>,
    out_proj: Affine<B>,
    norm1: Affine1<B>,
    norm2: Affine1<B>,
    linear1: Affine<B>,
    linear2: Affine<B>,
}

/// A linear layer with a bias, the weight `[in, out]`.
#[derive(Module, Debug)]
pub struct Affine<B: Backend> {
    w: Param<Tensor<B, 2>>,
    b: Param<Tensor<B, 1>>,
}

/// A LayerNorm with a bias.
#[derive(Module, Debug)]
pub struct Affine1<B: Backend> {
    w: Param<Tensor<B, 1>>,
    b: Param<Tensor<B, 1>>,
}

/// The whole compat model.
#[derive(Module, Debug)]
pub struct Compat<B: Backend> {
    tok_embeddings: Param<Tensor<B, 2>>,
    embed_norm: Param<Tensor<B, 1>>,
    layers: Vec<EncoderLayer<B>>,
    final_norm: Param<Tensor<B, 1>>,
    type_emb: Param<Tensor<B, 2>>,
    head: Vec<HeadLayer<B>>,
    scorer_norm: Affine1<B>,
    scorer_in: Affine<B>,
    scorer_out: Affine<B>,
    act_in: Affine<B>,
    act_out: Affine<B>,
    #[module(skip)]
    shape: Shape,
}

/// What the forward pass needs to know besides the weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    /// Hidden size.
    pub d: usize,
    /// GeGLU inner size.
    pub inter: usize,
    /// Attention heads, of [`HEAD_DIM`] each.
    pub heads: usize,
    /// Sliding window width, of which each token sees half on either side.
    pub window: usize,
    /// RoPE base on global layers.
    pub rope_global: f64,
    /// RoPE base on sliding window layers.
    pub rope_local: f64,
    /// The encoder's LayerNorm epsilon.
    pub norm_eps: f64,
}

/// One question, laid out by `kime_tok`'s `compat_sequence`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question {
    /// Token ids, `[CLS] ... [SEP]`.
    pub ids: Vec<u32>,
    /// The position of each option's mask token.
    pub markers: Vec<u32>,
    /// 0 for choice, 1 for score, 2 for noul.
    pub qtype: usize,
}

/// A batch of questions on a device, padded to the longest.
#[derive(Debug, Clone)]
pub struct Batch<B: Backend> {
    ids: Tensor<B, 2, Int>,
    qtype: Tensor<B, 1, Int>,
    /// `[n, 1, t, t]`, zero where a query may see a key and [`MASKED`] elsewhere.
    full: Tensor<B, 4>,
    local: Tensor<B, 4>,
    rope_global: (Tensor<B, 4>, Tensor<B, 4>),
    rope_local: (Tensor<B, 4>, Tensor<B, 4>),
    /// Row of each marker in the `[n * t, d]` hidden states.
    markers: Tensor<B, 1, Int>,
    /// Row of each question's CLS token.
    cls: Tensor<B, 1, Int>,
    /// Markers per question, in order.
    pub options: Vec<usize>,
}

/// RoPE tables `[1, 1, len, HEAD_DIM]` with each half repeated, computed the way `kime_cpu`
/// computes them so both start from the same f32 angles.
fn rope<B: Backend>(theta: f64, len: usize, dev: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = HEAD_DIM / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta.powf(f64::from((2 * i) as f32 / HEAD_DIM as f32)) as f32))
        .collect();
    let (mut cos, mut sin) =
        (Vec::with_capacity(len * HEAD_DIM), Vec::with_capacity(len * HEAD_DIM));
    for p in 0..len {
        for _ in 0..2 {
            for &f in &inv {
                let a = f64::from(p as f32 * f);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
    }
    let t = |v: Vec<f32>| Tensor::from_data(TensorData::new(v, [1, 1, len, HEAD_DIM]), dev);
    (t(cos), t(sin))
}

impl<B: Backend> Batch<B> {
    /// Pads `qs` into one batch for `shape`.
    ///
    /// # Panics
    ///
    /// If `qs` is empty, a question has no tokens, or a marker is past its sequence.
    #[must_use]
    pub fn new(qs: &[Question], shape: &Shape, dev: &B::Device) -> Self {
        let t = qs.iter().map(|q| q.ids.len()).max().unwrap_or(0);
        Self::padded(qs, t, shape, dev)
    }

    /// Pads `qs` to `t` tokens, so batches of different questions can share one shape and the
    /// kernels compiled for it.
    ///
    /// # Panics
    ///
    /// As [`Batch::new`], and if a question is longer than `t`.
    #[must_use]
    pub fn padded(qs: &[Question], t: usize, shape: &Shape, dev: &B::Device) -> Self {
        assert!(!qs.is_empty(), "an empty batch");
        let n = qs.len();
        assert!(t > 0, "a question with no tokens");
        assert!(qs.iter().all(|q| q.ids.len() <= t), "a question longer than {t}");
        let mut ids = vec![0i64; n * t];
        let (mut full, mut local) = (vec![0f32; n * t * t], vec![0f32; n * t * t]);
        let half = shape.window / 2;
        let (mut markers, mut cls, mut options) =
            (Vec::new(), Vec::with_capacity(n), Vec::with_capacity(n));
        for (s, q) in qs.iter().enumerate() {
            let len = q.ids.len();
            assert!(len > 0, "a question with no tokens");
            for (i, &id) in q.ids.iter().enumerate() {
                ids[s * t + i] = i64::from(id);
            }
            for i in 0..t {
                for j in 0..t {
                    let at = (s * t + i) * t + j;
                    if j >= len {
                        full[at] = MASKED;
                    }
                    if j >= len || i.abs_diff(j) > half {
                        local[at] = MASKED;
                    }
                }
            }
            for &m in &q.markers {
                assert!((m as usize) < len, "marker {m} is past its sequence");
                markers.push((s * t + m as usize) as i64);
            }
            cls.push((s * t) as i64);
            options.push(q.markers.len());
        }
        let int1 = |v: Vec<i64>| {
            let k = v.len();
            Tensor::<B, 1, Int>::from_data(TensorData::new(v, [k]), dev)
        };
        let mask = |v: Vec<f32>| Tensor::from_data(TensorData::new(v, [n, 1, t, t]), dev);
        let qtype = qs.iter().map(|q| q.qtype as i64).collect();
        Self {
            ids: Tensor::from_data(TensorData::new(ids, [n, t]), dev),
            qtype: int1(qtype),
            full: mask(full),
            local: mask(local),
            rope_global: rope(shape.rope_global, t, dev),
            rope_local: rope(shape.rope_local, t, dev),
            markers: int1(markers),
            cls: int1(cls),
            options,
        }
    }
}

/// LayerNorm over the last dimension, with the biased variance as torch uses.
fn layer_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    w: &Param<Tensor<B, 1>>,
    b: Option<&Param<Tensor<B, 1>>>,
    eps: f64,
) -> Tensor<B, D> {
    let (var, mean) = x.clone().var_mean_bias(D - 1);
    let y = (x - mean) / (var + eps).sqrt();
    let y = y * w.val().unsqueeze::<D>();
    match b {
        Some(b) => y + b.val().unsqueeze::<D>(),
        None => y,
    }
}

/// `x cos + rotate_half(x) sin` over `[n, heads, t, HEAD_DIM]`.
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    (cos, sin): &(Tensor<B, 4>, Tensor<B, 4>),
) -> Tensor<B, 4> {
    let half = HEAD_DIM / 2;
    let a = x.clone().narrow(3, 0, half);
    let b = x.clone().narrow(3, half, half);
    let rot = Tensor::cat(vec![-b, a], 3);
    x * cos.clone() + rot * sin.clone()
}

/// Multi head attention over packed `[q k v]` rows `[n, t, 3d]`.
fn attention<B: Backend>(
    qkv: Tensor<B, 3>,
    heads: usize,
    rope: Option<&(Tensor<B, 4>, Tensor<B, 4>)>,
    bias: &Tensor<B, 4>,
) -> Tensor<B, 3> {
    let [n, t, _] = qkv.dims();
    let d = heads * HEAD_DIM;
    let split =
        |i: usize| qkv.clone().narrow(2, i * d, d).reshape([n, t, heads, HEAD_DIM]).swap_dims(1, 2);
    let (mut q, mut k, v) = (split(0), split(1), split(2));
    if let Some(r) = rope {
        q = apply_rope(q, r);
        k = apply_rope(k, r);
    }
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let scores = q.matmul(k.swap_dims(2, 3)) * scale + bias.clone();
    softmax(scores, 3).matmul(v).swap_dims(1, 2).reshape([n, t, d])
}

fn lin<B: Backend, const D: usize>(x: Tensor<B, D>, a: &Affine<B>) -> Tensor<B, D> {
    linear(x, a.w.val(), Some(a.b.val()))
}

impl<B: Backend> Compat<B> {
    /// The model's shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Loads every weight of a compat checkpoint onto `dev` as f32.
    ///
    /// # Panics
    ///
    /// If a tensor is missing, which cannot happen for tensors `kime_model::Model` has bound.
    #[must_use]
    pub fn load(spec: &LayaSpec, t: &Tensors, dev: &B::Device) -> Self {
        let get = |name: &str| {
            let v = t.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
            (v.to_f32(), v.shape.to_vec())
        };
        let p1 = |name: &str| {
            let (v, s) = get(name);
            Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(v, [s[0]]), dev))
        };
        // Checkpoint linears are [out, in], and Burn's are [in, out].
        let p2 = |name: &str| {
            let (v, s) = get(name);
            Param::from_tensor(
                Tensor::<B, 2>::from_data(TensorData::new(v, [s[0], s[1]]), dev).transpose(),
            )
        };
        let aff = |name: &str| Affine {
            w: p2(&format!("{name}.weight")),
            b: p1(&format!("{name}.bias")),
        };
        let aff1 = |name: &str| Affine1 {
            w: p1(&format!("{name}.weight")),
            b: p1(&format!("{name}.bias")),
        };
        let e = &spec.encoder;
        let layers = (0..e.layers)
            .map(|i| {
                let p = format!("encoder.layers.{i}");
                EncoderLayer {
                    attn_norm: (i > 0).then(|| p1(&format!("{p}.attn_norm.weight"))),
                    wqkv: p2(&format!("{p}.attn.Wqkv.weight")),
                    wo: p2(&format!("{p}.attn.Wo.weight")),
                    mlp_norm: p1(&format!("{p}.mlp_norm.weight")),
                    wi: p2(&format!("{p}.mlp.Wi.weight")),
                    mlp_wo: p2(&format!("{p}.mlp.Wo.weight")),
                    global: e.global[i],
                }
            })
            .collect();
        let head = (0..spec.agent.head_layers)
            .map(|l| {
                let p = format!("head.layers.{l}");
                HeadLayer {
                    in_proj: Affine {
                        w: p2(&format!("{p}.self_attn.in_proj_weight")),
                        b: p1(&format!("{p}.self_attn.in_proj_bias")),
                    },
                    out_proj: aff(&format!("{p}.self_attn.out_proj")),
                    norm1: aff1(&format!("{p}.norm1")),
                    norm2: aff1(&format!("{p}.norm2")),
                    linear1: aff(&format!("{p}.linear1")),
                    linear2: aff(&format!("{p}.linear2")),
                }
            })
            .collect();
        let (te, ts) = get("type_emb.weight");
        let (ee, es) = get("encoder.embeddings.tok_embeddings.weight");
        Self {
            tok_embeddings: Param::from_tensor(Tensor::from_data(
                TensorData::new(ee, [es[0], es[1]]),
                dev,
            )),
            embed_norm: p1("encoder.embeddings.norm.weight"),
            layers,
            final_norm: p1("encoder.final_norm.weight"),
            type_emb: Param::from_tensor(Tensor::from_data(
                TensorData::new(te, [ts[0], ts[1]]),
                dev,
            )),
            head,
            scorer_norm: aff1("scorer.0"),
            scorer_in: aff("scorer.1"),
            scorer_out: aff("scorer.3"),
            act_in: aff("act_head.0"),
            act_out: aff("act_head.2"),
            shape: Shape {
                d: e.d,
                inter: e.inter,
                heads: e.heads,
                window: e.window,
                rope_global: e.rope_global,
                rope_local: e.rope_local,
                norm_eps: e.norm_eps,
            },
        }
    }

    /// Every weight under its checkpoint name and in its checkpoint shape, in the order of
    /// `LayaSpec::expected` apart from `temperature`, which is not trained.
    #[must_use]
    pub fn weights(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        // Everything goes through as 2D, vectors as [1, len] with a note to drop the 1.
        let mut all: Vec<(String, Tensor<B, 2>, bool)> = Vec::new();
        let mut put = |name: String, (t, vector): (Tensor<B, 2>, bool)| all.push((name, t, vector));
        let one = |p: &Param<Tensor<B, 1>>| (p.val().unsqueeze::<2>(), true);
        // Linear weights go back to the checkpoint's [out, in].
        let lin = |p: &Param<Tensor<B, 2>>| (p.val().transpose(), false);
        put("encoder.embeddings.tok_embeddings.weight".into(), (self.tok_embeddings.val(), false));
        put("encoder.embeddings.norm.weight".into(), one(&self.embed_norm));
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("encoder.layers.{i}");
            if let Some(w) = &l.attn_norm {
                put(format!("{p}.attn_norm.weight"), one(w));
            }
            put(format!("{p}.attn.Wqkv.weight"), lin(&l.wqkv));
            put(format!("{p}.attn.Wo.weight"), lin(&l.wo));
            put(format!("{p}.mlp_norm.weight"), one(&l.mlp_norm));
            put(format!("{p}.mlp.Wi.weight"), lin(&l.wi));
            put(format!("{p}.mlp.Wo.weight"), lin(&l.mlp_wo));
        }
        put("encoder.final_norm.weight".into(), one(&self.final_norm));
        put("type_emb.weight".into(), (self.type_emb.val(), false));
        for (i, l) in self.head.iter().enumerate() {
            let p = format!("head.layers.{i}");
            put(format!("{p}.self_attn.in_proj_weight"), lin(&l.in_proj.w));
            put(format!("{p}.self_attn.in_proj_bias"), one(&l.in_proj.b));
            for (n, a) in [
                ("self_attn.out_proj", &l.out_proj),
                ("linear1", &l.linear1),
                ("linear2", &l.linear2),
            ] {
                put(format!("{p}.{n}.weight"), lin(&a.w));
                put(format!("{p}.{n}.bias"), one(&a.b));
            }
            for (n, a) in [("norm1", &l.norm1), ("norm2", &l.norm2)] {
                put(format!("{p}.{n}.weight"), one(&a.w));
                put(format!("{p}.{n}.bias"), one(&a.b));
            }
        }
        put("scorer.0.weight".into(), one(&self.scorer_norm.w));
        put("scorer.0.bias".into(), one(&self.scorer_norm.b));
        let affines = [
            ("scorer.1", &self.scorer_in),
            ("scorer.3", &self.scorer_out),
            ("act_head.0", &self.act_in),
            ("act_head.2", &self.act_out),
        ];
        for (n, a) in affines {
            put(format!("{n}.weight"), lin(&a.w));
            put(format!("{n}.bias"), one(&a.b));
        }
        all.into_iter()
            .map(|(name, t, vector)| {
                let mut shape = t.dims().to_vec();
                if vector {
                    shape.remove(0);
                }
                (name, shape, t.into_data().convert::<f32>().to_vec::<f32>().unwrap_or_default())
            })
            .collect()
    }

    /// The hidden states after the decision head, `[n * t, d]`.
    fn hidden(&self, b: &Batch<B>) -> Tensor<B, 2> {
        let s = &self.shape;
        let [n, t] = b.ids.dims();
        let mut h = embedding(self.tok_embeddings.val(), b.ids.clone());
        h = layer_norm(h, &self.embed_norm, None, s.norm_eps);
        for l in &self.layers {
            let x = match &l.attn_norm {
                Some(w) => layer_norm(h.clone(), w, None, s.norm_eps),
                None => h.clone(),
            };
            let qkv = linear(x, l.wqkv.val(), None);
            let (r, bias) =
                if l.global { (&b.rope_global, &b.full) } else { (&b.rope_local, &b.local) };
            h = h + linear(attention(qkv, s.heads, Some(r), bias), l.wo.val(), None);
            let x = layer_norm(h.clone(), &l.mlp_norm, None, s.norm_eps);
            let u = linear(x, l.wi.val(), None);
            let act = gelu(u.clone().narrow(2, 0, s.inter)) * u.narrow(2, s.inter, s.inter);
            h = h + linear(act, l.mlp_wo.val(), None);
        }
        h = layer_norm(h, &self.final_norm, None, s.norm_eps);
        h = h + self.type_emb.val().select(0, b.qtype.clone()).unsqueeze_dim::<3>(1);
        for l in &self.head {
            let x = layer_norm(h.clone(), &l.norm1.w, Some(&l.norm1.b), TORCH_EPS);
            h = h + lin(attention(lin(x, &l.in_proj), s.heads, None, &b.full), &l.out_proj);
            let x = layer_norm(h.clone(), &l.norm2.w, Some(&l.norm2.b), TORCH_EPS);
            h = h + lin(relu(lin(x, &l.linear1)), &l.linear2);
        }
        h.reshape([n * t, s.d])
    }

    /// One logit per option of every question, in order, `[markers]`, and the hidden states for
    /// [`Compat::act`].
    #[must_use]
    pub fn forward(&self, b: &Batch<B>) -> (Tensor<B, 1>, Tensor<B, 2>) {
        let h = self.hidden(b);
        let m = h.clone().select(0, b.markers.clone());
        let m = layer_norm(m, &self.scorer_norm.w, Some(&self.scorer_norm.b), TORCH_EPS);
        let z = lin(gelu(lin(m, &self.scorer_in)), &self.scorer_out);
        let k = z.dims()[0];
        (z.reshape([k]), h)
    }

    /// The act head's two logits per question, `[n, 2]`, from the hidden states of
    /// [`Compat::forward`] and the four features `kime_cpu::act_features` gives for each
    /// question's logits, `[n, 4]`.
    #[must_use]
    pub fn act(&self, b: &Batch<B>, h: Tensor<B, 2>, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let cls = h.select(0, b.cls.clone());
        let x = Tensor::cat(vec![cls, features], 1);
        let a = gelu(lin(x, &self.act_in));
        debug_assert_eq!(a.dims()[1], ACT_HIDDEN);
        lin(a, &self.act_out)
    }
}

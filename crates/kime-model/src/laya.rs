//! The laya compat family: Laya 0.3.7's `DecisionModel` over a ModernBERT encoder.
//!
//! The graph, from spec/05-model.md, with `d` the hidden size:
//!
//! ```text
//! h   = LN(E[ids])                                     encoder.embeddings
//! for each encoder layer i:
//!     h = h + Wo(Attn(RoPE(q), RoPE(k), v))            [q k v] = Wqkv LN(h), no LN on layer 0
//!     h = h + Wo(GELU(a) * g)                          [a g] = Wi LN(h)
//! h   = LN(h)                                          encoder.final_norm
//! h   = h + type_emb[qtype]
//! for each head layer:                                 PyTorch TransformerEncoderLayer, norm first
//!     h = h + out_proj(Attn(in_proj LN1(h)))           full attention, biases, no RoPE
//!     h = h + linear2(ReLU(linear1(LN2(h))))
//! m   = h[markers]
//! z   = scorer.3(GELU(scorer.1(LN(m))))                one logit per option
//! act = act_head.2(GELU(act_head.0([h[0], top1, top1 - top2, entropy, k / 255])))
//! ```
//!
//! Encoder layers have no biases and LayerNorms without bias. Attention is global on every third
//! layer and a 128 token sliding window on the rest, and GELU is the exact erf form throughout.
//! [`LayaGraph`] binds each weight in that description to a tensor in a checkpoint, and binding is
//! also how a checkpoint is checked: every missing tensor, extra tensor and wrong shape is named.

use serde_json::Value;

use kime_tensor::plan::{Epilogue, Graph, Op, Rows};

use crate::error::{Error, Result};
use crate::tensors::Tensors;

/// Head dimension of every attention in the family.
pub const HEAD_DIM: usize = 64;

/// PyTorch's LayerNorm default, which the decision head and the scorer use.
pub const TORCH_EPS: f64 = 1e-5;

/// Width of the act head's hidden layer.
pub const ACT_HIDDEN: usize = 256;

/// The encoder's shape, from `encoder/config.json` (a HF ModernBERT config).
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Hidden size.
    pub d: usize,
    /// GeGLU inner size. `mlp.Wi` has twice as many rows.
    pub inter: usize,
    /// Attention heads.
    pub heads: usize,
    /// Encoder layers.
    pub layers: usize,
    /// Vocabulary size, the rows of the embedding table.
    pub vocab: usize,
    /// Whether each layer attends globally, false meaning the sliding window.
    pub global: Vec<bool>,
    /// Sliding window width in tokens, 128 for both published encoders.
    pub window: usize,
    /// RoPE base on global layers.
    pub rope_global: f64,
    /// RoPE base on sliding window layers.
    pub rope_local: f64,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
}

/// Laya's own config, `rl_agent_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// The HF id of the encoder the checkpoint was trained from.
    pub encoder: String,
    /// Transformer layers in the decision head.
    pub head_layers: usize,
    /// Longest sequence.
    pub max_len: usize,
    /// Budget for the question head and its options.
    pub head_max_len: usize,
    /// Per type temperatures, choice, score, noul.
    pub temperature: [f64; 3],
    /// Temperatures per type and option count bucket, such as `choice:3-5`.
    pub temperature_by_options: Vec<(String, f64)>,
}

/// Both configs of a compat checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct LayaSpec {
    /// The name kime serves it under, `laya`, `laya-multilingual` or `laya-typed-decisions` for
    /// the published ones.
    pub id: String,
    /// The encoder.
    pub encoder: EncoderConfig,
    /// The decision head.
    pub agent: AgentConfig,
}

fn get<'a>(v: &'a Value, key: &str, file: &str) -> Result<&'a Value> {
    v.get(key).ok_or_else(|| Error::format(format!("{file}: missing {key:?}")))
}

fn usize_of(v: &Value, key: &str, file: &str) -> Result<usize> {
    get(v, key, file)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::format(format!("{file}: {key:?} is not a non negative integer")))
}

fn f64_of(v: &Value, key: &str, file: &str) -> Result<f64> {
    get(v, key, file)?
        .as_f64()
        .ok_or_else(|| Error::format(format!("{file}: {key:?} is not a number")))
}

impl EncoderConfig {
    /// Reads a HF ModernBERT config. Both the transformers 5 layout (`layer_types`,
    /// `rope_parameters`) and the older one (`global_attn_every_n_layers`, `global_rope_theta`,
    /// `local_rope_theta`) are accepted, since checkpoints in the wild have either.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] naming the missing or bad field.
    pub fn from_json(v: &Value) -> Result<Self> {
        const F: &str = "encoder/config.json";
        let d = usize_of(v, "hidden_size", F)?;
        let heads = usize_of(v, "num_attention_heads", F)?;
        let layers = usize_of(v, "num_hidden_layers", F)?;
        if heads == 0 || d != heads * HEAD_DIM {
            return Err(Error::format(format!(
                "{F}: hidden_size {d} is not {heads} heads of {HEAD_DIM}"
            )));
        }
        let global = match v.get("layer_types").and_then(Value::as_array) {
            Some(types) => types
                .iter()
                .map(|t| match t.as_str() {
                    Some("full_attention") => Ok(true),
                    Some("sliding_attention") => Ok(false),
                    _ => Err(Error::format(format!("{F}: unknown layer type {t}"))),
                })
                .collect::<Result<Vec<_>>>()?,
            None => {
                let every = usize_of(v, "global_attn_every_n_layers", F)?.max(1);
                (0..layers).map(|i| i % every == 0).collect()
            }
        };
        if global.len() != layers {
            return Err(Error::format(format!(
                "{F}: {} layer types for {layers} layers",
                global.len()
            )));
        }
        let (rope_global, rope_local) = match v.get("rope_parameters") {
            Some(p) => (
                f64_of(get(p, "full_attention", F)?, "rope_theta", F)?,
                f64_of(get(p, "sliding_attention", F)?, "rope_theta", F)?,
            ),
            None => (f64_of(v, "global_rope_theta", F)?, f64_of(v, "local_rope_theta", F)?),
        };
        Ok(Self {
            d,
            inter: usize_of(v, "intermediate_size", F)?,
            heads,
            layers,
            vocab: usize_of(v, "vocab_size", F)?,
            global,
            window: usize_of(v, "local_attention", F)?,
            rope_global,
            rope_local,
            norm_eps: v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5),
        })
    }
}

impl AgentConfig {
    /// Reads `rl_agent_config.json`.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] naming the missing or bad field.
    pub fn from_json(v: &Value) -> Result<Self> {
        const F: &str = "rl_agent_config.json";
        let t = get(v, "temperature", F)?
            .as_array()
            .filter(|a| a.len() == 3)
            .and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?]))
            .ok_or_else(|| Error::format(format!("{F}: temperature must be three numbers")))?;
        let by_options = match v.get("temperature_by_options") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Object(m)) => m
                .iter()
                .map(|(k, t)| {
                    t.as_f64().map(|t| (k.clone(), t)).ok_or_else(|| {
                        Error::format(format!("{F}: temperature_by_options[{k:?}] is not a number"))
                    })
                })
                .collect::<Result<_>>()?,
            Some(_) => {
                return Err(Error::format(format!("{F}: temperature_by_options is not an object")));
            }
        };
        Ok(Self {
            encoder: get(v, "encoder", F)?.as_str().unwrap_or_default().to_string(),
            head_layers: usize_of(v, "head_layers", F)?,
            max_len: usize_of(v, "max_len", F)?,
            head_max_len: usize_of(v, "head_max_len", F)?,
            temperature: t,
            temperature_by_options: by_options,
        })
    }
}

impl LayaSpec {
    /// Builds the spec from the two config files. The id follows the encoder: the published
    /// checkpoints get their Laya names and anything else is `laya-custom`. The typed-decisions
    /// checkpoint shares the English encoder and is told apart by its `model_name`.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] from either config.
    pub fn from_json(agent_json: &Value, encoder: &Value) -> Result<Self> {
        let agent = AgentConfig::from_json(agent_json)?;
        let encoder = EncoderConfig::from_json(encoder)?;
        let typed = agent_json.get("model_name").and_then(Value::as_str);
        let id = match agent.encoder.as_str() {
            "answerdotai/ModernBERT-large" if typed == Some("laya-typed-decisions") => {
                "laya-typed-decisions"
            }
            "answerdotai/ModernBERT-large" => "laya",
            "jhu-clsp/mmBERT-base" => "laya-multilingual",
            _ => "laya-custom",
        };
        Ok(Self { id: id.to_string(), encoder, agent })
    }

    /// Every tensor the checkpoint must hold, with its shape, in Laya's module order.
    #[must_use]
    pub fn expected(&self) -> Vec<(String, Vec<usize>)> {
        let e = &self.encoder;
        let d = e.d;
        let mut out: Vec<(String, Vec<usize>)> = Vec::new();
        let mut put = |name: String, shape: &[usize]| out.push((name, shape.to_vec()));
        put("encoder.embeddings.tok_embeddings.weight".into(), &[e.vocab, d]);
        put("encoder.embeddings.norm.weight".into(), &[d]);
        for i in 0..e.layers {
            let p = format!("encoder.layers.{i}");
            if i > 0 {
                put(format!("{p}.attn_norm.weight"), &[d]);
            }
            put(format!("{p}.attn.Wqkv.weight"), &[3 * d, d]);
            put(format!("{p}.attn.Wo.weight"), &[d, d]);
            put(format!("{p}.mlp_norm.weight"), &[d]);
            put(format!("{p}.mlp.Wi.weight"), &[2 * e.inter, d]);
            put(format!("{p}.mlp.Wo.weight"), &[d, e.inter]);
        }
        put("encoder.final_norm.weight".into(), &[d]);
        put("type_emb.weight".into(), &[3, d]);
        for l in 0..self.agent.head_layers {
            let p = format!("head.layers.{l}");
            put(format!("{p}.self_attn.in_proj_weight"), &[3 * d, d]);
            put(format!("{p}.self_attn.in_proj_bias"), &[3 * d]);
            put(format!("{p}.self_attn.out_proj.weight"), &[d, d]);
            put(format!("{p}.self_attn.out_proj.bias"), &[d]);
            put(format!("{p}.linear1.weight"), &[4 * d, d]);
            put(format!("{p}.linear1.bias"), &[4 * d]);
            put(format!("{p}.linear2.weight"), &[d, 4 * d]);
            put(format!("{p}.linear2.bias"), &[d]);
            for n in ["norm1", "norm2"] {
                put(format!("{p}.{n}.weight"), &[d]);
                put(format!("{p}.{n}.bias"), &[d]);
            }
        }
        put("scorer.0.weight".into(), &[d]);
        put("scorer.0.bias".into(), &[d]);
        put("scorer.1.weight".into(), &[d, d]);
        put("scorer.1.bias".into(), &[d]);
        put("scorer.3.weight".into(), &[1, d]);
        put("scorer.3.bias".into(), &[1]);
        put("act_head.0.weight".into(), &[256, d + 4]);
        put("act_head.0.bias".into(), &[256]);
        put("act_head.2.weight".into(), &[2, 256]);
        put("act_head.2.bias".into(), &[2]);
        put("temperature".into(), &[3]);
        out
    }
}

/// A tensor bound into the graph, as its index in [`Tensors`].
pub type W = usize;

/// One encoder layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncoderLayer {
    /// None on layer 0, where ModernBERT skips the norm.
    pub attn_norm: Option<W>,
    /// `[3d, d]`, rows are q then k then v.
    pub wqkv: W,
    /// `[d, d]`.
    pub wo: W,
    /// `[d]`.
    pub mlp_norm: W,
    /// `[2I, d]`, rows are the GELU input then the gate.
    pub wi: W,
    /// `[d, I]`.
    pub mlp_wo: W,
    /// Global attention, or the sliding window.
    pub global: bool,
    /// RoPE base for this layer.
    pub rope_theta: f64,
}

/// One layer of the decision head, a PyTorch `TransformerEncoderLayer` with `norm_first`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct HeadLayer {
    pub in_proj_w: W,
    pub in_proj_b: W,
    pub out_proj_w: W,
    pub out_proj_b: W,
    pub norm1_w: W,
    pub norm1_b: W,
    pub norm2_w: W,
    pub norm2_b: W,
    pub linear1_w: W,
    pub linear1_b: W,
    pub linear2_w: W,
    pub linear2_b: W,
}

/// A linear layer or LayerNorm with a bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Affine {
    /// The weight.
    pub w: W,
    /// The bias.
    pub b: W,
}

/// The whole compat graph with every weight bound.
#[derive(Debug, Clone, PartialEq)]
pub struct LayaGraph {
    /// `[vocab, d]`.
    pub tok_embeddings: W,
    /// `[d]`.
    pub embed_norm: W,
    /// Encoder layers in order.
    pub layers: Vec<EncoderLayer>,
    /// `[d]`.
    pub final_norm: W,
    /// `[3, d]`, one row per question type.
    pub type_emb: W,
    /// Decision head layers in order.
    pub head: Vec<HeadLayer>,
    /// LayerNorm, `scorer.0`.
    pub scorer_norm: Affine,
    /// `scorer.1`, `[d, d]`.
    pub scorer_in: Affine,
    /// `scorer.3`, `[1, d]`.
    pub scorer_out: Affine,
    /// `act_head.0`, `[256, d + 4]`.
    pub act_in: Affine,
    /// `act_head.2`, `[2, 256]`.
    pub act_out: Affine,
    /// `[3]`, the calibrated temperatures stored in the checkpoint.
    pub temperature: W,
}

impl LayaGraph {
    /// Binds every weight of the graph to `tensors`, checking names, shapes and dtypes.
    ///
    /// # Errors
    ///
    /// [`Error::Mismatch`] listing every missing tensor, extra tensor, wrong shape and non float
    /// dtype, in that order.
    ///
    /// # Panics
    ///
    /// Never. Binding looks up only names the check above has found.
    pub fn bind(spec: &LayaSpec, tensors: &Tensors) -> Result<Self> {
        let expected = spec.expected();
        let mut problems = Vec::new();
        for (name, shape) in &expected {
            match tensors.get(name) {
                None => problems.push(format!("missing {name} {shape:?}")),
                Some(v) if v.shape != shape.as_slice() => {
                    problems.push(format!("{name} has shape {:?}, expected {shape:?}", v.shape));
                }
                Some(v) if !v.dtype.is_float() => {
                    problems.push(format!("{name} is {}, expected a float type", v.dtype));
                }
                Some(_) => {}
            }
        }
        let known: std::collections::HashSet<&str> =
            expected.iter().map(|(n, _)| n.as_str()).collect();
        for e in tensors.entries() {
            if !known.contains(e.name.as_str()) {
                problems.push(format!("unexpected {} {:?}", e.name, e.shape));
            }
        }
        if !problems.is_empty() {
            return Err(Error::Mismatch(problems));
        }
        let w = |name: &str| tensors.index(name).expect("checked above");
        let aff = |p: &str| Affine { w: w(&format!("{p}.weight")), b: w(&format!("{p}.bias")) };
        let enc = &spec.encoder;
        let layers = (0..enc.layers)
            .map(|i| {
                let p = format!("encoder.layers.{i}");
                EncoderLayer {
                    attn_norm: (i > 0).then(|| w(&format!("{p}.attn_norm.weight"))),
                    wqkv: w(&format!("{p}.attn.Wqkv.weight")),
                    wo: w(&format!("{p}.attn.Wo.weight")),
                    mlp_norm: w(&format!("{p}.mlp_norm.weight")),
                    wi: w(&format!("{p}.mlp.Wi.weight")),
                    mlp_wo: w(&format!("{p}.mlp.Wo.weight")),
                    global: enc.global[i],
                    rope_theta: if enc.global[i] { enc.rope_global } else { enc.rope_local },
                }
            })
            .collect();
        let head = (0..spec.agent.head_layers)
            .map(|l| {
                let p = format!("head.layers.{l}");
                HeadLayer {
                    in_proj_w: w(&format!("{p}.self_attn.in_proj_weight")),
                    in_proj_b: w(&format!("{p}.self_attn.in_proj_bias")),
                    out_proj_w: w(&format!("{p}.self_attn.out_proj.weight")),
                    out_proj_b: w(&format!("{p}.self_attn.out_proj.bias")),
                    norm1_w: w(&format!("{p}.norm1.weight")),
                    norm1_b: w(&format!("{p}.norm1.bias")),
                    norm2_w: w(&format!("{p}.norm2.weight")),
                    norm2_b: w(&format!("{p}.norm2.bias")),
                    linear1_w: w(&format!("{p}.linear1.weight")),
                    linear1_b: w(&format!("{p}.linear1.bias")),
                    linear2_w: w(&format!("{p}.linear2.weight")),
                    linear2_b: w(&format!("{p}.linear2.bias")),
                }
            })
            .collect();
        Ok(Self {
            tok_embeddings: w("encoder.embeddings.tok_embeddings.weight"),
            embed_norm: w("encoder.embeddings.norm.weight"),
            layers,
            final_norm: w("encoder.final_norm.weight"),
            type_emb: w("type_emb.weight"),
            head,
            scorer_norm: aff("scorer.0"),
            scorer_in: aff("scorer.1"),
            scorer_out: aff("scorer.3"),
            act_in: aff("act_head.0"),
            act_out: aff("act_head.2"),
            temperature: w("temperature"),
        })
    }

    /// The forward pass as ops for a backend to lower, the graph in the module comment. Values are
    /// token rows until the markers are gathered, and a backend runs each op on the rows the batch
    /// has.
    #[must_use]
    pub fn plan(&self, spec: &LayaSpec) -> Graph {
        let e = &spec.encoder;
        let d = e.d;
        let mut g = Graph::default();
        let tok = |g: &mut Graph, w| g.val(Rows::Tokens, w);
        let emb = tok(&mut g, d);
        let h = tok(&mut g, d);
        g.push(Op::Embed { table: self.tok_embeddings, out: emb });
        g.push(Op::LayerNorm { x: emb, w: self.embed_norm, b: None, eps: e.norm_eps, out: h });
        let gemm = |g: &mut Graph, a, w, b, epilogue, out| {
            g.push(Op::Gemm { a, w, b, epilogue, out });
        };
        for l in &self.layers {
            let x = match l.attn_norm {
                Some(n) => {
                    let x = tok(&mut g, d);
                    g.push(Op::LayerNorm { x: h, w: n, b: None, eps: e.norm_eps, out: x });
                    x
                }
                None => h,
            };
            let qkv = tok(&mut g, 3 * d);
            gemm(&mut g, x, l.wqkv, None, Epilogue::None, qkv);
            g.push(Op::Rope { qkv, theta: l.rope_theta });
            let att = tok(&mut g, d);
            let window = (!l.global).then_some(e.window / 2);
            g.push(Op::Attention { qkv, window, out: att });
            gemm(&mut g, att, l.wo, None, Epilogue::Accumulate, h);
            let x = tok(&mut g, d);
            g.push(Op::LayerNorm { x: h, w: l.mlp_norm, b: None, eps: e.norm_eps, out: x });
            let u = tok(&mut g, 2 * e.inter);
            gemm(&mut g, x, l.wi, None, Epilogue::None, u);
            let a = tok(&mut g, e.inter);
            g.push(Op::GeGlu { x: u, out: a });
            gemm(&mut g, a, l.mlp_wo, None, Epilogue::Accumulate, h);
        }
        let h2 = tok(&mut g, d);
        g.push(Op::LayerNorm { x: h, w: self.final_norm, b: None, eps: e.norm_eps, out: h2 });
        let h = h2;
        g.push(Op::AddType { h, table: self.type_emb });
        for l in &self.head {
            let x = tok(&mut g, d);
            g.push(Op::LayerNorm {
                x: h,
                w: l.norm1_w,
                b: Some(l.norm1_b),
                eps: TORCH_EPS,
                out: x,
            });
            let qkv = tok(&mut g, 3 * d);
            gemm(&mut g, x, l.in_proj_w, Some(l.in_proj_b), Epilogue::None, qkv);
            let att = tok(&mut g, d);
            g.push(Op::Attention { qkv, window: None, out: att });
            gemm(&mut g, att, l.out_proj_w, Some(l.out_proj_b), Epilogue::Accumulate, h);
            let x = tok(&mut g, d);
            g.push(Op::LayerNorm {
                x: h,
                w: l.norm2_w,
                b: Some(l.norm2_b),
                eps: TORCH_EPS,
                out: x,
            });
            let f = tok(&mut g, 4 * d);
            gemm(&mut g, x, l.linear1_w, Some(l.linear1_b), Epilogue::Relu, f);
            gemm(&mut g, f, l.linear2_w, Some(l.linear2_b), Epilogue::Accumulate, h);
        }
        let m = g.val(Rows::Markers, d);
        g.push(Op::GatherMarkers { h, out: m });
        let mn = g.val(Rows::Markers, d);
        let sn = self.scorer_norm;
        g.push(Op::LayerNorm { x: m, w: sn.w, b: Some(sn.b), eps: TORCH_EPS, out: mn });
        let z = g.val(Rows::Markers, d);
        gemm(&mut g, mn, self.scorer_in.w, Some(self.scorer_in.b), Epilogue::Gelu, z);
        let logits = g.val(Rows::Markers, 1);
        gemm(&mut g, z, self.scorer_out.w, Some(self.scorer_out.b), Epilogue::None, logits);
        let f = g.val(Rows::Seqs, d + 4);
        g.push(Op::ActFeatures { h, logits, out: f });
        let a = g.val(Rows::Seqs, ACT_HIDDEN);
        gemm(&mut g, f, self.act_in.w, Some(self.act_in.b), Epilogue::Gelu, a);
        let act = g.val(Rows::Seqs, 2);
        gemm(&mut g, a, self.act_out.w, Some(self.act_out.b), Epilogue::None, act);
        g.logits = Some(logits);
        g.act = Some(act);
        g
    }
}

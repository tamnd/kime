//! The kime-v1 family: the split encoder of spec/05-model.md.
//!
//! ```text
//! s   = LN(E[state_ids])                               shared embedding and norm
//! for each state layer i:                              ModernBERT layers, no LN before layer 0
//!     s = s + Wo(Attn(RoPE(q), RoPE(k), v))            global every third layer, else a window
//!     s = s + Wo(GELU(a) * g)
//! S   = LN(s)                                          state.final_norm
//! K_j = S Wk_j,  V_j = S Wv_j                          per question layer j, kv_heads of 64, no RoPE
//! pooled = normalize(mean(S))
//!
//! q   = LN(E[question_ids]) + type_emb[qtype]
//! for each question layer j:
//!     q = q + Wo(Attn(RoPE(q), RoPE(k), v, question mask))
//!     q = q + Wo_c(CrossAttn(Wq_c LN(q), K_j, V_j))    query heads grouped onto kv_heads
//!     q = q + Wo(GELU(a) * g)
//! Q   = LN(q)                                          question.final_norm
//! m_i = Q[marker_i] + ord_emb([i/(k-1), (i/(k-1))^2, sin(pi i/(k-1)), cos(pi i/(k-1))])   score only
//! z_i = Wb(GELU(Wa LN(m_i))) + type_bias[qtype]
//! ```
//!
//! A question row is a header and option segments, each segment starting with its marker. The
//! header takes positions 0 to h - 1 and every segment h onwards, so RoPE gives no order between
//! options. In the question mask a header token sees the header and every marker, and an option
//! token sees the header, its own segment and every marker. Question layers attend globally with
//! the global RoPE base.
//!
//! Every linear and LayerNorm has no bias, as in ModernBERT. The state tower and the embedding use
//! ModernBERT's tensor names under `state.`, so a ModernBERT or Ettin checkpoint maps onto them.

use serde_json::Value;

use crate::error::{Error, Result};
use crate::tensors::{Tensors, check};

/// Head dimension of every attention in the family.
pub const HEAD_DIM: usize = 64;

/// Features of the ordinal embedding.
pub const ORD_FEATURES: usize = 4;

/// The special tokens kime adds after the base vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct Specials {
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    pub opt: u32,
    pub no: u32,
    pub yes: u32,
    pub cls_s: u32,
    pub cls_q: u32,
}

/// A kime-v1 checkpoint's `kime.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct V1Spec {
    /// The model id, such as `kime-v1-s-en`.
    pub id: String,
    /// The tokenizer's special ids.
    pub specials: Specials,
    /// Hidden size.
    pub d: usize,
    /// Attention heads, `d / 64`.
    pub heads: usize,
    /// GeGLU inner size.
    pub inter: usize,
    /// Rows of the embedding table.
    pub vocab: usize,
    /// State tower layers.
    pub state_layers: usize,
    /// Whether each state layer attends globally, false meaning the sliding window.
    pub global: Vec<bool>,
    /// Sliding window width in tokens.
    pub window: usize,
    /// RoPE base on global layers and in the question tower.
    pub rope_global: f64,
    /// RoPE base on sliding window layers.
    pub rope_local: f64,
    /// Top state layers that run over all segments in segment mode.
    pub segment_top_layers: usize,
    /// Longest state.
    pub state_max_tokens: usize,
    /// Question tower layers.
    pub question_layers: usize,
    /// Key and value heads of the cross attention.
    pub kv_heads: usize,
    /// Most option segments in one question row.
    pub max_options: usize,
    /// Longest question row.
    pub question_max_tokens: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
}

fn get<'a>(v: &'a Value, path: &str) -> Result<&'a Value> {
    v.pointer(path).ok_or_else(|| Error::format(format!("kime.json: missing {path}")))
}

fn usize_at(v: &Value, path: &str) -> Result<usize> {
    get(v, path)?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::format(format!("kime.json: {path} is not a non negative integer")))
}

fn u32_at(v: &Value, path: &str) -> Result<u32> {
    let n = usize_at(v, path)?;
    u32::try_from(n).map_err(|_| Error::format(format!("kime.json: {path} is past u32")))
}

fn f64_at(v: &Value, path: &str) -> Result<f64> {
    get(v, path)?
        .as_f64()
        .ok_or_else(|| Error::format(format!("kime.json: {path} is not a number")))
}

impl V1Spec {
    /// Reads `kime.json`.
    ///
    /// # Errors
    ///
    /// [`Error::Format`] naming the missing or bad field, a family other than `kime-v1`, a shape
    /// that does not add up, or a GELU other than `erf`.
    pub fn from_json(v: &Value) -> Result<Self> {
        let family = get(v, "/family")?.as_str().unwrap_or_default();
        if family != "kime-v1" {
            return Err(Error::format(format!("kime.json: family {family:?} is not kime-v1")));
        }
        let gelu = v.get("gelu").and_then(Value::as_str).unwrap_or("erf");
        if gelu != "erf" {
            return Err(Error::format(format!(
                "kime.json: gelu {gelu:?} is not supported, only erf"
            )));
        }
        let d = usize_at(v, "/dims/d")?;
        let heads = usize_at(v, "/dims/heads")?;
        let head_dim = usize_at(v, "/dims/head_dim")?;
        if head_dim != HEAD_DIM || heads == 0 || d != heads * HEAD_DIM {
            return Err(Error::format(format!(
                "kime.json: d {d} is not {heads} heads of {HEAD_DIM} (head_dim {head_dim})"
            )));
        }
        let state_layers = usize_at(v, "/state_tower/layers")?;
        let every = usize_at(v, "/state_tower/global_every")?.max(1);
        let kv_heads = usize_at(v, "/question_tower/kv_heads")?;
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(Error::format(format!(
                "kime.json: {heads} heads do not group onto {kv_heads} kv heads"
            )));
        }
        let segment_top_layers = usize_at(v, "/state_tower/segment_top_layers")?;
        if segment_top_layers > state_layers {
            return Err(Error::format(format!(
                "kime.json: segment_top_layers {segment_top_layers} is more than the {state_layers} state layers"
            )));
        }
        let specials = Specials {
            cls: u32_at(v, "/tokenizer/cls")?,
            sep: u32_at(v, "/tokenizer/sep")?,
            pad: u32_at(v, "/tokenizer/pad")?,
            opt: u32_at(v, "/tokenizer/specials/opt")?,
            no: u32_at(v, "/tokenizer/specials/no")?,
            yes: u32_at(v, "/tokenizer/specials/yes")?,
            cls_s: u32_at(v, "/tokenizer/specials/cls_s")?,
            cls_q: u32_at(v, "/tokenizer/specials/cls_q")?,
        };
        let vocab = usize_at(v, "/dims/vocab")?;
        let s = specials;
        if let Some(id) = [s.cls, s.sep, s.pad, s.opt, s.no, s.yes, s.cls_s, s.cls_q]
            .into_iter()
            .find(|&id| id as usize >= vocab)
        {
            return Err(Error::format(format!(
                "kime.json: special id {id} is past the vocabulary {vocab}"
            )));
        }
        Ok(Self {
            id: get(v, "/id")?.as_str().unwrap_or("kime-v1-custom").to_string(),
            specials,
            d,
            heads,
            inter: usize_at(v, "/dims/inter")?,
            vocab,
            state_layers,
            global: (0..state_layers).map(|i| i % every == 0).collect(),
            window: usize_at(v, "/state_tower/window")?,
            rope_global: f64_at(v, "/state_tower/rope_theta_global")?,
            rope_local: f64_at(v, "/state_tower/rope_theta_local")?,
            segment_top_layers,
            state_max_tokens: usize_at(v, "/state_tower/max_tokens")?,
            question_layers: usize_at(v, "/question_tower/layers")?,
            kv_heads,
            max_options: usize_at(v, "/question_tower/max_options_per_chunk")?,
            question_max_tokens: usize_at(v, "/question_tower/max_tokens")?,
            norm_eps: v.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5),
        })
    }

    /// Width of the cross attention keys and values, `kv_heads` heads of 64.
    #[must_use]
    pub fn kv(&self) -> usize {
        self.kv_heads * HEAD_DIM
    }

    /// Every tensor the checkpoint must hold, with its shape.
    #[must_use]
    pub fn expected(&self) -> Vec<(String, Vec<usize>)> {
        let (d, inter, kv) = (self.d, self.inter, self.kv());
        let mut out: Vec<(String, Vec<usize>)> = Vec::new();
        let mut put = |name: String, shape: &[usize]| out.push((name, shape.to_vec()));
        put("state.embeddings.tok_embeddings.weight".into(), &[self.vocab, d]);
        put("state.embeddings.norm.weight".into(), &[d]);
        for i in 0..self.state_layers {
            let p = format!("state.layers.{i}");
            if i > 0 {
                put(format!("{p}.attn_norm.weight"), &[d]);
            }
            put(format!("{p}.attn.Wqkv.weight"), &[3 * d, d]);
            put(format!("{p}.attn.Wo.weight"), &[d, d]);
            put(format!("{p}.mlp_norm.weight"), &[d]);
            put(format!("{p}.mlp.Wi.weight"), &[2 * inter, d]);
            put(format!("{p}.mlp.Wo.weight"), &[d, inter]);
        }
        put("state.final_norm.weight".into(), &[d]);
        put("type_emb.weight".into(), &[3, d]);
        for j in 0..self.question_layers {
            let p = format!("question.layers.{j}");
            put(format!("{p}.attn_norm.weight"), &[d]);
            put(format!("{p}.attn.Wqkv.weight"), &[3 * d, d]);
            put(format!("{p}.attn.Wo.weight"), &[d, d]);
            put(format!("{p}.cross_norm.weight"), &[d]);
            put(format!("{p}.cross.Wq.weight"), &[d, d]);
            put(format!("{p}.cross.Wk.weight"), &[kv, d]);
            put(format!("{p}.cross.Wv.weight"), &[kv, d]);
            put(format!("{p}.cross.Wo.weight"), &[d, d]);
            put(format!("{p}.mlp_norm.weight"), &[d]);
            put(format!("{p}.mlp.Wi.weight"), &[2 * inter, d]);
            put(format!("{p}.mlp.Wo.weight"), &[d, inter]);
        }
        put("question.final_norm.weight".into(), &[d]);
        put("ord_emb.weight".into(), &[d, ORD_FEATURES]);
        put("scorer.norm.weight".into(), &[d]);
        put("scorer.Wa.weight".into(), &[d, d]);
        put("scorer.Wb.weight".into(), &[1, d]);
        put("scorer.type_bias".into(), &[3]);
        out
    }
}

/// A tensor bound into the model, as its index in [`Tensors`].
pub type W = usize;

/// One state tower layer, a ModernBERT layer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub struct StateLayer {
    /// None on layer 0, where ModernBERT skips the norm.
    pub attn_norm: Option<W>,
    pub wqkv: W,
    pub wo: W,
    pub mlp_norm: W,
    pub wi: W,
    pub mlp_wo: W,
    pub global: bool,
    pub rope_theta: f64,
}

/// One question tower layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct QuestionLayer {
    pub attn_norm: W,
    pub wqkv: W,
    pub wo: W,
    pub cross_norm: W,
    pub cross_q: W,
    pub cross_k: W,
    pub cross_v: W,
    pub cross_o: W,
    pub mlp_norm: W,
    pub wi: W,
    pub mlp_wo: W,
}

/// The whole model with every weight bound.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct V1Graph {
    pub tok_embeddings: W,
    pub embed_norm: W,
    pub state: Vec<StateLayer>,
    pub state_norm: W,
    pub type_emb: W,
    pub question: Vec<QuestionLayer>,
    pub question_norm: W,
    pub ord_emb: W,
    pub scorer_norm: W,
    pub scorer_a: W,
    pub scorer_b: W,
    pub type_bias: W,
}

impl V1Graph {
    /// Binds every weight to `tensors`, checking names, shapes and dtypes.
    ///
    /// # Errors
    ///
    /// [`Error::Mismatch`] listing every missing tensor, extra tensor, wrong shape and non float
    /// dtype.
    ///
    /// # Panics
    ///
    /// Never. Binding looks up only names the check has found.
    pub fn bind(spec: &V1Spec, tensors: &Tensors) -> Result<Self> {
        check(&spec.expected(), tensors)?;
        let w = |name: &str| tensors.index(name).expect("checked above");
        let state = (0..spec.state_layers)
            .map(|i| {
                let p = format!("state.layers.{i}");
                StateLayer {
                    attn_norm: (i > 0).then(|| w(&format!("{p}.attn_norm.weight"))),
                    wqkv: w(&format!("{p}.attn.Wqkv.weight")),
                    wo: w(&format!("{p}.attn.Wo.weight")),
                    mlp_norm: w(&format!("{p}.mlp_norm.weight")),
                    wi: w(&format!("{p}.mlp.Wi.weight")),
                    mlp_wo: w(&format!("{p}.mlp.Wo.weight")),
                    global: spec.global[i],
                    rope_theta: if spec.global[i] { spec.rope_global } else { spec.rope_local },
                }
            })
            .collect();
        let question = (0..spec.question_layers)
            .map(|j| {
                let p = format!("question.layers.{j}");
                QuestionLayer {
                    attn_norm: w(&format!("{p}.attn_norm.weight")),
                    wqkv: w(&format!("{p}.attn.Wqkv.weight")),
                    wo: w(&format!("{p}.attn.Wo.weight")),
                    cross_norm: w(&format!("{p}.cross_norm.weight")),
                    cross_q: w(&format!("{p}.cross.Wq.weight")),
                    cross_k: w(&format!("{p}.cross.Wk.weight")),
                    cross_v: w(&format!("{p}.cross.Wv.weight")),
                    cross_o: w(&format!("{p}.cross.Wo.weight")),
                    mlp_norm: w(&format!("{p}.mlp_norm.weight")),
                    wi: w(&format!("{p}.mlp.Wi.weight")),
                    mlp_wo: w(&format!("{p}.mlp.Wo.weight")),
                }
            })
            .collect();
        Ok(Self {
            tok_embeddings: w("state.embeddings.tok_embeddings.weight"),
            embed_norm: w("state.embeddings.norm.weight"),
            state,
            state_norm: w("state.final_norm.weight"),
            type_emb: w("type_emb.weight"),
            question,
            question_norm: w("question.final_norm.weight"),
            ord_emb: w("ord_emb.weight"),
            scorer_norm: w("scorer.norm.weight"),
            scorer_a: w("scorer.Wa.weight"),
            scorer_b: w("scorer.Wb.weight"),
            type_bias: w("scorer.type_bias"),
        })
    }
}

/// Seeded random weights for every tensor of `spec`, in [`V1Spec::expected`] order, for tests
/// and the reference in tools/ref/kime_ref.py, which draws the same values.
///
/// One splitmix64 stream runs over all tensors in order. Each draw gives `u = (z >> 40) / 2^24 * 2
/// - 1` in [-1, 1), exact in f32. A norm weight is `1 + u / 10`, `scorer.type_bias` is `u / 10`,
/// and any other weight is `u / sqrt(columns)`, taken in f64 and rounded once.
#[must_use]
pub fn random_weights(spec: &V1Spec, seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f64 / f64::from(1u32 << 24) * 2.0 - 1.0
    };
    spec.expected()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let cols = *shape.last().unwrap_or(&1) as f64;
            let norm = name.ends_with("norm.weight");
            let bias = name == "scorer.type_bias";
            let v = (0..n)
                .map(|_| {
                    let u = next();
                    (if norm {
                        1.0 + u / 10.0
                    } else if bias {
                        u / 10.0
                    } else {
                        u / cols.sqrt()
                    }) as f32
                })
                .collect();
            (name, shape, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kime_tensor::Blob;
    use serde_json::{Map, json};

    pub(crate) fn tiny() -> Value {
        json!({
            "format": "kime/1", "family": "kime-v1", "id": "kime-v1-tiny", "version": "0.0.0",
            "tokenizer": {"kind": "bpe-bytelevel", "cls": 1, "sep": 2, "pad": 0,
                "specials": {"opt": 3, "no": 4, "yes": 5, "cls_s": 6, "cls_q": 7}},
            "dims": {"d": 256, "heads": 4, "head_dim": 64, "inter": 320, "vocab": 400},
            "state_tower": {"layers": 4, "global_every": 3, "window": 16, "rope_theta_global": 160000.0,
                "rope_theta_local": 10000.0, "segment_top_layers": 3, "max_tokens": 1024},
            "question_tower": {"layers": 2, "kv_heads": 2, "max_options_per_chunk": 32, "max_tokens": 512},
            "gelu": "erf", "norm_eps": 1e-5
        })
    }

    fn load(spec: &V1Spec, drop: usize) -> Tensors {
        let mut header = Map::new();
        let mut data = Vec::new();
        for (i, (name, shape, v)) in random_weights(spec, 7).into_iter().enumerate() {
            if i == drop {
                continue;
            }
            let start = data.len();
            v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes()));
            header.insert(
                name,
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let head = Value::Object(header).to_string().into_bytes();
        let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&head);
        bytes.extend_from_slice(&data);
        crate::safetensors::load(Blob::owned(bytes)).unwrap().0
    }

    #[test]
    fn spec_and_binding() {
        let spec = V1Spec::from_json(&tiny()).unwrap();
        assert_eq!(spec.global, [true, false, false, true]);
        assert_eq!(spec.kv(), 128);
        let g = V1Graph::bind(&spec, &load(&spec, usize::MAX)).unwrap();
        assert_eq!((g.state.len(), g.question.len()), (4, 2));
        assert!(g.state[0].attn_norm.is_none() && g.state[1].attn_norm.is_some());
        let Err(Error::Mismatch(p)) = V1Graph::bind(&spec, &load(&spec, 5)) else { panic!() };
        assert_eq!(p.len(), 1);
        assert!(p[0].starts_with("missing state.layers.0.mlp.Wi.weight"), "{p:?}");
    }

    #[test]
    fn bad_specs() {
        let mut v = tiny();
        v["family"] = json!("laya");
        assert!(V1Spec::from_json(&v).is_err());
        let mut v = tiny();
        v["question_tower"]["kv_heads"] = json!(3);
        assert!(V1Spec::from_json(&v).unwrap_err().to_string().contains("kv heads"));
        let mut v = tiny();
        v["tokenizer"]["specials"]["cls_q"] = json!(400);
        assert!(V1Spec::from_json(&v).unwrap_err().to_string().contains("past the vocabulary"));
    }

    #[test]
    fn random_weights_are_fixed() {
        let spec = V1Spec::from_json(&tiny()).unwrap();
        let w = random_weights(&spec, 7);
        assert_eq!(w.len(), spec.expected().len());
        // The first values the reference draws for the same seed, see tools/ref/kime_ref.py.
        let first = [-0.013_771_288_f32, -0.060_401_47, 0.050_095_08, 0.010_366_283];
        assert_eq!(w[0].2[..4], first);
        assert!(w[1].2.iter().all(|v| (0.9..=1.1).contains(v)), "a norm weight is near 1");
    }
}

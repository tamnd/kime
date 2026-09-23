//! The Laya compat graph on the reference kernels: ModernBERT or mmBERT, the type embedding, the
//! two layer decision head, the option scorer and the act head, as `DecisionModel.forward` in
//! Laya 0.3.7 runs them in FP32.
//!
//! This is the correctness reference. It allocates its buffers per call and converts the F16
//! weights to f32 once at load, which costs 1.7 GB for Laya English. The fast paths are checked
//! against it.

use kime_model::Model;
use kime_model::laya::{Affine, LayaGraph, LayaSpec};

use crate::attention::{HEAD, attention};
use crate::gemm::linear;
use crate::ops::{Rope, add, gelu, geglu, layer_norm};
use crate::par;

/// PyTorch's LayerNorm default, which the head and scorer use.
const TORCH_EPS: f64 = 1e-5;

/// One question, laid out as Laya lays it out.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a> {
    /// Token ids, `[CLS] ... [SEP]`.
    pub ids: &'a [u32],
    /// The position of each option's mask token.
    pub markers: &'a [u32],
    /// 0 for choice, 1 for score, 2 for noul.
    pub qtype: usize,
}

/// What the model says about one question, before temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// One logit per marker.
    pub logits: Vec<f32>,
    /// The act head's two logits.
    pub act: [f32; 2],
}

/// A compat checkpoint ready to run on the CPU.
#[derive(Debug)]
pub struct Compat {
    spec: LayaSpec,
    graph: LayaGraph,
    w: Vec<Vec<f32>>,
    threads: usize,
}

impl Compat {
    /// Converts the weights of `model` to f32, on `threads` threads, which the forward pass uses
    /// too.
    #[must_use]
    pub fn new(model: &Model, threads: usize) -> Self {
        let t = &model.tensors;
        let w = par::map(t.entries().len(), threads, |i| t.view(i).to_f32());
        Self { spec: model.spec.clone(), graph: model.graph.clone(), w, threads: threads.max(1) }
    }

    /// The checkpoint's configuration.
    #[must_use]
    pub fn spec(&self) -> &LayaSpec {
        &self.spec
    }

    fn w(&self, i: usize) -> &[f32] {
        &self.w[i]
    }

    /// `y = x wᵀ + b` for a bound weight.
    fn linear(&self, x: &[f32], k: usize, w: usize, b: Option<usize>) -> Vec<f32> {
        let m = x.len() / k;
        let n = self.w[w].len() / k;
        let mut y = vec![0f32; m * n];
        linear(x, m, k, self.w(w), n, b.map(|b| self.w(b)), &mut y, self.threads);
        y
    }

    fn affine(&self, x: &[f32], k: usize, a: Affine) -> Vec<f32> {
        self.linear(x, k, a.w, Some(a.b))
    }

    /// Runs a batch. Sequences are packed end to end, so a batch costs what its tokens cost.
    ///
    /// # Panics
    ///
    /// If a token id is past the vocabulary, a marker is past its sequence, or a qtype is not 0,
    /// 1 or 2.
    #[must_use]
    pub fn forward(&self, batch: &[Input<'_>]) -> Vec<Output> {
        let e = &self.spec.encoder;
        let g = &self.graph;
        let d = e.d;
        let heads = e.heads;
        assert_eq!(d, heads * HEAD);
        let mut cu = vec![0usize];
        for x in batch {
            cu.push(cu.last().unwrap() + x.ids.len());
        }
        let t = *cu.last().unwrap();
        let longest = batch.iter().map(|x| x.ids.len()).max().unwrap_or(0);

        // Embeddings, then the embedding norm.
        let emb = self.w(g.tok_embeddings);
        let mut h = Vec::with_capacity(t * d);
        for x in batch {
            for &id in x.ids {
                let id = id as usize;
                assert!(id < e.vocab, "token id {id} is past the vocabulary");
                h.extend_from_slice(&emb[id * d..(id + 1) * d]);
            }
        }
        let mut x = vec![0f32; t * d];
        layer_norm(&h, d, self.w(g.embed_norm), None, e.norm_eps, &mut x);
        std::mem::swap(&mut h, &mut x);

        // The encoder.
        let ropes: Vec<(f64, Rope)> = [e.rope_global, e.rope_local]
            .iter()
            .map(|&theta| (theta, Rope::new(theta, HEAD, longest)))
            .collect();
        let mut att = vec![0f32; t * d];
        let mut act = vec![0f32; t * e.inter];
        for layer in &g.layers {
            let xin = match layer.attn_norm {
                Some(n) => {
                    layer_norm(&h, d, self.w(n), None, e.norm_eps, &mut x);
                    &x
                }
                None => &h,
            };
            let mut qkv = self.linear(xin, d, layer.wqkv, None);
            let rope = &ropes.iter().find(|r| r.0.to_bits() == layer.rope_theta.to_bits()).unwrap().1;
            for s in 0..batch.len() {
                for (pos, i) in (cu[s]..cu[s + 1]).enumerate() {
                    let row = &mut qkv[i * 3 * d..(i + 1) * 3 * d];
                    for head in row[..2 * d].as_chunks_mut::<HEAD>().0 {
                        rope.apply(head, pos);
                    }
                }
            }
            let window = (!layer.global).then_some(e.window / 2);
            attention(&qkv, heads, &cu, window, &mut att, self.threads);
            add(&mut h, &self.linear(&att, d, layer.wo, None));
            layer_norm(&h, d, self.w(layer.mlp_norm), None, e.norm_eps, &mut x);
            let u = self.linear(&x, d, layer.wi, None);
            geglu(&u, e.inter, &mut act);
            add(&mut h, &self.linear(&act, e.inter, layer.mlp_wo, None));
        }
        layer_norm(&h, d, self.w(g.final_norm), None, e.norm_eps, &mut x);
        std::mem::swap(&mut h, &mut x);

        // The question type, added to every position.
        let types = self.w(g.type_emb);
        for (s, input) in batch.iter().enumerate() {
            assert!(input.qtype < 3, "qtype {} is not 0, 1 or 2", input.qtype);
            let te = &types[input.qtype * d..(input.qtype + 1) * d];
            for i in cu[s]..cu[s + 1] {
                add(&mut h[i * d..(i + 1) * d], te);
            }
        }

        // The decision head, PyTorch TransformerEncoderLayers with norm_first and ReLU.
        for l in &g.head {
            layer_norm(&h, d, self.w(l.norm1_w), Some(self.w(l.norm1_b)), TORCH_EPS, &mut x);
            let qkv = self.linear(&x, d, l.in_proj_w, Some(l.in_proj_b));
            attention(&qkv, heads, &cu, None, &mut att, self.threads);
            add(&mut h, &self.linear(&att, d, l.out_proj_w, Some(l.out_proj_b)));
            layer_norm(&h, d, self.w(l.norm2_w), Some(self.w(l.norm2_b)), TORCH_EPS, &mut x);
            let mut f = self.linear(&x, d, l.linear1_w, Some(l.linear1_b));
            f.iter_mut().for_each(|v| *v = v.max(0.0));
            let ffn = f.len() / t.max(1);
            add(&mut h, &self.linear(&f, ffn, l.linear2_w, Some(l.linear2_b)));
        }

        // The scorer, on every marker of every question at once.
        let mut m = Vec::new();
        for (s, input) in batch.iter().enumerate() {
            for &p in input.markers {
                let p = p as usize;
                assert!(p < input.ids.len(), "marker {p} is past its sequence");
                m.extend_from_slice(&h[(cu[s] + p) * d..(cu[s] + p + 1) * d]);
            }
        }
        let mut mn = vec![0f32; m.len()];
        let sn = g.scorer_norm;
        layer_norm(&m, d, self.w(sn.w), Some(self.w(sn.b)), TORCH_EPS, &mut mn);
        let mut z = self.affine(&mn, d, g.scorer_in);
        z.iter_mut().for_each(|v| *v = gelu(*v));
        let logits = self.affine(&z, d, g.scorer_out);

        // The act head: the CLS row and four numbers about the option distribution.
        let mut feats = Vec::with_capacity(batch.len() * (d + 4));
        let mut at = 0;
        let mut out = Vec::with_capacity(batch.len());
        for (s, input) in batch.iter().enumerate() {
            let k = input.markers.len();
            let l = logits[at..at + k].to_vec();
            at += k;
            let first = cu[s] * d;
            if cu[s + 1] > cu[s] {
                feats.extend_from_slice(&h[first..first + d]);
            } else {
                feats.extend(std::iter::repeat_n(0.0, d));
            }
            feats.extend_from_slice(&act_features(&l));
            out.push(Output { logits: l, act: [0.0; 2] });
        }
        let a = self.affine(&feats, d + 4, g.act_in);
        let a: Vec<f32> = a.iter().map(|&v| gelu(v)).collect();
        let a = self.affine(&a, a.len() / batch.len().max(1), g.act_out);
        for (s, o) in out.iter_mut().enumerate() {
            o.act = [a[2 * s], a[2 * s + 1]];
        }
        out
    }
}

/// `[top1, top1 - top2, entropy / ln k, k / 255]` over the softmax of the logits, with `k` at
/// least 2, as Laya computes them. A single option has a top2 of 0. No options at all is an error
/// in Laya, and gives zeros here apart from the count.
#[must_use]
pub fn act_features(logits: &[f32]) -> [f32; 4] {
    let kf = logits.len().max(2) as f32;
    if logits.is_empty() {
        return [0.0, 0.0, 0.0, kf / 255.0];
    }
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let sum: f32 = e.iter().sum();
    let p: Vec<f32> = e.iter().map(|x| x / sum).collect();
    let ent = -p.iter().map(|&q| q * q.max(1e-9).ln()).sum::<f32>() / kf.ln();
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let top1 = sorted[0];
    let top2 = sorted.get(1).copied().unwrap_or(0.0);
    [top1, top1 - top2, ent, kf / 255.0]
}

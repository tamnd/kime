//! The kime-v1 split encoder on the CPU in FP32, as spec/05-model.md and
//! [`kime_model::kime_v1`] describe it.
//!
//! [`Split::state`] runs the state tower once and keeps the state memory, the keys and values of
//! every question layer. [`Split::question`] runs one question row against a state memory. This
//! is the plain forward the reference is checked against: state layers use the packed attention
//! of the compat path, and the question tower's masked self attention and cross attention are
//! computed directly, row by row. Batching rows and fused kernels come with the scheduler.

use kime_model::Tensors;
use kime_model::kime_v1::{HEAD_DIM, ORD_FEATURES, V1Graph, V1Spec};

use crate::attention::attention;
use crate::gemm::linear;
use crate::ops::{Rope, add, geglu, gelu, layer_norm};
use crate::par;

/// What a question row reads from its state: the final state representation and, per question
/// layer, its keys and values.
#[derive(Debug, Clone)]
pub struct Memory {
    /// State tokens.
    pub tokens: usize,
    /// `[tokens, d]`, the state after the final norm.
    pub states: Vec<f32>,
    /// Per question layer, `[tokens, kv]` keys and values.
    pub kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// The mean of `states` over tokens, L2 normalized.
    pub pooled: Vec<f32>,
}

/// One question row: a header and option segments, each starting with its marker token.
#[derive(Debug, Clone, Copy)]
pub struct Row<'a> {
    /// 0 choice, 1 score, 2 noul.
    pub qtype: usize,
    /// `[CLS_Q] type_text instructions [SEP]`.
    pub header: &'a [u32],
    /// `[OPT] option [SEP]` for each option, or `[NO] ...` and `[YES] ...` for a noul.
    pub options: &'a [Vec<u32>],
}

/// A kime-v1 checkpoint ready to run on the CPU.
#[derive(Debug)]
pub struct Split {
    spec: V1Spec,
    graph: V1Graph,
    w: Vec<Vec<f32>>,
    threads: usize,
}

impl Split {
    /// Converts the weights to f32 on `threads` threads, which the forward uses too.
    #[must_use]
    pub fn new(spec: &V1Spec, graph: &V1Graph, t: &Tensors, threads: usize) -> Self {
        let w = par::map(t.entries().len(), threads, |i| t.view(i).to_f32());
        Self { spec: spec.clone(), graph: graph.clone(), w, threads: threads.max(1) }
    }

    /// The checkpoint's configuration.
    #[must_use]
    pub fn spec(&self) -> &V1Spec {
        &self.spec
    }

    fn w(&self, i: usize) -> &[f32] {
        &self.w[i]
    }

    fn linear(&self, x: &[f32], k: usize, w: usize) -> Vec<f32> {
        let m = x.len() / k;
        let n = self.w[w].len() / k;
        let mut y = vec![0f32; m * n];
        linear(x, m, k, self.w(w), n, None, &mut y, self.threads);
        y
    }

    fn norm(&self, x: &[f32], w: usize) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        layer_norm(x, self.spec.d, self.w(w), None, self.spec.norm_eps, &mut out);
        out
    }

    fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.spec.d;
        let emb = self.w(self.graph.tok_embeddings);
        let mut h = Vec::with_capacity(ids.len() * d);
        for &id in ids {
            let id = id as usize;
            assert!(id < self.spec.vocab, "token id {id} is past the vocabulary");
            h.extend_from_slice(&emb[id * d..(id + 1) * d]);
        }
        self.norm(&h, self.graph.embed_norm)
    }

    fn mlp(&self, h: &mut [f32], norm: usize, wi: usize, wo: usize) {
        let s = &self.spec;
        let x = self.norm(h, norm);
        let u = self.linear(&x, s.d, wi);
        let mut act = vec![0f32; u.len() / 2];
        geglu(&u, s.inter, &mut act);
        add(h, &self.linear(&act, s.inter, wo));
    }

    /// Runs the state tower over `[CLS_S] state [SEP]` and builds the state memory.
    ///
    /// # Panics
    ///
    /// If a token id is past the vocabulary.
    #[must_use]
    pub fn state(&self, ids: &[u32]) -> Memory {
        let s = &self.spec;
        let (d, t) = (s.d, ids.len());
        let mut h = self.embed(ids);
        let ropes = [Rope::new(s.rope_global, HEAD_DIM, t), Rope::new(s.rope_local, HEAD_DIM, t)];
        let mut att = vec![0f32; t * d];
        for layer in &self.graph.state {
            let x = match layer.attn_norm {
                Some(n) => self.norm(&h, n),
                None => h.clone(),
            };
            let mut qkv = self.linear(&x, d, layer.wqkv);
            let rope = &ropes[usize::from(!layer.global)];
            for (pos, row) in qkv.chunks_exact_mut(3 * d).enumerate() {
                for head in row[..2 * d].as_chunks_mut::<HEAD_DIM>().0 {
                    rope.apply(head, pos);
                }
            }
            let window = (!layer.global).then_some(s.window / 2);
            attention(&qkv, s.heads, &[0, t], window, &mut att, self.threads);
            add(&mut h, &self.linear(&att, d, layer.wo));
            self.mlp(&mut h, layer.mlp_norm, layer.wi, layer.mlp_wo);
        }
        let states = self.norm(&h, self.graph.state_norm);
        let kv = self
            .graph
            .question
            .iter()
            .map(|l| (self.linear(&states, d, l.cross_k), self.linear(&states, d, l.cross_v)))
            .collect();
        let mut pooled = vec![0f64; d];
        for row in states.chunks_exact(d) {
            pooled.iter_mut().zip(row).for_each(|(p, &v)| *p += f64::from(v));
        }
        let len = pooled.iter().map(|v| v * v).sum::<f64>().sqrt();
        let pooled =
            pooled.iter().map(|v| if len > 0.0 { (v / len) as f32 } else { 0.0 }).collect();
        Memory { tokens: t, states, kv, pooled }
    }

    /// The option logits of one question row against `mem`.
    ///
    /// # Panics
    ///
    /// If a token id is past the vocabulary, the qtype is not 0, 1 or 2, or an option segment is
    /// empty.
    #[must_use]
    pub fn question(&self, mem: &Memory, row: Row<'_>) -> Vec<f32> {
        let s = &self.spec;
        let g = &self.graph;
        let d = s.d;
        assert!(row.qtype < 3, "qtype {} is not 0, 1 or 2", row.qtype);
        assert!(row.options.iter().all(|o| !o.is_empty()), "an option segment is empty");

        // Tokens, positions, segments and markers.
        let hlen = row.header.len();
        let mut ids = row.header.to_vec();
        let mut pos: Vec<usize> = (0..hlen).collect();
        let mut seg = vec![0usize; hlen];
        let mut markers = Vec::with_capacity(row.options.len());
        for (i, o) in row.options.iter().enumerate() {
            markers.push(ids.len());
            ids.extend_from_slice(o);
            pos.extend(hlen..hlen + o.len());
            seg.extend(std::iter::repeat_n(i + 1, o.len()));
        }
        let t = ids.len();
        let mut is_marker = vec![false; t];
        markers.iter().for_each(|&m| is_marker[m] = true);
        let sees =
            |a: usize, b: usize| seg[b] == 0 || is_marker[b] || (seg[a] != 0 && seg[a] == seg[b]);

        let mut h = self.embed(&ids);
        let te = &self.w(g.type_emb)[row.qtype * d..(row.qtype + 1) * d];
        h.chunks_exact_mut(d).for_each(|r| add(r, te));

        let rope = Rope::new(s.rope_global, HEAD_DIM, pos.iter().max().map_or(0, |m| m + 1));
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let group = s.heads / s.kv_heads;
        let kv = s.kv();
        for (l, (k_mem, v_mem)) in g.question.iter().zip(&mem.kv) {
            // Self attention under the question mask.
            let x = self.norm(&h, l.attn_norm);
            let mut qkv = self.linear(&x, d, l.wqkv);
            for (i, r) in qkv.chunks_exact_mut(3 * d).enumerate() {
                for head in r[..2 * d].as_chunks_mut::<HEAD_DIM>().0 {
                    rope.apply(head, pos[i]);
                }
            }
            let mut att = vec![0f32; t * d];
            for a in 0..t {
                for hd in 0..s.heads {
                    let q = &qkv[a * 3 * d + hd * HEAD_DIM..][..HEAD_DIM];
                    let keys: Vec<usize> = (0..t).filter(|&b| sees(a, b)).collect();
                    let scores: Vec<f32> = keys
                        .iter()
                        .map(|&b| dot(q, &qkv[b * 3 * d + d + hd * HEAD_DIM..][..HEAD_DIM]) * scale)
                        .collect();
                    let vals =
                        keys.iter().map(|&b| &qkv[b * 3 * d + 2 * d + hd * HEAD_DIM..][..HEAD_DIM]);
                    mix(&scores, vals, &mut att[a * d + hd * HEAD_DIM..][..HEAD_DIM]);
                }
            }
            add(&mut h, &self.linear(&att, d, l.wo));

            // Cross attention into the state memory, query heads grouped onto the kv heads.
            let x = self.norm(&h, l.cross_norm);
            let q = self.linear(&x, d, l.cross_q);
            let mut att = vec![0f32; t * d];
            for a in 0..t {
                for hd in 0..s.heads {
                    let kh = hd / group;
                    let qa = &q[a * d + hd * HEAD_DIM..][..HEAD_DIM];
                    let scores: Vec<f32> = (0..mem.tokens)
                        .map(|b| dot(qa, &k_mem[b * kv + kh * HEAD_DIM..][..HEAD_DIM]) * scale)
                        .collect();
                    let vals =
                        (0..mem.tokens).map(|b| &v_mem[b * kv + kh * HEAD_DIM..][..HEAD_DIM]);
                    mix(&scores, vals, &mut att[a * d + hd * HEAD_DIM..][..HEAD_DIM]);
                }
            }
            add(&mut h, &self.linear(&att, d, l.cross_o));

            self.mlp(&mut h, l.mlp_norm, l.wi, l.mlp_wo);
        }
        let hq = self.norm(&h, g.question_norm);

        // The scorer on the markers, with the ordinal embedding for a score.
        let k = markers.len();
        let mut m = Vec::with_capacity(k * d);
        let ord = self.w(g.ord_emb);
        for (i, &at) in markers.iter().enumerate() {
            let mut v = hq[at * d..(at + 1) * d].to_vec();
            if row.qtype == 1 {
                let f = ord_features(i, k);
                for (j, o) in v.iter_mut().enumerate() {
                    *o += (0..ORD_FEATURES).map(|c| ord[j * ORD_FEATURES + c] * f[c]).sum::<f32>();
                }
            }
            m.extend_from_slice(&v);
        }
        let x = self.norm(&m, g.scorer_norm);
        let mut a = self.linear(&x, d, g.scorer_a);
        a.iter_mut().for_each(|v| *v = gelu(*v));
        let bias = self.w(g.type_bias)[row.qtype];
        self.linear(&a, d, g.scorer_b).into_iter().map(|z| z + bias).collect()
    }
}

/// The ordinal features of level `i` of `k`: `[x, x^2, sin(pi x), cos(pi x)]` with
/// `x = i / (k - 1)`, and `x = 0` for a single level.
#[must_use]
pub fn ord_features(i: usize, k: usize) -> [f32; ORD_FEATURES] {
    let x = if k > 1 { i as f64 / (k - 1) as f64 } else { 0.0 };
    let a = std::f64::consts::PI * x;
    [x as f32, (x * x) as f32, a.sin() as f32, a.cos() as f32]
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `out = softmax(scores) · values`, the softmax in f64.
fn mix<'a>(scores: &[f32], values: impl Iterator<Item = &'a [f32]>, out: &mut [f32]) {
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = scores.iter().map(|&s| f64::from(s - top).exp()).collect();
    let z: f64 = e.iter().sum();
    let mut acc = vec![0f64; out.len()];
    for (p, v) in e.iter().zip(values) {
        acc.iter_mut().zip(v).for_each(|(a, &x)| *a += p / z * f64::from(x));
    }
    out.iter_mut().zip(acc).for_each(|(o, a)| *o = a as f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinal_features() {
        let near = |a: [f32; ORD_FEATURES], b: [f32; ORD_FEATURES]| {
            a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-7)
        };
        assert!(near(ord_features(0, 5), [0.0, 0.0, 0.0, 1.0]));
        assert!(near(ord_features(4, 5), [1.0, 1.0, 0.0, -1.0]));
        assert!(near(ord_features(2, 5), [0.5, 0.25, 1.0, 0.0]));
        assert!(near(ord_features(0, 1), [0.0, 0.0, 0.0, 1.0]));
    }
}

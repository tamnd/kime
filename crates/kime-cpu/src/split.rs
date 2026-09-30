//! The kime-v1 split encoder on the CPU in FP32, as spec/05-model.md and
//! [`kime_model::kime_v1`] describe it.
//!
//! [`Split::state`] runs the state tower once and keeps the state memory, the keys and values of
//! every question layer. [`Split::state_segments`] does the same in segment mode, taking the
//! output of the low layers from a [`SegmentCache`] for every segment it has seen before. [`Split::question`] runs one question row against a state memory. This
//! is the plain forward the reference is checked against: state layers use the packed attention
//! of the compat path, and the question tower's masked self attention and cross attention are
//! computed directly, row by row. Batching rows and fused kernels come with the scheduler.

use std::collections::HashMap;

use kime_model::Tensors;
use kime_model::kime_v1::{HEAD_DIM, ORD_FEATURES, StateLayer, V1Graph, V1Spec};

use crate::attention::attention;
use crate::gemm::{Gemm, pack, scratch_len};
use crate::ops::{Rope, add, geglu, gelu, layer_norm};
use crate::par;
use kime_tensor::Epilogue;

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
    /// Per question layer and kv head, the keys `[tokens, 64]` and the values transposed
    /// `[64, tokens]`, packed for the GEMM so every question reads them as they are.
    pub cross: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
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

/// The output of the low state layers of segments, keyed by the blake3 of the model id and the
/// segment's token ids, with counts of what was found and what had to be computed.
///
/// It holds at most its byte budget in two generations of half the budget each: new entries go
/// in the young one, and when that is full the old one is dropped and the young one takes its
/// place. A hit in the old generation moves the entry back.
#[derive(Debug)]
pub struct SegmentCache {
    young: HashMap<[u8; 32], Vec<f32>>,
    old: HashMap<[u8; 32], Vec<f32>>,
    young_bytes: usize,
    budget: usize,
    /// Segments found in the cache.
    pub hits: usize,
    /// Segments computed.
    pub misses: usize,
    /// Tokens the low layers ran on.
    pub tokens_computed: usize,
}

impl Default for SegmentCache {
    fn default() -> Self {
        Self::with_budget(usize::MAX)
    }
}

impl SegmentCache {
    /// A cache that holds at most `bytes` of segment outputs.
    #[must_use]
    pub fn with_budget(bytes: usize) -> Self {
        Self {
            young: HashMap::new(),
            old: HashMap::new(),
            young_bytes: 0,
            budget: bytes,
            hits: 0,
            misses: 0,
            tokens_computed: 0,
        }
    }

    /// Segments held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.young.len() + self.old.len()
    }

    /// Whether no segment is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held, at most the budget.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.young_bytes + self.old.values().map(|v| v.len() * 4).sum::<usize>()
    }

    /// Clears the counts and keeps the entries.
    pub fn reset_counts(&mut self) {
        (self.hits, self.misses, self.tokens_computed) = (0, 0, 0);
    }

    fn get(&mut self, k: &[u8; 32]) -> Option<Vec<f32>> {
        if let Some(v) = self.young.get(k) {
            return Some(v.clone());
        }
        let v = self.old.remove(k)?;
        self.put(*k, v.clone());
        Some(v)
    }

    fn put(&mut self, k: [u8; 32], v: Vec<f32>) {
        let n = v.len() * 4;
        if n > self.budget / 2 {
            return;
        }
        if self.young_bytes + n > self.budget / 2 {
            self.old = std::mem::take(&mut self.young);
            self.young_bytes = 0;
        }
        self.young_bytes += n;
        if let Some(old) = self.young.insert(k, v) {
            self.young_bytes -= old.len() * 4;
        }
    }
}

fn segment_key(model: &str, ids: &[u32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(&(model.len() as u64).to_le_bytes());
    h.update(model.as_bytes());
    ids.iter().for_each(|id| {
        h.update(&id.to_le_bytes());
    });
    *h.finalize().as_bytes()
}

/// A kime-v1 checkpoint ready to run on the CPU.
#[derive(Debug)]
pub struct Split {
    spec: V1Spec,
    graph: V1Graph,
    w: Vec<Vec<f32>>,
    /// The rows of every weight, its first dimension.
    rows: Vec<usize>,
    threads: usize,
}

impl Split {
    /// Converts the weights to f32 on `threads` threads, which the forward uses too. The
    /// weights of the linears are packed for the GEMM once here, so a short question row does
    /// not pay for it on every call.
    #[must_use]
    pub fn new(spec: &V1Spec, graph: &V1Graph, t: &Tensors, threads: usize) -> Self {
        let g = graph;
        let mut linears: Vec<usize> = vec![g.scorer_a, g.scorer_b];
        for l in &g.state {
            linears.extend([l.wqkv, l.wo, l.wi, l.mlp_wo]);
        }
        for l in &g.question {
            linears
                .extend([l.wqkv, l.wo, l.cross_q, l.cross_k, l.cross_v, l.cross_o, l.wi, l.mlp_wo]);
        }
        let w = par::map(t.entries().len(), threads, |i| {
            let v = t.view(i).to_f32();
            let shape = &t.entries()[i].shape;
            if linears.contains(&i) { pack(&v, shape[0], shape[1]) } else { v }
        });
        let rows = t.entries().iter().map(|e| e.shape.first().copied().unwrap_or(1)).collect();
        Self { spec: spec.clone(), graph: graph.clone(), w, rows, threads: threads.max(1) }
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
        self.linear_on(x, k, w, self.threads)
    }

    /// `x wᵀ` for the packed weight `w` of `n` rows, `n` given since packing can pad.
    fn linear_on(&self, x: &[f32], k: usize, w: usize, threads: usize) -> Vec<f32> {
        let m = x.len() / k;
        let n = self.rows[w];
        let mut y = vec![0f32; m * n];
        gemm(x, m, k, self.w(w), n, &mut y, threads);
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

    fn mlp(&self, h: &mut [f32], norm: usize, wi: usize, wo: usize, threads: usize) {
        let s = &self.spec;
        let x = self.norm(h, norm);
        let u = self.linear_on(&x, s.d, wi, threads);
        let mut act = vec![0f32; u.len() / 2];
        geglu(&u, s.inter, &mut act);
        add(h, &self.linear_on(&act, s.inter, wo, threads));
    }

    /// Runs `layers` on `h` in place. `cu` holds the sequence boundaries, as in [`attention`],
    /// and `pos` the position of every token.
    fn layers(&self, h: &mut [f32], layers: &[StateLayer], pos: &[usize], cu: &[usize]) {
        let s = &self.spec;
        let (d, t) = (s.d, pos.len());
        let len = pos.iter().max().map_or(0, |m| m + 1);
        let ropes =
            [Rope::new(s.rope_global, HEAD_DIM, len), Rope::new(s.rope_local, HEAD_DIM, len)];
        let mut att = vec![0f32; t * d];
        for layer in layers {
            let x = match layer.attn_norm {
                Some(n) => self.norm(h, n),
                None => h.to_vec(),
            };
            let mut qkv = self.linear(&x, d, layer.wqkv);
            let rope = &ropes[usize::from(!layer.global)];
            for (row, &p) in qkv.chunks_exact_mut(3 * d).zip(pos) {
                for head in row[..2 * d].as_chunks_mut::<HEAD_DIM>().0 {
                    rope.apply(head, p);
                }
            }
            let window = (!layer.global).then_some(s.window / 2);
            attention(&qkv, s.heads, cu, window, &mut att, self.threads);
            add(h, &self.linear(&att, d, layer.wo));
            self.mlp(h, layer.mlp_norm, layer.wi, layer.mlp_wo, self.threads);
        }
    }

    /// The final norm, the keys and values of every question layer and the pooled embedding.
    fn memory(&self, h: &[f32]) -> Memory {
        let d = self.spec.d;
        let states = self.norm(h, self.graph.state_norm);
        let kv: Vec<(Vec<f32>, Vec<f32>)> = self
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
        let tokens = h.len() / d;
        let kvw = self.spec.kv();
        let cross = kv
            .iter()
            .map(|(k, v): &(Vec<f32>, Vec<f32>)| {
                (0..self.spec.kv_heads)
                    .map(|kh| {
                        let mut kt = Vec::with_capacity(tokens * HEAD_DIM);
                        let mut vt = vec![0f32; HEAD_DIM * tokens];
                        for b in 0..tokens {
                            kt.extend_from_slice(&k[b * kvw + kh * HEAD_DIM..][..HEAD_DIM]);
                            for (c, &x) in
                                v[b * kvw + kh * HEAD_DIM..][..HEAD_DIM].iter().enumerate()
                            {
                                vt[c * tokens + b] = x;
                            }
                        }
                        (pack(&kt, tokens, HEAD_DIM), pack(&vt, HEAD_DIM, tokens))
                    })
                    .collect()
            })
            .collect();
        Memory { tokens, states, kv, pooled, cross }
    }

    /// Runs the state tower over `[CLS_S] state [SEP]` and builds the state memory.
    ///
    /// # Panics
    ///
    /// If a token id is past the vocabulary.
    #[must_use]
    pub fn state(&self, ids: &[u32]) -> Memory {
        let t = ids.len();
        let mut h = self.embed(ids);
        let pos: Vec<usize> = (0..t).collect();
        self.layers(&mut h, &self.graph.state, &pos, &[0, t]);
        self.memory(&h)
    }

    /// Runs the state tower in segment mode. The low layers run on each segment alone, with
    /// positions from 0, and only for segments `cache` does not hold. Then every token gets its
    /// segment's index embedding and the top `segment_top_layers` run over the whole state.
    ///
    /// # Panics
    ///
    /// If a segment is empty or a token id is past the vocabulary.
    #[must_use]
    pub fn state_segments(&self, segments: &[&[u32]], cache: &mut SegmentCache) -> Memory {
        let s = &self.spec;
        let d = s.d;
        assert!(segments.iter().all(|g| !g.is_empty()), "a state segment is empty");
        let low = s.state_layers - s.segment_top_layers;
        let keys: Vec<[u8; 32]> = segments.iter().map(|g| segment_key(&s.id, g)).collect();

        // The low layers on every segment not in the cache, all in one packed pass. A segment
        // that occurs twice is computed once.
        let mut outs: Vec<Option<Vec<f32>>> = vec![None; segments.len()];
        let mut todo: Vec<usize> = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            if let Some(v) = cache.get(k) {
                outs[i] = Some(v);
                cache.hits += 1;
            } else if todo.iter().any(|&j| keys[j] == *k) {
                cache.hits += 1;
            } else {
                todo.push(i);
            }
        }
        if !todo.is_empty() {
            let ids: Vec<u32> = todo.iter().flat_map(|&i| segments[i].iter().copied()).collect();
            let pos: Vec<usize> = todo.iter().flat_map(|&i| 0..segments[i].len()).collect();
            let mut cu = vec![0];
            todo.iter().for_each(|&i| cu.push(cu[cu.len() - 1] + segments[i].len()));
            let mut h = self.embed(&ids);
            self.layers(&mut h, &self.graph.state[..low], &pos, &cu);
            for (n, &i) in todo.iter().enumerate() {
                let v = h[cu[n] * d..cu[n + 1] * d].to_vec();
                cache.put(keys[i], v.clone());
                outs[i] = Some(v);
            }
            cache.misses += todo.len();
            cache.tokens_computed += ids.len();
        }

        // The top layers over the whole state.
        let seg = self.w(self.graph.segment_emb);
        let mut h = Vec::with_capacity(segments.iter().map(|g| g.len()).sum::<usize>() * d);
        for (i, k) in keys.iter().enumerate() {
            let row = i.min(s.max_segments - 1);
            let e = &seg[row * d..(row + 1) * d];
            let at = h.len();
            // A repeat of a segment computed in this call takes the first one's output.
            let first = keys.iter().position(|x| x == k).unwrap_or(i);
            let out = outs[i].as_ref().or(outs[first].as_ref()).expect("found or computed above");
            h.extend_from_slice(out);
            h[at..].chunks_exact_mut(d).for_each(|r| add(r, e));
        }
        let t = h.len() / d;
        let pos: Vec<usize> = (0..t).collect();
        self.layers(&mut h, &self.graph.state[low..], &pos, &[0, t]);
        self.memory(&h)
    }

    /// The option logits of one question row against `mem`.
    ///
    /// # Panics
    ///
    /// If a token id is past the vocabulary, the qtype is not 0, 1 or 2, or an option segment is
    /// empty.
    #[must_use]
    pub fn question(&self, mem: &Memory, row: Row<'_>) -> Vec<f32> {
        self.question_on(mem, row, self.threads)
    }

    /// The option logits of many rows, each against its own memory. Rows are short, so they
    /// run side by side on one thread each rather than one after another on all of them.
    ///
    /// # Panics
    ///
    /// As [`Split::question`].
    #[must_use]
    pub fn questions(&self, rows: &[(&Memory, Row<'_>)]) -> Vec<Vec<f32>> {
        if rows.len() == 1 {
            return vec![self.question(rows[0].0, rows[0].1)];
        }
        par::map(rows.len(), self.threads, |i| self.question_on(rows[i].0, rows[i].1, 1))
    }

    fn question_on(&self, mem: &Memory, row: Row<'_>, threads: usize) -> Vec<f32> {
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
        for (l, cross) in g.question.iter().zip(&mem.cross) {
            // Self attention under the question mask.
            let x = self.norm(&h, l.attn_norm);
            let mut qkv = self.linear_on(&x, d, l.wqkv, threads);
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
            add(&mut h, &self.linear_on(&att, d, l.wo, threads));

            // Cross attention into the state memory, query heads grouped onto the kv heads: for
            // each kv head the scores of all its query heads are one GEMM against the packed
            // keys, and the mix of values one more against the packed transposed values.
            let x = self.norm(&h, l.cross_norm);
            let q = self.linear_on(&x, d, l.cross_q, threads);
            let mut att = vec![0f32; t * d];
            let n = mem.tokens;
            for (kh, (kp, vp)) in cross.iter().enumerate() {
                let heads = kh * group..(kh + 1) * group;
                let mut qs = Vec::with_capacity(group * t * HEAD_DIM);
                for hd in heads.clone() {
                    for a in 0..t {
                        qs.extend(
                            q[a * d + hd * HEAD_DIM..][..HEAD_DIM].iter().map(|&v| v * scale),
                        );
                    }
                }
                let m = group * t;
                let mut p = vec![0f32; m * n];
                gemm(&qs, m, HEAD_DIM, kp, n, &mut p, threads);
                for row in p.chunks_exact_mut(n) {
                    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f64> = row.iter().map(|&s| f64::from(s - top).exp()).collect();
                    let z: f64 = e.iter().sum();
                    row.iter_mut().zip(e).for_each(|(r, e)| *r = (e / z) as f32);
                }
                let mut o = vec![0f32; m * HEAD_DIM];
                gemm(&p, m, n, vp, HEAD_DIM, &mut o, threads);
                for (g, hd) in heads.enumerate() {
                    for a in 0..t {
                        att[a * d + hd * HEAD_DIM..][..HEAD_DIM]
                            .copy_from_slice(&o[(g * t + a) * HEAD_DIM..][..HEAD_DIM]);
                    }
                }
            }
            add(&mut h, &self.linear_on(&att, d, l.cross_o, threads));

            self.mlp(&mut h, l.mlp_norm, l.wi, l.mlp_wo, threads);
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
        let mut a = self.linear_on(&x, d, g.scorer_a, threads);
        a.iter_mut().for_each(|v| *v = gelu(*v));
        let bias = self.w(g.type_bias)[row.qtype];
        self.linear_on(&a, d, g.scorer_b, threads).into_iter().map(|z| z + bias).collect()
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
/// `y = x wᵀ` for `x` `[m, k]` and `w` `[n, k]` as [`pack`] lays it out.
fn gemm(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, y: &mut [f32], threads: usize) {
    let g = Gemm { x, m, k, w, n, b: None, ep: Epilogue::None };
    let len = scratch_len(k, n);
    g.run(y, threads, |tasks, f| par::for_each(tasks, threads, |t| f(t, &mut vec![0.0; len])));
}

fn mix<'a>(scores: &[f32], values: impl Iterator<Item = &'a [f32]>, out: &mut [f32]) {
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = scores.iter().map(|&s| f64::from(s - top).exp()).collect();
    let z: f64 = e.iter().sum();
    let mut acc = vec![0f64; out.len()];
    for (p, v) in e.iter().zip(values) {
        let w = p / z;
        acc.iter_mut().zip(v).for_each(|(a, &x)| *a += w * f64::from(x));
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

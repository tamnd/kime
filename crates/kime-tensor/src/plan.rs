//! The op graph a model builder emits and a backend lowers, see spec/07-engine.md.
//!
//! A graph names its activations as [`Val`]s whose size is a row count times a width, and the row
//! count is one of the three a batch has: tokens, sequences or markers. Nothing in a graph knows the
//! batch size. A backend lowers the graph once per [`Bucket`](crate::Bucket), which fixes the row
//! counts, and [`layout`] gives every value an offset in one arena, reusing space once a value's
//! last reader has run.

/// An activation in a graph, an index into [`Graph::vals`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Val(pub u32);

/// A weight, the index of a tensor in the checkpoint the graph was built from.
pub type W = usize;

/// What a value has one row per.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rows {
    /// One row per token of the batch.
    Tokens,
    /// One row per sequence.
    Seqs,
    /// One row per marker.
    Markers,
}

/// The shape of a value: `rows` rows of `width` f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// What there is one row per.
    pub rows: Rows,
    /// Elements per row.
    pub width: usize,
}

/// What a GEMM does to each output element after the bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epilogue {
    /// Store it.
    None,
    /// Store the exact GELU of it.
    Gelu,
    /// Store max(0, x).
    Relu,
    /// Add it to what `out` already holds, the residual connection.
    Accumulate,
}

/// One step of a graph. Every op runs on the rows the batch has, not the bucket's padded count.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `out[i] = table[ids[i]]`, token rows.
    Embed {
        /// `[vocab, width]`.
        table: W,
        /// Token rows.
        out: Val,
    },
    /// LayerNorm over each row, statistics in f64.
    LayerNorm {
        /// Input.
        x: Val,
        /// Scale.
        w: W,
        /// Shift, if the norm has one.
        b: Option<W>,
        /// Epsilon.
        eps: f64,
        /// Output, the same shape as `x`.
        out: Val,
    },
    /// `out = epilogue(a wᵀ + b)` with `w` as `[n, k]`.
    Gemm {
        /// Input, `k` wide.
        a: Val,
        /// `[n, k]`.
        w: W,
        /// `[n]`, if there is one.
        b: Option<W>,
        /// What happens to each result.
        epilogue: Epilogue,
        /// Output, `n` wide, with as many rows as `a`.
        out: Val,
    },
    /// Rotary embedding applied in place to the q and k parts of fused `[q | k | v]` rows, with
    /// positions counted from the start of each sequence.
    Rope {
        /// Token rows, `3 heads 64` wide.
        qkv: Val,
        /// The base.
        theta: f64,
    },
    /// Self attention within each sequence over fused `[q | k | v]` rows.
    Attention {
        /// Token rows, `3 heads 64` wide.
        qkv: Val,
        /// Keys at most this far away on either side, or all of them.
        window: Option<usize>,
        /// Token rows, `heads 64` wide.
        out: Val,
    },
    /// `out = gelu(x[:, ..n]) * x[:, n..]` with `x` twice as wide as `out`.
    GeGlu {
        /// Input.
        x: Val,
        /// Output.
        out: Val,
    },
    /// Adds `table[qtype[s]]` to every row of sequence `s`, in place.
    AddType {
        /// Token rows.
        h: Val,
        /// `[types, width]`.
        table: W,
    },
    /// The row of each marker.
    GatherMarkers {
        /// Token rows.
        h: Val,
        /// Marker rows.
        out: Val,
    },
    /// Per sequence, its first row followed by `[top1, top1 - top2, entropy / ln k, k / 255]` over
    /// the softmax of its markers' logits, Laya's act head input.
    ActFeatures {
        /// Token rows.
        h: Val,
        /// Marker rows, one wide.
        logits: Val,
        /// Sequence rows, `h` width plus four.
        out: Val,
    },
}

impl Op {
    /// Values the op reads, then values it writes. An in place op lists the value in both.
    #[must_use]
    pub fn uses(&self) -> (Vec<Val>, Vec<Val>) {
        match *self {
            Op::Embed { out, .. } => (vec![], vec![out]),
            Op::LayerNorm { x, out, .. } | Op::GeGlu { x, out } => (vec![x], vec![out]),
            Op::Gemm { a, epilogue, out, .. } => {
                if epilogue == Epilogue::Accumulate {
                    (vec![a, out], vec![out])
                } else {
                    (vec![a], vec![out])
                }
            }
            Op::Rope { qkv, .. } => (vec![qkv], vec![qkv]),
            Op::Attention { qkv, out, .. } => (vec![qkv], vec![out]),
            Op::AddType { h, .. } => (vec![h], vec![h]),
            Op::GatherMarkers { h, out } => (vec![h], vec![out]),
            Op::ActFeatures { h, logits, out } => (vec![h, logits], vec![out]),
        }
    }
}

/// A model's forward pass as ops over values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    /// The shape of every value.
    pub vals: Vec<Shape>,
    /// The ops, in the order they run.
    pub ops: Vec<Op>,
    /// Marker rows, one wide, one logit per option.
    pub logits: Option<Val>,
    /// Sequence rows, two wide, the act head.
    pub act: Option<Val>,
}

impl Graph {
    /// A new value.
    ///
    /// # Panics
    ///
    /// Past 2^32 values.
    pub fn val(&mut self, rows: Rows, width: usize) -> Val {
        self.vals.push(Shape { rows, width });
        Val(u32::try_from(self.vals.len() - 1).expect("fewer than 2^32 values"))
    }

    /// The shape of `v`.
    #[must_use]
    pub fn shape(&self, v: Val) -> Shape {
        self.vals[v.0 as usize]
    }

    /// Appends an op.
    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }
}

/// Values packed into one arena, in f32 elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Where each value starts, indexed like [`Graph::vals`].
    pub offsets: Vec<usize>,
    /// The arena's length.
    pub len: usize,
}

/// Offsets are multiples of this, 64 bytes, a cache line.
pub const ALIGN: usize = 16;

/// Lays the values of `graph` out in one arena for the row counts `rows`, by a greedy interval
/// colouring over their live ranges. A value lives from the first op that touches it to the last,
/// and the outputs live to the end. Two values whose ranges overlap never share space, so an op's
/// inputs and outputs never alias unless the op is in place on one value. Values are placed
/// largest first at the lowest offset that fits.
#[must_use]
pub fn layout(graph: &Graph, rows: impl Fn(Rows) -> usize) -> Layout {
    let n = graph.vals.len();
    let mut live = vec![(usize::MAX, 0usize); n];
    for (i, op) in graph.ops.iter().enumerate() {
        let (r, w) = op.uses();
        for v in r.into_iter().chain(w) {
            let l = &mut live[v.0 as usize];
            l.0 = l.0.min(i);
            l.1 = l.1.max(i);
        }
    }
    for v in [graph.logits, graph.act].into_iter().flatten() {
        let l = &mut live[v.0 as usize];
        l.0 = l.0.min(graph.ops.len());
        l.1 = usize::MAX;
    }
    let size: Vec<usize> =
        graph.vals.iter().map(|s| (rows(s.rows) * s.width).next_multiple_of(ALIGN)).collect();
    let mut order: Vec<usize> = (0..n).filter(|&v| live[v].0 != usize::MAX).collect();
    order.sort_by_key(|&v| (std::cmp::Reverse(size[v]), v));
    let mut offsets = vec![0usize; n];
    let mut placed: Vec<usize> = Vec::new();
    let mut len = 0;
    for v in order {
        let (a, b) = live[v];
        let mut taken: Vec<(usize, usize)> = placed
            .iter()
            .filter(|&&u| live[u].0 <= b && a <= live[u].1)
            .map(|&u| (offsets[u], offsets[u] + size[u]))
            .collect();
        taken.sort_unstable();
        let mut at = 0;
        for (lo, hi) in taken {
            if at + size[v] <= lo {
                break;
            }
            at = at.max(hi);
        }
        offsets[v] = at;
        len = len.max(at + size[v]);
        placed.push(v);
    }
    Layout { offsets, len }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(width: usize, steps: usize) -> Graph {
        let mut g = Graph::default();
        let mut x = g.val(Rows::Tokens, width);
        g.push(Op::Embed { table: 0, out: x });
        for _ in 0..steps {
            let y = g.val(Rows::Tokens, width);
            g.push(Op::LayerNorm { x, w: 1, b: None, eps: 1e-5, out: y });
            x = y;
        }
        g.logits = Some(x);
        g
    }

    #[test]
    fn a_chain_needs_two_buffers() {
        let g = chain(10, 20);
        let l = layout(&g, |_| 7);
        let one = (7 * 10usize).next_multiple_of(ALIGN);
        assert_eq!(l.len, 2 * one);
        for (i, op) in g.ops.iter().enumerate().skip(1) {
            let Op::LayerNorm { x, out, .. } = *op else { unreachable!() };
            assert_ne!(l.offsets[x.0 as usize], l.offsets[out.0 as usize], "op {i} aliases");
        }
    }

    #[test]
    fn live_values_never_overlap() {
        // A residual stream that lives throughout, with temporaries of several sizes around it.
        let mut g = Graph::default();
        let h = g.val(Rows::Tokens, 8);
        g.push(Op::Embed { table: 0, out: h });
        for i in 0..6 {
            let a = g.val(Rows::Tokens, 8 * (i % 3 + 1));
            let b = g.val(Rows::Seqs, 3);
            g.push(Op::Gemm { a: h, w: 0, b: None, epilogue: Epilogue::None, out: a });
            g.push(Op::GatherMarkers { h: a, out: b });
            g.push(Op::Gemm { a, w: 0, b: None, epilogue: Epilogue::Accumulate, out: h });
        }
        g.act = Some(h);
        let rows = |r| match r {
            Rows::Tokens => 33,
            Rows::Seqs => 5,
            Rows::Markers => 9,
        };
        let l = layout(&g, rows);
        let mut live = vec![(usize::MAX, 0); g.vals.len()];
        for (i, op) in g.ops.iter().enumerate() {
            let (r, w) = op.uses();
            for v in r.into_iter().chain(w) {
                let x = &mut live[v.0 as usize];
                *x = (x.0.min(i), x.1.max(i));
            }
        }
        live[h.0 as usize].1 = usize::MAX;
        for u in 0..g.vals.len() {
            for v in 0..u {
                let time = live[u].0 <= live[v].1 && live[v].0 <= live[u].1;
                let su = rows(g.vals[u].rows) * g.vals[u].width;
                let sv = rows(g.vals[v].rows) * g.vals[v].width;
                let space = l.offsets[u] < l.offsets[v] + sv && l.offsets[v] < l.offsets[u] + su;
                assert!(!(time && space), "values {u} and {v} overlap");
                assert!(l.offsets[u] + su <= l.len);
            }
        }
        assert!(l.len < g.vals.iter().map(|s| rows(s.rows) * s.width).sum::<usize>());
    }
}

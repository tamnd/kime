//! Multi head self attention over a batch of sequences packed end to end.
//!
//! The input is the fused projection, one row of `[q | k | v]` per token with each part `heads`
//! vectors of `HEAD` wide, the layout of ModernBERT's `Wqkv` and of `in_proj` in PyTorch's
//! `MultiheadAttention`. `cu` holds the sequence boundaries, so `cu[s]..cu[s + 1]` are the rows of
//! sequence `s`, and no token sees another sequence. A window `w` limits each token to keys at most
//! `w` positions away on either side, which is ModernBERT's local attention with `w = 64`.
//!
//! Everything is f32, the way PyTorch's attention computes it, and each sum runs in a fixed order
//! inside one task, so the result does not depend on the thread count either.

use crate::ops::exp_neg;
use crate::par::{self, Shared};

/// Width of one head. Every model kime runs uses 64.
pub const HEAD: usize = 64;
/// Query rows per task.
pub const QB: usize = 32;

/// Attention for all heads of all sequences. `qkv` is `[t, 3 heads HEAD]` and `out` is
/// `[t, heads HEAD]`.
///
/// # Panics
///
/// If the shapes do not match, or `cu` is not a non decreasing list starting at 0 and ending at
/// `t`.
pub fn attention(
    qkv: &[f32],
    heads: usize,
    cu: &[usize],
    window: Option<usize>,
    out: &mut [f32],
    threads: usize,
) {
    let d = heads * HEAD;
    let stride = 3 * d;
    let t = out.len() / d.max(1);
    assert_eq!(out.len(), t * d);
    assert_eq!(qkv.len(), t * stride);
    assert!(cu.first() == Some(&0) && cu.last() == Some(&t), "cu does not cover the batch");
    assert!(cu.windows(2).all(|p| p[0] <= p[1]), "cu is not sorted");
    if t == 0 || heads == 0 {
        return;
    }

    // One task per (sequence, head, block of queries).
    let mut tasks = Vec::new();
    for s in 0..cu.len() - 1 {
        for q0 in (cu[s]..cu[s + 1]).step_by(QB) {
            for h in 0..heads {
                tasks.push((s, h, q0));
            }
        }
    }
    let shared = Shared::new(out);
    par::for_each(tasks.len(), threads, |task| {
        let (s, h, q0) = tasks[task];
        // SAFETY: rows q0 to q0 + QB of head h belong to this task alone.
        unsafe { block(qkv, heads, (cu[s], cu[s + 1]), q0, h, window, &mut Vec::new(), &shared) };
    });
}

/// Attention for query rows `q0..q0 + QB` of head `h` of the sequence in rows `lo..hi`, with `p`
/// as scratch. This is one task of [`attention`], exposed so a plan can run it with scratch it
/// owns.
///
/// Queries go `QT` at a time. Their scores are taken against `KT` keys at once, with all sixteen
/// running sums in registers, and the weighted sum of values keeps a `QT` by `CT` tile of the
/// output in registers while it walks the keys, so each key and value row is loaded once per
/// group of queries instead of once per query. The softmax is in f32 with a vectorized exp, the
/// sums in eight lanes, in key order, so the result does not depend on the thread count.
///
/// # Panics
///
/// If `qkv` does not hold the rows it names.
///
/// # Safety
///
/// No other thread may touch those rows of head `h` in `out` meanwhile.
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn block(
    qkv: &[f32],
    heads: usize,
    (lo, hi): (usize, usize),
    q0: usize,
    h: usize,
    window: Option<usize>,
    p: &mut Vec<f32>,
    out: &Shared<'_>,
) {
    let d = heads * HEAD;
    let stride = 3 * d;
    let scale = 1.0 / (HEAD as f32).sqrt();
    let q1 = (q0 + QB).min(hi);
    if q0 >= q1 {
        return;
    }
    let span = |i: usize| match window {
        Some(w) => (i.saturating_sub(w).max(lo), (i + w + 1).min(hi)),
        None => (lo, hi),
    };
    let (ka, kb) = (span(q0).0, span(q1 - 1).1);
    // Keys rounded up to whole tiles. Past kb a tile reads the last key again, with weight zero.
    let np = (kb - ka).next_multiple_of(KT);
    let key = |j: usize| (ka + j).min(kb - 1) * stride + h * HEAD;
    p.clear();
    p.resize(QT * np, 0.0);
    let s = &mut p[..];
    for i0 in (q0..q1).step_by(QT) {
        let m = (q1 - i0).min(QT);
        // The keys any of these queries sees, in whole tiles.
        let t0 = (span(i0).0 - ka) / KT * KT;
        let t1 = (span(i0 + m - 1).1 - ka).next_multiple_of(KT);
        let mut qs = [[0f32; HEAD]; QT];
        for (r, q) in qs.iter_mut().enumerate().take(m) {
            let src = &qkv[(i0 + r) * stride + h * HEAD..][..HEAD];
            q.iter_mut().zip(src).for_each(|(q, &x)| *q = x * scale);
        }
        for j in (t0..t1).step_by(KT) {
            let ks = std::array::from_fn(|t| qkv[d + key(j + t)..][..HEAD].try_into().unwrap());
            let t = scores(&qs, ks);
            for (r, t) in t.iter().enumerate() {
                s[r * np + j..][..KT].copy_from_slice(t);
            }
        }
        // Softmax over each query's own keys, with zeros elsewhere in the tiles.
        let mut inv = [0f32; QT];
        for (r, inv) in inv.iter_mut().enumerate() {
            let row = &mut s[r * np + t0..r * np + t1];
            if r >= m {
                row.fill(0.0);
                continue;
            }
            let (a, b) = span(i0 + r);
            let (a, b) = (a - ka - t0, b - ka - t0);
            row[..a].fill(0.0);
            row[b..].fill(0.0);
            let row = &mut row[a..b];
            let mx = lanes(row, f32::NEG_INFINITY, f32::max)
                .into_iter()
                .fold(f32::NEG_INFINITY, f32::max);
            row.iter_mut().for_each(|x| *x = exp_neg(*x - mx));
            *inv = 1.0 / lanes(row, 0.0, |a, b| a + b).iter().sum::<f32>();
        }
        for c in (0..HEAD).step_by(CT) {
            let mut o = [[0f32; CT]; QT];
            for j in t0..t1 {
                let v: &[f32; CT] = qkv[2 * d + key(j) + c..][..CT].try_into().unwrap();
                for (r, or) in o.iter_mut().enumerate() {
                    let e = s[r * np + j];
                    or.iter_mut().zip(v).for_each(|(o, &v)| *o = madd(*o, e, v));
                }
            }
            for (r, (o, inv)) in o.iter().zip(inv).enumerate().take(m) {
                for (x, &v) in o.iter().enumerate() {
                    // SAFETY: the caller owns row i0 + r of head h.
                    unsafe { out.set((i0 + r) * d + h * HEAD + c + x, v * inv) };
                }
            }
        }
    }
}

/// Floats of scratch each task of [`block`] needs over sequences of at most `tokens` tokens.
#[must_use]
pub fn scratch_len(tokens: usize) -> usize {
    QT * tokens.next_multiple_of(KT)
}

/// Queries taken together.
const QT: usize = 4;
/// Keys per tile.
const KT: usize = 4;
/// Value channels per tile.
const CT: usize = 16;

/// Scores of the `QT` queries in `qs` against the `KT` keys in `ks`, four channels at a time in
/// each of the sixteen pairs, then the four lanes added.
#[inline(always)]
fn scores(qs: &[[f32; HEAD]; QT], ks: [&[f32; HEAD]; KT]) -> [[f32; KT]; QT] {
    let mut t = [[[0f32; 4]; KT]; QT];
    for c in (0..HEAD).step_by(4) {
        for (tr, q) in t.iter_mut().zip(qs) {
            for (tk, k) in tr.iter_mut().zip(ks) {
                for l in 0..4 {
                    tk[l] = madd(tk[l], q[c + l], k[c + l]);
                }
            }
        }
    }
    t.map(|tr| tr.map(|[a, b, c, d]| (a + c) + (b + d)))
}

/// `x` folded with `f` into eight running values, one per lane, so the loop vectorizes where a
/// single running value would wait on itself every step.
#[inline(always)]
fn lanes(x: &[f32], init: f32, f: impl Fn(f32, f32) -> f32) -> [f32; 8] {
    let mut acc = [init; 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for c in chunks {
        acc.iter_mut().zip(c).for_each(|(a, &v)| *a = f(*a, v));
    }
    for (a, &v) in acc.iter_mut().zip(rest) {
        *a = f(*a, v);
    }
    acc
}

/// `c + a * b`, fused where the machine always has it and two roundings elsewhere, since x86
/// builds without FMA would turn `mul_add` into a call.
#[inline(always)]
fn madd(c: f32, a: f32, b: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    return a.mul_add(b, c);
    #[cfg(not(target_arch = "aarch64"))]
    return c + a * b;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, close};

    fn naive(qkv: &[f32], heads: usize, cu: &[usize], window: Option<usize>) -> Vec<f32> {
        let d = heads * HEAD;
        let t = *cu.last().unwrap();
        let mut out = vec![0f32; t * d];
        for s in 0..cu.len() - 1 {
            for i in cu[s]..cu[s + 1] {
                for h in 0..heads {
                    let keys: Vec<usize> = (cu[s]..cu[s + 1])
                        .filter(|&j| window.is_none_or(|w| i.abs_diff(j) <= w))
                        .collect();
                    let sc: Vec<f64> = keys
                        .iter()
                        .map(|&j| {
                            (0..HEAD)
                                .map(|c| {
                                    f64::from(qkv[i * 3 * d + h * HEAD + c])
                                        * f64::from(qkv[j * 3 * d + d + h * HEAD + c])
                                })
                                .sum::<f64>()
                                / 8.0
                        })
                        .collect();
                    let mx = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let z: f64 = sc.iter().map(|x| (x - mx).exp()).sum();
                    for c in 0..HEAD {
                        let v: f64 = keys
                            .iter()
                            .zip(&sc)
                            .map(|(&j, x)| {
                                (x - mx).exp() / z
                                    * f64::from(qkv[j * 3 * d + 2 * d + h * HEAD + c])
                            })
                            .sum();
                        out[i * d + h * HEAD + c] = v as f32;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn matches_naive_with_empty_and_single_sequences() {
        let mut rng = Rng(5);
        let heads = 2;
        let d = heads * HEAD;
        // Lengths 0, 1, 3, 0, 70 and 1, so windows cut both inside and at the edges.
        let cu = [0, 0, 1, 4, 4, 74, 75];
        let t = 75;
        let qkv: Vec<f32> = rng.vec(t * 3 * d).iter().map(|x| x * 3.0).collect();
        for window in [None, Some(0), Some(1), Some(64)] {
            let want = naive(&qkv, heads, &cu, window);
            for threads in [1, 4] {
                let mut out = vec![f32::NAN; t * d];
                attention(&qkv, heads, &cu, window, &mut out, threads);
                close(&out, &want, 1e-5, &format!("{window:?}"));
            }
        }
        let mut none: Vec<f32> = Vec::new();
        attention(&[], heads, &[0], None, &mut none, 2);
    }
}

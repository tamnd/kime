//! Multi head self attention over a batch of sequences packed end to end.
//!
//! The input is the fused projection, one row of `[q | k | v]` per token with each part `heads`
//! vectors of `HEAD` wide, the layout of ModernBERT's `Wqkv` and of `in_proj` in PyTorch's
//! `MultiheadAttention`. `cu` holds the sequence boundaries, so `cu[s]..cu[s + 1]` are the rows of
//! sequence `s`, and no token sees another sequence. A window `w` limits each token to keys at most
//! `w` positions away on either side, which is ModernBERT's local attention with `w = 64`.
//!
//! Scores go through the same dot as the GEMMs. The softmax sum and the probability weighted sum
//! of values are taken in f64 in key order and rounded once, so the result does not depend on the
//! thread count either.

use crate::gemm::dot;
use crate::par::{self, Shared};

/// Width of one head. Every model kime runs uses 64.
pub const HEAD: usize = 64;
/// Query rows per task.
const QB: usize = 32;

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
    let scale = 1.0 / (HEAD as f32).sqrt();
    let shared = Shared::new(out);
    par::for_each(tasks.len(), threads, |task| {
        let (s, h, q0) = tasks[task];
        let (lo, hi) = (cu[s], cu[s + 1]);
        let mut p = Vec::new();
        let mut acc = [0f64; HEAD];
        for i in q0..(q0 + QB).min(hi) {
            let (a, b) = match window {
                Some(w) => (i.saturating_sub(w).max(lo), (i + w + 1).min(hi)),
                None => (lo, hi),
            };
            let q = &qkv[i * stride + h * HEAD..][..HEAD];
            p.clear();
            p.extend((a..b).map(|j| dot(q, &qkv[j * stride + d + h * HEAD..][..HEAD]) * scale));
            let mx = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f64;
            for x in &mut p {
                *x = (*x - mx).exp();
                sum += f64::from(*x);
            }
            acc.fill(0.0);
            for (j, &e) in (a..b).zip(&p) {
                let pj = f64::from(e) / sum;
                let v = &qkv[j * stride + 2 * d + h * HEAD..][..HEAD];
                for c in 0..HEAD {
                    acc[c] = pj.mul_add(f64::from(v[c]), acc[c]);
                }
            }
            for (c, &v) in acc.iter().enumerate() {
                // SAFETY: row i, head h belongs to this task alone.
                unsafe { shared.set(i * d + h * HEAD + c, v as f32) };
            }
        }
    });
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

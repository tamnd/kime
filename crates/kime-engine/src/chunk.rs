//! Choices scored in chunks, and the joint rerank of their best options, as spec/04-semantics.md
//! describes.
//!
//! A choice whose options do not all fit one sequence is cut into chunks, each laid out with the
//! same question header, and the logits of all chunks go through one softmax. When there were
//! several chunks and more than [`TOP`] options, the [`TOP`] most likely are scored again together
//! in one more sequence, and that pass shares out their mass:
//!
//! ```text
//! p_final[i] = P_top * p_joint[i]   for i in the top options
//! p_final[i] = p_chunked[i]         otherwise
//! ```

/// Options per chunk when a choice does not fit one sequence and the request names no size.
pub(crate) const CHUNK: usize = 32;

/// Options scored again together after a chunked pass.
pub(crate) const TOP: usize = 16;

/// The option indices of each chunk, in order: as few chunks of at most `size` as hold `k`
/// options, with sizes that differ by at most one, so no chunk is left with a lone option.
pub(crate) fn chunks(k: usize, size: usize) -> Vec<Vec<usize>> {
    let n = k.div_ceil(size.max(1)).max(1);
    let (base, extra) = (k / n, k % n);
    let mut out = Vec::with_capacity(n);
    let mut at = 0;
    for c in 0..n {
        let len = base + usize::from(c < extra);
        out.push((at..at + len).collect());
        at += len;
    }
    out
}

/// The [`TOP`] options with the highest logits, in option order. Ties go to the earlier option.
pub(crate) fn top(logits: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    idx.truncate(TOP);
    idx.sort_unstable();
    idx
}

/// The softmax of `logits / t`.
pub(crate) fn softmax(logits: &[f32], t: f64) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = logits.iter().map(|&v| ((f64::from(v) - f64::from(top)) / t).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

/// The final distribution from the chunked logits at temperature `t`, and the joint logits of the
/// options at `picked` at temperature `t_joint`.
pub(crate) fn merge(
    chunked: &[f32],
    t: f64,
    picked: &[usize],
    joint: &[f32],
    t_joint: f64,
) -> Vec<f64> {
    let mut p = softmax(chunked, t);
    let mass: f64 = picked.iter().map(|&i| p[i]).sum();
    for (&i, q) in picked.iter().zip(softmax(joint, t_joint)) {
        p[i] = mass * q;
    }
    p
}

/// Logits that give `p` back through a softmax at temperature `t`, so an answer is built from a
/// merged distribution the way it is from one pass.
pub(crate) fn logits(p: &[f64], t: f64) -> Vec<f32> {
    p.iter().map(|&v| (t * v.max(1e-30).ln()) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_sizes() {
        let sizes = |k, s| chunks(k, s).iter().map(Vec::len).collect::<Vec<_>>();
        assert_eq!(sizes(64, 32), [32, 32]);
        assert_eq!(sizes(33, 32), [17, 16]);
        assert_eq!(sizes(100, 32), [25, 25, 25, 25]);
        assert_eq!(sizes(5, 8), [5]);
        assert_eq!(sizes(64, 8), [8; 8]);
        let all: Vec<usize> = chunks(255, 32).into_iter().flatten().collect();
        assert_eq!(all, (0..255).collect::<Vec<_>>());
    }

    #[test]
    fn top_keeps_option_order() {
        let l: Vec<f32> = (0..40).map(|i| ((i * 7) % 40) as f32).collect();
        let t = top(&l);
        assert_eq!(t.len(), TOP);
        assert!(t.windows(2).all(|w| w[0] < w[1]));
        assert!(t.iter().all(|&i| l[i] >= 24.0));
        assert_eq!(top(&[1.0, 1.0, 1.0]), [0, 1, 2]);
    }

    #[test]
    fn merge_keeps_the_tail_and_the_total() {
        let chunked = [2.0, 1.0, 0.5, -1.0, 3.0];
        let p = merge(&chunked, 1.5, &[0, 4], &[1.0, -1.0], 1.0);
        let before = softmax(&chunked, 1.5);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(p[1..4], before[1..4]);
        assert!(p[0] > p[4], "the joint pass reorders the top");
        assert!((p[0] + p[4] - before[0] - before[4]).abs() < 1e-12);
    }

    #[test]
    fn logits_round_trip() {
        let p = [0.5, 0.25, 0.125, 0.125];
        for t in [0.5, 1.0, 2.7] {
            let back = softmax(&logits(&p, t), t);
            for (a, b) in p.iter().zip(back) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }
}

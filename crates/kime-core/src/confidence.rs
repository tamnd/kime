//! The confidence formulas from spec/04-semantics.md.
//!
//! Two families. The `jev` formulas are the default because they reproduce the numbers TypeSafe's
//! API returns, which is what a client switching from Jev compares against. The `entropy` formula
//! is what Laya returns. Both take the calibrated distribution for one question, in option order,
//! and none of them looks at labels.

/// Choice confidence as Jev computes it: how far the top probability sits above uniform, scaled so
/// that uniform is 0 and certainty is 1.
///
/// `clip((max(p) - 1/k) / (1 - 1/k), 0, 1)`, and 1.0 for a single option, where there is nothing
/// to be unsure about.
#[must_use]
pub fn choice_jev(p: &[f64]) -> f64 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let uniform = 1.0 / k as f64;
    let top = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((top - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
}

/// Normalized entropy confidence, as Laya computes it: `clip(1 - H(p) / ln k, 0, 1)`.
///
/// `0 ln 0` is taken as 0, so a distribution with exact zeros in it is not a NaN. Used for choice
/// and score alike.
#[must_use]
pub fn entropy(p: &[f64]) -> f64 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let h: f64 = p.iter().filter(|&&x| x > 0.0).map(|&x| -x * x.ln()).sum();
    (1.0 - h / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Score confidence as Jev computes it, in its mean absolute deviation form.
///
/// With `m` the most likely level, `dist = sum(p_i * |i - m|)` is how much probability sits away
/// from it and how far. `mad` is the same distance for a uniform distribution around the middle
/// level, so a spread as wide as uniform scores 0. For a single level the answer is 1.0.
#[must_use]
pub fn score_jev(p: &[f64]) -> f64 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let m = argmax(p) as f64;
    let dist: f64 = p.iter().enumerate().map(|(i, &x)| x * (i as f64 - m).abs()).sum();
    let centre = (k as f64 - 1.0) / 2.0;
    let mad = (0..k).map(|i| (i as f64 - centre).abs()).sum::<f64>() / k as f64;
    (1.0 - dist / mad).max(0.0)
}

/// The score value itself, `sum(i * p_i)`. A real number that can fall between levels.
#[must_use]
pub fn expected_level(p: &[f64]) -> f64 {
    p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum()
}

/// Noul confidence with extensions on, `max(noul, 1 - noul)`, which matches Laya. The default
/// response has no noul confidence at all, to match Jev.
#[must_use]
pub fn noul(p_true: f64) -> f64 {
    p_true.max(1.0 - p_true)
}

/// The index of the largest probability, the first one on a tie. Option order is the tie break
/// everywhere in spec/03-api.md, so this is the one argmax the crate uses.
#[must_use]
pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > p[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 5e-3
    }

    // The worked example in spec/04-semantics.md. The first draft of that section said the entropy
    // confidence was 0.72, and this test is how it was found to be 0.71.
    #[test]
    fn the_worked_example() {
        let p = [0.91, 0.02, 0.03, 0.04];
        assert_eq!(argmax(&p), 0);
        assert!(close(choice_jev(&p), 0.88));
        assert!(close(entropy(&p), 0.713));
    }

    #[test]
    fn uniform_is_zero_and_certain_is_one() {
        for k in 2..=10 {
            let uniform = vec![1.0 / k as f64; k];
            assert!(choice_jev(&uniform).abs() < 1e-12);
            assert!(entropy(&uniform).abs() < 1e-12);
            let mut certain = vec![0.0; k];
            certain[k / 2] = 1.0;
            assert!((choice_jev(&certain) - 1.0).abs() < 1e-12);
            assert!((entropy(&certain) - 1.0).abs() < 1e-12);
            assert!((score_jev(&certain) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn one_option_is_certain() {
        assert!((choice_jev(&[1.0]) - 1.0).abs() < 1e-12);
        assert!((entropy(&[1.0]) - 1.0).abs() < 1e-12);
        assert!((score_jev(&[1.0]) - 1.0).abs() < 1e-12);
    }

    // Reversing the levels mirrors the distribution, and a mirrored distribution is exactly as
    // confident as the original. spec/15-testing.md tests the model for this, and the formula has
    // to hold it first.
    #[test]
    fn score_confidence_is_mirror_symmetric() {
        let p = [0.1, 0.6, 0.2, 0.1];
        let mut q = p;
        q.reverse();
        assert!((score_jev(&p) - score_jev(&q)).abs() < 1e-12);
        assert!((expected_level(&p) + expected_level(&q) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn spread_lowers_score_confidence() {
        let peaked = [0.05, 0.9, 0.05];
        let spread = [0.3, 0.4, 0.3];
        assert!(score_jev(&peaked) > score_jev(&spread));
    }

    #[test]
    fn noul_confidence_is_symmetric() {
        assert!((noul(0.2) - noul(0.8)).abs() < 1e-12);
        assert!((noul(0.5) - 0.5).abs() < 1e-12);
    }
}

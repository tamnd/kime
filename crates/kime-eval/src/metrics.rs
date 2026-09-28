//! The metrics of spec/13-benchmarks.md over single label questions.
//!
//! Each one follows the definition in Laya's benchmark notebook (`research/scripts/
//! build_benchmark_nb.py`, `hard_metrics`), so a number here and a number in Laya's published
//! results mean the same thing: confidence is the largest probability, ECE uses 15 equal bins with
//! the first one closed on the left, Brier sums over every option, and NLL clamps the gold
//! probability at 1e-12.

/// One answered question: the gold option, the probabilities in option order, and what the
/// suite knows beyond the gold label.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The index of the gold option.
    pub gold: usize,
    /// The model's probabilities, in option order.
    pub probs: Vec<f64>,
    /// Soft labels in option order, when the suite has them (typed-decisions).
    pub soft: Option<Vec<f64>>,
    /// The gold score for a score question, which can fall between levels.
    pub gold_score: Option<f64>,
}

impl Row {
    /// The index of the largest probability, the first one on a tie.
    #[must_use]
    pub fn pred(&self) -> usize {
        let mut best = 0;
        for (i, p) in self.probs.iter().enumerate() {
            if *p > self.probs[best] {
                best = i;
            }
        }
        best
    }

    /// The largest probability.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        self.probs.get(self.pred()).copied().unwrap_or(0.0)
    }

    /// Whether the largest probability is on the gold option.
    #[must_use]
    pub fn correct(&self) -> bool {
        self.pred() == self.gold
    }
}

/// The single label metrics of a set of rows.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Hard {
    pub n: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub ece: f64,
    pub brier: f64,
    pub nll: f64,
    pub aurc: f64,
    pub mean_confidence: f64,
    pub acc_at_50: f64,
    pub acc_at_80: f64,
}

/// Expected calibration error over `bins` equal bins of confidence.
#[must_use]
pub fn ece(conf: &[f64], correct: &[bool], bins: usize) -> f64 {
    let n = conf.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut e = 0.0;
    for i in 0..bins {
        let (lo, hi) = (i as f64 / bins as f64, (i + 1) as f64 / bins as f64);
        let (mut k, mut c, mut a) = (0usize, 0.0, 0.0);
        for (x, ok) in conf.iter().zip(correct) {
            let inside = if i == 0 { *x >= lo } else { *x > lo } && *x <= hi;
            if inside {
                k += 1;
                c += x;
                a += f64::from(u8::from(*ok));
            }
        }
        if k > 0 {
            e += k as f64 / n as f64 * (c / k as f64 - a / k as f64).abs();
        }
    }
    e
}

fn macro_f1(rows: &[Row]) -> f64 {
    let mut classes: Vec<usize> = rows.iter().flat_map(|r| [r.gold, r.pred()]).collect();
    classes.sort_unstable();
    classes.dedup();
    let mut sum = 0.0;
    for c in &classes {
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
        for r in rows {
            match (r.pred() == *c, r.gold == *c) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                _ => {}
            }
        }
        sum += 2.0 * tp as f64 / (2 * tp + fp + fn_).max(1) as f64;
    }
    sum / classes.len().max(1) as f64
}

/// The row indices by confidence, highest first, ties in row order.
fn by_confidence(rows: &[Row]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|a, b| rows[*b].confidence().total_cmp(&rows[*a].confidence()));
    order
}

/// Accuracy over the most confident `coverage` share of the rows, at least one row.
#[must_use]
pub fn accuracy_at_coverage(rows: &[Row], coverage: f64) -> f64 {
    let order = by_confidence(rows);
    let k = ((rows.len() as f64 * coverage) as usize).max(1).min(rows.len());
    order[..k].iter().filter(|i| rows[**i].correct()).count() as f64 / k as f64
}

/// The single label metrics of `rows`. Every field is NaN when there are none.
#[must_use]
pub fn hard(rows: &[Row]) -> Hard {
    let n = rows.len();
    if n == 0 {
        return Hard { n: 0, ..Hard::default() };
    }
    let conf: Vec<f64> = rows.iter().map(Row::confidence).collect();
    let correct: Vec<bool> = rows.iter().map(Row::correct).collect();
    let mean = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / n as f64;
    let order = by_confidence(rows);
    let (mut wrong, mut aurc) = (0.0, 0.0);
    for (i, r) in order.iter().enumerate() {
        wrong += f64::from(u8::from(!correct[*r]));
        aurc += wrong / (i + 1) as f64;
    }
    Hard {
        n,
        accuracy: correct.iter().filter(|c| **c).count() as f64 / n as f64,
        macro_f1: macro_f1(rows),
        ece: ece(&conf, &correct, 15),
        brier: mean(&|r| {
            r.probs
                .iter()
                .enumerate()
                .map(|(i, p)| (p - f64::from(u8::from(i == r.gold))).powi(2))
                .sum()
        }),
        nll: mean(&|r| -r.probs.get(r.gold).copied().unwrap_or(0.0).max(1e-12).ln()),
        aurc: aurc / n as f64,
        mean_confidence: conf.iter().sum::<f64>() / n as f64,
        acc_at_50: accuracy_at_coverage(rows, 0.5),
        acc_at_80: accuracy_at_coverage(rows, 0.8),
    }
}

/// The metrics that need more than the gold label, each over the rows that have what it needs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Extra {
    /// The probability mass on the soft label, `sum(p * g)`.
    pub soft_accuracy: Option<f64>,
    /// Brier against the soft label.
    pub soft_brier: Option<f64>,
    /// Mean absolute error of the expected level against the gold score.
    pub score_mae: Option<f64>,
    /// The share of score questions whose expected level is within 1 of the gold score.
    pub within_1: Option<f64>,
}

/// The soft label and score metrics of `rows`, with the soft label renormalized as Laya does.
#[must_use]
pub fn extra(rows: &[Row]) -> Extra {
    let (mut acc, mut brier, mut mae, mut within) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in rows {
        if let Some(g) = &r.soft {
            let gs: f64 = g.iter().sum();
            if gs > 0.0 {
                let mut p: Vec<f64> =
                    (0..g.len()).map(|i| r.probs.get(i).copied().unwrap_or(0.0)).collect();
                let ps = p.iter().sum::<f64>().max(1e-12);
                p.iter_mut().for_each(|x| *x /= ps);
                acc.push(p.iter().zip(g).map(|(p, g)| p * g / gs).sum::<f64>());
                brier.push(p.iter().zip(g).map(|(p, g)| (p - g / gs).powi(2)).sum::<f64>());
            }
        }
        if let Some(s) = r.gold_score {
            let exp: f64 = r.probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
            mae.push((exp - s).abs());
            within.push(f64::from(u8::from((exp - s).abs() <= 1.0)));
        }
    }
    let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
    Extra {
        soft_accuracy: mean(&acc),
        soft_brier: mean(&brier),
        score_mae: mean(&mae),
        within_1: mean(&within),
    }
}

/// A small seeded generator, so a bootstrap interval is the same on every run and machine.
#[derive(Debug, Clone)]
pub struct SplitMix(pub u64);

impl SplitMix {
    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform index below `n`.
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// The 95 percent bootstrap interval of accuracy and of ECE, from `resamples` resamples of the
/// rows with a fixed seed.
#[must_use]
pub fn bootstrap(rows: &[Row], resamples: usize, seed: u64) -> ([f64; 2], [f64; 2]) {
    if rows.is_empty() {
        return ([f64::NAN; 2], [f64::NAN; 2]);
    }
    let conf: Vec<f64> = rows.iter().map(Row::confidence).collect();
    let correct: Vec<bool> = rows.iter().map(Row::correct).collect();
    let mut rng = SplitMix(seed);
    let (mut accs, mut eces) = (Vec::with_capacity(resamples), Vec::with_capacity(resamples));
    let (mut c, mut k) = (vec![0.0; rows.len()], vec![false; rows.len()]);
    for _ in 0..resamples {
        for j in 0..rows.len() {
            let i = rng.below(rows.len());
            c[j] = conf[i];
            k[j] = correct[i];
        }
        accs.push(k.iter().filter(|x| **x).count() as f64 / rows.len() as f64);
        eces.push(ece(&c, &k, 15));
    }
    let ci = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        [at(0.025), at(0.975)]
    };
    (ci(&mut accs), ci(&mut eces))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(gold: usize, probs: &[f64]) -> Row {
        Row { gold, probs: probs.to_vec(), soft: None, gold_score: None }
    }

    #[test]
    fn matches_the_numpy_definitions() {
        // Worked by hand with Laya's numpy code on the same four rows.
        let rows =
            [row(0, &[0.9, 0.1]), row(1, &[0.6, 0.4]), row(1, &[0.2, 0.8]), row(0, &[0.3, 0.7])];
        let h = hard(&rows);
        assert_eq!(h.n, 4);
        assert!((h.accuracy - 0.5).abs() < 1e-12);
        let brier = (0.01 + 0.01 + 0.36 + 0.36 + 0.04 + 0.04 + 0.49 + 0.49) / 4.0;
        assert!((h.brier - brier).abs() < 1e-12);
        let nll = -(0.9f64.ln() + 0.4f64.ln() + 0.8f64.ln() + 0.3f64.ln()) / 4.0;
        assert!((h.nll - nll).abs() < 1e-12);
        // Confidences 0.9, 0.6, 0.8, 0.7 each fall in their own bin.
        let e = (0.1 + 0.6 + 0.2 + 0.7) / 4.0;
        assert!((h.ece - e).abs() < 1e-12, "{}", h.ece);
        // By confidence: 0.9 right, 0.8 right, 0.7 wrong, 0.6 wrong.
        assert!((h.aurc - (0.0 + 0.0 + 1.0 / 3.0 + 0.5) / 4.0).abs() < 1e-12);
        assert!((h.acc_at_50 - 1.0).abs() < 1e-12);
        assert!((h.acc_at_80 - 2.0 / 3.0).abs() < 1e-12);
        assert!((h.macro_f1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn soft_and_score() {
        let mut r = row(2, &[0.1, 0.2, 0.7]);
        r.soft = Some(vec![0.0, 1.0, 3.0]);
        r.gold_score = Some(1.0);
        let e = extra(&[r]);
        assert!((e.soft_accuracy.unwrap() - (0.2 * 0.25 + 0.7 * 0.75)).abs() < 1e-12);
        assert!((e.score_mae.unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(e.within_1, Some(1.0));
    }

    #[test]
    fn bootstrap_is_stable() {
        let rows: Vec<Row> = (0..200).map(|i| row(i % 3, &[0.5, 0.3, 0.2])).collect();
        let (a, _) = bootstrap(&rows, 500, 7);
        assert_eq!(bootstrap(&rows, 500, 7).0.map(f64::to_bits), a.map(f64::to_bits));
        assert!(a[0] < 0.34 && a[1] > 0.33 && a[0] > 0.2 && a[1] < 0.45, "{a:?}");
    }
}

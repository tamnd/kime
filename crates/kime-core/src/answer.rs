//! Answers, built from a question's option logits and act logits the way Laya builds them.
//!
//! This is Laya's `Agent._decode_answers` (unchanged from 0.3.7 to 0.3.11), done in the same float
//! types numpy uses so the rounded numbers match: the calibrated softmax and the entropy in f32,
//! the expected score in f64, and sums in numpy's pairwise order. `exp` and `log` are numpy's own
//! AVX2 routines rather than libm's, since they differ in the last bit often enough to move a
//! rounded probability: 2 of the 400 parity responses changed in the fourth decimal with libm.

use serde_json::{Map, Value, json};

use crate::request::{Criteria, QType, Question};

/// The `model` field of a compat response.
pub const LAYA_MODEL: &str = "laya-rl-agent";

/// The range Laya 0.3.9 and later confine a checkpoint's temperatures to. The English checkpoint
/// ships 0.1006 for `choice:11+`, which would sharpen the logits tenfold.
pub const TEMPERATURE_RANGE: (f64, f64) = (0.5, 5.0);

/// A temperature Laya would use: `t` confined to [`TEMPERATURE_RANGE`], or 1 when it is not a
/// finite number.
#[must_use]
pub fn clamp_temperature(t: f64) -> f64 {
    if t.is_finite() { t.clamp(TEMPERATURE_RANGE.0, TEMPERATURE_RANGE.1) } else { 1.0 }
}

/// Laya's `temp_bucket`: the key of `temperature_by_options` for a type and an option count.
#[must_use]
pub fn temperature_bucket(t: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", t.as_str())
}

/// The calibration of a compat checkpoint, from its `rl_agent_config.json`, already clamped.
#[derive(Debug, Clone, PartialEq)]
pub struct Temperatures {
    by_type: [f64; 3],
    by_options: Vec<(String, f64)>,
}

impl Temperatures {
    /// From the checkpoint's `temperature` (choice, score, noul) and `temperature_by_options`.
    #[must_use]
    pub fn new(by_type: [f64; 3], by_options: &[(String, f64)]) -> Self {
        Self {
            by_type: by_type.map(clamp_temperature),
            by_options: by_options
                .iter()
                .map(|(k, t)| (k.clone(), clamp_temperature(*t)))
                .collect(),
        }
    }

    /// The temperature for a question of type `t` with `k` options.
    #[must_use]
    pub fn get(&self, t: QType, k: usize) -> f64 {
        let bucket = temperature_bucket(t, k);
        self.by_options
            .iter()
            .find(|(b, _)| *b == bucket)
            .map_or(self.by_type[t.index()], |(_, t)| *t)
    }
}

impl Default for Temperatures {
    fn default() -> Self {
        Self::new([1.0; 3], &[])
    }
}

/// One question's answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// A choice question.
    Choice {
        /// The most likely label, the first one on a tie.
        choice: String,
        /// Each label with its calibrated probability, in option order.
        probabilities: Vec<(String, f64)>,
        /// Normalized entropy confidence.
        confidence: f64,
        /// The act head's probability that acting on this answer is right.
        act_probability: f64,
    },
    /// A score question.
    Score {
        /// The expected level, which can fall between levels.
        score: f64,
        /// The levels exactly as sent.
        legend: Vec<Value>,
        /// The probability of each level.
        probabilities: Vec<f64>,
        /// Normalized entropy confidence.
        confidence: f64,
        /// See [`Answer::Choice`].
        act_probability: f64,
    },
    /// A noul question.
    Noul {
        /// The probability the statement is true.
        noul: f64,
        /// `max(noul, 1 - noul)`.
        confidence: f64,
        /// See [`Answer::Choice`].
        act_probability: f64,
    },
}

/// Rounds to `digits` places the way Python's `round` does: correctly, ties to even.
#[must_use]
pub fn py_round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

impl Answer {
    /// The answer in Laya's JSON shape, with every number rounded to 4 places.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let r = |x: f64| py_round(x, 4);
        match self {
            Answer::Choice { choice, probabilities, confidence, act_probability } => json!({
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities.iter().map(|(k, p)| (k.clone(), json!(r(*p)))).collect::<Map<_, _>>(),
                "confidence": r(*confidence),
                "action": {"act_probability": r(*act_probability)},
            }),
            Answer::Score { score, legend, probabilities, confidence, act_probability } => json!({
                "type": "score",
                "score": r(*score),
                "legend": legend.iter().enumerate().map(|(i, v)| (i.to_string(), v.clone())).collect::<Map<_, _>>(),
                "probabilities": probabilities.iter().enumerate().map(|(i, p)| (i.to_string(), json!(r(*p)))).collect::<Map<_, _>>(),
                "confidence": r(*confidence),
                "action": {"act_probability": r(*act_probability)},
            }),
            Answer::Noul { noul, confidence, act_probability } => json!({
                "type": "noul",
                "noul": r(*noul),
                "confidence": r(*confidence),
                "action": {"act_probability": r(*act_probability)},
            }),
        }
    }
}

/// The answers to one request.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// The model that answered.
    pub model: String,
    /// The answers, in request order.
    pub answers: Vec<(String, Answer)>,
    /// Tokens read over all the request's sequences.
    pub input_tokens: usize,
}

impl Response {
    /// The answer to question `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Answer> {
        self.answers.iter().find(|(q, _)| q == id).map(|(_, a)| a)
    }

    /// The response in Laya's JSON shape.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let answers: Map<_, _> =
            self.answers.iter().map(|(id, a)| (id.clone(), a.to_json())).collect();
        json!({
            "model": self.model,
            "answers": answers,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
        })
    }
}

/// numpy's pairwise sum, which is the order `ndarray.sum` adds contiguous values in.
fn pairwise<T: Copy + Default + std::ops::Add<Output = T>>(x: &[T]) -> T {
    let n = x.len();
    if n < 8 {
        return x.iter().fold(T::default(), |a, &b| a + b);
    }
    if n <= 128 {
        let mut r = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]];
        let body = n - n % 8;
        for c in x[8..body].as_chunks::<8>().0 {
            for j in 0..8 {
                r[j] = r[j] + c[j];
            }
        }
        let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &x[body..] {
            s = s + v;
        }
        return s;
    }
    let half = n / 2 - (n / 2) % 8;
    pairwise(&x[..half]) + pairwise(&x[half..])
}

/// numpy's float32 `exp` on x86 with AVX2 and FMA (`simd_exp_FLOAT` in
/// `loops_exponent_log.dispatch.c.src`), one lane at a time: Cody-Waite reduction by ln 2, a 5/2
/// rational approximation, then the power of two added straight into the exponent bits.
// The constants are numpy's, digits and all, so they can be checked against its source.
#[allow(clippy::excessive_precision)]
fn np_exp(x: f32) -> f32 {
    const XMAX: f32 = 88.722_84;
    const XMIN: f32 = -103.972_08;
    const MAGIC: f32 = 12_582_912.0;
    const P: [f32; 6] = [
        9.999_999_999_980_870_924_916e-1,
        7.257_664_613_233_124_478_488e-1,
        2.473_615_434_895_520_810_817e-1,
        5.114_512_081_637_298_353_406e-2,
        6.757_896_990_527_504_603_057e-3,
        5.082_762_527_590_693_718_096e-4,
    ];
    const Q: [f32; 3] = [1.0, -2.742_335_390_411_667_452_936e-1, 2.159_509_375_685_829_852_307e-2];
    if x.is_nan() {
        return x;
    }
    if x >= XMAX {
        return f32::INFINITY;
    }
    if x <= XMIN {
        return 0.0;
    }
    let q = (x * std::f32::consts::LOG2_E + MAGIC) - MAGIC;
    let r = q.mul_add(-1.428_606_77e-6, q.mul_add(-6.931_457_52e-1, x));
    let r = q.mul_add(0.0, r);
    let num =
        P[5].mul_add(r, P[4]).mul_add(r, P[3]).mul_add(r, P[2]).mul_add(r, P[1]).mul_add(r, P[0]);
    let den = Q[2].mul_add(r, Q[1]).mul_add(r, Q[0]);
    let poly = num / den;
    // q is a whole number here, and at most 128 in size.
    let shift =
        |poly: f32, q: f32| f32::from_bits(poly.to_bits().wrapping_add(((q as i32) << 23) as u32));
    if q <= -125.0 {
        let diff = (-(q + 125.0)) as u32;
        shift(poly, -125.0) / (1u32 << diff) as f32
    } else {
        shift(poly, q)
    }
}

/// numpy's float32 `log` on x86 with AVX2 and FMA (`simd_log_FLOAT`), for a positive finite `x`:
/// the mantissa scaled into (sqrt(1/2), sqrt(2)], a 5/5 rational approximation of `log(1 + m)`,
/// plus the exponent times ln 2.
#[allow(clippy::excessive_precision)]
fn np_log(x: f32) -> f32 {
    const P: [f32; 6] = [
        0.0,
        9.999_999_999_999_998_702_752e-1,
        2.112_677_543_073_053_063_722,
        1.480_000_633_576_506_585_156,
        3.808_837_741_388_407_920_751e-1,
        2.589_979_117_907_922_693_523e-2,
    ];
    const Q: [f32; 6] = [
        1.0,
        2.612_677_543_073_109_236_779,
        2.453_006_071_784_736_363_091,
        9.864_942_958_519_418_960_339e-1,
        1.546_476_374_983_906_719_538e-1,
        5.875_095_403_124_574_342_950e-3,
    ];
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let (bits, bias) = if x < f32::MIN_POSITIVE {
        ((x * f32::from_bits(0x7180_0000)).to_bits(), 100.0)
    } else {
        (x.to_bits(), 0.0)
    };
    let mut exponent = ((bits >> 23) as i32 - 126) as f32 - bias;
    let mut m = f32::from_bits((bits & 0x7f_ffff) | (126 << 23));
    if m <= std::f32::consts::FRAC_1_SQRT_2 {
        m += m;
        exponent -= 1.0;
    }
    let m = m - 1.0;
    let num =
        P[5].mul_add(m, P[4]).mul_add(m, P[3]).mul_add(m, P[2]).mul_add(m, P[1]).mul_add(m, P[0]);
    let den =
        Q[5].mul_add(m, Q[4]).mul_add(m, Q[3]).mul_add(m, Q[2]).mul_add(m, Q[1]).mul_add(m, Q[0]);
    exponent.mul_add(std::f32::consts::LN_2, num / den)
}

/// The calibrated distribution over `logits` at temperature `t`, in f32 as numpy computes it.
fn softmax(logits: &[f32], t: f64) -> Vec<f32> {
    let t = t as f32;
    let z: Vec<f32> = logits.iter().map(|&l| l / t).collect();
    let top = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&v| np_exp(v - top)).collect();
    let s = pairwise(&e);
    e.iter().map(|&v| v / s).collect()
}

/// Laya's `confidence_from_probs`, in f32.
fn entropy_confidence(p: &[f32]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let terms: Vec<f32> = p.iter().map(|&v| v * np_log(v.clamp(1e-12, 1.0))).collect();
    let ent = -pairwise(&terms);
    f64::from((1.0 - ent / (k as f64).ln() as f32).clamp(0.0, 1.0))
}

/// The act head's first probability, as `torch.softmax` gives it.
#[must_use]
pub fn act_probability(act: [f32; 2]) -> f64 {
    let top = act[0].max(act[1]);
    let e = act.map(|v| (v - top).exp());
    f64::from(e[0] * (1.0 / (e[0] + e[1])))
}

/// The answer to `q` from its option logits, in option order, and its two act logits.
///
/// # Panics
///
/// If `logits` does not have one value per option.
#[must_use]
pub fn laya_answer(q: &Question, logits: &[f32], act: [f32; 2], temps: &Temperatures) -> Answer {
    let k = q.criteria.len();
    assert_eq!(logits.len(), k, "one logit per option");
    let p = softmax(logits, temps.get(q.qtype, k));
    let act_probability = act_probability(act);
    match &q.criteria {
        Criteria::Choice(opts) => {
            let mut best = 0;
            for (i, &v) in p.iter().enumerate() {
                if v > p[best] {
                    best = i;
                }
            }
            Answer::Choice {
                choice: opts[best].label.clone(),
                probabilities: opts
                    .iter()
                    .zip(&p)
                    .map(|(o, &v)| (o.label.clone(), f64::from(v)))
                    .collect(),
                confidence: entropy_confidence(&p),
                act_probability,
            }
        }
        Criteria::Score(levels) => {
            let weighted: Vec<f64> =
                p.iter().enumerate().map(|(i, &v)| i as f64 * f64::from(v)).collect();
            Answer::Score {
                score: pairwise(&weighted),
                legend: levels.clone(),
                probabilities: p.iter().map(|&v| f64::from(v)).collect(),
                confidence: entropy_confidence(&p),
                act_probability,
            }
        }
        Criteria::Noul { .. } => {
            let yes = f64::from(p[1]);
            Answer::Noul { noul: yes, confidence: yes.max(1.0 - yes), act_probability }
        }
    }
}

#[cfg(test)]
// These values are exact, so comparing them exactly is the point.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_order() {
        // Nine values: eight lanes plus one left over, which a left fold adds in another order.
        let x = [1e8f32, 1.0, -1e8, 1.0, 0.5, 0.25, 3.0, 7.0, 1.0];
        let lanes = ((1e8f32 + 1.0) + (-1e8 + 1.0)) + ((0.5 + 0.25) + (3.0 + 7.0)) + 1.0;
        assert_eq!(pairwise(&x).to_bits(), lanes.to_bits());
    }

    #[test]
    fn numpy_exp_log() {
        // Values numpy 2.5 gives on an AVX2 machine, where libm's differ in the last bit.
        for (x, want) in [(-0.0, 1.0f32), (-1.0, 0.367_879_43), (-20.0, 2.061_153_6e-9)] {
            assert!((np_exp(x) - want).abs() <= want * 3e-7, "exp {x}");
        }
        assert_eq!(np_exp(0.0), 1.0);
        assert_eq!(np_log(1.0), 0.0);
        assert!((np_log(1e-12) - 1e-12f32.ln()).abs() < 1e-5);
        assert!((np_log(0.3) - 0.3f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn rounding() {
        assert_eq!(py_round(0.25, 1), 0.2);
        assert_eq!(py_round(0.965_449_999, 4), 0.9654);
        assert_eq!(py_round(1.0, 4), 1.0);
    }

    #[test]
    fn buckets_and_clamp() {
        let t = Temperatures::new([1.5, 1.2, 2.0], &[("choice:11+".into(), 0.1)]);
        assert_eq!(t.get(QType::Choice, 12), 0.5);
        assert_eq!(t.get(QType::Choice, 4), 1.5);
        assert_eq!(temperature_bucket(QType::Score, 6), "score:6-10");
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
    }
}

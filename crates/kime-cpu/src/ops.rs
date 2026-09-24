//! The elementwise and per row ops of the compat graph, in FP32.
//!
//! Statistics and transcendental functions are taken in f64 and rounded once, so each result is
//! within an ulp or so of the exact value. PyTorch computes them in f32 with its own vectorized
//! approximations, and the difference to it is at that level either way.

/// LayerNorm over rows of `d`: `(x - mean) / sqrt(var + eps) * w + b`, with the biased variance
/// PyTorch uses.
///
/// # Panics
///
/// If a length does not match.
pub fn layer_norm(x: &[f32], d: usize, w: &[f32], b: Option<&[f32]>, eps: f64, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    assert_eq!(w.len(), d);
    if d == 0 {
        return;
    }
    for (row, o) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / d as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            let n = ((f64::from(row[i]) - mean) * rstd) as f32;
            o[i] = match b {
                Some(b) => n * w[i] + b[i],
                None => n * w[i],
            };
        }
    }
}

/// The exact GELU, `x/2 (1 + erf(x/√2))`, which is what `nn.GELU()` and ModernBERT's `"gelu"`
/// compute.
///
/// It is taken in f32 without branches or calls so the loops over it vectorize. The normal CDF
/// comes from `erfc`, `1 - erfc(z)/2` for `x >= 0` and `erfc(z)/2` below, so the negative side
/// keeps its relative accuracy instead of cancelling in `1 + erf`. The error is a few ulp, about
/// what PyTorch's own vectorized erf gives.
#[inline]
#[must_use]
pub fn gelu(x: f32) -> f32 {
    let q = erfc_sqrt_half(x.abs());
    let p = if x >= 0.0 { 1.0 - 0.5 * q } else { 0.5 * q };
    // The CDF stops at a = 13 below, so past it the product would grow with x instead of vanishing.
    if x < -13.0 { -0.0 } else { x * p }
}

/// `erfc(a/√2)` for `a >= 0` with relative error around 1e-7: the Chebyshev fit from Numerical
/// Recipes, `k exp(-z² + P(k))` with `z = a/√2` and `k = 1/(1 + z/2)`. The exponent is taken from
/// `a` in two parts, `ah` keeping 12 bits so `ah²/2` is exact, since rounding `z` first would put
/// an error of `z²` ulp on the result. Past `a = 13` the value is below `1e-37`, so `a` stops
/// there, which keeps the exponent in range for [`exp`].
#[inline]
fn erfc_sqrt_half(a: f32) -> f32 {
    let a = a.min(13.0);
    let k = 1.0 / (1.0 + 0.5 * std::f32::consts::FRAC_1_SQRT_2 * a);
    let mut r = 0.170_872_77f32;
    for c in [
        -0.822_152_23,
        1.488_515_9,
        -1.135_204,
        0.278_868_07,
        -0.186_288_06,
        0.096_784_18,
        0.374_091_96,
        1.000_023_7,
        -1.265_512_2,
    ] {
        r = r * k + c;
    }
    let ah = f32::from_bits(a.to_bits() & 0xffff_f000);
    k * exp(-0.5 * ah * ah, 0.5 * (ah - a) * (ah + a) + r)
}

/// `e^y` for `y <= 0` without calls, so softmax loops vectorize. Below `-87` it gives about
/// `1.6e-38` instead of going on down to zero.
#[inline]
#[must_use]
pub fn exp_neg(y: f32) -> f32 {
    exp(y.max(-87.0), 0.0)
}

/// `e^(hi + lo)` without calls, for `|lo|` a few units at most and `hi` exact: `2^n e^r` with `|r| <= ln2/2` and a
/// degree 6 polynomial for `e^r`. Keeping `lo` apart until after the reduction saves the bits a
/// single f32 argument near `-20` would lose. The sum has to stay between `-87` and `88`, since
/// `2^n` is built from its bits. There is no fused multiply add, since x86 builds without FMA
/// would turn it into a call.
#[inline]
fn exp(hi: f32, lo: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23, rounds to an integer when added
    let n = ((hi + lo) * std::f32::consts::LOG2_E + MAGIC) - MAGIC;
    // ln 2 split in two, the first part short enough that n times it is exact.
    let r = hi - n * 0.693_359_4;
    let r = r + n * 2.121_944_4e-4 + lo;
    let mut e = 1.0 / 720.0f32;
    for c in [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        e = e * r + c;
    }
    e * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// ModernBERT's gated MLP input: `u` is rows of `2i`, the first half goes through GELU and is
/// multiplied by the second half.
///
/// # Panics
///
/// If a length does not match.
pub fn geglu(u: &[f32], inter: usize, out: &mut [f32]) {
    assert_eq!(u.len(), 2 * out.len());
    if inter == 0 {
        return;
    }
    for (row, o) in u.chunks_exact(2 * inter).zip(out.chunks_exact_mut(inter)) {
        let (a, g) = row.split_at(inter);
        o.iter_mut().zip(a).zip(g).for_each(|((o, &a), &g)| *o = gelu(a) * g);
    }
}

/// `x += y`.
///
/// # Panics
///
/// If the lengths differ.
pub fn add(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// Rotary position tables for one base, as Hugging Face builds them for the default rope type.
#[derive(Debug, Clone)]
pub struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// Tables for positions `0..len` over heads of `dim` (even).
    ///
    /// # Panics
    ///
    /// If `dim` is odd.
    #[must_use]
    pub fn new(theta: f64, dim: usize, len: usize) -> Self {
        assert!(dim.is_multiple_of(2));
        let half = dim / 2;
        // inv_freq = 1 / theta ** (arange(0, dim, 2) / dim) in f32, then pos * inv_freq in f32.
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                let e = (2 * i) as f32 / dim as f32;
                1.0 / (theta.powf(f64::from(e)) as f32)
            })
            .collect();
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for p in 0..len {
            for &f in &inv {
                let a = f64::from(p as f32 * f);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        Self { half, cos, sin }
    }

    /// Positions the tables cover.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cos.len() / self.half.max(1)
    }

    /// True when the tables cover no positions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cos.is_empty()
    }

    /// Rotates one head vector at position `pos` in place: `x cos + rotate_half(x) sin`.
    ///
    /// # Panics
    ///
    /// If `pos` is past the tables or `x` is not one head long.
    #[inline]
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let h = self.half;
        assert_eq!(x.len(), 2 * h);
        let c = &self.cos[pos * h..(pos + 1) * h];
        let s = &self.sin[pos * h..(pos + 1) * h];
        for i in 0..h {
            let (a, b) = (x[i], x[i + h]);
            // Two products then a sum, not a fused multiply add, the rounding torch does.
            x[i] = a * c[i] + -b * s[i];
            x[i + h] = b * c[i] + a * s[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, close};

    #[test]
    fn layer_norm_against_the_formula() {
        let mut rng = Rng(3);
        for (rows, d) in [(0, 4), (1, 1), (3, 7), (2, 1024)] {
            let x = rng.vec(rows * d);
            let w = rng.vec(d);
            let b = rng.vec(d);
            let mut out = vec![0f32; rows * d];
            layer_norm(&x, d, &w, Some(&b), 1e-5, &mut out);
            let mut want = vec![0f32; rows * d];
            for r in 0..rows {
                let row = &x[r * d..(r + 1) * d];
                let mean: f32 = row.iter().sum::<f32>() / d as f32;
                let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                for i in 0..d {
                    want[r * d + i] = (row[i] - mean) / (var + 1e-5).sqrt() * w[i] + b[i];
                }
            }
            close(&out, &want, 1e-4, "layer_norm");
        }
    }

    #[test]
    fn gelu_values() {
        // x/2 (1 + erf(x/√2)) from Python's math.erf.
        let cases = [
            (0.0, 0.0),
            (1.0, 0.841_344_746_068_542_9),
            (-1.0, -0.158_655_253_931_457_07),
            (3.0, 2.995_950_305_905_11),
            (-6.0, -5.919_525_869_479_969e-9),
        ];
        for (x, want) in cases {
            let got = gelu(x as f32);
            assert!((f64::from(got) - want).abs() <= want.abs() * 1e-6 + 1e-12, "{x}: {got}");
        }
        let mut out = [0f32; 2];
        geglu(&[1.0, -1.0, 2.0, 3.0], 2, &mut out);
        close(&out, &[gelu(1.0) * 2.0, gelu(-1.0) * 3.0], 0.0, "geglu");
    }

    #[test]
    fn gelu_matches_f64_erf_over_the_range() {
        // Relative error against libm's f64 erf, and absolute error far down the negative tail
        // where the value is below anything that reaches an output.
        let (mut worst, mut at) = (0f64, 0f32);
        let mut x = -12f32;
        while x < 12.0 {
            let want =
                0.5 * f64::from(x) * libm::erfc(-f64::from(x) * std::f64::consts::FRAC_1_SQRT_2);
            let got = f64::from(gelu(x));
            let err = (got - want).abs() / want.abs().max(1e-30);
            let err = if x < -8.0 { (got - want).abs() * 1e6 } else { err };
            if err > worst {
                (worst, at) = (err, x);
            }
            x += 1.0 / 4096.0 + x.abs() * 1e-4;
        }
        assert!(worst < 6e-7, "worst {worst:e} at {at}");
        for x in [13.0, 20.0, 100.0, 1e30, f32::MAX, f32::INFINITY] {
            assert_eq!(gelu(x).to_bits(), x.to_bits(), "{x}");
            let g = gelu(-x);
            assert!(g <= 0.0 && g > -1e-30, "{}: {g}", -x);
        }
        assert!(gelu(f32::NAN).is_nan());
    }

    #[test]
    fn rope_rotates() {
        let r = Rope::new(10000.0, 4, 3);
        assert_eq!(r.len(), 3);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        r.apply(&mut x, 0);
        close(&x, &[1.0, 2.0, 3.0, 4.0], 0.0, "position 0");
        // Position 1: pair 0 turns by 1 radian and pair 1 by 1/100.
        let mut x = [1.0, 2.0, 3.0, 4.0];
        r.apply(&mut x, 1);
        let (c0, s0, c1, s1) = (1f32.cos(), 1f32.sin(), 0.01f32.cos(), 0.01f32.sin());
        close(
            &x,
            &[c0 - 3.0 * s0, 2.0 * c1 - 4.0 * s1, 3.0 * c0 + s0, 4.0 * c1 + 2.0 * s1],
            1e-6,
            "rope",
        );
    }
}

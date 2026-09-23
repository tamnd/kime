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
#[inline]
#[must_use]
pub fn gelu(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
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
        for i in 0..inter {
            o[i] = gelu(a[i]) * g[i];
        }
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
        close(&x, &[c0 - 3.0 * s0, 2.0 * c1 - 4.0 * s1, 3.0 * c0 + s0, 4.0 * c1 + 2.0 * s1], 1e-6, "rope");
    }
}

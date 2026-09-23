//! FP32 matrix products in the layout of a PyTorch `Linear`: `y = x wᵀ + b`, with `x` as `[m, k]`
//! and `w` as `[n, k]`, both row major.
//!
//! Every output is one dot product, and it is computed the same way wherever it lands: eight
//! running sums over `k` in steps of eight with fused multiply adds, the eight added in a fixed
//! tree, then the leftover `k % 8` terms in order, then the bias. Tiling and threading only decide
//! which dots run side by side, so the result is the same bit for bit for any thread count and
//! any split, and the same on x86 with FMA as on ARM.

use crate::par::{self, Shared};

const LANES: usize = 8;
/// Elements of `k` summed in f32 before the running sums move to f64.
const BLOCK: usize = 64;
/// Rows of `x` per micro tile.
const MR: usize = 4;
/// Rows of `w` per micro tile.
const NR: usize = 3;
/// Rows of `w` per task, a multiple of NR sized so a task's weights stay in L2.
const NB: usize = 48;
/// Rows of `x` per task.
const MB: usize = 128;

/// `dot(a, b)` in the order described in the module docs.
///
/// # Panics
///
/// If the lengths differ.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    return tile::<neon::Neon, 1, 1>([a], [b])[0][0];
    #[cfg(target_arch = "x86_64")]
    if has_fma() {
        // SAFETY: the features dot_fma enables were detected on this machine.
        return unsafe { dot_fma(a, b) };
    }
    #[allow(unreachable_code)]
    tile::<[f32; LANES], 1, 1>([a], [b])[0][0]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    tile::<avx::Avx, 1, 1>([a], [b])[0][0]
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// Eight f32 lanes with a fused multiply add, the one vector op the dots need. Each lane is its
/// own running sum, so every implementation gives the same bits.
trait V8: Copy {
    fn zero() -> Self;
    fn load(s: &[f32; LANES]) -> Self;
    /// `self + a * b`, rounded once.
    fn fma(self, a: Self, b: Self) -> Self;
    fn lanes(self) -> [f32; LANES];
}

impl V8 for [f32; LANES] {
    #[inline(always)]
    fn zero() -> Self {
        [0.0; LANES]
    }
    #[inline(always)]
    fn load(s: &[f32; LANES]) -> Self {
        *s
    }
    #[inline(always)]
    fn fma(self, a: Self, b: Self) -> Self {
        std::array::from_fn(|l| a[l].mul_add(b[l], self[l]))
    }
    #[inline(always)]
    fn lanes(self) -> [f32; LANES] {
        self
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    use super::{LANES, V8};

    #[derive(Clone, Copy)]
    pub(super) struct Neon(float32x4_t, float32x4_t);

    impl V8 for Neon {
        #[inline(always)]
        fn zero() -> Self {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { Self(vdupq_n_f32(0.0), vdupq_n_f32(0.0)) }
        }
        #[inline(always)]
        fn load(s: &[f32; LANES]) -> Self {
            // SAFETY: both loads read four floats inside the eight the reference covers.
            unsafe { Self(vld1q_f32(s.as_ptr()), vld1q_f32(s.as_ptr().add(4))) }
        }
        #[inline(always)]
        fn fma(self, a: Self, b: Self) -> Self {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { Self(vfmaq_f32(self.0, a.0, b.0), vfmaq_f32(self.1, a.1, b.1)) }
        }
        #[inline(always)]
        fn lanes(self) -> [f32; LANES] {
            let mut out = [0f32; LANES];
            // SAFETY: both stores write four floats inside the eight of out.
            unsafe {
                vst1q_f32(out.as_mut_ptr(), self.0);
                vst1q_f32(out.as_mut_ptr().add(4), self.1);
            }
            out
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    use super::{LANES, V8};

    /// Only built inside functions that enable avx2 and fma, after detecting them.
    #[derive(Clone, Copy)]
    pub(super) struct Avx(__m256);

    impl V8 for Avx {
        #[inline(always)]
        fn zero() -> Self {
            // SAFETY: Avx values only exist on machines where AVX was detected.
            unsafe { Self(_mm256_setzero_ps()) }
        }
        #[inline(always)]
        fn load(s: &[f32; LANES]) -> Self {
            // SAFETY: an unaligned load of the eight floats of s, on a machine with AVX.
            unsafe { Self(_mm256_loadu_ps(s.as_ptr())) }
        }
        #[inline(always)]
        fn fma(self, a: Self, b: Self) -> Self {
            // SAFETY: FMA was detected before any Avx value was made.
            unsafe { Self(_mm256_fmadd_ps(a.0, b.0, self.0)) }
        }
        #[inline(always)]
        fn lanes(self) -> [f32; LANES] {
            let mut out = [0f32; LANES];
            // SAFETY: an unaligned store of eight floats into out, on a machine with AVX.
            unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
            out
        }
    }
}

/// The dots of `R` rows of x against `C` rows of w, all of the same length.
#[inline(always)]
fn tile<V: V8, const R: usize, const C: usize>(x: [&[f32]; R], w: [&[f32]; C]) -> [[f32; C]; R] {
    let k = w[0].len();
    let body = k - k % LANES;
    let mut wide = [[[0f64; LANES]; C]; R];
    let mut p = 0;
    while p < body {
        let end = (p + BLOCK).min(body);
        let mut acc = [[V::zero(); C]; R];
        while p < end {
            let xv: [V; R] =
                std::array::from_fn(|r| V::load(x[r][p..p + LANES].try_into().unwrap()));
            let wv: [V; C] =
                std::array::from_fn(|c| V::load(w[c][p..p + LANES].try_into().unwrap()));
            for r in 0..R {
                for c in 0..C {
                    acc[r][c] = acc[r][c].fma(xv[r], wv[c]);
                }
            }
            p += LANES;
        }
        for r in 0..R {
            for c in 0..C {
                let l = acc[r][c].lanes();
                for i in 0..LANES {
                    wide[r][c][i] += f64::from(l[i]);
                }
            }
        }
    }
    let mut out = [[0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            let v = wide[r][c];
            let mut s = ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
            for q in body..k {
                s = f64::from(x[r][q]).mul_add(f64::from(w[c][q]), s);
            }
            out[r][c] = s as f32;
        }
    }
    out
}

struct Args<'a> {
    x: &'a [f32],
    w: &'a [f32],
    b: Option<&'a [f32]>,
    k: usize,
    n: usize,
    y: &'a Shared<'a>,
}

impl Args<'_> {
    #[inline(always)]
    fn run<V: V8, const R: usize, const C: usize>(&self, i: usize, j: usize) {
        let k = self.k;
        let xs = std::array::from_fn(|r| &self.x[(i + r) * k..(i + r + 1) * k]);
        let ws = std::array::from_fn(|c| &self.w[(j + c) * k..(j + c + 1) * k]);
        let out = tile::<V, R, C>(xs, ws);
        for (r, row) in out.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let v = match self.b {
                    Some(b) => v + b[j + c],
                    None => v,
                };
                // SAFETY: each (i, j) pair belongs to exactly one task and one tile within it.
                unsafe { self.y.set((i + r) * self.n + j + c, v) };
            }
        }
    }

    #[inline(always)]
    fn block<V: V8>(&self, rows: (usize, usize), cols: (usize, usize)) {
        let mut j = cols.0;
        while j < cols.1 {
            let wide = cols.1 - j >= NR;
            let mut i = rows.0;
            while i + MR <= rows.1 {
                if wide {
                    self.run::<V, MR, NR>(i, j);
                } else {
                    for c in j..cols.1 {
                        self.run::<V, MR, 1>(i, c);
                    }
                }
                i += MR;
            }
            for r in i..rows.1 {
                if wide {
                    self.run::<V, 1, NR>(r, j);
                } else {
                    for c in j..cols.1 {
                        self.run::<V, 1, 1>(r, c);
                    }
                }
            }
            j += NR;
        }
    }

    fn block_dispatch(&self, rows: (usize, usize), cols: (usize, usize)) {
        #[cfg(target_arch = "aarch64")]
        return self.block::<neon::Neon>(rows, cols);
        #[cfg(target_arch = "x86_64")]
        if has_fma() {
            // SAFETY: the features block_fma enables were detected on this machine.
            unsafe { self.block_fma(rows, cols) };
            return;
        }
        #[allow(unreachable_code)]
        self.block::<[f32; LANES]>(rows, cols);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn block_fma(&self, rows: (usize, usize), cols: (usize, usize)) {
        self.block::<avx::Avx>(rows, cols);
    }
}

/// `y = x wᵀ + b` with `x` as `[m, k]`, `w` as `[n, k]`, `b` as `[n]` and `y` as `[m, n]`.
///
/// # Panics
///
/// If a length does not match the shape.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    b: Option<&[f32]>,
    y: &mut [f32],
    threads: usize,
) {
    assert_eq!(x.len(), m * k, "x is not [m, k]");
    assert_eq!(w.len(), n * k, "w is not [n, k]");
    assert_eq!(y.len(), m * n, "y is not [m, n]");
    if let Some(b) = b {
        assert_eq!(b.len(), n, "b is not [n]");
    }
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for row in y.chunks_exact_mut(n) {
            match b {
                Some(b) => row.copy_from_slice(b),
                None => row.fill(0.0),
            }
        }
        return;
    }
    let shared = Shared::new(y);
    let args = Args { x, w, b, k, n, y: &shared };
    let mt = m.div_ceil(MB);
    let nt = n.div_ceil(NB);
    par::for_each(mt * nt, threads, |t| {
        let (bi, bj) = (t % mt, t / mt);
        let rows = (bi * MB, ((bi + 1) * MB).min(m));
        let cols = (bj * NB, ((bj + 1) * NB).min(n));
        args.block_dispatch(rows, cols);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, close};

    fn naive(x: &[f32], m: usize, k: usize, w: &[f32], n: usize, b: Option<&[f32]>) -> Vec<f32> {
        let mut y = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let s: f64 =
                    (0..k).map(|p| f64::from(x[i * k + p]) * f64::from(w[j * k + p])).sum();
                y[i * n + j] = (s + b.map_or(0.0, |b| f64::from(b[j]))) as f32;
            }
        }
        y
    }

    #[test]
    fn matches_naive_on_awkward_shapes() {
        let mut rng = Rng(7);
        let shapes = [
            (0, 8, 5),
            (1, 1, 1),
            (1, 7, 3),
            (3, 16, 2),
            (4, 9, 3),
            (5, 64, 7),
            (130, 33, 50),
            (129, 1028, 49),
            (17, 0, 4),
        ];
        for (m, k, n) in shapes {
            let x = rng.vec(m * k);
            let w = rng.vec(n * k);
            let b = rng.vec(n);
            for bias in [None, Some(&b[..])] {
                let want = naive(&x, m, k, &w, n, bias);
                for threads in [1, 3] {
                    let mut y = vec![f32::NAN; m * n];
                    linear(&x, m, k, &w, n, bias, &mut y, threads);
                    close(&y, &want, 1e-5, &format!("{m}x{k}x{n}"));
                }
            }
        }
    }

    #[test]
    fn same_bits_for_any_split() {
        let mut rng = Rng(11);
        let (m, k, n) = (37, 200, 101);
        let x = rng.vec(m * k);
        let w = rng.vec(n * k);
        let mut one = vec![0f32; m * n];
        linear(&x, m, k, &w, n, None, &mut one, 1);
        for threads in [2, 5, 16] {
            let mut y = vec![0f32; m * n];
            linear(&x, m, k, &w, n, None, &mut y, threads);
            assert!(y.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // And the same bits as a single dot on the row.
        for i in [0, 36] {
            for j in [0, 50, 100] {
                let d = dot(&x[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
                assert_eq!(d.to_bits(), one[i * n + j].to_bits());
            }
        }
    }
}

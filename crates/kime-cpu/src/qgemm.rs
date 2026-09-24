//! INT8 matrix products, `y = x wᵀ + b` with both sides rounded to eight bits: the weights per
//! output channel and the activations per row, each symmetric with a scale of `max |v| / 127`.
//! The products are summed in i32, which is exact, so a sum does not depend on the order it is
//! taken in. Every kernel, every split and every machine gives the same bits, and a row's result
//! does not depend on the rows around it.
//!
//! Both sides are stored the way Arm's `smmla` reads them. Rows go in tiles of eight, and a tile
//! holds, for each group of eight values of `k`, its four row pairs one after the other, each pair
//! as eight values from its first row then eight from its second: `[rows / 8][k / 8][4][2][8]`,
//! with the rows past the end and the tail of `k` padded with zeros. One `smmla` multiplies a pair
//! of rows of `x` by a pair of rows of `w` over eight values of `k` into a 2 by 2 block of sums,
//! so a tile of `x` against a tile of `w` is sixteen of them for each group, all from 128 bytes
//! read in order.

use kime_tensor::Epilogue;

use crate::ops::gelu;
use crate::par::Shared;

/// Values of `k` per group.
const KG: usize = 8;
/// Rows per tile, on both sides.
const TILE: usize = 8;
/// Rows of `x` per task, a multiple of TILE.
const MB: usize = 64;

/// A matrix rounded to INT8 row by row, in the layout of the module docs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QMatrix {
    /// Rows.
    pub rows: usize,
    /// Values per row.
    pub k: usize,
    /// `[rows.div_ceil(8)][k.div_ceil(8)][4][2][8]`.
    pub q: Vec<i8>,
    /// `max |row| / 127` for each row, so a row is its values times its scale.
    pub scale: Vec<f32>,
}

impl QMatrix {
    /// Rounds `x`, `[rows, k]`, to INT8.
    ///
    /// # Panics
    ///
    /// If `x` is not `[rows, k]`.
    #[must_use]
    pub fn quantize(x: &[f32], rows: usize, k: usize) -> Self {
        assert_eq!(x.len(), rows * k, "x is not [rows, k]");
        let mut q = vec![0; tiled_len(rows, k)];
        let mut scale = vec![0.0; rows];
        for (r, row) in x.chunks_exact(k.max(1)).take(rows).enumerate() {
            scale[r] = quantize_row(row, r, &mut q);
        }
        Self { rows, k, q, scale }
    }

    /// The values as f32, `[rows, k]`, for tests and for checking what the rounding lost.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.k];
        for r in 0..self.rows {
            for p in 0..self.k {
                out[r * self.k + p] = f32::from(self.q[at(r, p, self.k)]) * self.scale[r];
            }
        }
        out
    }
}

/// Bytes of a tiled matrix of `rows` by `k`.
#[must_use]
pub fn tiled_len(rows: usize, k: usize) -> usize {
    rows.div_ceil(TILE) * TILE * k.div_ceil(KG) * KG
}

/// Where value `p` of row `r` sits in a tiled matrix with rows of `k`.
#[inline(always)]
fn at(r: usize, p: usize, k: usize) -> usize {
    let (tile, pair, half) = (r / TILE, r % TILE / 2, r % 2);
    tile * k.div_ceil(KG) * TILE * KG + p / KG * TILE * KG + pair * 2 * KG + half * KG + p % KG
}

/// Rounds `row` into row `r` of the tiled `q` and returns its scale.
fn quantize_row(row: &[f32], r: usize, q: &mut [i8]) -> f32 {
    let k = row.len();
    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        for p in 0..k {
            q[at(r, p, k)] = 0;
        }
        return if amax == 0.0 { 0.0 } else { f32::NAN };
    }
    let inv = 127.0 / amax;
    let base = at(r, 0, k);
    for (g, chunk) in row.chunks(KG).enumerate() {
        let o = base + g * TILE * KG;
        for (e, &v) in chunk.iter().enumerate() {
            // |v| <= amax, so the product is within [-127, 127] and the cast does not clamp.
            q[o + e] = (v * inv).round() as i8;
        }
    }
    amax / 127.0
}

/// Bytes of scratch a task needs for its rows of `x`, as f32 values.
#[must_use]
pub fn scratch_len(k: usize) -> usize {
    tiled_len(MB, k).div_ceil(4) + MB
}

/// One INT8 GEMM with its epilogue, `y = ep(x wᵀ + b)`, where `x` is rounded row by row as each
/// task reads it and `w` was rounded once with [`QMatrix::quantize`].
#[derive(Debug, Clone, Copy)]
pub struct QGemm<'a> {
    /// `[m, k]`.
    pub x: &'a [f32],
    /// Rows of `x` and `y`.
    pub m: usize,
    /// `[n, k]`, rounded.
    pub w: &'a QMatrix,
    /// `[n]`.
    pub b: Option<&'a [f32]>,
    /// What happens to each result.
    pub ep: Epilogue,
}

impl QGemm<'_> {
    /// Runs the GEMM into `y`, handing `spawn` a task count and the task body to run for each.
    /// A task gets [`scratch_len`] floats of scratch of its own. The results are exact sums, so
    /// the split only has to keep the threads busy.
    ///
    /// # Panics
    ///
    /// If a length does not match the shape.
    pub fn run(
        &self,
        y: &mut [f32],
        threads: usize,
        spawn: impl FnOnce(usize, &(dyn Fn(usize, &mut [f32]) + Sync)),
    ) {
        let Self { x, m, w, b, ep } = *self;
        let (k, n) = (w.k, w.rows);
        assert_eq!(x.len(), m * k, "x is not [m, k]");
        assert_eq!(y.len(), m * n, "y is not [m, n]");
        if let Some(b) = b {
            assert_eq!(b.len(), n, "b is not [n]");
        }
        if m == 0 || n == 0 {
            return;
        }
        let y = Shared::new(y);
        let mt = m.div_ceil(MB);
        let tiles = n.div_ceil(TILE);
        // Column blocks of whole tiles, enough of them for three tasks a thread.
        let per = tiles.div_ceil((3 * threads.max(1)).div_ceil(mt)).max(1);
        let nt = tiles.div_ceil(per);
        let kernel = pick();
        spawn(mt * nt, &|t, scratch| {
            let (bi, bj) = (t % mt, t / mt);
            let (r0, rows) = (bi * MB, MB.min(m - bi * MB));
            let (scale, bytes) = scratch[..scratch_len(k)].split_at_mut(MB);
            // SAFETY: any bit pattern is an i8, and i8 needs no alignment.
            let (_, bytes, _) = unsafe { bytes.align_to_mut::<i8>() };
            let xq = &mut bytes[..tiled_len(MB, k)];
            for r in 0..rows {
                scale[r] = quantize_row(&x[(r0 + r) * k..(r0 + r + 1) * k], r, xq);
            }
            // The scratch is reused, so the rows past the end of x in the last tile and the tail of
            // k hold whatever was there. The first give sums that are dropped, and the second meet
            // the zeros w is padded with.
            let tile_bytes = k.div_ceil(KG) * TILE * KG;
            for tj in bj * per..((bj + 1) * per).min(tiles) {
                let wt = &w.q[tj * tile_bytes..(tj + 1) * tile_bytes];
                for ti in 0..rows.div_ceil(TILE) {
                    let xt = &xq[ti * tile_bytes..(ti + 1) * tile_bytes];
                    let sums = kernel(xt, wt);
                    for (i, row) in sums.iter().enumerate().take(rows - ti * TILE) {
                        let (r, j0) = (ti * TILE + i, tj * TILE);
                        for (jj, &s) in row.iter().enumerate().take(n - j0) {
                            let j = j0 + jj;
                            let v = s as f32 * (scale[r] * w.scale[j]);
                            let v = match b {
                                Some(b) => v + b[j],
                                None => v,
                            };
                            let at = (r0 + r) * n + j;
                            // SAFETY: each output belongs to exactly one task.
                            unsafe {
                                y.set(
                                    at,
                                    match ep {
                                        Epilogue::None => v,
                                        Epilogue::Gelu => gelu(v),
                                        Epilogue::Relu => v.max(0.0),
                                        Epilogue::Accumulate => y.get(at) + v,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        });
    }
}

/// A tile of `x` against a tile of `w`: `[8][8]` sums, row `i` of `x` against row `j` of `w`.
type Kernel = fn(&[i8], &[i8]) -> [[i32; TILE]; TILE];

fn pick() -> Kernel {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("i8mm") {
        return |x, w| {
            // SAFETY: i8mm was detected on this machine.
            unsafe { i8mm::tile(x, w) }
        };
    }
    tile_scalar
}

/// The portable kernel, and the one the others are tested against.
fn tile_scalar(x: &[i8], w: &[i8]) -> [[i32; TILE]; TILE] {
    assert_eq!(x.len(), w.len());
    let mut out = [[0i32; TILE]; TILE];
    for (xg, wg) in x.as_chunks::<{ TILE * KG }>().0.iter().zip(w.as_chunks::<{ TILE * KG }>().0) {
        for (i, row) in out.iter_mut().enumerate() {
            let xr = &xg[i / 2 * 2 * KG + i % 2 * KG..][..KG];
            for (j, o) in row.iter_mut().enumerate() {
                let wr = &wg[j / 2 * 2 * KG + j % 2 * KG..][..KG];
                *o += xr.iter().zip(wr).map(|(&a, &b)| i32::from(a) * i32::from(b)).sum::<i32>();
            }
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
mod i8mm {
    use std::arch::aarch64::{int8x16_t, int32x4_t, vdupq_n_s32, vld1q_s8, vst1q_s32};
    use std::arch::asm;

    use super::{KG, TILE};

    /// `acc += a bᵀ` for a and b each two rows of eight, the 2 by 2 result row major.
    /// `vmmlaq_s32` does the same but is not stable yet.
    #[inline(always)]
    fn mmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        // SAFETY: smmla reads and writes registers only, and the callers run where i8mm exists.
        unsafe {
            asm!(
                "smmla {d:v}.4s, {a:v}.16b, {b:v}.16b",
                d = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            );
        }
        acc
    }

    /// # Safety
    ///
    /// The CPU must have i8mm, and `x` and `w` must be whole tiles of the same length.
    #[target_feature(enable = "neon,i8mm")]
    pub(super) unsafe fn tile(x: &[i8], w: &[i8]) -> [[i32; TILE]; TILE] {
        assert!(x.len() == w.len() && x.len().is_multiple_of(TILE * KG));
        let mut acc = [[vdupq_n_s32(0); 4]; 4];
        for (xg, wg) in
            x.as_chunks::<{ TILE * KG }>().0.iter().zip(w.as_chunks::<{ TILE * KG }>().0)
        {
            // SAFETY: each group is 64 bytes, four loads of 16.
            let wv = unsafe { [0, 1, 2, 3].map(|b| vld1q_s8(wg.as_ptr().add(16 * b))) };
            for (a, acc) in acc.iter_mut().enumerate() {
                // SAFETY: as above.
                let xa = unsafe { vld1q_s8(xg.as_ptr().add(16 * a)) };
                for (acc, &wb) in acc.iter_mut().zip(&wv) {
                    *acc = mmla(*acc, xa, wb);
                }
            }
        }
        let mut out = [[0i32; TILE]; TILE];
        for (a, acc) in acc.iter().enumerate() {
            for (b, &v) in acc.iter().enumerate() {
                let mut s = [0i32; 4];
                // SAFETY: s holds four i32.
                unsafe { vst1q_s32(s.as_mut_ptr(), v) };
                out[2 * a][2 * b] = s[0];
                out[2 * a][2 * b + 1] = s[1];
                out[2 * a + 1][2 * b] = s[2];
                out[2 * a + 1][2 * b + 1] = s[3];
            }
        }
        out
    }
}

/// `y = x wᵀ + b` in INT8, rounding `w` on every call. For tests and benchmarks.
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
    let w = QMatrix::quantize(w, n, k);
    let g = QGemm { x, m, w: &w, b, ep: Epilogue::None };
    let len = scratch_len(k);
    g.run(y, threads, |tasks, f| {
        crate::par::for_each(tasks, threads, |t| f(t, &mut vec![0.0; len]));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    #[test]
    fn quantize_round_trips_within_half_a_step() {
        let mut rng = Rng(3);
        for (rows, k) in [(1, 1), (3, 7), (9, 17), (16, 64), (5, 0)] {
            let x = rng.vec(rows * k);
            let q = QMatrix::quantize(&x, rows, k);
            let back = q.dequantize();
            for r in 0..rows {
                for p in 0..k {
                    let (a, b) = (x[r * k + p], back[r * k + p]);
                    assert!((a - b).abs() <= q.scale[r] * 0.5 + 1e-7, "{rows}x{k} {r},{p}");
                }
            }
        }
        let q = QMatrix::quantize(&[0.0, 0.0, 2.0, -1.0], 2, 2);
        assert_eq!((q.scale[0], q.q[at(1, 0, 2)], q.q[at(1, 1, 2)]), (0.0, 127, -64));
    }

    /// The exact result from the rounded values, in f64.
    fn want(x: &QMatrix, w: &QMatrix, b: Option<&[f32]>) -> Vec<f32> {
        let k = x.k;
        let mut y = vec![0.0; x.rows * w.rows];
        for i in 0..x.rows {
            for j in 0..w.rows {
                let s: i64 =
                    (0..k).map(|p| i64::from(x.q[at(i, p, k)]) * i64::from(w.q[at(j, p, k)])).sum();
                let v = s as f32 * (x.scale[i] * w.scale[j]);
                y[i * w.rows + j] = b.map_or(v, |b| v + b[j]);
            }
        }
        y
    }

    #[test]
    fn exact_on_awkward_shapes() {
        let mut rng = Rng(7);
        for (m, k, n) in
            [(1, 1, 1), (3, 5, 2), (9, 17, 9), (64, 64, 64), (70, 300, 130), (130, 8, 7)]
        {
            let (x, w, b) = (rng.vec(m * k), rng.vec(n * k), rng.vec(n));
            let (xq, wq) = (QMatrix::quantize(&x, m, k), QMatrix::quantize(&w, n, k));
            for bias in [None, Some(&b[..])] {
                let want = want(&xq, &wq, bias);
                for threads in [1, 3, 16] {
                    let mut y = vec![f32::NAN; m * n];
                    linear(&x, m, k, &w, n, bias, &mut y, threads);
                    assert!(
                        y.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{m}x{k}x{n} on {threads}"
                    );
                }
            }
        }
    }

    #[test]
    fn kernels_agree() {
        let mut rng = Rng(9);
        for k in [8, 64, 1024] {
            let (x, w) = (rng.vec(TILE * k), rng.vec(TILE * k));
            let (x, w) = (QMatrix::quantize(&x, TILE, k), QMatrix::quantize(&w, TILE, k));
            assert_eq!(pick()(&x.q, &w.q), tile_scalar(&x.q, &w.q), "k {k}");
        }
    }

    #[test]
    fn close_to_f32() {
        let mut rng = Rng(11);
        let (m, k, n) = (20, 1024, 96);
        let (x, w) = (rng.vec(m * k), rng.vec(n * k));
        let mut y = vec![0.0; m * n];
        linear(&x, m, k, &w, n, None, &mut y, 4);
        let mut exact = vec![0.0; m * n];
        crate::gemm::linear(&x, m, k, &w, n, None, &mut exact, 4);
        // Uniform values in [-1, 1) give sums with a spread of about sqrt(k / 9), and the
        // rounding adds an error of about a hundredth of that.
        let spread = (k as f32 / 9.0).sqrt();
        let err = y.iter().zip(&exact).fold(0f32, |e, (a, b)| e.max((a - b).abs()));
        assert!(err < 0.05 * spread, "error {err} against a spread of {spread}");
    }

    #[test]
    fn epilogues() {
        let mut rng = Rng(5);
        let (m, k, n) = (70, 40, 90);
        let (x, w, b, y0) = (rng.vec(m * k), rng.vec(n * k), rng.vec(n), rng.vec(m * n));
        let (xq, wq) = (QMatrix::quantize(&x, m, k), QMatrix::quantize(&w, n, k));
        let lin = want(&xq, &wq, Some(&b));
        for ep in [Epilogue::None, Epilogue::Gelu, Epilogue::Relu, Epilogue::Accumulate] {
            let mut y = y0.clone();
            let g = QGemm { x: &x, m, w: &wq, b: Some(&b), ep };
            g.run(&mut y, 4, |tasks, f| {
                crate::par::for_each(tasks, 4, |t| f(t, &mut vec![0.0; scratch_len(k)]));
            });
            for (i, (&got, (&v, &y))) in y.iter().zip(lin.iter().zip(&y0)).enumerate() {
                let want = match ep {
                    Epilogue::None => v,
                    Epilogue::Gelu => gelu(v),
                    Epilogue::Relu => v.max(0.0),
                    Epilogue::Accumulate => y + v,
                };
                assert_eq!(got.to_bits(), want.to_bits(), "{ep:?} at {i}");
            }
        }
    }
}

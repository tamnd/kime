//! FP32 matrix products in the layout of a PyTorch `Linear`: `y = x wᵀ + b`, with `x` as `[m, k]`
//! and `w` as `[n, k]`, both row major.
//!
//! The weights are packed once into panels of 16 rows, `[n / 16][k][16]` with the last panel
//! padded with zeros, and a micro kernel keeps a 6 by 16 block of outputs in registers while it
//! walks `k`, one broadcast of `x` and two vector loads of the panel per step. Every output is
//! summed the same way wherever it lands: in order over `k` with fused multiply adds in f32,
//! moved to an f64 sum every 64 steps, then rounded to f32 and given its bias. Tiling and threading
//! only decide which outputs run side by side, so the result is the same bit for bit for any
//! thread count and any split, and the same on x86 with FMA as on ARM.
//!
//! On macOS the GEMM goes to Accelerate instead, whose `sgemm` runs on the AMX units at two to
//! four times what the NEON kernel reaches. Its sums are in an order of its own that changes with
//! the number of rows in the call, so it is always called on blocks of exactly 64 rows, the last
//! one padded with zeros. With the row count fixed, a row's result does not depend on the rows
//! around it or on its place in the block, so the bits still do not depend on the batch or the
//! split. They are not the same bits as on other machines, which the parity tests allow for.
//!
//! [`dot`], which attention uses for its scores, sums in eight lanes instead and is not meant to
//! match the GEMM bit for bit.

use kime_tensor::Epilogue;

use crate::ops::gelu;
use crate::par::{self, Shared};

const LANES: usize = 8;
/// Elements of `k` summed in f32 before the running sums move to f64.
const BLOCK: usize = 64;
/// Rows of `x` per micro tile.
const MR: usize = 6;
/// Rows of `w` per panel, two vectors.
pub const NR: usize = 16;
/// Rows of `w` per task, a multiple of NR sized so a task's panels stay in L2.
#[cfg(not(target_os = "macos"))]
const NB: usize = 4 * NR;
/// Rows of `x` per task, a multiple of MR.
#[cfg(not(target_os = "macos"))]
const MB: usize = 24 * MR;

/// `dot(a, b)` in the order described in the module docs.
///
/// # Panics
///
/// If the lengths differ.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    return dot_v::<neon::Neon>(a, b);
    #[cfg(target_arch = "x86_64")]
    if has_fma() {
        // SAFETY: the features dot_fma enables were detected on this machine.
        return unsafe { dot_fma(a, b) };
    }
    #[allow(unreachable_code)]
    dot_v::<[f32; LANES]>(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    dot_v::<avx::Avx>(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// Eight f32 lanes with a fused multiply add, the one vector op the kernels need. Each lane is
/// its own running sum, so every implementation gives the same bits.
trait V8: Copy {
    fn zero() -> Self;
    fn splat(v: f32) -> Self;
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
    fn splat(v: f32) -> Self {
        [v; LANES]
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
        fn splat(v: f32) -> Self {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { Self(vdupq_n_f32(v), vdupq_n_f32(v)) }
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
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
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
        fn splat(v: f32) -> Self {
            // SAFETY: Avx values only exist on machines where AVX was detected.
            unsafe { Self(_mm256_set1_ps(v)) }
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

/// `a · b` in eight lanes: running sums over `k` in steps of eight with fused multiply adds, moved
/// to f64 every 64 elements, the eight added in a fixed tree, then the leftover `k % 8` terms in
/// order.
#[inline(always)]
fn dot_v<V: V8>(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let body = k - k % LANES;
    let mut wide = [0f64; LANES];
    let mut p = 0;
    while p < body {
        let end = (p + BLOCK).min(body);
        let mut acc = V::zero();
        while p < end {
            acc = acc.fma(
                V::load(a[p..p + LANES].try_into().unwrap()),
                V::load(b[p..p + LANES].try_into().unwrap()),
            );
            p += LANES;
        }
        for (w, l) in wide.iter_mut().zip(acc.lanes()) {
            *w += f64::from(l);
        }
    }
    let v = wide;
    let mut s = ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
    for q in body..k {
        s = f64::from(a[q]).mul_add(f64::from(b[q]), s);
    }
    s as f32
}

/// Lays out `w`, `[n, k]` row major, the way [`Gemm`] reads it: panels of 16 rows, or as it is
/// on macOS, where Accelerate reads it.
///
/// # Panics
///
/// If `w` is not `[n, k]`.
#[must_use]
pub fn pack(w: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), n * k, "w is not [n, k]");
    if cfg!(target_os = "macos") { w.to_vec() } else { pack_panels(w, n, k) }
}

/// Values in `w` as [`pack`] lays it out.
fn packed_len(n: usize, k: usize) -> usize {
    if cfg!(target_os = "macos") { n * k } else { n.div_ceil(NR) * NR * k }
}

/// Floats of scratch each task of a GEMM with these sizes needs.
#[must_use]
pub fn scratch_len(k: usize, n: usize) -> usize {
    #[cfg(target_os = "macos")]
    return blas::ROWS * (k + 3 * n.min(blas::COLS)) + 1;
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (k, n);
        0
    }
}

/// `w` in panels: `[n.div_ceil(16)][k][16]`, with the rows past `n` zero.
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn pack_panels(w: &[f32], n: usize, k: usize) -> Vec<f32> {
    let panels = n.div_ceil(NR);
    let mut out = vec![0f32; panels * k * NR];
    if k == 0 {
        return out;
    }
    for (p, panel) in out.chunks_exact_mut(k * NR).enumerate() {
        for c in 0..NR.min(n - p * NR) {
            let row = &w[(p * NR + c) * k..][..k];
            for (q, &v) in row.iter().enumerate() {
                panel[q * NR + c] = v;
            }
        }
    }
    out
}

/// Rows `i..i + R` of `x` against one panel, the sums before the bias.
///
/// # Safety
///
/// Rows `i..i + R` must be in `x`, which is `[_, k]`, and `panel` must hold `k * NR` values.
#[inline(always)]
#[cfg_attr(target_os = "macos", allow(dead_code))]
unsafe fn kernel<V: V8, const R: usize>(
    x: &[f32],
    k: usize,
    i: usize,
    panel: &[f32],
) -> [[f32; NR]; R] {
    let xs: [*const f32; R] = std::array::from_fn(|r| x.as_ptr().wrapping_add((i + r) * k));
    let pw = panel.as_ptr();
    let mut wide = [[0f64; NR]; R];
    let mut q = 0;
    while q < k {
        let end = (q + BLOCK).min(k);
        let mut acc = [[V::zero(); 2]; R];
        while q < end {
            // SAFETY: q < k, so the 16 values of step q are in the panel and x[i + r][q] in x.
            let (w0, w1) = unsafe {
                let at = pw.add(q * NR);
                (
                    V::load(&*at.cast::<[f32; LANES]>()),
                    V::load(&*at.add(LANES).cast::<[f32; LANES]>()),
                )
            };
            for r in 0..R {
                // SAFETY: as above.
                let xv = V::splat(unsafe { *xs[r].add(q) });
                acc[r][0] = acc[r][0].fma(xv, w0);
                acc[r][1] = acc[r][1].fma(xv, w1);
            }
            q += 1;
        }
        for r in 0..R {
            for h in 0..2 {
                for (w, l) in wide[r][h * LANES..][..LANES].iter_mut().zip(acc[r][h].lanes()) {
                    *w += f64::from(l);
                }
            }
        }
    }
    wide.map(|row| row.map(|v| v as f32))
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
struct Args<'a> {
    x: &'a [f32],
    w: &'a [f32],
    b: Option<&'a [f32]>,
    ep: Epilogue,
    k: usize,
    n: usize,
    y: &'a Shared<'a>,
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
impl Args<'_> {
    #[inline(always)]
    fn run<V: V8, const R: usize>(&self, i: usize, p: usize) {
        let k = self.k;
        let panel = &self.w[p * k * NR..][..k * NR];
        // SAFETY: the caller keeps i + R within m, and the panel was sliced to k * NR.
        let out = unsafe { kernel::<V, R>(self.x, k, i, panel) };
        let cols = NR.min(self.n - p * NR);
        for (r, row) in out.iter().enumerate() {
            for (c, &v) in row[..cols].iter().enumerate() {
                self.put(i + r, p * NR + c, v);
            }
        }
    }

    /// Adds the bias, applies the epilogue and stores element `(i, j)`.
    #[inline(always)]
    fn put(&self, i: usize, j: usize, v: f32) {
        let v = match self.b {
            Some(b) => v + b[j],
            None => v,
        };
        let at = i * self.n + j;
        let v = match self.ep {
            Epilogue::None => v,
            Epilogue::Gelu => gelu(v),
            Epilogue::Relu => v.max(0.0),
            // SAFETY: each (i, j) pair belongs to exactly one task and one tile within it.
            Epilogue::Accumulate => v + unsafe { self.y.get(at) },
        };
        // SAFETY: as above.
        unsafe { self.y.set(at, v) };
    }

    /// Finishes columns `j0..` of row `i` of `y` from their sums, as [`put`](Self::put) does for one element, with
    /// the choices made once for the row.
    #[cfg(target_os = "macos")]
    fn put_row(&self, i: usize, j0: usize, sums: &[f32]) {
        // SAFETY: each stretch of a row belongs to exactly one task.
        let y = unsafe { self.y.slice_mut(i * self.n + j0, sums.len()) };
        let biased = |j: usize, v: f32| match self.b {
            Some(b) => v + b[j0 + j],
            None => v,
        };
        match (self.b, self.ep) {
            (None, Epilogue::None) => y.copy_from_slice(sums),
            (_, Epilogue::None) => {
                y.iter_mut().zip(sums).enumerate().for_each(|(j, (y, &v))| *y = biased(j, v))
            }
            (_, Epilogue::Gelu) => {
                y.iter_mut().zip(sums).enumerate().for_each(|(j, (y, &v))| *y = gelu(biased(j, v)))
            }
            (_, Epilogue::Relu) => y
                .iter_mut()
                .zip(sums)
                .enumerate()
                .for_each(|(j, (y, &v))| *y = biased(j, v).max(0.0)),
            (Some(b), Epilogue::Accumulate) => {
                y.iter_mut().zip(sums).zip(&b[j0..]).for_each(|((y, &v), &b)| *y += v + b)
            }
            (None, Epilogue::Accumulate) => y.iter_mut().zip(sums).for_each(|(y, &v)| *y += v),
        }
    }

    /// Rows `rows` against the panels `panels`.
    #[inline(always)]
    fn block<V: V8>(&self, rows: (usize, usize), panels: (usize, usize)) {
        for p in panels.0..panels.1 {
            let mut i = rows.0;
            while i + MR <= rows.1 {
                self.run::<V, MR>(i, p);
                i += MR;
            }
            match rows.1 - i {
                0 => {}
                1 => self.run::<V, 1>(i, p),
                2 => self.run::<V, 2>(i, p),
                3 => self.run::<V, 3>(i, p),
                4 => self.run::<V, 4>(i, p),
                _ => self.run::<V, 5>(i, p),
            }
        }
    }

    fn block_dispatch(&self, rows: (usize, usize), panels: (usize, usize)) {
        #[cfg(target_arch = "aarch64")]
        return self.block::<neon::Neon>(rows, panels);
        #[cfg(target_arch = "x86_64")]
        if has_fma() {
            // SAFETY: the features block_fma enables were detected on this machine.
            unsafe { self.block_fma(rows, panels) };
            return;
        }
        #[allow(unreachable_code)]
        self.block::<[f32; LANES]>(rows, panels);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn block_fma(&self, rows: (usize, usize), panels: (usize, usize)) {
        self.block::<avx::Avx>(rows, panels);
    }
}

/// `y = x wᵀ + b` with `x` as `[m, k]`, `w` as `[n, k]`, `b` as `[n]` and `y` as `[m, n]`. This
/// packs `w` on every call, so a caller with a fixed weight should [`pack`] it once and run a
/// [`Gemm`].
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
    let w = pack(w, n, k);
    let g = Gemm { x, m, k, w: &w, n, b, ep: Epilogue::None };
    let len = scratch_len(k, n);
    g.run(y, threads, |tasks, f| par::for_each(tasks, threads, |t| f(t, &mut vec![0.0; len])));
}

/// One GEMM with its epilogue, `y = ep(x wᵀ + b)`, split into tiles that any thread may run.
/// Every output element is computed the same way whatever the split, so the split is free to
/// follow the thread count.
#[derive(Debug, Clone, Copy)]
pub struct Gemm<'a> {
    /// `[m, k]`.
    pub x: &'a [f32],
    /// Rows of `x` and `y`.
    pub m: usize,
    /// The reduction length.
    pub k: usize,
    /// `[n, k]` as [`pack`] lays it out.
    pub w: &'a [f32],
    /// Columns of `y`.
    pub n: usize,
    /// `[n]`.
    pub b: Option<&'a [f32]>,
    /// What happens to each result.
    pub ep: Epilogue,
}

impl Gemm<'_> {
    /// Rows per task: the largest block that still gives every thread a few tasks.
    #[cfg(not(target_os = "macos"))]
    fn row_block(&self, threads: usize) -> usize {
        let nt = self.n.div_ceil(NB);
        [MB, 12 * MR, 6 * MR, 3 * MR]
            .into_iter()
            .find(|&mb| self.m.div_ceil(mb) * nt >= 3 * threads)
            .unwrap_or(MR)
    }

    /// Runs the GEMM into `y`, handing `spawn` a task count and the task body to run for each.
    /// A task gets [`scratch_len`] floats of scratch of its own.
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
        let Self { x, m, k, w, n, b, ep } = *self;
        assert_eq!(x.len(), m * k, "x is not [m, k]");
        assert_eq!(w.len(), packed_len(n, k), "w is not [n, k] packed");
        assert_eq!(y.len(), m * n, "y is not [m, n]");
        if let Some(b) = b {
            assert_eq!(b.len(), n, "b is not [n]");
        }
        if m == 0 || n == 0 {
            return;
        }
        let shared = Shared::new(y);
        let args = Args { x, w, b, ep, k, n, y: &shared };
        if k == 0 {
            for i in 0..m {
                for j in 0..n {
                    args.put(i, j, 0.0);
                }
            }
            return;
        }
        #[cfg(target_os = "macos")]
        {
            blas::run(&args, m, threads, spawn);
        }
        #[cfg(not(target_os = "macos"))]
        self.run_panels(&args, threads, spawn);
    }

    #[cfg(not(target_os = "macos"))]
    fn run_panels(
        &self,
        args: &Args<'_>,
        threads: usize,
        spawn: impl FnOnce(usize, &(dyn Fn(usize, &mut [f32]) + Sync)),
    ) {
        let (m, n) = (self.m, self.n);
        let mb = self.row_block(threads);
        let mt = m.div_ceil(mb);
        let (panels, per) = (n.div_ceil(NR), NB / NR);
        let nt = panels.div_ceil(per);
        spawn(mt * nt, &|t, _| {
            let (bi, bj) = (t % mt, t / mt);
            let rows = (bi * mb, ((bi + 1) * mb).min(m));
            args.block_dispatch(rows, (bj * per, ((bj + 1) * per).min(panels)));
        });
    }
}

#[cfg(target_os = "macos")]
mod blas {
    use super::Args;

    /// Rows per call. The bits of a row depend on the row count of the call, so it never changes.
    pub(super) const ROWS: usize = 64;
    /// Elements of `k` per call. Accelerate sums in f32, so the calls cover `k` in blocks of this
    /// and their results are added in f64, which keeps long rows as accurate as the NEON kernel.
    const KB: usize = 128;
    /// Columns per task are a multiple of this when the columns are split.
    pub(super) const COLS: usize = 256;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    /// `len` f64 values inside `buf`, which holds at least `2 len + 1` f32 values.
    fn f64s(buf: &mut [f32], len: usize) -> &mut [f64] {
        // SAFETY: any bit pattern is an f64, and align_to_mut only hands out aligned values.
        let (_, mid, _) = unsafe { buf.align_to_mut::<f64>() };
        &mut mid[..len]
    }

    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    /// One task per block of 64 rows of `x`, each a few `sgemm` calls into the task's scratch,
    /// then the bias and the epilogue on the way to `y`.
    ///
    /// The columns are split in chunks of 256 whatever the number of rows, since Accelerate picks
    /// its kernels by the width of the call and a width that followed the batch would change bits.
    pub(super) fn run(
        args: &Args<'_>,
        m: usize,
        threads: usize,
        spawn: impl FnOnce(usize, &(dyn Fn(usize, &mut [f32]) + Sync)),
    ) {
        let (k, n) = (args.k, args.n);
        let dim = |v: usize| i32::try_from(v).expect("GEMM sizes fit in an i32");
        let _ = threads;
        let (mt, cols) = (m.div_ceil(ROWS), COLS.min(n));
        spawn(mt * n.div_ceil(cols), &|t, scratch| {
            let (bi, bj) = (t % mt, t / mt);
            let (r0, rows) = (bi * ROWS, ROWS.min(m - bi * ROWS));
            let (j0, nc) = (bj * cols, cols.min(n - bj * cols));
            let (xs, rest) = scratch[..ROWS * (k + 3 * cols) + 1].split_at_mut(ROWS * k);
            let (c, wide) = rest.split_at_mut(ROWS * cols);
            let (c, wide) = (&mut c[..ROWS * nc], &mut f64s(wide, ROWS * cols)[..ROWS * nc]);
            wide.fill(0.0);
            let x = if rows == ROWS {
                &args.x[r0 * k..(r0 + ROWS) * k]
            } else {
                xs[..rows * k].copy_from_slice(&args.x[r0 * k..(r0 + rows) * k]);
                xs[rows * k..].fill(0.0);
                &xs[..]
            };
            let mut p = 0;
            while p < k {
                let kc = KB.min(k - p);
                // SAFETY: x is ROWS by k and w is n by k, read from column p for kc columns, and
                // c is ROWS by n, all row major and dense.
                unsafe {
                    cblas_sgemm(
                        ROW_MAJOR,
                        NO_TRANS,
                        TRANS,
                        dim(ROWS),
                        dim(nc),
                        dim(kc),
                        1.0,
                        x.as_ptr().add(p),
                        dim(k),
                        args.w.as_ptr().add(j0 * k + p),
                        dim(k),
                        0.0,
                        c.as_mut_ptr(),
                        dim(nc),
                    );
                }
                wide.iter_mut().zip(c.iter()).for_each(|(w, &v)| *w += f64::from(v));
                p += kc;
            }
            c.iter_mut().zip(wide.iter()).for_each(|(c, &w)| *c = w as f32);
            for r in 0..rows {
                args.put_row(r0 + r, j0, &c[r * nc..(r + 1) * nc]);
            }
        });
    }
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
                    // Accelerate sums in f32 alone, so it drifts a little further on long rows.
                    let tol = if cfg!(target_os = "macos") { 2e-5 } else { 1e-5 };
                    close(&y, &want, tol, &format!("{m}x{k}x{n}"));
                }
            }
        }
    }

    #[test]
    fn epilogues_on_every_column() {
        let mut rng = Rng(5);
        let (m, k, n) = (70, 40, 600);
        let (x, w, b, y0) = (rng.vec(m * k), rng.vec(n * k), rng.vec(n), rng.vec(m * n));
        let packed = pack(&w, n, k);
        let lin = naive(&x, m, k, &w, n, Some(&b));
        for ep in [Epilogue::None, Epilogue::Gelu, Epilogue::Relu, Epilogue::Accumulate] {
            let want: Vec<f32> = lin
                .iter()
                .zip(&y0)
                .map(|(&v, &y)| match ep {
                    Epilogue::None => v,
                    Epilogue::Gelu => gelu(v),
                    Epilogue::Relu => v.max(0.0),
                    Epilogue::Accumulate => y + v,
                })
                .collect();
            let mut y = y0.clone();
            let g = Gemm { x: &x, m, k, w: &packed, n, b: Some(&b), ep };
            let len = scratch_len(k, n);
            g.run(&mut y, 4, |tasks, f| par::for_each(tasks, 4, |t| f(t, &mut vec![0.0; len])));
            close(&y, &want, 1e-4, &format!("{ep:?}"));
        }
    }

    #[test]
    fn same_bits_for_any_split() {
        let mut rng = Rng(11);
        let (m, k, n) = (137, 300, 600);
        let x = rng.vec(m * k);
        let w = rng.vec(n * k);
        let mut one = vec![0f32; m * n];
        linear(&x, m, k, &w, n, None, &mut one, 1);
        for threads in [2, 5, 10, 16] {
            let mut y = vec![0f32; m * n];
            linear(&x, m, k, &w, n, None, &mut y, threads);
            assert!(y.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // And the same bits for a row wherever it sits in the batch.
        for i in [0, 5, 70, 136] {
            let mut row = vec![0f32; n];
            linear(&x[i * k..(i + 1) * k], 1, k, &w, n, None, &mut row, 1);
            assert!(row.iter().zip(&one[i * n..]).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn dot_matches_naive() {
        let mut rng = Rng(3);
        for k in [0, 1, 7, 8, 64, 65, 200] {
            let (a, b) = (rng.vec(k), rng.vec(k));
            let want = naive(&a, 1, k, &b, 1, None);
            close(&[dot(&a, &b)], &want, 1e-5, &format!("dot {k}"));
        }
    }
}

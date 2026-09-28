//! GEMMs through cuBLASLt, set up once per plan step so a run is one call with no descriptor work.
//!
//! Every GEMM is a PyTorch `Linear`, `y = x wᵀ` with `x` as `[m, k]` and `w` as `[n, k]`, both
//! row major. cuBLASLt is column major, so it computes `yᵀ = w xᵀ`: `w` read as a `k × n` column
//! major matrix and transposed, `x` as `k × m`, and `y` as `n × m`, which is `y` row major.

use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtr};

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;

use kime_tensor::{Error, Result};

use crate::{WORKSPACE, dev};

fn err(e: lt::CublasError) -> Error {
    Error::Device(format!("cublasLt: {e:?}"))
}

/// A cuBLASLt handle.
pub(crate) struct Handle(pub(crate) sys::cublasLtHandle_t);

// SAFETY: a cuBLASLt handle may be used from any thread. The backend only uses it from inside
// `Backend::lower` and `Backend::run`, and the executor never runs two of those at once on one
// backend because both go through `&mut Executor`.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

impl Handle {
    pub(crate) fn new() -> Result<Self> {
        lt::create_handle().map(Self).map_err(err)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle was created in `new` and is destroyed once, here.
        let _ = unsafe { lt::destroy_handle(self.0) };
    }
}

/// Element types a GEMM can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Ty {
    F16,
    F32,
}

impl Ty {
    fn cuda(self) -> sys::cudaDataType {
        match self {
            Ty::F16 => sys::cudaDataType::CUDA_R_16F,
            Ty::F32 => sys::cudaDataType::CUDA_R_32F,
        }
    }
}

/// How many of cuBLASLt's ranked algorithms a GEMM keeps to choose from.
pub(crate) const CANDIDATES: usize = 8;

/// The rows cuBLASLt ranks the reference algorithm for, whatever rows a GEMM has. A GEMM of a
/// given inner size, width and types gives the reference's bits in every bucket, so each output
/// sums its inner products in the same order and a row gets the same bits whatever else is in the
/// batch. See [`Gemm::faster`] for how a bucket can still run another algorithm.
pub(crate) const RANKED_ROWS: usize = 256;

type Algo = sys::cublasLtMatmulAlgo_t;

/// One GEMM with its descriptors, the algorithms cuBLASLt ranks for its shape, and the one it runs.
pub(crate) struct Gemm {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    /// `x` and `y` at [`RANKED_ROWS`] rows.
    rb: sys::cublasLtMatrixLayout_t,
    rc: sys::cublasLtMatrixLayout_t,
    algos: Vec<Algo>,
    /// Index into `algos` of the reference, the algorithm whose bits every bucket gives.
    pub(crate) pick: usize,
    /// What runs: the reference, or one ranked for this GEMM's rows that gives the same bits.
    algo: Algo,
    ab: Ty,
    out: Ty,
    beta: f32,
    workspace: usize,
    /// Rows, inner size and columns.
    pub(crate) dims: (usize, usize, usize),
    /// Device addresses of `w`, `x` and `y`.
    pub(crate) w: u64,
    pub(crate) x: u64,
    pub(crate) y: u64,
}

// SAFETY: the descriptors are plain host objects owned by this value, used only through `&self`
// from one thread at a time as the Handle comment explains.
unsafe impl Send for Gemm {}

/// cuBLASLt's ranking of algorithms for `desc` with `x` as `b` and `y` as `c`, with no split K and
/// at most `workspace` bytes, best first.
fn rank(
    h: &Handle,
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    workspace: usize,
) -> Result<Vec<Algo>> {
    let pref = lt::create_matmul_pref().map_err(err)?;
    let ws = workspace as u64;
    // SAFETY: an all zero result is a valid value for cuBLASLt to overwrite.
    let mut found: [sys::cublasLtMatmulHeuristicResult_t; CANDIDATES] =
        unsafe { std::mem::zeroed() };
    let mut count = 0;
    // SAFETY: every descriptor is live, the attribute buffers are a u64 and a u32, and `found` has
    // room for the CANDIDATES results asked for.
    let status = unsafe {
        let set = lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&raw const ws).cast::<c_void>(),
            size_of::<u64>(),
        );
        // No split K: only algorithms that sum each output over the whole inner size in one
        // pass, so a row's result does not depend on how many other rows are in the batch.
        let none: u32 = 0;
        let set = set.and_then(|()| {
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                (&raw const none).cast::<c_void>(),
                size_of::<u32>(),
            )
        });
        let status = set.and_then(|()| {
            sys::cublasLtMatmulAlgoGetHeuristic(
                h.0,
                desc,
                a,
                b,
                c,
                c,
                pref,
                CANDIDATES as i32,
                found.as_mut_ptr(),
                &raw mut count,
            )
            .result()
        });
        let _ = lt::destroy_matmul_pref(pref);
        status
    };
    status.map_err(err)?;
    let ok = sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let n = usize::try_from(count).unwrap_or(0).min(CANDIDATES);
    Ok(found[..n].iter().filter(|r| r.state == ok).map(|r| r.algo).collect())
}

impl Gemm {
    /// Sets up `y = x wᵀ` (plus `y` when `accumulate`) for `m` rows, with `w` and `x` of type
    /// `ab` and `y` of type `c`, running the algorithm cuBLASLt ranks `pick`th (0 is its first
    /// choice) for [`RANKED_ROWS`] rows, or the first after it that takes `m` rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        h: &Handle,
        (m, k, n): (usize, usize, usize),
        ab: Ty,
        c: Ty,
        accumulate: bool,
        workspace: usize,
        (w, x, y): (u64, u64, u64),
        pick: usize,
    ) -> Result<Self> {
        let desc =
            lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, Ty::F32.cuda())
                .map_err(err)?;
        let null = std::ptr::null_mut();
        let mut g = Self {
            desc,
            a: null,
            b: null,
            c: null,
            rb: null,
            rc: null,
            algos: Vec::new(),
            pick: 0,
            // SAFETY: an all zero algorithm is plain data, replaced below before anything runs.
            algo: unsafe { std::mem::zeroed() },
            ab,
            out: c,
            beta: if accumulate { 1.0 } else { 0.0 },
            workspace,
            dims: (m, k, n),
            w,
            x,
            y,
        };
        // CUBLAS_OP_T. The enum lives in cuBLAS proper, which this crate does not load.
        let op: i32 = 1;
        // SAFETY: desc is live and the buffer is one cublasOperation_t, an int32.
        unsafe {
            lt::set_matmul_desc_attribute(
                g.desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&raw const op).cast::<c_void>(),
                size_of::<i32>(),
            )
            .map_err(err)?;
        }
        let (m, k, n) = (m as u64, k as u64, n as u64);
        let r = RANKED_ROWS as u64;
        g.a = lt::create_matrix_layout(ab.cuda(), k, n, k as i64).map_err(err)?;
        g.b = lt::create_matrix_layout(ab.cuda(), k, m, k as i64).map_err(err)?;
        g.c = lt::create_matrix_layout(c.cuda(), n, m, n as i64).map_err(err)?;
        g.rb = lt::create_matrix_layout(ab.cuda(), k, r, k as i64).map_err(err)?;
        g.rc = lt::create_matrix_layout(c.cuda(), n, r, n as i64).map_err(err)?;
        g.algos = rank(h, g.desc, g.a, g.rb, g.rc, workspace)?;
        let first = if pick < g.algos.len() { pick } else { 0 };
        let order = (first..g.algos.len()).chain(0..first);
        g.pick = order.into_iter().find(|&i| g.takes(h, &g.algos[i])).ok_or_else(|| {
            err(lt::CublasError(sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED))
        })?;
        g.algo = g.algos[g.pick];
        Ok(g)
    }

    /// Whether `algo` runs this GEMM at its own rows within the workspace.
    fn takes(&self, h: &Handle, algo: &Algo) -> bool {
        let ok = sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
        // SAFETY: an all zero result is a valid value for cuBLASLt to overwrite, and every
        // descriptor is live.
        unsafe {
            let mut r: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            sys::cublasLtMatmulAlgoCheck(
                h.0, self.desc, self.a, self.b, self.c, self.c, algo, &raw mut r,
            ) == ok
                && r.workspaceSize <= self.workspace
        }
    }

    /// How many algorithms there are to pick from.
    pub(crate) fn candidates(&self) -> usize {
        self.algos.len()
    }

    /// Makes the `pick`th ranked algorithm the reference and the one that runs.
    pub(crate) fn set_pick(&mut self, pick: usize) {
        self.pick = pick;
        self.algo = self.algos[pick];
    }

    /// Runs the first algorithm cuBLASLt ranks for this GEMM's own rows that gives the reference's
    /// bits, if it ranks one above the reference.
    ///
    /// The reference is ranked for [`RANKED_ROWS`], which suits one question, and in a bucket of
    /// thousands of rows it can run at half the speed of cuBLASLt's pick for those rows. Two
    /// algorithms that walk the inner size in the same order give the same bits, and `probe` runs
    /// each candidate against the reference on the same inputs to find one. A row still gets the
    /// same bits in every bucket, so this only changes the speed.
    pub(crate) fn faster(
        &mut self,
        h: &Handle,
        probe: &mut Probe,
        s: &Arc<CudaStream>,
        ws: u64,
    ) -> Result<()> {
        if self.dims.0 == RANKED_ROWS {
            return Ok(());
        }
        let reference = self.algos[self.pick];
        for algo in rank(h, self.desc, self.a, self.b, self.c, self.workspace)? {
            if algo.data == reference.data {
                break;
            }
            if self.takes(h, &algo) && probe.same(h, self, &reference, &algo, s, ws)? {
                self.algo = algo;
                break;
            }
        }
        Ok(())
    }

    /// Enqueues `algo` on `stream` with `x` as `b` and `y` as `c`.
    ///
    /// # Safety
    ///
    /// `x` and `y` must point at buffers of the shapes `b` and `c` give, and `workspace` at
    /// `size` bytes no other work on the stream uses meanwhile.
    #[allow(clippy::too_many_arguments)]
    unsafe fn matmul(
        &self,
        h: &Handle,
        algo: &Algo,
        (b, c): (sys::cublasLtMatrixLayout_t, sys::cublasLtMatrixLayout_t),
        (x, y): (u64, u64),
        workspace: u64,
        size: usize,
        stream: sys::cudaStream_t,
    ) -> Result<()> {
        let alpha = 1f32;
        // SAFETY: by the caller, and the descriptors are live.
        unsafe {
            lt::matmul(
                h.0,
                self.desc,
                (&raw const alpha).cast(),
                (&raw const self.beta).cast(),
                self.w as *const c_void,
                self.a,
                x as *const c_void,
                b,
                y as *const c_void,
                c,
                y as *mut c_void,
                c,
                algo,
                workspace as *mut c_void,
                size,
                stream,
            )
            .map_err(err)
        }
    }

    /// Enqueues the GEMM on `stream`.
    ///
    /// # Safety
    ///
    /// The three addresses must still point at buffers of the shapes given to [`Gemm::new`], and
    /// `workspace` at `size` bytes no other work on the stream uses meanwhile.
    pub(crate) unsafe fn run(
        &self,
        h: &Handle,
        workspace: u64,
        size: usize,
        stream: sys::cudaStream_t,
    ) -> Result<()> {
        // SAFETY: by the caller.
        unsafe {
            self.matmul(h, &self.algo, (self.b, self.c), (self.x, self.y), workspace, size, stream)
        }
    }
}

impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: each descriptor was created in `new` and is destroyed once, here. A null layout
        // is one `new` never got to create.
        unsafe {
            for l in [self.a, self.b, self.c, self.rb, self.rc] {
                if !l.is_null() {
                    let _ = lt::destroy_matrix_layout(l);
                }
            }
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

/// What a GEMM's shape and types are, for [`Probe`].
type Shape = (usize, usize, Ty, Ty, bool);

/// Tells whether two algorithms give the same bits, by running both on the same inputs at
/// [`RANKED_ROWS`] rows, and remembers the answer for each shape and pair.
#[derive(Default)]
pub(crate) struct Probe {
    seen: HashMap<(Shape, [u64; 8], [u64; 8]), bool>,
}

impl Probe {
    fn same(
        &mut self,
        h: &Handle,
        g: &Gemm,
        reference: &Algo,
        algo: &Algo,
        s: &Arc<CudaStream>,
        ws: u64,
    ) -> Result<bool> {
        let (_, k, n) = g.dims;
        let shape = (k, n, g.ab, g.out, g.beta != 0.0);
        let key = (shape, reference.data, algo.data);
        if let Some(&same) = self.seen.get(&key) {
            return Ok(same);
        }
        let r = RANKED_ROWS;
        let x = s.clone_htod(&noise(g.ab, r * k, 1)).map_err(dev)?;
        let y0 = noise(g.out, r * n, 2);
        let mut outs = Vec::new();
        for a in [reference, algo] {
            // A fresh `y` for each, so both accumulate onto the same values.
            let y = s.clone_htod(&y0).map_err(dev)?;
            let (xp, yp) = (x.device_ptr(s).0, y.device_ptr(s).0);
            // SAFETY: `x` and `y` hold RANKED_ROWS rows of the GEMM's types, `w` is the GEMM's own
            // weight, and the workspace is used by nothing else on the stream while lowering.
            let run = unsafe {
                g.matmul(h, a, (g.rb, g.rc), (xp, yp), ws, WORKSPACE, s.cu_stream().cast())
            };
            // An algorithm ranked for many rows may not take RANKED_ROWS, and then it cannot be
            // shown to match.
            if run.is_err() {
                self.seen.insert(key, false);
                return Ok(false);
            }
            outs.push(s.clone_dtoh(&y).map_err(dev)?);
        }
        let same = outs[0] == outs[1];
        self.seen.insert(key, same);
        Ok(same)
    }
}

/// `len` values of type `t` as bytes, from a fixed seed: signed, with magnitudes from 1/4 to 2 so
/// no product or sum is near the edge of the type.
fn noise(t: Ty, len: usize, seed: u64) -> Vec<u8> {
    let mut v = 0x9e37_79b9_7f4a_7c15_u64 ^ seed;
    let mut next = || {
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        v
    };
    let mut out = Vec::with_capacity(len * 4);
    for _ in 0..len {
        let r = next();
        let sign = (r >> 63) as u32;
        let exp = (r >> 40) as u32 % 3;
        match t {
            Ty::F16 => {
                let bits = (sign << 15) | ((13 + exp) << 10) | (r as u32 & 0x3ff);
                out.extend_from_slice(&(bits as u16).to_le_bytes());
            }
            Ty::F32 => {
                let bits = (sign << 31) | ((125 + exp) << 23) | (r as u32 & 0x7f_ffff);
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

//! GEMMs through cuBLASLt, set up once per plan step so a run is one call with no descriptor work.
//!
//! Every GEMM is a PyTorch `Linear`, `y = x wᵀ` with `x` as `[m, k]` and `w` as `[n, k]`, both
//! row major. cuBLASLt is column major, so it computes `yᵀ = w xᵀ`: `w` read as a `k × n` column
//! major matrix and transposed, `x` as `k × m`, and `y` as `n × m`, which is `y` row major.

use std::ffi::c_void;
use std::mem::size_of;

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;

use kime_tensor::{Error, Result};

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

/// One GEMM with its descriptors, the algorithms cuBLASLt ranks for its shape, and the one it runs.
pub(crate) struct Gemm {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    algos: Vec<sys::cublasLtMatmulAlgo_t>,
    /// Index into `algos` of the one that runs.
    pub(crate) pick: usize,
    beta: f32,
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

impl Gemm {
    /// Sets up `y = x wᵀ` (plus `y` when `accumulate`) for `m` rows, with `w` and `x` of type
    /// `ab` and `y` of type `c`, running the algorithm cuBLASLt ranks `pick`th (0 is its first
    /// choice) or its first when it offers fewer.
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
        let mut g = Self {
            desc,
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            algos: Vec::new(),
            pick: 0,
            beta: if accumulate { 1.0 } else { 0.0 },
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
        g.a = lt::create_matrix_layout(ab.cuda(), k, n, k as i64).map_err(err)?;
        g.b = lt::create_matrix_layout(ab.cuda(), k, m, k as i64).map_err(err)?;
        g.c = lt::create_matrix_layout(c.cuda(), n, m, n as i64).map_err(err)?;
        let pref = lt::create_matmul_pref().map_err(err)?;
        let ws = workspace as u64;
        // SAFETY: an all zero result is a valid value for cuBLASLt to overwrite.
        let mut found: [sys::cublasLtMatmulHeuristicResult_t; CANDIDATES] =
            unsafe { std::mem::zeroed() };
        let mut count = 0;
        // SAFETY: every descriptor is live, the attribute buffers are a u64 and a u32, and `found` has room
        // for the CANDIDATES results asked for.
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
                    g.desc,
                    g.a,
                    g.b,
                    g.c,
                    g.c,
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
        g.algos = found[..n].iter().filter(|r| r.state == ok).map(|r| r.algo).collect();
        if g.algos.is_empty() {
            return Err(err(lt::CublasError(sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED)));
        }
        g.pick = if pick < g.algos.len() { pick } else { 0 };
        Ok(g)
    }

    /// How many algorithms there are to pick from.
    pub(crate) fn candidates(&self) -> usize {
        self.algos.len()
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
                self.x as *const c_void,
                self.b,
                self.y as *const c_void,
                self.c,
                self.y as *mut c_void,
                self.c,
                &raw const self.algos[self.pick],
                workspace as *mut c_void,
                size,
                stream,
            )
            .map_err(err)
        }
    }
}

impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: each descriptor was created in `new` and is destroyed once, here. A null layout
        // is one `new` never got to create.
        unsafe {
            for l in [self.a, self.b, self.c] {
                if !l.is_null() {
                    let _ = lt::destroy_matrix_layout(l);
                }
            }
            let _ = lt::destroy_matmul_desc(self.desc);
        }
    }
}

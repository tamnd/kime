//! The NVIDIA backend, from spec/08-cuda.md.
//!
//! This first version runs the compat graph with FP16 weights and activations and FP32
//! accumulation. The residual stream, the head's inputs and everything after the scorer stay in
//! FP32. The GEMMs go through cuBLASLt, and the other ops are our own kernels in
//! `kernels/compat.cu`, compiled with NVRTC for the device when the backend starts, so the build
//! needs no CUDA toolkit and the binary runs on any machine with a driver.
//!
//! Every launch is sized for the plan's bucket and every kernel reads the batch's real counts from
//! a device buffer, so a run is a fixed sequence of launches that a CUDA graph can capture.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

mod compat;
mod lt;
mod plan;

pub use compat::executor;
pub use plan::{CudaPlan, Weights};

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use kime_tensor::{Error, Result};

const SOURCE: &str = include_str!("../kernels/compat.cu");

/// Bytes of scratch cuBLASLt may use.
const WORKSPACE: usize = 32 << 20;

fn dev(e: impl std::fmt::Debug) -> Error {
    Error::Device(format!("{e:?}"))
}

/// The kernels of `kernels/compat.cu`.
pub(crate) struct Kernels {
    pub(crate) embed: CudaFunction,
    /// Indexed by `2 * (input is f16) + (output is f16)`.
    pub(crate) ln: [CudaFunction; 4],
    /// f32, f16.
    pub(crate) bias_act: [CudaFunction; 2],
    /// f32, f16.
    pub(crate) rope: [CudaFunction; 2],
    /// Indexed like `ln`.
    pub(crate) attention: [CudaFunction; 4],
    /// Indexed like `ln`.
    pub(crate) geglu: [CudaFunction; 4],
    pub(crate) add_type: CudaFunction,
    /// f32, f16.
    pub(crate) gather: [CudaFunction; 2],
    pub(crate) act_features: CudaFunction,
    pub(crate) to_f16: CudaFunction,
}

impl Kernels {
    fn load(m: &Arc<CudaModule>) -> Result<Self> {
        let f = |name: &str| m.load_function(name).map_err(dev);
        Ok(Self {
            embed: f("embed")?,
            ln: [f("ln_f32_f32")?, f("ln_f32_f16")?, f("ln_f16_f32")?, f("ln_f16_f16")?],
            bias_act: [f("bias_act_f32")?, f("bias_act_f16")?],
            rope: [f("rope_f32")?, f("rope_f16")?],
            attention: [
                f("attention_f32_f32")?,
                f("attention_f32_f16")?,
                f("attention_f16_f32")?,
                f("attention_f16_f16")?,
            ],
            geglu: [
                f("geglu_f32_f32")?,
                f("geglu_f32_f16")?,
                f("geglu_f16_f32")?,
                f("geglu_f16_f16")?,
            ],
            add_type: f("add_type")?,
            gather: [f("gather_f32")?, f("gather_f16")?],
            act_features: f("act_features")?,
            to_f16: f("to_f16")?,
        })
    }
}

/// How much of the graph runs in FP16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// FP32 values, weights and GEMMs. Matches Laya to about 1e-4 in the logits.
    #[default]
    F32,
    /// FP16 weights and GEMM inputs with FP32 accumulation, the residual stream, attention's qkv
    /// and the heads in FP32.
    F16,
}

/// One GPU, with its stream, kernels and cuBLASLt handle.
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    k: Kernels,
    lt: lt::Handle,
    workspace: CudaSlice<u8>,
    name: String,
    arch: (i32, i32),
    precision: Precision,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("name", &self.name)
            .field("arch", &self.arch)
            .field("precision", &self.precision)
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
    /// Opens GPU `ordinal` and compiles the kernels for it.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when there is no such GPU, no driver, or the kernels do not compile.
    pub fn new(ordinal: usize, precision: Precision) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).map_err(dev)?;
        // SAFETY: all work goes to one stream, in order, and the plan synchronizes that stream
        // before it reads results or frees buffers, so cudarc's per buffer events are not needed.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(dev)?;
        let arch = ctx.compute_capability().map_err(dev)?;
        let opts = CompileOptions { arch: Some(arch_flag(arch)), ..Default::default() };
        let ptx = compile_ptx_with_opts(SOURCE, opts).map_err(dev)?;
        let module = ctx.load_module(ptx).map_err(dev)?;
        let k = Kernels::load(&module)?;
        let lt = lt::Handle::new()?;
        let workspace = stream.alloc_zeros::<u8>(WORKSPACE).map_err(dev)?;
        let name = ctx.name().map_err(dev)?;
        Ok(Self { stream, k, lt, workspace, name, arch, precision })
    }

    /// The GPU's name, as the driver reports it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What runs in FP16.
    #[must_use]
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Compute capability, major and minor.
    #[must_use]
    pub fn arch(&self) -> (i32, i32) {
        self.arch
    }

    fn workspace_ptr(&self) -> u64 {
        self.workspace.device_ptr(&self.stream).0
    }
}

/// The virtual architecture NVRTC compiles for. The driver JITs the PTX for the actual device,
/// so a newer GPU than any listed gets the newest one here.
fn arch_flag((major, minor): (i32, i32)) -> &'static str {
    match (major, minor) {
        (..=7, _) => "compute_75",
        (8, 0) => "compute_80",
        (8, 6..=8) => "compute_86",
        (8, _) => "compute_89",
        _ => "compute_90",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_flags() {
        assert_eq!(arch_flag((7, 5)), "compute_75");
        assert_eq!(arch_flag((8, 6)), "compute_86");
        assert_eq!(arch_flag((8, 9)), "compute_89");
        assert_eq!(arch_flag((12, 0)), "compute_90");
    }
}

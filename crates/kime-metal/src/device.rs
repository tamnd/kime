//! The device, its queue, the compiled kernels and shared buffers.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLCompileOptions, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions,
};

use kime_tensor::{Error, Result};

const SOURCE: &str = include_str!("../kernels/compat.metal");

pub(crate) fn dev(e: impl std::fmt::Display) -> Error {
    Error::Device(e.to_string())
}

/// The element type of weights and activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// FP32 throughout.
    F32,
    /// FP16 storage with FP32 accumulation, and FP32 where the CUDA backend keeps it.
    F16,
}

/// A buffer in memory the CPU and the GPU share.
#[derive(Clone)]
pub(crate) struct Buffer(pub(crate) Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: Metal buffers may be used from any thread; Apple's guide lists them with devices and
// queues as the thread safe objects. What is in them is only touched through `&mut` of the plan or
// the weights that own them, or by the GPU while the thread that committed the work waits.
unsafe impl Send for Buffer {}
// SAFETY: as for Send.
unsafe impl Sync for Buffer {}

impl Buffer {
    pub(crate) fn ptr(&self) -> *mut u8 {
        self.0.contents().as_ptr().cast()
    }

    pub(crate) fn len(&self) -> usize {
        self.0.length()
    }
}

pub(crate) type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The kernels of `kernels/compat.metal`.
pub(crate) struct Kernels {
    pub(crate) embed: Pipeline,
    /// Indexed by `2 * (input is f16) + (output is f16)`.
    pub(crate) ln: [Pipeline; 4],
    /// A, W and C: f32 f32 f32, then FP16 weights with f32 f16, f16 f32 and f16 f16.
    pub(crate) gemm: [Pipeline; 4],
    /// f32, f16.
    pub(crate) rope: [Pipeline; 2],
    /// Indexed like `ln`.
    pub(crate) attention: [Pipeline; 4],
    /// Indexed like `ln`.
    pub(crate) geglu: [Pipeline; 4],
    pub(crate) add_type: Pipeline,
    /// f32, f16.
    pub(crate) gather: [Pipeline; 2],
    pub(crate) act_features: Pipeline,
    pub(crate) mean_pool: Pipeline,
}

/// The Apple GPU.
pub struct MetalBackend {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) k: Kernels,
    pub(crate) precision: Precision,
    name: String,
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device", &self.name)
            .field("precision", &self.precision)
            .finish_non_exhaustive()
    }
}

impl MetalBackend {
    /// Opens the system's default GPU and compiles the kernels for it.
    ///
    /// # Errors
    ///
    /// [`Error::Device`] when there is no GPU or the kernels do not compile.
    pub fn new(precision: Precision) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| dev("no Metal device"))?;
        let queue = device.newCommandQueue().ok_or_else(|| dev("no command queue"))?;
        let opts = MTLCompileOptions::new();
        // Safe math, so that exp, log and sqrt are the precise ones unless a kernel asks for the
        // fast ones by name.
        opts.setMathMode(MTLMathMode::Safe);
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&opts))
            .map_err(|e| dev(format!("kernels: {}", e.localizedDescription())))?;
        let f = |name: &str| -> Result<Pipeline> {
            let func = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| dev(format!("no kernel {name}")))?;
            device
                .newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| dev(format!("{name}: {}", e.localizedDescription())))
        };
        let k = Kernels {
            embed: f("embed")?,
            ln: [f("ln_f32_f32")?, f("ln_f32_f16")?, f("ln_f16_f32")?, f("ln_f16_f16")?],
            gemm: [
                f("gemm_f32_f32_f32")?,
                f("gemm_f32_f16_f16")?,
                f("gemm_f16_f16_f32")?,
                f("gemm_f16_f16_f16")?,
            ],
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
            mean_pool: f("mean_pool")?,
        };
        let name = device.name().to_string();
        Ok(Self { device, queue, k, precision, name })
    }

    /// The GPU's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// A zeroed shared buffer of `len` bytes, at least one.
    pub(crate) fn buffer(&self, len: usize) -> Result<Buffer> {
        let b = self
            .device
            .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| dev(format!("cannot allocate {len} bytes")))?;
        Ok(Buffer(b))
    }

    /// A shared buffer holding `data`.
    pub(crate) fn buffer_from<T: Copy>(&self, data: &[T]) -> Result<Buffer> {
        let b = self.buffer(size_of_val(data))?;
        // SAFETY: the buffer was just made with room for `data`, and nothing else can see it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), b.ptr(), size_of_val(data));
        }
        Ok(b)
    }
}

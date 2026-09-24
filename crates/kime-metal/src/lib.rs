//! The Apple GPU backend, from spec/09-apple.md. macOS only; on other targets the crate is empty.
//!
//! This first version runs the compat graph with the kernels in `kernels/compat.metal`, compiled
//! from source when the backend starts, so the build needs no Xcode. The GEMMs run on simdgroup
//! matrices with FP32 accumulation and fuse the bias, the activation and the residual add into
//! their store. Weights and activations are FP32, or FP16 with the same placement as the CUDA
//! backend. Buffers live in shared memory, so the batch's index tables are written in place and the
//! outputs are read in place, with no copies.
//!
//! Every dispatch is sized for the plan's bucket and every kernel reads the batch's real counts
//! from a buffer, so a run is a fixed list of dispatches. They are encoded into one command buffer
//! per run for now; spec/09-apple.md wants them in an indirect command buffer, which comes next.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

#[cfg(target_os = "macos")]
mod compat;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod plan;

#[cfg(target_os = "macos")]
pub use compat::executor;
#[cfg(target_os = "macos")]
pub use device::{MetalBackend, Precision};
#[cfg(target_os = "macos")]
pub use plan::{MetalPlan, Weights};

//! The NVIDIA backend. Twelve kernels, CUDA graphs per shape bucket, and cuBLASLt where it wins. The toolkit dependency sits behind a `cuda` feature once it arrives, so that a machine without the toolkit still builds the workspace. See spec/08-cuda.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

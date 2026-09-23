//! The CPU backend. FP32 reference kernels first, because every other backend is tested against them, then INT8 kernels for AVX2, AVX-512 VNNI, AMX and NEON i8mm. See spec/10-cpu.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

pub mod attention;
pub mod compat;
pub mod gemm;
pub mod ops;
pub mod par;
#[cfg(test)]
mod testing;

pub use compat::{Compat, Input, Output};

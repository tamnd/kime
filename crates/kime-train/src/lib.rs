//! Training, distillation and calibration fitting. Outside the inference graph on purpose, so nothing it pulls in reaches the server binary. See spec/12-training.md.
//!
//! [`model`] is the compat graph as a Burn module, and [`data`] reads the training shards. The backend is picked at build time: CUDA with
//! the `cuda` feature, Metal with `metal`, and the CPU otherwise.

#![forbid(unsafe_code)]

pub mod data;
pub mod export;
pub mod loss;
pub mod model;
pub mod rng;
pub mod train;

/// The backend forward passes run on.
#[cfg(feature = "cuda")]
pub type Base = burn::backend::Cuda;
/// The backend forward passes run on.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub type Base = burn::backend::Metal;
/// The backend forward passes run on.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub type Base = burn::backend::NdArray;

/// [`Base`] with gradients.
pub type Train = burn::backend::Autodiff<Base>;

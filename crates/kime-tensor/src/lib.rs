//! Device neutral tensor handles, dtypes, arenas, the `Backend` trait and the plan executor. A plan is a flat precomputed list of ops for one model at one shape bucket, and the hot path through it does no allocation, no locking and no dynamic dispatch per op. See spec/07-engine.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

pub mod backend;
pub mod blob;
pub mod bucket;
pub mod dtype;
pub mod plan;

pub use backend::{Backend, Batch, BatchBuf, Caps, Error, Executor, HostTensor, Outputs, Result};
pub use blob::Blob;
pub use bucket::{Bucket, Buckets};
pub use dtype::DType;
pub use plan::{Epilogue, Graph, Layout, Op, Rows, Shape, Val, W, layout};

//! ModelSpec, checkpoint formats, weight loading, the `.kime` packed format and the graph builders for both model families: the laya compat graph and the kime-v1 split encoder. See spec/05-model.md and spec/07-engine.md.
//!
//! Every file is untrusted input. The parsers take a `&[u8]` and check each offset and length
//! before using it, so a malformed or hostile file is an error and never a read out of bounds.

#![forbid(unsafe_code)]

mod error;
pub mod laya;
mod model;
pub mod pack;
pub mod safetensors;
mod tensors;

pub use error::Error;
pub use model::{LAYA_FILES, Model};
pub use tensors::{Entry, Tensors, View};

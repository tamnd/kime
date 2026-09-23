//! The public crate. It re-exports the in process API from `kime-engine` and the request and answer types from `kime-core`, so a Rust program depends on this one crate and nothing else. See spec/07-engine.md and spec/14-sdks-cli.md.

#![forbid(unsafe_code)]

pub use kime_core::*;

//! The public crate. It re-exports the in process API from `kime-engine` and the request and answer types from `kime-core`, so a Rust program depends on this one crate and nothing else. See spec/07-engine.md and spec/14-sdks-cli.md.
//!
//! ```no_run
//! use kime::{Device, Kime, Request, Response, State};
//! # fn main() -> Result<(), kime::Error> {
//! # let value = serde_json::json!({"customer": "I was charged twice"});
//! # let (req1, req2) = (Request::new("a"), Request::new("b"));
//! let kime = Kime::builder()
//!     .model("laya")                  // alias, local path or hf:// ref
//!     .device(Device::Auto)           // Cuda(0), Cpu { threads }, Auto
//!     .preload(true)
//!     .build()?;
//!
//! let req = Request::new(State::json(&value))
//!     .choice("department", "Which team should handle this", [("billing", "Payment issues"), ("technical", "Bugs")])
//!     .score("urgency", "How urgent", ["not urgent", "soon", "critical"])
//!     .noul("churn", "The customer threatens to leave");
//!
//! let res: Response = kime.decide(&req)?;          // blocking
//! let many = kime.decide_batch(&[req1, req2])?;    // batch
//! # Ok(()) }
//! # async fn later(kime: Kime, req: Request) -> Result<(), kime::Error> {
//! let res = kime.decide_async(&req).await?;        // async, any executor
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]

pub use kime_core::answer::{Answer, Response};
pub use kime_core::request::{Request, State};
pub use kime_core::*;
pub use kime_engine::{Builder, Decision, Device, Error, Kime, Precision, hub};

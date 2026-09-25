//! Request and answer types, validation, state rendering, confidence formulas, rounding,
//! calibration, email cleaning, presets and browser agent steps.
//!
//! Everything here is pure computation on the host with no device and no I/O, which is why it is
//! the crate every other one can depend on. See spec/03-api.md, spec/04-semantics.md and
//! spec/06-input.md.

#![forbid(unsafe_code)]

/// The date this version of kime was released, which `/v1/models` gives as every model's
/// `release_date`. Jev's clients want a string there, and a checkpoint carries no date of its own.
pub const RELEASE_DATE: &str = "2026-09-25";

pub mod agent;
pub mod answer;
pub mod confidence;
pub mod email;
pub mod presets;
pub mod pyjson;
pub mod render;
pub mod request;
pub mod round;

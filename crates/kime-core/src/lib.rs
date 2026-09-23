//! Request and answer types, validation, state rendering, confidence formulas, rounding,
//! calibration, email cleaning and presets.
//!
//! Everything here is pure computation on the host with no device and no I/O, which is why it is
//! the crate every other one can depend on. See spec/03-api.md, spec/04-semantics.md and
//! spec/06-input.md.

#![forbid(unsafe_code)]

pub mod answer;
pub mod confidence;
pub mod pyjson;
pub mod render;
pub mod request;
pub mod round;

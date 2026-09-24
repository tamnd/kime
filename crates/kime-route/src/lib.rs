//! Script and language detection and the Router, which picks the English or the multilingual model for a request with Laya's rules plus a real language identifier. See spec/11-serving.md.

#![forbid(unsafe_code)]

pub mod lang;
pub mod lid;
pub mod router;

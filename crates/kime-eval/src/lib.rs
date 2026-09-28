//! Datasets, metrics, benchmark runners and the scorecard. The workloads and targets are in spec/13-benchmarks.md, and the comparisons against other engines on other machines live in tamnd/kime-bench.
//!
//! [`suite`] reads quality suites and turns answers into rows, [`metrics`] computes the numbers
//! over rows, and [`report`] writes them out. `kime eval` in kime-cli runs them.

#![forbid(unsafe_code)]

pub mod metrics;
pub mod report;
pub mod suite;

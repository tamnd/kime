//! Device neutral tensor handles, dtypes, arenas, the `Backend` trait and the plan executor. A plan is a flat precomputed list of ops for one model at one shape bucket, and the hot path through it does no allocation, no locking and no dynamic dispatch per op. See spec/07-engine.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

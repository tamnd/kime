//! The Apple Neural Engine backend. Models are exported to MIL at fixed shape buckets and run through Core ML, with a placement check at load so that a silent fallback to the CPU is an error rather than a slow afternoon. macOS only; on other targets the crate is empty. See spec/09-apple.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

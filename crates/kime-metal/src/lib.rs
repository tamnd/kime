//! The Apple GPU backend. Kernels in Metal Shading Language, encoded once per bucket into indirect command buffers so that a request costs one commit. macOS only; on other targets the crate is empty. See spec/09-apple.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

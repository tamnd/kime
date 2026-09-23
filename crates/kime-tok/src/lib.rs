//! The two tokenizer families kime needs: byte level BPE for the ModernBERT vocabulary and SentencePiece style BPE with byte fallback for the Gemma 2 vocabulary. Ids must match Hugging Face tokenizers exactly, which is tested on every commit on a sample and nightly on ten million lines. See spec/06-input.md.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

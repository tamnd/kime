//! Read only bytes that are either a mapped file or owned memory.
//!
//! Weights are mapped rather than read (spec/07-engine.md), so opening a checkpoint costs page
//! table entries and not a copy. Mapping needs `unsafe`, which is why this lives here and not in
//! `kime-model`: the parsers there only ever see a `&[u8]`.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;

/// Bytes a checkpoint is parsed from.
#[derive(Debug)]
pub struct Blob(Inner);

#[derive(Debug)]
enum Inner {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Blob {
    /// Maps `path` read only. Empty files are returned as owned, because mapping zero bytes fails
    /// on some platforms.
    ///
    /// # Errors
    ///
    /// Any error from opening or mapping the file.
    pub fn map(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() == 0 {
            return Ok(Self(Inner::Owned(Vec::new())));
        }
        // SAFETY: the mapping is read only and kime never writes through it. The remaining hazard
        // is another process truncating or rewriting the file while it is mapped, which the OS
        // turns into a signal or changed bytes rather than a Rust level data race on our side.
        // Every loader treats the bytes as untrusted and checks bounds before each read, so changed
        // bytes can give wrong weights but not an out of bounds access. This is the same contract
        // safetensors, candle and llama.cpp use for weight files.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self(Inner::Mapped(map)))
    }

    /// Wraps bytes already in memory, for tests and for files that arrive over the network.
    #[must_use]
    pub fn owned(bytes: Vec<u8>) -> Self {
        Self(Inner::Owned(bytes))
    }

    /// Whether the bytes come from a mapping.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        matches!(self.0, Inner::Mapped(_))
    }
}

impl Deref for Blob {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.0 {
            Inner::Mapped(m) => m,
            Inner::Owned(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_a_file() {
        let path = std::env::temp_dir().join(format!("kime-blob-{}", std::process::id()));
        std::fs::write(&path, b"hello").unwrap();
        let b = Blob::map(&path).unwrap();
        assert!(b.is_mapped());
        assert_eq!(&*b, b"hello");
        drop(b);
        std::fs::write(&path, b"").unwrap();
        assert_eq!(Blob::map(&path).unwrap().len(), 0);
        std::fs::remove_file(path).unwrap();
    }
}

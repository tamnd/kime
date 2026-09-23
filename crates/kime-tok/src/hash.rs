//! A small multiplicative hasher for integer and short string keys. The merge table is looked up once per candidate
//! pair on the hot path, and SipHash costs more there than the rest of the lookup.

use std::hash::{BuildHasherDefault, Hasher};

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    // The multiply leaves the best mixed bits at the top, and the map indexes with the bottom
    // ones, which would otherwise depend only on the first bytes of a key. rustc-hash rotates for
    // the same reason.
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    // Strings are hashed eight bytes at a time. The vocabulary map is keyed by token text, and a
    // multiply per byte made building a 256k token vocabulary noticeably slow.
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, tail) = bytes.as_chunks::<8>();
        for c in chunks {
            self.write_u64(u64::from_le_bytes(*c));
        }
        let mut last = [0u8; 8];
        last[..tail.len()].copy_from_slice(tail);
        self.write_u64(u64::from_le_bytes(last) ^ ((tail.len() as u64) << 59));
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

pub(crate) type FxBuild = BuildHasherDefault<Fx>;
pub(crate) type FxMap<K, V> = std::collections::HashMap<K, V, FxBuild>;

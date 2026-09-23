//! A small multiplicative hasher for integer keys. The merge table is looked up once per candidate
//! pair on the hot path, and SipHash costs more there than the rest of the lookup.

use std::hash::{BuildHasherDefault, Hasher};

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
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

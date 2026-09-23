//! Element types as they appear in checkpoints.

use std::fmt;

/// The element types kime reads from checkpoints. Weights are floats. The integer types exist
/// because safetensors files may carry them, and a loader that knows their size can check a file
/// it will not use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE binary32.
    F32,
    /// IEEE binary16.
    F16,
    /// bfloat16.
    BF16,
    /// Signed 64 bit integers.
    I64,
    /// Signed 32 bit integers.
    I32,
    /// Signed 8 bit integers.
    I8,
    /// Unsigned 8 bit integers.
    U8,
    /// Booleans, one byte each.
    Bool,
}

impl DType {
    /// Bytes per element.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// The name safetensors uses, which kime also uses in `.kime` headers.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
        }
    }

    /// The inverse of [`DType::name`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            _ => return None,
        })
    }

    /// Whether this is one of the float types weights come in.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Reads element `i` of little endian `bytes` as f32. Panics if `bytes` is too short, and
    /// returns NaN for the integer types, which no weight uses.
    #[must_use]
    pub fn read_f32(self, bytes: &[u8], i: usize) -> f32 {
        let s = self.size();
        let b = &bytes[i * s..(i + 1) * s];
        match self {
            Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Self::F16 => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
            Self::BF16 => half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
            _ => f32::NAN,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for d in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I8,
            DType::U8,
            DType::Bool,
        ] {
            assert_eq!(DType::from_name(d.name()), Some(d));
        }
        assert_eq!(DType::from_name("F64"), None);
    }

    #[test]
    fn reads_halves() {
        let one = half::f16::from_f32(1.5).to_le_bytes();
        assert_eq!(DType::F16.read_f32(&one, 0), 1.5);
        let two = half::bf16::from_f32(-2.0).to_le_bytes();
        assert_eq!(DType::BF16.read_f32(&two, 0), -2.0);
    }
}

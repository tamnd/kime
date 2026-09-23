//! A set of named tensors over one blob of bytes, whichever format they came from.

use std::collections::HashMap;

use kime_tensor::{Blob, DType};

use crate::error::{Error, Result};

/// Where one tensor lives in its blob. Offsets are absolute and have been checked against the blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The tensor's name.
    pub name: String,
    /// Its element type.
    pub dtype: DType,
    /// Its shape, outermost first.
    pub shape: Vec<usize>,
    /// First byte.
    pub start: usize,
    /// One past the last byte.
    pub end: usize,
}

impl Entry {
    /// The number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A borrowed tensor.
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    /// The tensor's name.
    pub name: &'a str,
    /// Its element type.
    pub dtype: DType,
    /// Its shape.
    pub shape: &'a [usize],
    /// Its bytes, little endian, row major.
    pub bytes: &'a [u8],
}

impl View<'_> {
    /// The elements as f32, converting from f16 or bf16.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        (0..self.bytes.len() / self.dtype.size())
            .map(|i| self.dtype.read_f32(self.bytes, i))
            .collect()
    }
}

/// Named tensors backed by one blob, in the order their file lists them.
#[derive(Debug)]
pub struct Tensors {
    blob: Blob,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
}

impl Tensors {
    /// Builds the set from entries a parser has already checked against `blob`.
    pub(crate) fn new(blob: Blob, entries: Vec<Entry>) -> Result<Self> {
        let mut by_name = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            debug_assert!(e.start <= e.end && e.end <= blob.len());
            if by_name.insert(e.name.clone(), i).is_some() {
                return Err(Error::format(format!("tensor {:?} appears twice", e.name)));
            }
        }
        Ok(Self { blob, entries, by_name })
    }

    /// All entries, in file order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The index of a tensor by name.
    #[must_use]
    pub fn index(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// The tensor at `i`, in file order. Panics if `i` is out of range.
    #[must_use]
    pub fn view(&self, i: usize) -> View<'_> {
        let e = &self.entries[i];
        View { name: &e.name, dtype: e.dtype, shape: &e.shape, bytes: &self.blob[e.start..e.end] }
    }

    /// A tensor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<View<'_>> {
        self.index(name).map(|i| self.view(i))
    }

    /// The bytes behind every tensor.
    #[must_use]
    pub fn blob(&self) -> &Blob {
        &self.blob
    }

    /// Bytes of tensor data in total.
    #[must_use]
    pub fn data_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.end - e.start).sum()
    }
}

/// The byte length of a tensor, or None when the shape overflows.
pub(crate) fn byte_len(dtype: DType, shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(dtype.size(), |acc, &d| acc.checked_mul(d))
}

/// Checks that no two ranges overlap. `ranges` holds (start, end, what) and is sorted in place.
pub(crate) fn check_disjoint(ranges: &mut [(usize, usize, &str)]) -> Result<()> {
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        if w[1].0 < w[0].1 {
            return Err(Error::format(format!("{:?} overlaps {:?}", w[1].2, w[0].2)));
        }
    }
    Ok(())
}

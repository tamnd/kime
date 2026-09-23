//! The `.kime` packed format: one file holding a model's tensors and config files, laid out so the
//! tensors can be mapped and used in place.
//!
//! ```text
//! 0      4 KiB header block
//!          0..8    magic "KIME\x01\0\0\0"
//!          8..12   format version, u32 LE, 1
//!          12..16  zero
//!          16..24  JSON length, u64 LE
//!          24..32  data start, u64 LE, a multiple of 4 KiB
//!          32..40  data length, u64 LE, running to the end of the file
//!          40..72  blake3 of the JSON
//!          72..    zero
//! 4096   JSON: format, family, id, hash, tensors, files
//! data   tensors, then files, each starting on a 4 KiB boundary, offsets relative to data start
//! ```
//!
//! `hash` in the JSON is the blake3 of the whole data section and is the model hash kime reports.
//! The JSON's own hash sits in the header block, so a reader can trust the index before touching
//! the data, and checking the model hash is a separate, optional pass at memory speed.
//!
//! Every number is checked before use, as for safetensors: a header that points outside the file
//! is an error and never a read.

use std::io::Write;

use kime_tensor::{Blob, DType};
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::safetensors::MAX_HEADER;
use crate::tensors::{Entry, Tensors, byte_len, check_disjoint};

/// The first eight bytes of every `.kime` file.
pub const MAGIC: [u8; 8] = *b"KIME\x01\0\0\0";
/// The format version this build reads and writes.
pub const VERSION: u32 = 1;
/// Alignment of the header block, the data section and every tensor.
pub const ALIGN: usize = 4096;
const FIXED: usize = 72;

/// A file carried inside a `.kime`, such as a config or the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// Path relative to the model directory, with forward slashes.
    pub name: String,
    /// First byte, absolute.
    pub start: usize,
    /// One past the last byte, absolute.
    pub end: usize,
}

/// A parsed and checked `.kime` index.
#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    /// The whole JSON, for fields this module does not interpret.
    pub json: Map<String, Value>,
    /// The model family, `laya` for now.
    pub family: String,
    /// The model id.
    pub id: String,
    /// blake3 of the data section, as 64 hex digits.
    pub hash: String,
    /// Tensors, in their original order, with absolute offsets.
    pub tensors: Vec<Entry>,
    /// Carried files, with absolute offsets.
    pub files: Vec<FileEntry>,
    /// Where the data section starts.
    pub data_start: usize,
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

/// Whether `bytes` starts like a `.kime` file.
#[must_use]
pub fn is_kime(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

/// Parses and checks the index of a `.kime` file. The data section is not hashed here, see
/// [`verify`].
///
/// # Errors
///
/// [`Error::Format`] naming the first problem found.
///
/// # Panics
///
/// Never. Fixed header fields are read only after the length check.
pub fn parse(bytes: &[u8]) -> Result<Index> {
    let bad = |msg: String| Error::Format(format!(".kime: {msg}"));
    if bytes.len() < ALIGN {
        return Err(bad(format!("{} bytes is shorter than the header block", bytes.len())));
    }
    if !is_kime(bytes) {
        return Err(bad("wrong magic".into()));
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version != VERSION {
        return Err(bad(format!("format version {version}, this build reads {VERSION}")));
    }
    if bytes[12..16].iter().chain(&bytes[FIXED..ALIGN]).any(|&b| b != 0) {
        return Err(bad("reserved header bytes are not zero".into()));
    }
    let (json_len, data_start, data_len) =
        (u64_at(bytes, 16), u64_at(bytes, 24), u64_at(bytes, 32));
    if json_len > MAX_HEADER as u64 {
        return Err(bad(format!("index of {json_len} bytes is over the limit")));
    }
    let json_end = ALIGN + json_len as usize;
    if !data_start.is_multiple_of(ALIGN as u64)
        || data_start < json_end as u64
        || data_start.checked_add(data_len) != Some(bytes.len() as u64)
    {
        return Err(bad(format!(
            "data section {data_start}+{data_len} does not follow the index and end the {} byte file",
            bytes.len()
        )));
    }
    let data_start = data_start as usize;
    let data_len = data_len as usize;
    let json_bytes = &bytes[ALIGN..json_end];
    if blake3::hash(json_bytes).as_bytes() != &bytes[40..72] {
        return Err(bad("index does not match its hash".into()));
    }
    let Ok(Value::Object(json)) = serde_json::from_slice::<Value>(json_bytes) else {
        return Err(bad("index is not a JSON object".into()));
    };
    let str_field = |k: &str| {
        json.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| bad(format!("index has no {k:?}")))
    };
    if str_field("format")? != "kime/1" {
        return Err(bad("index format is not kime/1".into()));
    }
    let hash = str_field("hash")?;
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(bad("hash is not 64 hex digits".into()));
    }
    let range = |item: &Map<String, Value>, what: &str| -> Result<(usize, usize)> {
        let num = |k: &str| {
            item.get(k)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| bad(format!("{what}: {k:?} is not a non negative integer")))
        };
        let (off, len) = (num("offset")?, num("len")?);
        match off.checked_add(len) {
            Some(end) if end <= data_len => Ok((data_start + off, data_start + end)),
            _ => Err(bad(format!("{what}: {off}+{len} falls outside {data_len} data bytes"))),
        }
    };
    let list = |k: &str| {
        json.get(k).and_then(Value::as_array).ok_or_else(|| bad(format!("index has no {k:?} list")))
    };
    let mut tensors = Vec::new();
    for t in list("tensors")? {
        let t = t.as_object().ok_or_else(|| bad("a tensor entry is not an object".into()))?;
        let name = t
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("a tensor has no name".into()))?
            .to_string();
        let what = format!("tensor {name:?}");
        let dtype = t
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(DType::from_name)
            .ok_or_else(|| bad(format!("{what}: missing or unknown dtype")))?;
        let shape = t
            .get("shape")
            .and_then(Value::as_array)
            .and_then(|a| {
                a.iter().map(|d| usize::try_from(d.as_u64()?).ok()).collect::<Option<Vec<_>>>()
            })
            .ok_or_else(|| bad(format!("{what}: shape must be a list of non negative integers")))?;
        let (start, end) = range(t, &what)?;
        if !(start - data_start).is_multiple_of(ALIGN) {
            return Err(bad(format!("{what}: offset is not aligned to {ALIGN}")));
        }
        let want =
            byte_len(dtype, &shape).ok_or_else(|| bad(format!("{what}: shape overflows")))?;
        if end - start != want {
            return Err(bad(format!(
                "{what}: holds {} bytes but {dtype} {shape:?} needs {want}",
                end - start
            )));
        }
        tensors.push(Entry { name, dtype, shape, start, end });
    }
    let mut files = Vec::new();
    for f in list("files")? {
        let f = f.as_object().ok_or_else(|| bad("a file entry is not an object".into()))?;
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| safe_name(n))
            .ok_or_else(|| bad("a file has no name or an unsafe one".into()))?
            .to_string();
        let (start, end) = range(f, &format!("file {name:?}"))?;
        files.push(FileEntry { name, start, end });
    }
    let mut ranges: Vec<_> = tensors
        .iter()
        .map(|e| (e.start, e.end, e.name.as_str()))
        .chain(files.iter().map(|f| (f.start, f.end, f.name.as_str())))
        .collect();
    check_disjoint(&mut ranges).map_err(|e| bad(e.to_string()))?;
    Ok(Index {
        family: str_field("family")?,
        id: str_field("id")?,
        hash,
        tensors,
        files,
        data_start,
        json,
    })
}

/// A carried file name must stay inside the directory it is unpacked to.
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && name.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

/// Hashes the data section and compares it with the index. This reads every byte, on all cores, so
/// it runs at memory or disk speed.
///
/// # Errors
///
/// [`Error::Format`] if the data does not match.
pub fn verify(bytes: &[u8], index: &Index) -> Result<()> {
    let got = blake3::Hasher::new().update_rayon(&bytes[index.data_start..]).finalize().to_hex();
    if got.as_str() == index.hash {
        Ok(())
    } else {
        Err(Error::Format(format!(".kime: data hash is {got}, the index says {}", index.hash)))
    }
}

/// What to pack.
#[derive(Debug)]
pub struct Contents<'a> {
    /// Model family.
    pub family: &'a str,
    /// Model id.
    pub id: &'a str,
    /// The tensors, packed in their order.
    pub tensors: &'a Tensors,
    /// Files to carry, by relative name.
    pub files: Vec<(&'a str, &'a [u8])>,
    /// Extra index fields, such as the source safetensors metadata.
    pub extra: Map<String, Value>,
}

/// Writes a `.kime` file and returns its model hash.
///
/// # Errors
///
/// Any error from `out`, or [`Error::Format`] for a file name that could escape its directory.
///
/// # Panics
///
/// Never. The index holds only strings and numbers, which always serialize.
pub fn write(c: &Contents<'_>, out: &mut impl Write) -> Result<String> {
    let io = |e: std::io::Error| Error::Io("<output>".into(), e);
    for (name, _) in &c.files {
        if !safe_name(name) {
            return Err(Error::Format(format!("refusing to pack file name {name:?}")));
        }
    }
    // Lay out the data section, then hash it by streaming the same bytes the writer will write.
    let pieces: Vec<&[u8]> = (0..c.tensors.entries().len())
        .map(|i| c.tensors.view(i).bytes)
        .chain(c.files.iter().map(|(_, b)| *b))
        .collect();
    let mut offsets = Vec::with_capacity(pieces.len());
    let mut at = 0usize;
    for p in &pieces {
        offsets.push(at);
        at = (at + p.len()).next_multiple_of(ALIGN);
    }
    let data_len = at;
    let zeros = [0u8; ALIGN];
    let mut hasher = blake3::Hasher::new();
    let mut written = 0;
    for (p, &off) in pieces.iter().zip(&offsets) {
        hasher.update(&zeros[..off - written]);
        hasher.update(p);
        written = off + p.len();
    }
    hasher.update(&zeros[..data_len - written]);
    let hash = hasher.finalize().to_hex().to_string();

    let n = c.tensors.entries().len();
    let tensors: Vec<Value> = c
        .tensors
        .entries()
        .iter()
        .zip(&offsets)
        .map(|(e, &off)| {
            json!({"name": e.name, "dtype": e.dtype.name(), "shape": e.shape, "offset": off, "len": e.end - e.start})
        })
        .collect();
    let files: Vec<Value> = c
        .files
        .iter()
        .zip(&offsets[n..])
        .map(|((name, b), &off)| json!({"name": name, "offset": off, "len": b.len()}))
        .collect();
    let mut index = Map::new();
    index.insert("format".into(), "kime/1".into());
    index.insert("family".into(), c.family.into());
    index.insert("id".into(), c.id.into());
    index.insert("hash".into(), hash.clone().into());
    for (k, v) in &c.extra {
        index.insert(k.clone(), v.clone());
    }
    index.insert("tensors".into(), tensors.into());
    index.insert("files".into(), files.into());
    let json = serde_json::to_vec(&Value::Object(index)).expect("plain values");
    let data_start = (ALIGN + json.len()).next_multiple_of(ALIGN);

    let mut head = vec![0u8; ALIGN];
    head[..8].copy_from_slice(&MAGIC);
    head[8..12].copy_from_slice(&VERSION.to_le_bytes());
    head[16..24].copy_from_slice(&(json.len() as u64).to_le_bytes());
    head[24..32].copy_from_slice(&(data_start as u64).to_le_bytes());
    head[32..40].copy_from_slice(&(data_len as u64).to_le_bytes());
    head[40..72].copy_from_slice(blake3::hash(&json).as_bytes());
    out.write_all(&head).map_err(io)?;
    out.write_all(&json).map_err(io)?;
    out.write_all(&zeros[..data_start - ALIGN - json.len()]).map_err(io)?;
    let mut written = 0;
    for (p, &off) in pieces.iter().zip(&offsets) {
        out.write_all(&zeros[..off - written]).map_err(io)?;
        out.write_all(p).map_err(io)?;
        written = off + p.len();
    }
    out.write_all(&zeros[..data_len - written]).map_err(io)?;
    Ok(hash)
}

/// Opens the tensors of a `.kime` file held in `blob`.
///
/// # Errors
///
/// [`Error::Format`] if the index is invalid.
pub fn load(blob: Blob) -> Result<(Index, Tensors)> {
    let index = parse(&blob)?;
    let tensors = Tensors::new(blob, index.tensors.clone())?;
    Ok((index, tensors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors;

    fn sample() -> Tensors {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F16","shape":[3],"data_offsets":[8,14]}}"#;
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        v.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
        safetensors::load(Blob::owned(v)).unwrap().0
    }

    fn pack(t: &Tensors) -> Vec<u8> {
        let c = Contents {
            family: "laya",
            id: "test",
            tensors: t,
            files: vec![("encoder/config.json", b"{}")],
            extra: Map::new(),
        };
        let mut out = Vec::new();
        write(&c, &mut out).unwrap();
        out
    }

    #[test]
    fn round_trip() {
        let t = sample();
        let bytes = pack(&t);
        assert_eq!(bytes.len() % ALIGN, 0);
        let index = parse(&bytes).unwrap();
        verify(&bytes, &index).unwrap();
        assert_eq!(index.files[0].name, "encoder/config.json");
        assert_eq!(&bytes[index.files[0].start..index.files[0].end], b"{}");
        let (_, t2) = load(Blob::owned(bytes)).unwrap();
        for (a, b) in t.entries().iter().zip(t2.entries()) {
            assert_eq!((&a.name, a.dtype, &a.shape), (&b.name, b.dtype, &b.shape));
            assert_eq!(t2.blob()[b.start..b.end], t.blob()[a.start..a.end]);
            assert_eq!(b.start % ALIGN, 0);
        }
    }

    #[test]
    fn rejects_damage() {
        let good = pack(&sample());
        let mut bad = good.clone();
        let n = bad.len();
        bad[n - 1] ^= 1;
        let index = parse(&bad).unwrap();
        assert!(verify(&bad, &index).unwrap_err().to_string().contains("data hash"));
        let mut bad = good.clone();
        bad[ALIGN + 3] ^= 1;
        assert!(parse(&bad).unwrap_err().to_string().contains("does not match its hash"));
        let mut bad = good.clone();
        bad[100] = 1;
        assert!(parse(&bad).unwrap_err().to_string().contains("reserved"));
        let mut bad = good.clone();
        bad[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(parse(&bad).unwrap_err().to_string().contains("data section"));
        assert!(parse(&good[..ALIGN - 1]).is_err());
        assert!(
            !safe_name("../x") && !safe_name("a//b") && !safe_name("/etc") && safe_name("a/b.json")
        );
    }
}

//! A safetensors reader that treats the file as hostile.
//!
//! The format is an 8 byte little endian header length, a JSON header mapping names to dtype,
//! shape and a byte range, then the data. Every number in the header is checked before it is used:
//! the header length against the file, each range against the data, each range length against its
//! dtype and shape (with overflow checks), and the ranges against each other. A file that fails any
//! check is an error, never a read out of bounds (SECURITY.md).

use kime_tensor::{Blob, DType};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::tensors::{Entry, Tensors, byte_len, check_disjoint};

/// The largest header we accept. The biggest real header we know of is a few MB, and the
/// safetensors reference implementation uses the same limit.
pub const MAX_HEADER: usize = 100 << 20;

/// What a safetensors file holds besides tensor data.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    /// The tensors, in header order, with absolute offsets.
    pub entries: Vec<Entry>,
    /// The optional `__metadata__` object.
    pub metadata: Option<Map<String, Value>>,
}

/// Parses and checks the header of a safetensors file held in `bytes`.
///
/// # Errors
///
/// [`Error::Format`] naming the first problem found.
pub fn parse(bytes: &[u8]) -> Result<Header> {
    let Some((len, rest)) = bytes.split_first_chunk::<8>() else {
        return Err(Error::format("safetensors: file is shorter than its 8 byte header length"));
    };
    let n = u64::from_le_bytes(*len);
    if n > MAX_HEADER as u64 {
        return Err(Error::format(format!("safetensors: header of {n} bytes is over the limit")));
    }
    let n = n as usize;
    if n > rest.len() {
        return Err(Error::format(format!(
            "safetensors: header of {n} bytes runs past the end of a {} byte file",
            bytes.len()
        )));
    }
    let data_start = 8 + n;
    let data_len = bytes.len() - data_start;
    let header: Value = serde_json::from_slice(&rest[..n])
        .map_err(|e| Error::format(format!("safetensors: header is not JSON: {e}")))?;
    let Value::Object(map) = header else {
        return Err(Error::format("safetensors: header is not a JSON object"));
    };

    let mut out = Header::default();
    for (name, v) in map {
        if name == "__metadata__" {
            match v {
                Value::Object(m) if m.values().all(Value::is_string) => out.metadata = Some(m),
                _ => {
                    return Err(Error::format(
                        "safetensors: __metadata__ must map strings to strings",
                    ));
                }
            }
            continue;
        }
        let bad = |why: &str| Error::format(format!("safetensors: tensor {name:?}: {why}"));
        let Value::Object(t) = v else { return Err(bad("entry is not an object")) };
        if t.len() != 3 {
            return Err(bad("entry must have exactly dtype, shape and data_offsets"));
        }
        let dtype = t
            .get("dtype")
            .and_then(Value::as_str)
            .and_then(DType::from_name)
            .ok_or_else(|| bad("missing or unknown dtype"))?;
        let shape = t
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("missing shape"))?
            .iter()
            .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| bad("shape must be a list of non negative integers"))?;
        let offsets = t
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .and_then(|a| {
                let a0 = usize::try_from(a[0].as_u64()?).ok()?;
                let a1 = usize::try_from(a[1].as_u64()?).ok()?;
                Some((a0, a1))
            })
            .ok_or_else(|| bad("data_offsets must be two non negative integers"))?;
        let (s, e) = offsets;
        if s > e || e > data_len {
            return Err(bad(&format!(
                "data_offsets [{s}, {e}] fall outside {data_len} data bytes"
            )));
        }
        let want = byte_len(dtype, &shape).ok_or_else(|| bad("shape overflows"))?;
        if e - s != want {
            return Err(bad(&format!("holds {} bytes but {dtype} {shape:?} needs {want}", e - s)));
        }
        out.entries.push(Entry { name, dtype, shape, start: data_start + s, end: data_start + e });
    }

    // The format also requires the ranges to tile the data with no gaps. The reference
    // implementation rejects gaps, so a file with gaps did not come from it.
    let mut ranges: Vec<_> =
        out.entries.iter().map(|e| (e.start, e.end, e.name.as_str())).collect();
    check_disjoint(&mut ranges)?;
    let mut at = data_start;
    for &(s, e, name) in &ranges {
        if s != at {
            return Err(Error::format(format!("safetensors: gap before tensor {name:?}")));
        }
        at = e;
    }
    if at != bytes.len() {
        return Err(Error::format(format!(
            "safetensors: {} bytes after the last tensor",
            bytes.len() - at
        )));
    }
    Ok(out)
}

/// Opens a safetensors file held in `blob`.
///
/// # Errors
///
/// [`Error::Format`] if the header is invalid.
pub fn load(blob: Blob) -> Result<(Tensors, Option<Map<String, Value>>)> {
    let header = parse(&blob)?;
    Ok((Tensors::new(blob, header.entries)?, header.metadata))
}

/// Writes tensors as a safetensors file, in the given order, the way the reference implementation
/// lays them out: compact JSON, padded with spaces to a multiple of 8, then the data in order.
///
/// # Errors
///
/// Any error from `out`.
///
/// # Panics
///
/// Never. The header holds only strings and numbers, which always serialize.
pub fn write(
    tensors: &Tensors,
    metadata: Option<&Map<String, Value>>,
    out: &mut impl std::io::Write,
) -> std::io::Result<()> {
    // Data goes in increasing offset order in the source, which keeps a round trip byte exact.
    let mut order: Vec<usize> = (0..tensors.entries().len()).collect();
    order.sort_by_key(|&i| tensors.entries()[i].start);
    let mut offset = vec![(0usize, 0usize); order.len()];
    let mut at = 0;
    for &i in &order {
        let e = &tensors.entries()[i];
        offset[i] = (at, at + (e.end - e.start));
        at += e.end - e.start;
    }
    let mut header = Map::new();
    if let Some(m) = metadata {
        header.insert("__metadata__".into(), Value::Object(m.clone()));
    }
    for (i, e) in tensors.entries().iter().enumerate() {
        let mut t = Map::new();
        t.insert("dtype".into(), e.dtype.name().into());
        t.insert("shape".into(), e.shape.clone().into());
        t.insert("data_offsets".into(), vec![offset[i].0, offset[i].1].into());
        header.insert(e.name.clone(), Value::Object(t));
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).expect("a map of plain values");
    json.resize(json.len().next_multiple_of(8), b' ');
    out.write_all(&(json.len() as u64).to_le_bytes())?;
    out.write_all(&json)?;
    for &i in &order {
        out.write_all(tensors.view(i).bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn reads_and_writes() {
        let bytes = file(
            r#"{"b":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0, 0, 128, 63, 0, 60, 0, 64],
        );
        let (t, meta) = load(Blob::owned(bytes)).unwrap();
        assert!(meta.is_none());
        assert_eq!(t.entries()[0].name, "b");
        assert_eq!(t.get("a").unwrap().to_f32(), [1.0]);
        assert_eq!(t.get("b").unwrap().to_f32(), [1.0, 2.0]);
        let mut out = Vec::new();
        write(&t, None, &mut out).unwrap();
        let (t2, _) = load(Blob::owned(out)).unwrap();
        assert_eq!(t2.entries().len(), 2);
        assert_eq!(t2.get("b").unwrap().bytes, t.get("b").unwrap().bytes);
    }

    #[test]
    fn rejects_bad_headers() {
        let cases = [
            (vec![1, 2, 3], "shorter"),
            (u64::MAX.to_le_bytes().to_vec(), "over the limit"),
            (file("{}", b"")[..9].to_vec(), "runs past"),
            (file("[]", b""), "not a JSON object"),
            (file(r#"{"a":{"dtype":"F99","shape":[1],"data_offsets":[0,4]}}"#, &[0; 4]), "dtype"),
            (file(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#, &[0; 4]), "outside"),
            (file(r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#, &[0; 4]), "needs 8"),
            (
                file(
                    r#"{"a":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
                    b"",
                ),
                "overflows",
            ),
            (
                file(
                    r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
                    &[0; 3],
                ),
                "overlaps",
            ),
            (file(r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#, &[0; 3]), "gap"),
            (file(r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#, &[0; 3]), "after"),
            (file(r#"{"__metadata__":{"k":1}}"#, b""), "__metadata__"),
        ];
        for (bytes, want) in cases {
            let err = parse(&bytes).unwrap_err().to_string();
            assert!(err.contains(want), "{want:?} not in {err:?}");
        }
    }
}

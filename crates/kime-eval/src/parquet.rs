//! A small Parquet writer for the per question rows of a run, so `kime eval` can write the
//! Parquet file spec/13-benchmarks.md asks for without the arrow crates.
//!
//! It writes one row group with one uncompressed, plain encoded data page per column, which any
//! Parquet reader accepts. Strings, integers, doubles, booleans and lists of doubles are enough
//! for the rows. Lists use the two level layout, `required group (LIST) { repeated double }`.

/// A column of values, all of one type.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// UTF-8 strings.
    Str(Vec<String>),
    /// 64 bit integers.
    I64(Vec<i64>),
    /// Doubles.
    F64(Vec<f64>),
    /// Booleans.
    Bool(Vec<bool>),
    /// A list of doubles a row.
    F64List(Vec<Vec<f64>>),
}

impl Column {
    fn rows(&self) -> usize {
        match self {
            Self::Str(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::F64List(v) => v.len(),
        }
    }

    /// The physical type: INT64 2, DOUBLE 5, BOOLEAN 0, BYTE_ARRAY 6.
    fn physical(&self) -> i32 {
        match self {
            Self::Str(_) => 6,
            Self::I64(_) => 2,
            Self::F64(_) | Self::F64List(_) => 5,
            Self::Bool(_) => 0,
        }
    }
}

/// The Thrift compact protocol, as much of it as the Parquet footer and page headers need.
#[derive(Default)]
struct Compact {
    out: Vec<u8>,
    last: Vec<i16>,
}

const I32: u8 = 5;
const I64: u8 = 6;
const BINARY: u8 = 8;
const LIST: u8 = 9;
const STRUCT: u8 = 12;

impl Compact {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.out.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.out.push(v as u8);
    }

    fn zigzag(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }

    fn field(&mut self, id: i16, kind: u8) {
        let last = self.last.last_mut().expect("a field outside a struct");
        let delta = id - *last;
        *last = id;
        if (1..=15).contains(&delta) {
            self.out.push(((delta as u8) << 4) | kind);
        } else {
            self.out.push(kind);
            self.zigzag(i64::from(id));
        }
    }

    fn begin(&mut self) {
        self.last.push(0);
    }

    fn end(&mut self) {
        self.out.push(0);
        self.last.pop();
    }

    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, I32);
        self.zigzag(i64::from(v));
    }

    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, I64);
        self.zigzag(v);
    }

    fn bytes(&mut self, v: &[u8]) {
        self.varint(v.len() as u64);
        self.out.extend_from_slice(v);
    }

    fn string(&mut self, id: i16, v: &str) {
        self.field(id, BINARY);
        self.bytes(v.as_bytes());
    }

    fn list(&mut self, id: i16, kind: u8, n: usize) {
        self.field(id, LIST);
        if n < 15 {
            self.out.push(((n as u8) << 4) | kind);
        } else {
            self.out.push(0xf0 | kind);
            self.varint(n as u64);
        }
    }

    fn strukt(&mut self, id: i16) {
        self.field(id, STRUCT);
        self.begin();
    }
}

/// Levels of bit width 1 in the RLE and bit packed hybrid, as one bit packed run, with the
/// 4 byte length Parquet puts in front of levels in a v1 data page.
fn levels(bits: &[bool]) -> Vec<u8> {
    let groups = bits.len().div_ceil(8);
    let mut run = Vec::new();
    let mut h = Compact::default();
    h.varint(((groups as u64) << 1) | 1);
    run.extend_from_slice(&h.out);
    for g in 0..groups {
        let mut b = 0u8;
        for i in 0..8 {
            if bits.get(g * 8 + i).copied().unwrap_or(false) {
                b |= 1 << i;
            }
        }
        run.push(b);
    }
    let mut out = (run.len() as u32).to_le_bytes().to_vec();
    out.extend(run);
    out
}

/// The body of a column's data page and the number of values it holds, levels included.
fn page(c: &Column) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    match c {
        Column::Str(v) => {
            for s in v {
                out.extend_from_slice(&(s.len() as u32).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            (out, v.len())
        }
        Column::I64(v) => {
            v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
            (out, v.len())
        }
        Column::F64(v) => {
            v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
            (out, v.len())
        }
        Column::Bool(v) => {
            for chunk in v.chunks(8) {
                out.push(chunk.iter().enumerate().fold(0u8, |b, (i, &x)| b | (u8::from(x) << i)));
            }
            (out, v.len())
        }
        Column::F64List(v) => {
            // A value per element and an empty list as one null value: repetition level 0
            // starts a row, and definition level 1 means an element is there.
            let (mut rep, mut def) = (Vec::new(), Vec::new());
            for row in v {
                if row.is_empty() {
                    rep.push(false);
                    def.push(false);
                }
                for i in 0..row.len() {
                    rep.push(i > 0);
                    def.push(true);
                }
            }
            out.extend(levels(&rep));
            out.extend(levels(&def));
            v.iter().flatten().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
            (out, rep.len())
        }
    }
}

/// A Parquet file of `cols`, which must all have the same number of rows.
///
/// # Panics
///
/// If the columns differ in length or a page is over 2 GiB.
#[must_use]
pub fn write(cols: &[(&str, Column)]) -> Vec<u8> {
    let rows = cols.first().map_or(0, |(_, c)| c.rows());
    assert!(cols.iter().all(|(_, c)| c.rows() == rows), "columns of different lengths");
    let mut file = b"PAR1".to_vec();
    // Offset, size and number of values of each column chunk.
    let mut chunks = Vec::with_capacity(cols.len());
    for (_, c) in cols {
        let (body, values) = page(c);
        let size = i32::try_from(body.len()).expect("a page over 2 GiB");
        let mut h = Compact::default();
        h.begin();
        h.i32(1, 0);
        h.i32(2, size);
        h.i32(3, size);
        h.strukt(5);
        h.i32(1, i32::try_from(values).expect("too many values"));
        h.i32(2, 0);
        h.i32(3, 3);
        h.i32(4, 3);
        h.end();
        h.end();
        let offset = file.len() as i64;
        let total = (h.out.len() + body.len()) as i64;
        file.extend(h.out);
        file.extend(body);
        chunks.push((offset, total, values as i64));
    }
    let mut m = Compact::default();
    m.begin();
    m.i32(1, 1);
    let lists = cols.iter().filter(|(_, c)| matches!(c, Column::F64List(_))).count();
    m.list(2, STRUCT, 1 + cols.len() + lists);
    m.begin();
    m.string(4, "schema");
    m.i32(5, i32::try_from(cols.len()).expect("too many columns"));
    m.end();
    for (name, c) in cols {
        m.begin();
        if let Column::F64List(_) = c {
            m.i32(3, 0);
            m.string(4, name);
            m.i32(5, 1);
            m.i32(6, 3);
            m.end();
            m.begin();
            m.i32(1, 5);
            m.i32(3, 2);
            m.string(4, "element");
        } else {
            m.i32(1, c.physical());
            m.i32(3, 0);
            m.string(4, name);
            if let Column::Str(_) = c {
                m.i32(6, 0);
            }
        }
        m.end();
    }
    m.i64(3, rows as i64);
    m.list(4, STRUCT, 1);
    m.begin();
    m.list(1, STRUCT, cols.len());
    for ((name, c), (offset, total, values)) in cols.iter().zip(&chunks) {
        m.begin();
        m.i64(2, *offset);
        m.strukt(3);
        m.i32(1, c.physical());
        m.list(2, I32, 2);
        m.zigzag(0);
        m.zigzag(3);
        let path: Vec<&str> = match c {
            Column::F64List(_) => vec![name, "element"],
            _ => vec![name],
        };
        m.list(3, BINARY, path.len());
        path.iter().for_each(|p| m.bytes(p.as_bytes()));
        m.i32(4, 0);
        m.i64(5, *values);
        m.i64(6, *total);
        m.i64(7, *total);
        m.i64(9, *offset);
        m.end();
        m.end();
    }
    m.i64(2, chunks.iter().map(|c| c.1).sum());
    m.i64(3, rows as i64);
    m.end();
    m.string(6, concat!("kime-eval ", env!("CARGO_PKG_VERSION")));
    m.end();
    file.extend_from_slice(&m.out);
    file.extend_from_slice(&(m.out.len() as u32).to_le_bytes());
    file.extend_from_slice(b"PAR1");
    file
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_the_file() {
        let f = write(&[
            ("a", Column::I64(vec![1, 2])),
            ("b", Column::Str(vec!["x".into(), "yz".into()])),
        ]);
        assert_eq!(&f[..4], b"PAR1");
        assert_eq!(&f[f.len() - 4..], b"PAR1");
        let n = u32::from_le_bytes(f[f.len() - 8..f.len() - 4].try_into().unwrap()) as usize;
        assert!(n < f.len() - 12);
        assert!(f.windows(2).any(|w| w == b"yz"));
    }

    #[test]
    fn packs_levels_in_one_run() {
        assert_eq!(levels(&[true, false, true]), vec![2, 0, 0, 0, 3, 0b101]);
        assert_eq!(levels(&[true; 9]).len(), 4 + 1 + 2);
    }

    #[test]
    fn encodes_compact_integers() {
        let mut c = Compact::default();
        c.zigzag(-1);
        c.zigzag(1);
        c.zigzag(300);
        assert_eq!(c.out, vec![1, 2, 0xd8, 0x04]);
    }
}

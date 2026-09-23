#![no_main]

// A .kime file whose header block is always well formed and whose index hash always matches, so
// every input reaches the index parser. The input is the index JSON, a zero byte, then the data
// section. The data section length is rounded up to 4 KiB with zeros, as the writer does.

use kime_model::pack::{ALIGN, MAGIC, VERSION};
use kime_tensor::Blob;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    let (json, rest) = match data.iter().position(|&b| b == 0) {
        Some(i) => (&data[..i], &data[i + 1..]),
        None => (data, &[][..]),
    };
    let data_start = (ALIGN + json.len()).next_multiple_of(ALIGN);
    let data_len = rest.len().next_multiple_of(ALIGN);
    let mut file = vec![0u8; data_start + data_len];
    file[..8].copy_from_slice(&MAGIC);
    file[8..12].copy_from_slice(&VERSION.to_le_bytes());
    file[16..24].copy_from_slice(&(json.len() as u64).to_le_bytes());
    file[24..32].copy_from_slice(&(data_start as u64).to_le_bytes());
    file[32..40].copy_from_slice(&(data_len as u64).to_le_bytes());
    file[40..72].copy_from_slice(blake3::hash(json).as_bytes());
    file[ALIGN..ALIGN + json.len()].copy_from_slice(json);
    file[data_start..data_start + rest.len()].copy_from_slice(rest);
    if let Ok((index, t)) = kime_model::pack::load(Blob::owned(file)) {
        common::touch(&t);
        for f in &index.files {
            std::hint::black_box(&t.blob()[f.start..f.end]);
        }
    }
});

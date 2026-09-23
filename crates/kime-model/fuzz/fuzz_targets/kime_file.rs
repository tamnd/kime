#![no_main]

// Raw bytes as a .kime file. This mostly exercises the fixed header block, since random bytes
// rarely carry a matching index hash. kime_index covers what lies behind it.

use kime_tensor::Blob;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    if let Ok((index, t)) = kime_model::pack::load(Blob::owned(data.to_vec())) {
        common::touch(&t);
        let _ = kime_model::pack::verify(&t.blob()[..], &index);
    }
});

#![no_main]

use kime_tensor::Blob;
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    if let Ok((t, _)) = kime_model::safetensors::load(Blob::owned(data.to_vec())) {
        common::touch(&t);
    }
});

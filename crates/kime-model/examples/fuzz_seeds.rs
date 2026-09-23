//! Writes small valid inputs for the fuzz targets in `fuzz/`.
//!
//!     cargo run -p kime-model --example fuzz_seeds -- crates/kime-model/fuzz/corpus

use std::path::Path;

use kime_model::pack::{self, ALIGN, Contents};
use kime_model::safetensors;
use kime_tensor::Blob;
use serde_json::Map;

fn main() {
    let out = std::env::args().nth(1).expect("usage: fuzz_seeds <corpus dir>");
    let out = Path::new(&out);
    let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]},"t":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#;
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());
    st.extend_from_slice(&[0, 60, 0, 64, 0, 66, 0, 68, 0, 0, 128, 63]);
    let (tensors, _) = safetensors::load(Blob::owned(st.clone())).unwrap();
    let mut kime = Vec::new();
    let contents = Contents {
        family: "laya",
        id: "seed",
        tensors: &tensors,
        files: vec![("encoder/config.json", b"{}".as_slice())],
        extra: Map::new(),
    };
    pack::write(&contents, &mut kime).unwrap();
    let json_len = u64::from_le_bytes(kime[16..24].try_into().unwrap()) as usize;
    let data_start = u64::from_le_bytes(kime[24..32].try_into().unwrap()) as usize;
    let mut index = kime[ALIGN..ALIGN + json_len].to_vec();
    index.push(0);
    index.extend_from_slice(&kime[data_start..data_start + 3 * ALIGN]);
    for (dir, bytes) in [("safetensors", st), ("kime_file", kime), ("kime_index", index)] {
        std::fs::create_dir_all(out.join(dir)).unwrap();
        std::fs::write(out.join(dir).join("seed"), bytes).unwrap();
    }
    println!("wrote seeds to {}", out.display());
}

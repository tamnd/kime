//! Writing a trained model as a checkpoint directory kime serves.
//!
//! The directory is the base checkpoint with new weights: the same carried files, the same tensor
//! names, order and element types, so `kime serve --models` and `kime eval` load it like the base.
//! Weights the model does not train, such as the temperatures, are copied from the base as they
//! are. `rl_agent_config.json` gets a `kime_train` entry saying how the weights were made.

use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use burn::tensor::backend::Backend;
use half::{bf16, f16};
use kime_model::Model;
use kime_tensor::DType;
use serde_json::{Map, Value};

use crate::model::Compat;

/// Writes `model` into `dir` as a checkpoint shaped like `base`, with `note` under `kime_train`
/// in `rl_agent_config.json`.
///
/// # Errors
///
/// Any error writing the files, and [`io::ErrorKind::InvalidData`] when the model's weights do
/// not match the base's names and shapes.
pub fn save<B: Backend>(
    model: &Compat<B>,
    base: &Model,
    dir: &Path,
    note: Value,
) -> io::Result<()> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let mut weights: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> =
        model.weights().into_iter().map(|(n, s, v)| (n, (s, v))).collect();
    for name in base.file_names() {
        let bytes = base.file(name).unwrap_or_default();
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if name == "rl_agent_config.json" {
            let mut cfg: Map<String, Value> =
                serde_json::from_slice(bytes).map_err(|e| bad(format!("{name}: {e}")))?;
            cfg.insert("kime_train".into(), note.clone());
            let mut s = serde_json::to_string_pretty(&Value::Object(cfg))
                .map_err(|e| bad(e.to_string()))?;
            s.push('\n');
            fs::write(&path, s)?;
        } else {
            fs::write(&path, bytes)?;
        }
    }
    let t = &base.tensors;
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(t.entries().len());
    for (i, e) in t.entries().iter().enumerate() {
        let Some((shape, v)) = weights.remove(&e.name) else {
            data.push(t.view(i).bytes.to_vec());
            continue;
        };
        if shape != e.shape {
            return Err(bad(format!("{}: shape {shape:?}, the base has {:?}", e.name, e.shape)));
        }
        data.push(match e.dtype {
            DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            DType::F16 => v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect(),
            DType::BF16 => v.iter().flat_map(|&x| bf16::from_f32(x).to_le_bytes()).collect(),
            d => return Err(bad(format!("{}: cannot write {d:?}", e.name))),
        });
    }
    if let Some(name) = weights.keys().next() {
        return Err(bad(format!("{name}: not in the base checkpoint")));
    }
    let mut header = Map::new();
    if let Some(m) = &base.metadata {
        header.insert("__metadata__".into(), Value::Object(m.clone()));
    }
    let mut at = 0usize;
    for (e, d) in t.entries().iter().zip(&data) {
        let mut h = Map::new();
        h.insert("dtype".into(), e.dtype.name().into());
        h.insert("shape".into(), e.shape.clone().into());
        h.insert("data_offsets".into(), vec![at, at + d.len()].into());
        header.insert(e.name.clone(), Value::Object(h));
        at += d.len();
    }
    let mut json = serde_json::to_vec(&Value::Object(header)).map_err(|e| bad(e.to_string()))?;
    json.resize(json.len().next_multiple_of(8), b' ');
    let tmp = dir.join("model.safetensors.part");
    let mut out = BufWriter::with_capacity(1 << 22, fs::File::create(&tmp)?);
    out.write_all(&(json.len() as u64).to_le_bytes())?;
    out.write_all(&json)?;
    for d in &data {
        out.write_all(d)?;
    }
    out.into_inner().map_err(io::IntoInnerError::into_error)?.sync_all()?;
    fs::rename(tmp, dir.join("model.safetensors"))
}

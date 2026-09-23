//! A small random compat checkpoint, so the plan tests run everywhere without Laya's weights.

#![allow(dead_code)]

use kime_model::laya::{LayaGraph, LayaSpec};
use kime_model::{Tensors, safetensors};
use kime_tensor::{BatchBuf, Blob};
use serde_json::{Map, Value, json};

/// Three encoder layers with one global, two heads, a 16 token window and a 60 token vocabulary.
pub(crate) fn tiny() -> (LayaSpec, LayaGraph, Tensors) {
    let encoder = json!({
        "hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 3,
        "intermediate_size": 96, "vocab_size": 60, "local_attention": 16,
        "global_attn_every_n_layers": 3, "global_rope_theta": 160000.0, "local_rope_theta": 10000.0,
    });
    let agent = json!({
        "encoder": "tiny", "head_layers": 2, "max_len": 128, "head_max_len": 32,
        "temperature": [1.0, 1.0, 1.0],
    });
    let spec = LayaSpec::from_json(&agent, &encoder).unwrap();
    let mut rng = Rng(11);
    let mut header = Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape) in spec.expected() {
        let n: usize = shape.iter().product();
        let norm =
            name.contains("norm") && name.ends_with("weight") || name.starts_with("scorer.0.w");
        let start = data.len();
        for _ in 0..n {
            let v = if norm { 1.0 + 0.1 * rng.next() } else { 0.15 * rng.next() };
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let head = Value::Object(header).to_string().into_bytes();
    let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&head);
    bytes.extend_from_slice(&data);
    let (tensors, _) = safetensors::load(Blob::owned(bytes)).unwrap();
    let graph = LayaGraph::bind(&spec, &tensors).unwrap();
    (spec, graph, tensors)
}

/// xorshift, so the tests need no dependency.
pub(crate) struct Rng(pub u64);

impl Rng {
    pub(crate) fn u(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// In [-1, 1).
    pub(crate) fn next(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.u() % n.max(1) as u64) as usize
    }
}

/// One random question: ids, marker positions and a type. Lengths run from 0 to `max`.
pub(crate) fn question(rng: &mut Rng, max: usize) -> (Vec<u32>, Vec<u32>, u8) {
    let len = rng.below(max + 1);
    let ids = (0..len).map(|_| rng.below(60) as u32).collect();
    let k = if len == 0 { 0 } else { rng.below(6) };
    let markers = (0..k).map(|_| rng.below(len) as u32).collect();
    (ids, markers, rng.below(3) as u8)
}

/// Fills `buf` with the questions.
pub(crate) fn fill(buf: &mut BatchBuf, qs: &[(Vec<u32>, Vec<u32>, u8)]) {
    buf.clear();
    for q in qs {
        buf.push(&q.0, &q.1, q.2);
    }
}

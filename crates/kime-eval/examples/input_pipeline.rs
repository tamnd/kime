//! Times the compat input pipeline (parse, validate, render, tokenize, lay out) over the parity
//! cases, one request at a time on one thread.
//!
//!     cargo run --release -p kime-eval --example input_pipeline -- <models>/laya

use std::time::Instant;

use kime_core::render::{compat_question, compat_state};
use kime_core::request::{Limits, parse};
use kime_tok::Tokenizer;
use kime_tok::layout::{CompatBudget, Cut};
use serde_json::Value;

fn main() {
    let root = std::env::args().nth(1).expect("usage: input_pipeline <models>/laya");
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/parity/cases.jsonl");
    let bodies: Vec<String> = std::fs::read_to_string(path).unwrap().lines().map(str::to_string).collect();
    for (name, dir, budget) in [
        ("laya", "tokenizer", CompatBudget { max_len: 512, head_max_len: 192 }),
        ("laya-multilingual", "multilingual/tokenizer", CompatBudget { max_len: 1024, head_max_len: 256 }),
    ] {
        let tok = Tokenizer::from_dir(format!("{root}/{dir}")).unwrap();
        let mask = tok.mask_text().to_string();
        let mut samples = Vec::new();
        let mut tokens = 0usize;
        for pass in 0..20 {
            for body in &bodies {
                let t = Instant::now();
                let v: Value = serde_json::from_str(body).unwrap();
                let req = parse(&v, &Limits::LAYA).unwrap();
                let state_ids = tok.encode_state(&compat_state(&req.state, &mask));
                let mut n = 0;
                for q in &req.questions {
                    let text = compat_question(q, &mask);
                    n += tok.compat_sequence(&text.head, &text.options, &state_ids, budget, Cut::Tail).ids.len();
                }
                let dt = t.elapsed().as_nanos() as u64;
                if pass > 0 {
                    samples.push(dt);
                    tokens += n;
                }
            }
        }
        samples.sort_unstable();
        let n = samples.len();
        let bytes: usize = bodies.iter().map(String::len).sum();
        println!(
            "{name}: {} requests of {:.0} bytes and {:.0} tokens on average, p50 {:.1} us, p99 {:.1} us, max {:.1} us",
            bodies.len(),
            bytes as f64 / bodies.len() as f64,
            tokens as f64 / n as f64,
            samples[n / 2] as f64 / 1e3,
            samples[n * 99 / 100] as f64 / 1e3,
            samples[n - 1] as f64 / 1e3
        );
    }
}

//! Times the FP32 reference forward pass on the parity questions.
//!
//!     cargo run --release -p kime-cpu --example compat_bench -- <laya dir> <fixture.jsonl> [threads] [batch]
//!
//! It prints the load time, the latency of one question at a time (p50 and p99), and the
//! throughput when `batch` questions run together.

use std::time::Instant;

use kime_cpu::{Compat, Input, par};
use kime_model::Model;
use serde_json::Value;

type Question = (Vec<u32>, Vec<u32>, usize);

fn input(q: &Question) -> Input<'_> {
    Input { ids: &q.0, markers: &q.1, qtype: q.2 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let fixture = &args[2];
    let threads = args.get(3).map_or_else(par::available, |t| t.parse().unwrap());
    let batch: usize = args.get(4).map_or(16, |b| b.parse().unwrap());

    let mut qs: Vec<Question> = Vec::new();
    for line in std::fs::read_to_string(fixture).unwrap().lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        for q in case["questions"].as_array().into_iter().flatten() {
            let ids = |k: &str| {
                q[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
            };
            qs.push((ids("ids"), ids("markers"), q["qtype"].as_u64().unwrap() as usize));
        }
    }
    let tokens: usize = qs.iter().map(|q| q.0.len()).sum();

    let t = Instant::now();
    let model = Model::open(dir).unwrap();
    let compat = Compat::new(&model, threads);
    println!("load and convert: {:.0} ms, {threads} threads", t.elapsed().as_secs_f64() * 1e3);

    let _ = compat.forward(&[input(&qs[0])]);

    let mut lat = Vec::new();
    let t = Instant::now();
    for q in &qs {
        let s = Instant::now();
        let _ = compat.forward(&[input(q)]);
        lat.push(s.elapsed().as_secs_f64() * 1e3);
    }
    let one = t.elapsed().as_secs_f64();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((lat.len() - 1) as f64 * p) as usize];
    println!(
        "one at a time: {} questions, {tokens} tokens, {:.1} s, p50 {:.1} ms, p99 {:.1} ms, mean {:.1} ms, {:.0} tokens/s",
        qs.len(),
        one,
        pct(0.5),
        pct(0.99),
        one * 1e3 / qs.len() as f64,
        tokens as f64 / one
    );

    let t = Instant::now();
    for chunk in qs.chunks(batch) {
        let inputs: Vec<Input<'_>> = chunk.iter().map(input).collect();
        let _ = compat.forward(&inputs);
    }
    let b = t.elapsed().as_secs_f64();
    println!(
        "batches of {batch}: {:.1} s, {:.1} ms per question, {:.0} tokens/s",
        b,
        b * 1e3 / qs.len() as f64,
        tokens as f64 / b
    );
}

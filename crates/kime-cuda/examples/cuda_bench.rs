//! Times the compat graph on one GPU on the parity questions.
//!
//!     cargo run --release -p kime-cuda --example cuda_bench -- <laya dir> <fixture.jsonl> [f32|f16] [batch]
//!
//! It prints the load time, the latency of one question at a time (p50 and p99) and the
//! throughput when `batch` questions run together.

use std::time::Instant;

use kime_cuda::{Precision, executor};
use kime_model::Model;
use kime_tensor::{BatchBuf, Outputs};
use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let fixture = &args[2];
    let precision = match args.get(3).map(String::as_str) {
        None | Some("f32") => Precision::F32,
        Some("f16") => Precision::F16,
        Some(p) => panic!("precision {p} is not f32 or f16"),
    };
    let batch: usize = args.get(4).map_or(16, |b| b.parse().unwrap());

    let mut qs: Vec<(Vec<u32>, Vec<u32>, u8)> = Vec::new();
    for line in std::fs::read_to_string(fixture).unwrap().lines() {
        let case: Value = serde_json::from_str(line).unwrap();
        for q in case["questions"].as_array().into_iter().flatten() {
            let ids = |k: &str| {
                q[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
            };
            qs.push((ids("ids"), ids("markers"), q["qtype"].as_u64().unwrap() as u8));
        }
    }
    let tokens: usize = qs.iter().map(|q| q.0.len()).sum();

    let t = Instant::now();
    let model = Model::open(dir).unwrap();
    let mut exec = executor(&model, 0, precision).unwrap();
    println!(
        "load and upload: {:.0} ms, {} {:?}",
        t.elapsed().as_secs_f64() * 1e3,
        exec.backend().name(),
        precision
    );

    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    let mut run = |exec: &mut kime_tensor::Executor<_>, qs: &[(Vec<u32>, Vec<u32>, u8)]| {
        buf.clear();
        for q in qs {
            buf.push(&q.0, &q.1, q.2);
        }
        exec.run(&buf.batch(), &mut out).unwrap();
    };
    // Build every bucket's plan the timed runs will use.
    let t = Instant::now();
    for q in &qs {
        run(&mut exec, std::slice::from_ref(q));
    }
    for chunk in qs.chunks(batch) {
        run(&mut exec, chunk);
    }
    println!("warm up, lowering every bucket used: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);

    let mut lat = Vec::new();
    let t = Instant::now();
    for q in &qs {
        let s = Instant::now();
        run(&mut exec, std::slice::from_ref(q));
        lat.push(s.elapsed().as_secs_f64() * 1e3);
    }
    let one = t.elapsed().as_secs_f64();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((lat.len() - 1) as f64 * p) as usize];
    println!(
        "one at a time: {} questions, {tokens} tokens, {:.3} s, p50 {:.3} ms, p99 {:.3} ms, mean {:.3} ms, {:.0} tokens/s",
        qs.len(),
        one,
        pct(0.5),
        pct(0.99),
        one * 1e3 / qs.len() as f64,
        tokens as f64 / one
    );

    let t = Instant::now();
    for chunk in qs.chunks(batch) {
        run(&mut exec, chunk);
    }
    let b = t.elapsed().as_secs_f64();
    println!(
        "batches of {batch}: {:.3} s, {:.3} ms per question, {:.0} tokens/s",
        b,
        b * 1e3 / qs.len() as f64,
        tokens as f64 / b
    );
}

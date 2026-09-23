//! Times the compat graph on the plan executor on the parity questions.
//!
//!     cargo run --release -p kime-cpu --example plan_bench -- <laya dir> <fixture.jsonl> [threads] [batch]
//!
//! It prints the load time, the latency of one question at a time (p50 and p99), the throughput
//! when `batch` questions run together, and where the time goes by kind of step.

use std::time::Instant;

use kime_cpu::{executor, par};
use kime_model::Model;
use kime_tensor::{BatchBuf, Outputs};
use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let fixture = &args[2];
    let threads = args.get(3).map_or_else(par::available, |t| t.parse().unwrap());
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
    let mut exec = executor(&model, threads).unwrap();
    println!("load and convert: {:.0} ms, {threads} threads", t.elapsed().as_secs_f64() * 1e3);

    let (mut buf, mut out) = (BatchBuf::default(), Outputs::default());
    let mut run = |exec: &mut kime_tensor::Executor<_>, qs: &[(Vec<u32>, Vec<u32>, u8)]| {
        buf.clear();
        for q in qs {
            buf.push(&q.0, &q.1, q.2);
        }
        exec.run(&buf.batch(), &mut out).unwrap();
    };
    // Build every bucket's plan the timed runs will use.
    for q in &qs {
        run(&mut exec, std::slice::from_ref(q));
    }
    for chunk in qs.chunks(batch) {
        run(&mut exec, chunk);
    }
    for p in exec.plans_mut() {
        p.profile();
    }

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
        "one at a time: {} questions, {tokens} tokens, {:.2} s, p50 {:.2} ms, p99 {:.2} ms, mean {:.2} ms, {:.0} tokens/s",
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
        "batches of {batch}: {:.2} s, {:.2} ms per question, {:.0} tokens/s",
        b,
        b * 1e3 / qs.len() as f64,
        tokens as f64 / b
    );

    let mut total: Vec<(&str, u64)> = Vec::new();
    for p in exec.plans_mut() {
        for (k, ns) in p.timings() {
            match total.iter_mut().find(|t| t.0 == k) {
                Some(t) => t.1 += ns,
                None => total.push((k, ns)),
            }
        }
    }
    total.sort_by_key(|t| std::cmp::Reverse(t.1));
    let all: u64 = total.iter().map(|t| t.1).sum();
    let parts: Vec<String> = total
        .iter()
        .map(|(k, ns)| format!("{k} {:.1}%", *ns as f64 * 100.0 / all.max(1) as f64))
        .collect();
    println!("time by step: {}", parts.join(", "));
}

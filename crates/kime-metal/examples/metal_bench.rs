//! Times the compat graph on the GPU on the parity questions.
//!
//!     cargo run --release -p kime-metal --example metal_bench -- <laya dir> <fixture.jsonl> [batch] [f32|f16]
//!
//! It prints the load time, the latency of one question at a time (p50 and p99), the throughput
//! when `batch` questions run together, and then, in a separate pass where every step waits for
//! the GPU, where the time goes by kind of step. Before that it prints the energy per decision
//! from the SMC over again as many runs, next to the machine's idle draw.

#[cfg(target_os = "macos")]
fn main() {
    use std::time::Instant;

    use kime_metal::{Precision, executor};
    use kime_model::Model;
    use kime_tensor::{BatchBuf, Outputs};
    use serde_json::Value;

    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let fixture = &args[2];
    let batch: usize = args.get(3).map_or(16, |b| b.parse().unwrap());
    let precision = match args.get(4).map(String::as_str) {
        Some("f16") => Precision::F16,
        _ => Precision::F32,
    };

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
    let mut exec = executor(&model, precision).unwrap();
    println!("load and convert: {:.0} ms, {precision:?}", t.elapsed().as_secs_f64() * 1e3);

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
        "batches of {batch}: {:.2} s, {:.2} ms per question, {:.0} questions/s, {:.0} tokens/s",
        b,
        b * 1e3 / qs.len() as f64,
        qs.len() as f64 / b,
        tokens as f64 / b
    );

    match kime_metal::energy::Meter::open() {
        Ok(meter) => {
            let idle = meter.idle_watts(5.0).unwrap();
            println!("idle: {idle:.1} W");
            let mut each = |name: &str, f: &mut dyn FnMut(&mut kime_tensor::Executor<_>)| {
                let (_, s, j) = meter.measure(|| f(&mut exec)).unwrap();
                let n = qs.len() as f64;
                println!(
                    "energy, {name}: {:.1} W over {s:.2} s, {:.1} mJ per decision, {:.1} mJ above idle",
                    j / s,
                    j * 1e3 / n,
                    (j - idle * s) * 1e3 / n
                );
            };
            each("one at a time", &mut |e| {
                for q in &qs {
                    run(e, std::slice::from_ref(q));
                }
            });
            each(&format!("batches of {batch}"), &mut |e| {
                for chunk in qs.chunks(batch) {
                    run(e, chunk);
                }
            });
        }
        Err(e) => println!("no energy counter: {e}"),
    }

    for p in exec.plans_mut() {
        p.profile();
    }
    for chunk in qs.chunks(batch) {
        run(&mut exec, chunk);
    }
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
    println!("time by step, batched: {}", parts.join(", "));
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the Metal backend needs macOS");
}

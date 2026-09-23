//! Compares kime's ids with Hugging Face's on a corpus from `tools/ref/tok_corpus.py`, and times
//! both tokenizers over the whole corpus on one thread.
//!
//!     cargo run --release -p kime-tok --example corpus -- <models>/laya corpus.jsonl

use std::time::Instant;

use kime_tok::Tokenizer;
use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, models, corpus] = args.as_slice() else {
        eprintln!("usage: corpus <laya model dir> <corpus.jsonl>");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(corpus).expect("read the corpus");
    let rows: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).expect("a JSON line")).collect();
    let lines: Vec<&str> = rows.iter().map(|r| r["text"].as_str().expect("text")).collect();
    let bytes: usize = lines.iter().map(|l| l.len()).sum();
    let mut failed = false;
    for (name, dir) in [("laya", "tokenizer"), ("laya-multilingual", "multilingual/tokenizer")] {
        let t0 = Instant::now();
        let tok = Tokenizer::from_dir(format!("{models}/{dir}")).expect("load the tokenizer");
        let load = t0.elapsed();
        let mut out = Vec::new();
        let mut bad = 0usize;
        let mut shown = 0;
        for (row, line) in rows.iter().zip(&lines) {
            out.clear();
            tok.encode_into(line, &mut out);
            let want: Vec<u32> = row[name].as_array().expect("ids").iter().map(|v| v.as_u64().expect("id") as u32).collect();
            if out != want {
                bad += 1;
                if shown < 5 {
                    shown += 1;
                    eprintln!("{name} differs on {line:?}\n  want {want:?}\n  got  {out:?}");
                }
            }
        }
        // Timed separately, three times, best of three, so the comparison above is not in it.
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let mut n = 0usize;
            for line in &lines {
                out.clear();
                tok.encode_into(line, &mut out);
                n += out.len();
            }
            std::hint::black_box(n);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{name}: {} lines, {bad} differ, load {:.0} ms, encode {:.2} s, {:.0} lines/s, {:.1} MB/s",
            lines.len(),
            load.as_secs_f64() * 1e3,
            best,
            lines.len() as f64 / best,
            bytes as f64 / 1e6 / best
        );
        // Distinct 2 KB requests made of whole lines, the size the spec budgets 15 microseconds for.
        // Each is encoded once, so the word cache only helps as much as it would with real traffic.
        let mut requests = vec![String::new()];
        for line in &lines {
            let last = requests.last_mut().expect("never empty");
            if last.len() + line.len() + 1 > 2048 {
                requests.push(String::new());
            }
            let last = requests.last_mut().expect("never empty");
            last.push_str(line);
            last.push('\n');
        }
        requests.retain(|r| r.len() > 1900);
        let fresh = Tokenizer::from_dir(format!("{models}/{dir}")).expect("load the tokenizer");
        let t = Instant::now();
        out.clear();
        fresh.encode_into(&requests[0], &mut out);
        let cold = t.elapsed().as_nanos() as f64 / 1e3;
        let mut samples = Vec::with_capacity(requests.len());
        let mut tokens = 0usize;
        for r in &requests {
            let t = Instant::now();
            out.clear();
            fresh.encode_into(r, &mut out);
            samples.push(t.elapsed().as_nanos() as u64);
            tokens += out.len();
        }
        samples.sort_unstable();
        let n = samples.len();
        println!(
            "{name}: {n} distinct 2 KB requests, {:.0} tokens each, first {cold:.1} us, p50 {:.1} us, p99 {:.1} us",
            tokens as f64 / n as f64,
            samples[n / 2] as f64 / 1e3,
            samples[n * 99 / 100] as f64 / 1e3
        );
        failed |= bad > 0;
    }
    if failed {
        std::process::exit(1);
    }
}

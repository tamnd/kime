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
        failed |= bad > 0;
    }
    if failed {
        std::process::exit(1);
    }
}

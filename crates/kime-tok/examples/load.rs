//! Times building a tokenizer from its files, best of five.
//!
//!     cargo run --release -p kime-tok --example load -- <tokenizer dir>...

use std::time::Instant;

fn main() {
    for dir in std::env::args().skip(1) {
        let json = std::fs::read(format!("{dir}/tokenizer.json")).unwrap();
        let config = std::fs::read(format!("{dir}/tokenizer_config.json")).ok();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let tok = kime_tok::Tokenizer::from_bytes(&json, config.as_deref()).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(tok);
        }
        println!("{dir}: {:.1} MB, built in {:.1} ms", json.len() as f64 / 1e6, best * 1e3);
    }
}

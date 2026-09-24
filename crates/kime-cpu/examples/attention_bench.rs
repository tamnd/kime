//! Times attention on one thread for 16 sequences, with a local window and without, best of 10.
//!
//!     cargo run --release -p kime-cpu --example attention_bench [length]

use kime_cpu::attention::{HEAD, attention};

fn main() {
    let len: usize = std::env::args().nth(1).map_or(128, |a| a.parse().unwrap());
    let (heads, seqs) = (12, 16);
    let t = len * seqs;
    let d = heads * HEAD;
    let qkv: Vec<f32> = (0..t * 3 * d).map(|i| ((i * 7919) % 2001) as f32 / 500.0 - 2.0).collect();
    let cu: Vec<usize> = (0..=seqs).map(|s| s * len).collect();
    let mut out = vec![0f32; t * d];
    for window in [Some(64), None] {
        let mut best = f64::MAX;
        for _ in 0..10 {
            let s = std::time::Instant::now();
            attention(&qkv, heads, &cu, window, &mut out, 1);
            best = best.min(s.elapsed().as_secs_f64());
        }
        println!("{seqs} x {len} tokens, window {window:?}: {:.2} ms ({})", best * 1e3, out[7]);
    }
}

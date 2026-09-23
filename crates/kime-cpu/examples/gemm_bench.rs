//! GFLOPS of the GEMM at the compat shapes, with the weights packed once as a plan does.
//!
//!     cargo run --release -p kime-cpu --example gemm_bench -- [threads]

use std::time::Instant;

use kime_cpu::gemm::{Gemm, pack};
use kime_cpu::par;
use kime_tensor::Epilogue;

fn main() {
    let threads = std::env::args().nth(1).map_or_else(par::available, |t| t.parse().unwrap());
    // One question of about 60 tokens, then a batch of them, at the Laya English layer shapes.
    let shapes = [
        (60, 1024, 3072),
        (60, 1024, 1024),
        (60, 1024, 5248),
        (60, 2624, 1024),
        (960, 1024, 3072),
        (960, 1024, 5248),
        (960, 2624, 1024),
    ];
    for (m, k, n) in shapes {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.1).collect();
        let w = pack(&w, n, k);
        let g = Gemm { x: &x, m, k, w: &w, n, b: None, ep: Epilogue::None };
        let mut y = vec![0f32; m * n];
        let run = |y: &mut [f32]| g.run(y, threads, |t, f| par::for_each(t, threads, f));
        run(&mut y);
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            run(&mut y);
        }
        let s = t.elapsed().as_secs_f64() / f64::from(reps);
        let flops = 2.0 * (m * k * n) as f64;
        println!(
            "{m}x{k}x{n}: {:.2} ms, {:.0} GFLOPS on {threads} threads",
            s * 1e3,
            flops / s / 1e9
        );
    }
}

//! GFLOPS of the reference GEMM at the compat shapes.
//!
//!     cargo run --release -p kime-cpu --example gemm_bench -- [threads]

use std::time::Instant;

use kime_cpu::{gemm, par};

fn main() {
    let threads = std::env::args().nth(1).map_or_else(par::available, |t| t.parse().unwrap());
    for (m, k, n) in [(128, 1024, 3072), (128, 2624, 1024), (2048, 1024, 5248), (2048, 4096, 1024)]
    {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.1).collect();
        let mut y = vec![0f32; m * n];
        gemm::linear(&x, m, k, &w, n, None, &mut y, threads);
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            gemm::linear(&x, m, k, &w, n, None, &mut y, threads);
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

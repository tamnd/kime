//! GFLOPS of the FP32 and INT8 GEMMs at the compat shapes, with the weights packed or rounded once
//! as a plan does. The INT8 time includes rounding the activations.
//!
//!     cargo run --release -p kime-cpu --example gemm_bench -- [threads]

use std::time::Instant;

use kime_cpu::gemm::{self, Gemm, pack};
use kime_cpu::par;
use kime_cpu::qgemm::{self, QGemm, QMatrix};
use kime_tensor::Epilogue;

/// Seconds per call, from the best of five after a warm up.
fn time(mut run: impl FnMut()) -> f64 {
    run();
    (0..5)
        .map(|_| {
            let t = Instant::now();
            run();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

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
        let mut y = vec![0f32; m * n];

        let packed = pack(&w, n, k);
        let g = Gemm { x: &x, m, k, w: &packed, n, b: None, ep: Epilogue::None };
        let len = gemm::scratch_len(k, n);
        let f32s = time(|| {
            g.run(&mut y, threads, |t, f| par::for_each(t, threads, |i| f(i, &mut vec![0.0; len])));
        });

        let q = QMatrix::quantize(&w, n, k);
        let g = QGemm { x: &x, m, w: &q, b: None, ep: Epilogue::None };
        let len = qgemm::scratch_len(k);
        let i8s = time(|| {
            g.run(&mut y, threads, |t, f| par::for_each(t, threads, |i| f(i, &mut vec![0.0; len])));
        });

        let ops = 2.0 * (m * k * n) as f64;
        println!(
            "{m}x{k}x{n} on {threads} threads: FP32 {:.2} ms {:.0} GFLOPS, INT8 {:.2} ms {:.0} GOPS",
            f32s * 1e3,
            ops / f32s / 1e9,
            i8s * 1e3,
            ops / i8s / 1e9
        );
    }
}

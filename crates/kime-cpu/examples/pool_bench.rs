//! Time of the GEMM at the compat shapes when it runs through the pool the plan uses, and the
//! overhead of a job of four tiny tasks, which is what a layer norm or rope over one question costs
//! the pool.
//!
//!     cargo run --release -p kime-cpu --example pool_bench -- [threads]

use kime_cpu::gemm::{self, Gemm, pack};
use kime_cpu::pool::Pool;
use kime_tensor::Epilogue;
use std::time::Instant;
fn main() {
    let threads: usize = std::env::args().nth(1).map_or(8, |t| t.parse().unwrap());
    let pool = Pool::new(threads);
    for (m, k, n) in [(60, 1024, 3072), (60, 1024, 1024), (960, 1024, 3072), (960, 2624, 1024)] {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let w: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.1).collect();
        let packed = pack(&w, n, k);
        let g = Gemm { x: &x, m, k, w: &packed, n, b: None, ep: Epilogue::None };
        let mut y = vec![0f32; m * n];
        let scratch: Vec<std::sync::Mutex<Vec<f32>>> = (0..threads)
            .map(|_| std::sync::Mutex::new(vec![0.0; gemm::scratch_len(k, n)]))
            .collect();
        let mut run = || {
            g.run(&mut y, threads, |t, f| {
                pool.run(t, &|i, wk| f(i, &mut scratch[wk].lock().unwrap()))
            })
        };
        run();
        let reps = if m > 100 { 20 } else { 200 };
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..reps {
                run();
            }
            best = best.min(t.elapsed().as_secs_f64() / reps as f64);
        }
        println!("{m}x{k}x{n}: {:.3} ms", best * 1e3);
    }
    let sink = std::sync::atomic::AtomicU64::new(0);
    let t = Instant::now();
    for _ in 0..20000 {
        pool.run(4, &|i, _| {
            sink.fetch_add(i as u64, std::sync::atomic::Ordering::Relaxed);
        });
    }
    println!("small job of 4 tasks: {:.2} us", t.elapsed().as_secs_f64() / 20000.0 * 1e6);
}

fn main() {
    let n = 1152 * 256;
    let u: Vec<f32> = (0..2 * n).map(|i| ((i * 7919) % 2001) as f32 / 200.0 - 5.0).collect();
    let mut out = vec![0f32; n];
    let mut best = f64::MAX;
    for _ in 0..20 {
        let t = std::time::Instant::now();
        kime_cpu::ops::geglu(&u, 1152, &mut out);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{:.2} ns per element {}", best * 1e9 / n as f64, out[5]);
}

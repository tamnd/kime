//! Each kernel of `kernels/compat.cu` on its own against a naive version on the host, in f64. The
//! real row counts are below the launched ones, as in a bucket, and the rows past them are checked
//! to stay as they were. Attention is checked against dense attention with the mask built as a
//! matrix, with and without rope and a window. Skipped, with a note, without an NVIDIA GPU.

use cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg};

use crate::plan::{ATT_Q, ATT_W, LN_ROWS, rope_tables};
use crate::{CudaBackend, Precision};

/// What padding rows start as, and must still hold after a launch.
const SENTINEL: f32 = 7.0;

fn gpu() -> Option<CudaBackend> {
    match CudaBackend::new(0, Precision::F32) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipped, no GPU: {e}");
            None
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in [-a, a).
    fn vec(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| a * ((self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)).collect()
    }
}

/// FP16 bits to f32.
fn h2f(h: u16) -> f32 {
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from(h >> 10 & 0x1f), f32::from(h & 0x3ff));
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A device buffer of f32 or f16 values.
enum Buf {
    F32(CudaSlice<f32>),
    F16(CudaSlice<u16>),
}

impl Buf {
    fn ptr(&self, b: &CudaBackend) -> u64 {
        match self {
            Buf::F32(d) => d.device_ptr(&b.stream).0,
            Buf::F16(d) => d.device_ptr(&b.stream).0,
        }
    }

    fn read(&self, b: &CudaBackend) -> Vec<f32> {
        b.stream.synchronize().unwrap();
        match self {
            Buf::F32(d) => b.stream.clone_dtoh(d).unwrap(),
            Buf::F16(d) => b.stream.clone_dtoh(d).unwrap().into_iter().map(h2f).collect(),
        }
    }
}

fn up<T: DeviceRepr>(b: &CudaBackend, v: &[T]) -> CudaSlice<T> {
    b.stream.clone_htod(v).unwrap()
}

fn ptr<T>(b: &CudaBackend, d: &CudaSlice<T>) -> u64 {
    d.device_ptr(&b.stream).0
}

/// `v` on the device in FP16 when `half`, and the values the device holds.
fn input(b: &CudaBackend, v: &[f32], half: bool) -> (Buf, Vec<f32>) {
    if !half {
        return (Buf::F32(up(b, v)), v.to_vec());
    }
    let x = up(b, v);
    let out = b.stream.alloc_zeros::<u16>(v.len()).unwrap();
    let (po, px, len) = (ptr(b, &out), ptr(b, &x), v.len());
    let mut l = b.stream.launch_builder(&b.k.to_f16);
    l.arg(&po).arg(&px).arg(&len);
    // SAFETY: both buffers hold `len` values.
    unsafe { l.launch(LaunchConfig::for_num_elems(len as u32)) }.unwrap();
    let out = Buf::F16(out);
    let held = out.read(b);
    (out, held)
}

/// `len` values of [`SENTINEL`].
fn output(b: &CudaBackend, len: usize, half: bool) -> Buf {
    input(b, &vec![SENTINEL; len], half).0
}

fn rows(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

/// Checks `got` against `want` within `tol` scaled by the size of `want`, and that every value
/// from `real` on is still the sentinel. Returns the largest error.
fn check(what: &str, got: &[f32], want: &[f32], real: usize, tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if i >= real {
            assert_eq!(g.to_bits(), SENTINEL.to_bits(), "{what}: padding value {i} was written");
            continue;
        }
        let err = (f64::from(g) - f64::from(w)).abs();
        worst = worst.max(err);
        assert!(err <= tol * (1.0 + f64::from(w).abs()), "{what}: value {i} is {g}, want {w}");
    }
    worst
}

/// The FP16 value nearest `x`, as f32, through the device's own conversion.
fn round_half(b: &CudaBackend, v: &[f32]) -> Vec<f32> {
    input(b, v, true).1
}

/// Tolerance for an output type: FP16 outputs are within half an ulp of the rounded value.
fn tol(half_out: bool, f32_tol: f64) -> f64 {
    if half_out { 1e-3 } else { f32_tol }
}

/// erf to about 1e-7, Abramowitz and Stegun 7.1.26.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let p = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - p * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Rows of the batch: the row counts real and launched, and for tokens their sequences.
struct Layout {
    tokens: usize,
    launched: usize,
    seq: Vec<u32>,
    cu: Vec<u32>,
    pos: Vec<u32>,
}

impl Layout {
    fn new(lens: &[usize], launched: usize) -> Self {
        let (mut seq, mut cu, mut pos) = (Vec::new(), vec![0u32], Vec::new());
        for (s, &n) in lens.iter().enumerate() {
            for p in 0..n {
                seq.push(s as u32);
                pos.push(p as u32);
            }
            cu.push(seq.len() as u32);
        }
        let tokens = seq.len();
        assert!(tokens <= launched);
        seq.resize(launched, 0);
        pos.resize(launched, 0);
        Self { tokens, launched, seq, cu, pos }
    }

    fn seqs(&self) -> usize {
        self.cu.len() - 1
    }
}

#[test]
fn to_f16_rounds_to_nearest() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(3);
    let mut v = rng.vec(5000, 4.0);
    v.extend([0.0, -0.0, 1e-5, -3e-7, 65504.0, 1.0 + 1.0 / 2048.0, 1.0 + 3.0 / 2048.0]);
    let got = round_half(&b, &v);
    for (&x, &h) in v.iter().zip(&got) {
        let err = (x - h).abs();
        // Half an ulp: 2^-11 relative for normal values and 2^-25 for subnormal ones.
        assert!(err <= (x.abs() * 2f32.powi(-11)).max(2f32.powi(-25)), "{x} became {h}");
    }
    // Ties go to even.
    assert_eq!(got[5005].to_bits(), 1f32.to_bits());
    assert_eq!(got[5006].to_bits(), (1.0f32 + 2.0 / 1024.0).to_bits());
}

#[test]
fn embed_copies_table_rows() {
    let Some(b) = gpu() else { return };
    let (d, vocab, real, launched) = (200, 50, 13, 16);
    let mut rng = Rng(5);
    let table = rng.vec(vocab * d, 1.0);
    let mut ids: Vec<u32> = (0..launched).map(|_| rng.below(vocab) as u32).collect();
    ids[real..].fill(u32::MAX);
    let n = up(&b, &[real as u32, 1, 0]);
    let (t, i) = (up(&b, &table), up(&b, &ids));
    let out = output(&b, launched * d, false);
    let (po, pt, pi, pn, di) = (out.ptr(&b), ptr(&b, &t), ptr(&b, &i), ptr(&b, &n), d as i32);
    let mut l = b.stream.launch_builder(&b.k.embed);
    l.arg(&po).arg(&pt).arg(&pi).arg(&pn).arg(&di);
    // SAFETY: the buffers match the sizes the kernel reads and writes.
    unsafe { l.launch(rows(launched, 256)) }.unwrap();
    let want: Vec<f32> =
        (0..real).flat_map(|r| table[ids[r] as usize * d..][..d].to_vec()).collect();
    let mut want = want;
    want.resize(launched * d, 0.0);
    check("embed", &out.read(&b), &want, real * d, 0.0);
}

#[test]
fn layer_norm_matches_naive() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(7);
    for (kind, d, bias) in [(0, 200, true), (0, 1024, false), (1, 96, true)] {
        let (real, launched) = (13, 16);
        let (w, bv) = (rng.vec(d, 2.0), rng.vec(d, 1.0));
        let x0: Vec<f32> = rng
            .vec(launched * d, 3.0)
            .iter()
            .enumerate()
            .map(|(i, v)| v + (i / d) as f32)
            .collect();
        let mut n = [0u32; 3];
        n[kind as usize] = real as u32;
        let (n, wd, bd) = (up(&b, &n), up(&b, &w), up(&b, &bv));
        for variant in 0..4 {
            let (half_in, half_out) = (variant >= 2, variant % 2 == 1);
            let (x, xs) = input(&b, &x0, half_in);
            let out = output(&b, launched * d, half_out);
            let pb = if bias { ptr(&b, &bd) } else { 0 };
            let (po, px, pw, pn) = (out.ptr(&b), x.ptr(&b), ptr(&b, &wd), ptr(&b, &n));
            let (di, eps) = (d as i32, 1e-5f32);
            let mut l = b.stream.launch_builder(&b.k.ln[variant]);
            l.arg(&po).arg(&px).arg(&pw).arg(&pb).arg(&pn).arg(&kind).arg(&di).arg(&eps);
            // SAFETY: the buffers match the sizes the kernel reads and writes.
            unsafe { l.launch(rows(launched.div_ceil(LN_ROWS), 32 * LN_ROWS as u32)) }.unwrap();
            let mut want = vec![0f32; launched * d];
            for r in 0..real {
                let row = &xs[r * d..][..d];
                let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
                let var =
                    row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / d as f64;
                for c in 0..d {
                    let y = (f64::from(row[c]) - mean) / (var + 1e-5).sqrt() * f64::from(w[c]);
                    want[r * d + c] = (y + if bias { f64::from(bv[c]) } else { 0.0 }) as f32;
                }
            }
            let what = format!("layer norm {variant} d {d}");
            check(&what, &out.read(&b), &want, real * d, tol(half_out, 1e-5));
        }
    }
}

#[test]
fn bias_act_matches_naive() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(11);
    let (width, real, launched) = (3000, 5, 8);
    let bias = rng.vec(width, 1.0);
    let bd = up(&b, &bias);
    let n = up(&b, &[real as u32, 0, 0]);
    for half in [false, true] {
        for (act, with_bias) in [(0, true), (1, true), (2, true), (1, false)] {
            let mut y0 = rng.vec(real * width, 3.0);
            y0.resize(launched * width, SENTINEL);
            let (y, ys) = input(&b, &y0, half);
            let pb = if with_bias { ptr(&b, &bd) } else { 0 };
            let (py, pn, kind, w) = (y.ptr(&b), ptr(&b, &n), 0i32, width as i32);
            let mut l = b.stream.launch_builder(&b.k.bias_act[usize::from(half)]);
            l.arg(&py).arg(&pb).arg(&pn).arg(&kind).arg(&w).arg(&act);
            // SAFETY: the buffers match the sizes the kernel reads and writes.
            unsafe { l.launch(rows(launched, 256)) }.unwrap();
            let want: Vec<f32> = ys
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let x = f64::from(v) + if with_bias { f64::from(bias[i % width]) } else { 0.0 };
                    (match act {
                        1 => gelu(x),
                        2 => x.max(0.0),
                        _ => x,
                    }) as f32
                })
                .collect();
            let what = format!("bias act {act} half {half}");
            check(&what, &y.read(&b), &want, real * width, tol(half, 1e-5));
        }
    }
}

/// q and k of every row of `[q | k | v]` rotated in f64 from tables `[pos, 32]`.
fn rope_naive(
    x: &[f32],
    heads: usize,
    rows: usize,
    pos: &[u32],
    cos: &[f32],
    sin: &[f32],
) -> Vec<f64> {
    let d = heads * 64;
    let mut out: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    for (r, &p) in pos.iter().enumerate().take(rows) {
        for h in 0..2 * heads {
            let at = r * 3 * d + h * 64;
            for i in 0..32 {
                let (a, bv) = (out[at + i], out[at + i + 32]);
                let t = p as usize * 32 + i;
                let (c, s) = (f64::from(cos[t]), f64::from(sin[t]));
                out[at + i] = a * c - bv * s;
                out[at + i + 32] = bv * c + a * s;
            }
        }
    }
    out
}

#[test]
fn rope_matches_naive() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(13);
    let heads = 3;
    let lay = Layout::new(&[9, 1, 20], 32);
    let (cos, sin) = rope_tables(160_000.0, lay.launched);
    let (cd, sd, pd) = (up(&b, &cos), up(&b, &sin), up(&b, &lay.pos));
    let n = up(&b, &[lay.tokens as u32, lay.seqs() as u32, 0]);
    let width = 3 * heads * 64;
    for half in [false, true] {
        let mut x0 = rng.vec(lay.tokens * width, 2.0);
        x0.resize(lay.launched * width, SENTINEL);
        let (x, xs) = input(&b, &x0, half);
        let (px, pc, ps, pp, pn, h) =
            (x.ptr(&b), ptr(&b, &cd), ptr(&b, &sd), ptr(&b, &pd), ptr(&b, &n), heads as i32);
        let mut l = b.stream.launch_builder(&b.k.rope[usize::from(half)]);
        l.arg(&px).arg(&pc).arg(&ps).arg(&pp).arg(&pn).arg(&h);
        // SAFETY: the buffers match the sizes the kernel reads and writes.
        unsafe { l.launch(rows(lay.launched, 256)) }.unwrap();
        let want: Vec<f32> = rope_naive(&xs, heads, lay.tokens, &lay.pos, &cos, &sin)
            .into_iter()
            .map(|v| v as f32)
            .collect();
        check(
            &format!("rope half {half}"),
            &x.read(&b),
            &want,
            lay.tokens * width,
            tol(half, 1e-6),
        );
    }
}

/// Dense attention in f64 with the mask built as a matrix: row i sees key j when both are in the
/// same sequence and, with a window, at most `window` apart.
fn attention_naive(qkv: &[f64], lay: &Layout, heads: usize, window: i32) -> Vec<f32> {
    let (t, d) = (lay.tokens, heads * 64);
    let mask: Vec<Vec<bool>> = (0..t)
        .map(|i| {
            (0..t)
                .map(|j| {
                    lay.seq[i] == lay.seq[j] && (window < 0 || i.abs_diff(j) <= window as usize)
                })
                .collect()
        })
        .collect();
    let mut out = vec![0f32; lay.launched * d];
    for h in 0..heads {
        let q = |i: usize, c: usize| qkv[i * 3 * d + h * 64 + c];
        let k = |j: usize, c: usize| qkv[j * 3 * d + d + h * 64 + c];
        let v = |j: usize, c: usize| qkv[j * 3 * d + 2 * d + h * 64 + c];
        for i in 0..t {
            let s: Vec<f64> = (0..t)
                .map(|j| {
                    if mask[i][j] {
                        (0..64).map(|c| q(i, c) * k(j, c)).sum::<f64>() / 8.0
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|&x| (x - m).exp()).collect();
            let sum: f64 = p.iter().sum();
            for c in 0..64 {
                let o: f64 = (0..t).map(|j| p[j] * v(j, c)).sum();
                out[i * d + h * 64 + c] = (o / sum) as f32;
            }
        }
    }
    out
}

#[test]
fn attention_matches_dense_masked() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(17);
    let heads = 2;
    // An empty sequence, a single row, and sequences that cross key tiles and query blocks.
    let lay = Layout::new(&[37, 0, 1, 70, 5], 128);
    let (cos, sin) = rope_tables(10_000.0, lay.launched);
    let (cd, sd, pd) = (up(&b, &cos), up(&b, &sin), up(&b, &lay.pos));
    let (sq, cu) = (up(&b, &lay.seq), up(&b, &lay.cu));
    let n = up(&b, &[lay.tokens as u32, lay.seqs() as u32, 0]);
    let (width, d) = (3 * heads * 64, heads * 64);
    let mut x0 = rng.vec(lay.tokens * width, 2.0);
    x0.resize(lay.launched * width, 0.0);
    let mut worst = [0f64; 4];
    for (variant, worst) in worst.iter_mut().enumerate() {
        let (half_in, half_out) = (variant >= 2, variant % 2 == 1);
        let (x, xs) = input(&b, &x0, half_in);
        for (window, rope) in [(-1, false), (16, false), (-1, true), (16, true)] {
            let out = output(&b, lay.launched * d, half_out);
            let (pc, ps) = if rope { (ptr(&b, &cd), ptr(&b, &sd)) } else { (0, 0) };
            let (po, px, pp) = (out.ptr(&b), x.ptr(&b), ptr(&b, &pd));
            let (psq, pcu, pn, h) = (ptr(&b, &sq), ptr(&b, &cu), ptr(&b, &n), heads as i32);
            let mut l = b.stream.launch_builder(&b.k.attention[variant]);
            l.arg(&po).arg(&px).arg(&pc).arg(&ps).arg(&pp);
            l.arg(&psq).arg(&pcu).arg(&pn).arg(&h).arg(&window);
            let cfg = LaunchConfig {
                grid_dim: (lay.launched.div_ceil(ATT_Q) as u32, heads as u32, 1),
                block_dim: (32 * ATT_W, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the buffers match the sizes the kernel reads and writes.
            unsafe { l.launch(cfg) }.unwrap();
            let qkv = if rope {
                rope_naive(&xs, heads, lay.tokens, &lay.pos, &cos, &sin)
            } else {
                xs.iter().map(|&v| f64::from(v)).collect()
            };
            let want = attention_naive(&qkv, &lay, heads, window);
            let what = format!("attention {variant} window {window} rope {rope}");
            let e = check(&what, &out.read(&b), &want, lay.tokens * d, tol(half_out, 1e-4));
            *worst = worst.max(e);
        }
    }
    let [a, c, e, f] = worst;
    eprintln!(
        "attention max error: {a:.2e} f32 to f32, {c:.2e} f32 to f16, {e:.2e} f16 to f32, {f:.2e} f16 to f16"
    );
}

#[test]
fn geglu_matches_naive() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(19);
    let (inter, real, launched) = (1500, 6, 8);
    let n = up(&b, &[real as u32, 0, 0]);
    let mut x0 = rng.vec(real * 2 * inter, 3.0);
    x0.resize(launched * 2 * inter, 0.0);
    for variant in 0..4 {
        let (half_in, half_out) = (variant >= 2, variant % 2 == 1);
        let (x, xs) = input(&b, &x0, half_in);
        let out = output(&b, launched * inter, half_out);
        let (po, px, pn, i) = (out.ptr(&b), x.ptr(&b), ptr(&b, &n), inter as i32);
        let mut l = b.stream.launch_builder(&b.k.geglu[variant]);
        l.arg(&po).arg(&px).arg(&pn).arg(&i);
        // SAFETY: the buffers match the sizes the kernel reads and writes.
        unsafe { l.launch(rows(launched, 256)) }.unwrap();
        let mut want = vec![0f32; launched * inter];
        for r in 0..real {
            for c in 0..inter {
                let (g, u) = (xs[r * 2 * inter + c], xs[r * 2 * inter + inter + c]);
                want[r * inter + c] = (gelu(f64::from(g)) * f64::from(u)) as f32;
            }
        }
        check(&format!("geglu {variant}"), &out.read(&b), &want, real * inter, tol(half_out, 1e-5));
    }
}

#[test]
fn add_type_adds_the_sequence_type_row() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(23);
    let d = 160;
    let lay = Layout::new(&[4, 0, 7], 16);
    let table = rng.vec(3 * d, 1.0);
    let qtype = [2u32, 0, 1];
    let mut h0 = rng.vec(lay.tokens * d, 1.0);
    h0.resize(lay.launched * d, SENTINEL);
    let (h, t, sq, qt) = (up(&b, &h0), up(&b, &table), up(&b, &lay.seq), up(&b, &qtype));
    let n = up(&b, &[lay.tokens as u32, lay.seqs() as u32, 0]);
    let (ph, pt, psq, pqt, pn, di) =
        (ptr(&b, &h), ptr(&b, &t), ptr(&b, &sq), ptr(&b, &qt), ptr(&b, &n), d as i32);
    let mut l = b.stream.launch_builder(&b.k.add_type);
    l.arg(&ph).arg(&pt).arg(&psq).arg(&pqt).arg(&pn).arg(&di);
    // SAFETY: the buffers match the sizes the kernel reads and writes.
    unsafe { l.launch(rows(lay.launched, 256)) }.unwrap();
    let want: Vec<f32> = h0
        .iter()
        .enumerate()
        .map(|(i, &v)| v + table[qtype[lay.seq[i / d] as usize] as usize * d + i % d])
        .collect();
    check("add type", &Buf::F32(h).read(&b), &want, lay.tokens * d, 0.0);
}

#[test]
fn gather_copies_marker_rows() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(29);
    let (d, tokens, real, launched) = (96, 40, 7, 8);
    let h0 = rng.vec(tokens * d, 2.0);
    let mut idx: Vec<u32> = (0..launched).map(|_| rng.below(tokens) as u32).collect();
    idx[real..].fill(u32::MAX);
    let n = up(&b, &[tokens as u32, 1, real as u32]);
    let ri = up(&b, &idx);
    for half in [false, true] {
        let (h, hs) = input(&b, &h0, half);
        let out = output(&b, launched * d, half);
        let (po, ph, pr, pn, di) = (out.ptr(&b), h.ptr(&b), ptr(&b, &ri), ptr(&b, &n), d as i32);
        let mut l = b.stream.launch_builder(&b.k.gather[usize::from(half)]);
        l.arg(&po).arg(&ph).arg(&pr).arg(&pn).arg(&di);
        // SAFETY: the buffers match the sizes the kernel reads and writes.
        unsafe { l.launch(rows(launched, 256)) }.unwrap();
        let mut want: Vec<f32> =
            (0..real).flat_map(|m| hs[idx[m] as usize * d..][..d].to_vec()).collect();
        want.resize(launched * d, 0.0);
        check(&format!("gather half {half}"), &out.read(&b), &want, real * d, 0.0);
    }
}

#[test]
fn act_features_match_naive() {
    let Some(b) = gpu() else { return };
    let mut rng = Rng(31);
    let d = 64;
    // Sequences with 3, 0, 1, 2 and 5 markers, the second of them empty of tokens too.
    let lay = Layout::new(&[6, 0, 3, 4, 9], 32);
    let marks = [3usize, 0, 1, 2, 5];
    let mut mcu = vec![0u32];
    for m in marks {
        mcu.push(mcu.last().unwrap() + m as u32);
    }
    let (seqs, launched_seqs) = (lay.seqs(), 8);
    let h0 = rng.vec(lay.launched * d, 1.0);
    let logits = rng.vec(*mcu.last().unwrap() as usize, 4.0);
    let (h, lg, cu, mc) = (up(&b, &h0), up(&b, &logits), up(&b, &lay.cu), up(&b, &mcu));
    let n = up(&b, &[lay.tokens as u32, seqs as u32, logits.len() as u32]);
    let out = output(&b, launched_seqs * (d + 4), false);
    let (po, ph, pl, pc, pm, pn, di) =
        (out.ptr(&b), ptr(&b, &h), ptr(&b, &lg), ptr(&b, &cu), ptr(&b, &mc), ptr(&b, &n), d as i32);
    let mut l = b.stream.launch_builder(&b.k.act_features);
    l.arg(&po).arg(&ph).arg(&pl).arg(&pc).arg(&pm).arg(&pn).arg(&di);
    // SAFETY: the buffers match the sizes the kernel reads and writes.
    unsafe { l.launch(rows(launched_seqs, 256)) }.unwrap();
    let mut want = vec![0f32; launched_seqs * (d + 4)];
    for s in 0..seqs {
        let o = &mut want[s * (d + 4)..][..d + 4];
        let (a, e) = (lay.cu[s] as usize, lay.cu[s + 1] as usize);
        if e > a {
            o[..d].copy_from_slice(&h0[a * d..][..d]);
        }
        let l = &logits[mcu[s] as usize..mcu[s + 1] as usize];
        let kf = l.len().max(2) as f64;
        o[d + 3] = (kf / 255.0) as f32;
        if l.is_empty() {
            continue;
        }
        let m = l.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
        let sum: f64 = l.iter().map(|&x| (f64::from(x) - m).exp()).sum();
        let mut q: Vec<f64> = l.iter().map(|&x| (f64::from(x) - m).exp() / sum).collect();
        let ent: f64 = q.iter().map(|&p| p * p.max(1e-9).ln()).sum();
        q.sort_by(|x, y| y.total_cmp(x));
        let top2 = q.get(1).copied().unwrap_or(0.0);
        o[d] = q[0] as f32;
        o[d + 1] = (q[0] - top2) as f32;
        o[d + 2] = (-ent / kf.ln()) as f32;
    }
    check("act features", &out.read(&b), &want, seqs * (d + 4), 1e-5);
}

#[test]
fn gemm_rows_do_not_depend_on_the_row_count() {
    use crate::WORKSPACE;
    use crate::lt::{Gemm, Ty};
    let Some(b) = gpu() else { return };
    let mut rng = Rng(47);
    let ws = b.workspace_ptr();
    // Laya's encoder shapes: attention in and out, and the two MLP halves.
    for (k, n) in [(1024, 3072), (1024, 1024), (1024, 5248), (2624, 1024)] {
        for half in [false, true] {
            let most = 1024;
            let (w, _) = input(&b, &rng.vec(n * k, 0.05), half);
            let (x, _) = input(&b, &rng.vec(most * k, 1.0), half);
            let ab = if half { Ty::F16 } else { Ty::F32 };
            let mut first: Option<Vec<u32>> = None;
            for m in [1, 3, 16, 50, 128, 256, 300, 1024] {
                let y = output(&b, m * n, half);
                let g = Gemm::new(
                    &b.lt,
                    (m, k, n),
                    ab,
                    ab,
                    false,
                    WORKSPACE,
                    (w.ptr(&b), x.ptr(&b), y.ptr(&b)),
                    0,
                )
                .unwrap();
                // SAFETY: the buffers hold the shapes the GEMM was set up for.
                unsafe { g.run(&b.lt, ws, WORKSPACE, b.stream.cu_stream().cast()) }.unwrap();
                let row: Vec<u32> = y.read(&b)[..n].iter().map(|v| v.to_bits()).collect();
                match &first {
                    None => first = Some(row),
                    Some(f) => {
                        let diff = f.iter().zip(&row).filter(|(a, b)| a != b).count();
                        assert_eq!(
                            diff, 0,
                            "k {k} n {n} half {half}: row 0 at {m} rows differs from 1 row in {diff} of {n}"
                        );
                    }
                }
            }
        }
    }
}

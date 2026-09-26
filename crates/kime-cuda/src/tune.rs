//! Which of cuBLASLt's ranked algorithms each GEMM runs.
//!
//! cuBLASLt's first choice is not always its fastest for the small row counts of one question, so
//! `tuned.txt` pins another rank per GPU and GEMM shape. Pinning by rank keeps a plan the same on
//! every start, which timing on every start would not. A run with `KIME_CUDA_TUNE=1` times every
//! rank of every GEMM in each plan it lowers, uses the fastest, and prints the table lines to add.

use std::collections::HashMap;

use cudarc::driver::{CudaStream, sys};
use kime_tensor::Result;

use crate::lt::{self, Ty};
use crate::{WORKSPACE, dev};

const TABLE: &str = include_str!("tuned.txt");

/// A GEMM shape: rows, inner size, columns, input type, output type and whether it accumulates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub(crate) dims: (usize, usize, usize),
    pub(crate) ab: Ty,
    pub(crate) c: Ty,
    pub(crate) acc: bool,
}

impl Key {
    fn line(&self, gpu: &str, pick: usize) -> String {
        let (m, k, n) = self.dims;
        let ty = |t: Ty| if t == Ty::F16 { "f16" } else { "f32" };
        let acc = if self.acc { "add" } else { "set" };
        format!("{gpu} | {m} {k} {n} | {} {} {acc} | {pick}", ty(self.ab), ty(self.c))
    }
}

/// The table's picks for one GPU.
#[derive(Debug, Default)]
pub(crate) struct Picks(HashMap<Key, usize>);

impl Picks {
    /// The lines of `tuned.txt` for the GPU named `gpu`.
    pub(crate) fn for_gpu(gpu: &str) -> Self {
        Self(TABLE.lines().filter_map(|l| parse(l, gpu)).collect())
    }

    /// The rank to run for `key`, 0 when the table has none.
    pub(crate) fn get(&self, key: &Key) -> usize {
        self.0.get(key).copied().unwrap_or(0)
    }
}

fn parse(line: &str, gpu: &str) -> Option<(Key, usize)> {
    let line = line.split('#').next().unwrap_or("").trim();
    if line.is_empty() {
        return None;
    }
    let f: Vec<&str> = line.split('|').map(str::trim).collect();
    let [name, dims, types, pick] = f[..] else { return None };
    if name != gpu {
        return None;
    }
    let d: Vec<usize> =
        dims.split_whitespace().map(str::parse).collect::<std::result::Result<_, _>>().ok()?;
    let t: Vec<&str> = types.split_whitespace().collect();
    let ty = |s: &str| match s {
        "f16" => Some(Ty::F16),
        "f32" => Some(Ty::F32),
        _ => None,
    };
    let (&[m, k, n], &[ab, c, acc]) = (&d[..], &t[..]) else { return None };
    let acc = match acc {
        "add" => true,
        "set" => false,
        _ => return None,
    };
    let key = Key { dims: (m, k, n), ab: ty(ab)?, c: ty(c)?, acc };
    Some((key, pick.parse().ok()?))
}

/// Whether this run tunes, from `KIME_CUDA_TUNE`.
pub(crate) fn enabled() -> bool {
    std::env::var_os("KIME_CUDA_TUNE").is_some_and(|v| v != "0")
}

/// Times every rank of every GEMM, in plan order so each reads its weights cold as a real run
/// does, and sets each shape to its fastest. A rank has to beat cuBLASLt's first choice by 2% to
/// replace it, so noise does not fill the table. Prints a table line for every shape with the
/// time of every rank, since a shape runs one rank in every bucket and the line to keep is the
/// rank that does best across them.
///
/// The GEMMs read and write whatever the arena holds, which no run depends on: a run writes every
/// value before it reads it.
pub(crate) fn tune(
    h: &lt::Handle,
    s: &CudaStream,
    workspace: u64,
    gemms: &mut [lt::Gemm],
    keys: &[Key],
    gpu: &str,
) -> Result<()> {
    let ctx = s.context();
    let timed = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
    let marks =
        (0..=gemms.len()).map(|_| ctx.new_event(timed)).collect::<std::result::Result<Vec<_>, _>>();
    let marks = marks.map_err(dev)?;
    let most = gemms.iter().map(lt::Gemm::candidates).max().unwrap_or(0);
    // The first pass only loads each algorithm's kernels, which CUDA does on first use. After it,
    // each GEMM and rank keeps its fastest of five.
    let mut fastest = vec![vec![f32::INFINITY; most]; gemms.len()];
    for pass in 0..6 {
        for rank in 0..most {
            marks[0].record(s).map_err(dev)?;
            let mut ran = vec![false; gemms.len()];
            for (i, g) in gemms.iter_mut().enumerate() {
                if rank < g.candidates() {
                    let keep = g.pick;
                    g.pick = rank;
                    // SAFETY: the GEMM's buffers were set up for its shape when it was lowered,
                    // and the workspace is used by nothing else on the stream.
                    let run = unsafe { g.run(h, workspace, WORKSPACE, s.cu_stream().cast()) };
                    ran[i] = run.is_ok();
                    g.pick = keep;
                }
                marks[i + 1].record(s).map_err(dev)?;
            }
            s.synchronize().map_err(dev)?;
            if pass == 0 {
                continue;
            }
            for (i, f) in fastest.iter_mut().enumerate() {
                if ran[i] {
                    f[rank] = f[rank].min(marks[i].elapsed_ms(&marks[i + 1]).map_err(dev)?);
                }
            }
        }
    }
    let mut total: HashMap<Key, Vec<f32>> = HashMap::new();
    for (key, f) in keys.iter().zip(&fastest) {
        let t = total.entry(*key).or_insert_with(|| vec![0.0; most]);
        for (t, f) in t.iter_mut().zip(f) {
            *t += f;
        }
    }
    let mut best: HashMap<Key, usize> = HashMap::new();
    for (key, t) in &total {
        let (mut pick, mut at) = (0, t[0]);
        for (rank, &x) in t.iter().enumerate().skip(1) {
            if x < at && x < 0.98 * t[0] {
                (pick, at) = (rank, x);
            }
        }
        best.insert(*key, pick);
        let calls = keys.iter().filter(|k| *k == key).count() as f32;
        let each: Vec<String> = t.iter().map(|x| format!("{:.2}", 1e3 * x / calls)).collect();
        println!(
            "{}    # {} rows, us by rank: {}",
            key.line(gpu, pick),
            gemms[keys.iter().position(|k| k == key).unwrap_or(0)].dims.0,
            each.join(" ")
        );
    }
    for (g, key) in gemms.iter_mut().zip(keys) {
        g.pick = best[key];
    }
    Ok(())
}

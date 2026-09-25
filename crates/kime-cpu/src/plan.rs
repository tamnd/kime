//! The CPU backend: graphs lowered to a list of steps over one arena, run on a persistent pool.
//!
//! Lowering resolves every value to an arena offset, every weight to its converted tensor and every
//! RoPE base to its table, and checks every shape, so running a step is a match and a call. The
//! arena, the per worker attention scratch and the per batch index tables are sized for the bucket
//! when the plan is built, and nothing on the run path allocates.
//!
//! Each step runs on the rows the batch has rather than the bucket's padded count. Padding only
//! matters to a backend that captures a fixed shape, and on the CPU it would be wasted work. The
//! kernels are the reference ones, so a plan gives the same bits as [`Compat`](crate::Compat).
//!
//! A backend made [`with_int8`](CpuBackend::with_int8) runs the GEMMs over token rows in INT8
//! instead, which is every GEMM of the encoder and the decision head. The scorer and the act head
//! run on a row per option or per question and stay in FP32, as spec/10-cpu.md has it.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::time::Instant;

use kime_tensor::plan::{Epilogue, Graph, Op, Rows, Val, layout};
use kime_tensor::{Backend, Batch, Bucket, Caps, Error, HostTensor, Outputs, Result};

use crate::attention::{self, HEAD, QB};
use crate::gemm::{self, Gemm};
use crate::ops::{Rope, geglu, layer_norm};
use crate::par::{self, Shared};
use crate::pool::Pool;
use crate::qgemm::{self, QGemm, QMatrix};

/// A weight converted to f32.
#[derive(Debug)]
pub struct Tensor {
    /// Shape.
    pub shape: Vec<usize>,
    /// Row major values, empty for a weight only GEMMs read.
    pub data: Vec<f32>,
    /// The values as [`gemm::pack`] lays them out, for a weight FP32 GEMMs read, and empty
    /// otherwise.
    pub packed: Vec<f32>,
    /// The values rounded to INT8, for a weight INT8 GEMMs read.
    pub quant: Option<QMatrix>,
}

/// Every weight of a checkpoint, shared by all the plans built from it.
#[derive(Debug, Clone)]
pub struct Weights(Arc<[Tensor]>);

/// The CPU backend, which owns its threads.
#[derive(Debug)]
pub struct CpuBackend {
    pool: Pool,
    int8: bool,
}

impl CpuBackend {
    /// A backend on `threads` threads, the calling thread included.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        Self { pool: Pool::new(threads.max(1)), int8: false }
    }

    /// The same backend with the GEMMs over token rows in INT8 when `on`: weights rounded per
    /// output channel at upload, activations per row as they are read, sums in i32. See
    /// [`qgemm`].
    #[must_use]
    pub fn with_int8(mut self, on: bool) -> Self {
        self.int8 = on;
        self
    }

    /// Whether the GEMMs over token rows run in INT8.
    #[must_use]
    pub fn int8(&self) -> bool {
        self.int8
    }

    /// Whether a GEMM reading `a` runs in INT8.
    fn int8_rows(&self, rows: Rows) -> bool {
        self.int8 && rows == Rows::Tokens
    }

    /// Threads.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }
}

/// A value resolved to its place in the arena.
#[derive(Debug, Clone, Copy)]
struct Loc {
    off: usize,
    rows: Rows,
    width: usize,
}

#[derive(Debug, Clone, Copy)]
enum Step {
    Embed { table: usize, out: Loc },
    LayerNorm { x: Loc, w: usize, b: Option<usize>, eps: f64, out: Loc },
    Gemm { a: Loc, w: usize, b: Option<usize>, ep: Epilogue, out: Loc },
    Gemm8 { a: Loc, w: usize, b: Option<usize>, ep: Epilogue, out: Loc },
    Rope { qkv: Loc, rope: usize },
    Attention { qkv: Loc, window: Option<usize>, out: Loc },
    GeGlu { x: Loc, out: Loc },
    AddType { h: Loc, table: usize },
    Gather { h: Loc, out: Loc },
    ActFeatures { h: Loc, logits: Loc, out: Loc },
}

impl Step {
    fn name(&self) -> &'static str {
        match self {
            Step::Embed { .. } => "embed",
            Step::LayerNorm { .. } => "layer norm",
            Step::Gemm { .. } => "gemm",
            Step::Gemm8 { .. } => "gemm int8",
            Step::Rope { .. } => "rope",
            Step::Attention { .. } => "attention",
            Step::GeGlu { .. } => "geglu",
            Step::AddType { .. } => "type embedding",
            Step::Gather { .. } => "gather markers",
            Step::ActFeatures { .. } => "act features",
        }
    }

    /// Where the step writes.
    fn out(&self) -> Loc {
        match *self {
            Step::Embed { out, .. }
            | Step::LayerNorm { out, .. }
            | Step::Gemm { out, .. }
            | Step::Gemm8 { out, .. }
            | Step::Attention { out, .. }
            | Step::GeGlu { out, .. }
            | Step::Gather { out, .. }
            | Step::ActFeatures { out, .. } => out,
            Step::Rope { qkv, .. } => qkv,
            Step::AddType { h, .. } => h,
        }
    }
}

/// What one step wrote in one run, from [`CpuPlan::dumps`].
#[derive(Debug, Clone)]
pub struct Dump {
    /// The kind of step.
    pub name: &'static str,
    /// Whether the rows are tokens, sequences or markers.
    pub rows: Rows,
    /// Values per row.
    pub width: usize,
    /// The live rows, row major.
    pub data: Vec<f32>,
}

/// One slot per worker, each touched only by its own worker.
struct PerWorker<T>(Vec<UnsafeCell<T>>);

// SAFETY: slot w is only reached through `get(w)` from the task running on worker w, and the pool
// never runs two tasks on one worker at once.
unsafe impl<T: Send> Sync for PerWorker<T> {}

impl<T> PerWorker<T> {
    /// # Safety
    ///
    /// Only the task running on `worker` may call this, and only for its own worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self, worker: usize) -> &mut T {
        // SAFETY: the caller is the only user of slot `worker` right now.
        unsafe { &mut *self.0[worker].get() }
    }
}

/// A graph lowered for one bucket.
pub struct CpuPlan {
    w: Weights,
    bucket: Bucket,
    steps: Vec<Step>,
    arena: Vec<f32>,
    ropes: Vec<Rope>,
    logits: Loc,
    act: Loc,
    /// Token starts of each sequence, then marker starts, then the sequence of each token, then
    /// the query blocks attention runs. Rebuilt per batch in place.
    cu: Vec<usize>,
    mcu: Vec<usize>,
    row_seq: Vec<u32>,
    blocks: Vec<(u32, u32)>,
    scratch: PerWorker<Vec<f32>>,
    /// Nanoseconds per step, summed over runs, when profiling.
    profile: Option<Vec<u64>>,
    /// Every step's output from the last run, when dumping.
    dumps: Option<Vec<Dump>>,
}

impl std::fmt::Debug for CpuPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuPlan")
            .field("bucket", &self.bucket)
            .field("steps", &self.steps.len())
            .field("arena", &self.arena.len())
            .finish_non_exhaustive()
    }
}

impl CpuPlan {
    /// The bucket it was built for.
    #[must_use]
    pub fn bucket(&self) -> Bucket {
        self.bucket
    }

    /// Arena size in bytes.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.arena.len() * 4
    }

    /// Starts timing every step, which costs two clock reads per step.
    pub fn profile(&mut self) {
        self.profile = Some(vec![0; self.steps.len()]);
    }

    /// Keeps a copy of what every step writes from now on, for finding the first step where two
    /// runs differ. It allocates on every run, so it is for tests and debugging only.
    pub fn dump(&mut self) {
        self.dumps = Some(Vec::new());
    }

    /// What every step wrote in the last run since [`CpuPlan::dump`], in order.
    #[must_use]
    pub fn dumps(&self) -> &[Dump] {
        self.dumps.as_deref().unwrap_or_default()
    }

    /// Time per kind of step since [`CpuPlan::profile`], in nanoseconds, largest first.
    #[must_use]
    pub fn timings(&self) -> Vec<(&'static str, u64)> {
        let mut by: Vec<(&'static str, u64)> = Vec::new();
        for (s, &ns) in self.steps.iter().zip(self.profile.iter().flatten()) {
            match by.iter_mut().find(|b| b.0 == s.name()) {
                Some(b) => b.1 += ns,
                None => by.push((s.name(), ns)),
            }
        }
        by.sort_by_key(|b| std::cmp::Reverse(b.1));
        by
    }
}

/// A pointer to an arena that steps carve disjoint slices from.
#[derive(Clone, Copy)]
struct Arena(*mut f32);

// SAFETY: the plan hands out slices of the arena only as the layout allows: values live at the
// same time never overlap, and within a step each task writes rows no other task touches.
unsafe impl Send for Arena {}
// SAFETY: as above.
unsafe impl Sync for Arena {}

impl Arena {
    /// # Safety
    ///
    /// The range must be in the arena and not written by anyone else while the slice lives.
    unsafe fn slice<'a>(self, off: usize, len: usize) -> &'a [f32] {
        // SAFETY: by the caller.
        unsafe { std::slice::from_raw_parts(self.0.add(off), len) }
    }

    /// # Safety
    ///
    /// The range must be in the arena and not touched by anyone else while the slice lives.
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut<'a>(self, off: usize, len: usize) -> &'a mut [f32] {
        // SAFETY: by the caller.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

/// Rows per task for the row wise steps.
const ROWS: usize = 16;

impl Backend for CpuBackend {
    type Weights = Weights;
    type Plan = CpuPlan;

    fn caps(&self) -> Caps {
        Caps { name: "cpu", threads: self.threads(), graphs: false, unified_memory: true }
    }

    fn weight_bytes(&self, w: &Weights) -> usize {
        w.0.iter()
            .map(|t| {
                let q = t.quant.as_ref().map_or(0, |q| q.q.len() + 4 * q.scale.len());
                4 * (t.data.len() + t.packed.len()) + q
            })
            .sum()
    }

    fn plan_bytes(&self, p: &CpuPlan) -> usize {
        p.arena_bytes()
    }

    fn upload(&self, tensors: &[HostTensor<'_>], graph: &Graph) -> Result<Weights> {
        // Weights the GEMMs read are packed or rounded once here, and kept row major only if
        // something else reads them too.
        let (mut gemm, mut other) = (vec![false; tensors.len()], vec![false; tensors.len()]);
        let mut gemm8 = vec![false; tensors.len()];
        let mark = |flags: &mut Vec<bool>, w: Option<usize>| {
            if let Some(f) = w.and_then(|w| flags.get_mut(w)) {
                *f = true;
            }
        };
        for op in &graph.ops {
            match *op {
                Op::Gemm { a, w, b, .. } => {
                    match self.int8_rows(graph.shape(a).rows) {
                        true => mark(&mut gemm8, Some(w)),
                        false => mark(&mut gemm, Some(w)),
                    }
                    mark(&mut other, b);
                }
                Op::Embed { table, .. } | Op::AddType { table, .. } => {
                    mark(&mut other, Some(table))
                }
                Op::LayerNorm { w, b, .. } => {
                    mark(&mut other, Some(w));
                    mark(&mut other, b);
                }
                Op::Rope { .. }
                | Op::Attention { .. }
                | Op::GeGlu { .. }
                | Op::GatherMarkers { .. }
                | Op::ActFeatures { .. } => {}
            }
        }
        let t = par::map(tensors.len(), self.threads(), |i| {
            let h = &tensors[i];
            let n = h.bytes.len() / h.dtype.size();
            if n != h.shape.iter().product::<usize>() {
                return Err(Error::Unsupported(format!(
                    "tensor {i} has {n} values for {:?}",
                    h.shape
                )));
            }
            let data: Vec<f32> = (0..n).map(|j| h.dtype.read_f32(h.bytes, j)).collect();
            let packed = match (gemm[i], h.shape) {
                (true, &[rows, cols]) => gemm::pack(&data, rows, cols),
                _ => Vec::new(),
            };
            let quant = match (gemm8[i], h.shape) {
                (true, &[rows, cols]) => Some(QMatrix::quantize(&data, rows, cols)),
                _ => None,
            };
            // The row major values go when every reader has its own copy.
            let copied = (!gemm[i] || !packed.is_empty()) && (!gemm8[i] || quant.is_some());
            let data = if (gemm[i] || gemm8[i]) && !other[i] && copied { Vec::new() } else { data };
            Ok(Tensor { shape: h.shape.to_vec(), data, packed, quant })
        });
        Ok(Weights(t.into_iter().collect::<Result<Vec<_>>>()?.into()))
    }

    fn lower(&self, w: &Weights, graph: &Graph, bucket: Bucket) -> Result<CpuPlan> {
        let lay = layout(graph, |r| bucket.rows(r));
        let loc = |v: Val| {
            let s = graph.shape(v);
            Loc { off: lay.offsets[v.0 as usize], rows: s.rows, width: s.width }
        };
        let bad = |m: String| Err(Error::Unsupported(m));
        let shape = |i: usize| -> Result<&[usize]> {
            match w.0.get(i) {
                Some(t) => Ok(&t.shape),
                None => Err(Error::Unsupported(format!("weight {i} is not in the checkpoint"))),
            }
        };
        let mut ropes: Vec<(u64, Rope)> = Vec::new();
        let mut scratch = attention::scratch_len(bucket.tokens);
        let mut steps = Vec::with_capacity(graph.ops.len());
        for (i, op) in graph.ops.iter().enumerate() {
            let step = match *op {
                Op::Embed { table, out } => {
                    let out = loc(out);
                    if shape(table)?.get(1) != Some(&out.width) || out.rows != Rows::Tokens {
                        return bad(format!("op {i}: embedding table does not match its output"));
                    }
                    Step::Embed { table, out }
                }
                Op::LayerNorm { x, w: nw, b, eps, out } => {
                    let (x, out) = (loc(x), loc(out));
                    let ok = shape(nw)? == [x.width]
                        && b.map_or(Ok(true), |b| shape(b).map(|s| s == [x.width]))?
                        && x.width == out.width
                        && x.rows == out.rows;
                    if !ok {
                        return bad(format!("op {i}: layer norm shapes do not match"));
                    }
                    Step::LayerNorm { x, w: nw, b, eps, out }
                }
                Op::Gemm { a, w: gw, b, epilogue, out } => {
                    let (a, out) = (loc(a), loc(out));
                    let ok = shape(gw)? == [out.width, a.width]
                        && b.map_or(Ok(true), |b| shape(b).map(|s| s == [out.width]))?
                        && a.rows == out.rows;
                    if !ok {
                        return bad(format!("op {i}: gemm shapes do not match"));
                    }
                    if self.int8_rows(a.rows) {
                        if w.0[gw].quant.is_none() {
                            return bad(format!("op {i}: gemm weight {gw} was not rounded"));
                        }
                        scratch = scratch.max(qgemm::scratch_len(a.width));
                        Step::Gemm8 { a, w: gw, b, ep: epilogue, out }
                    } else {
                        if w.0[gw].packed.is_empty() && a.width * out.width > 0 {
                            return bad(format!("op {i}: gemm weight {gw} was not packed"));
                        }
                        scratch = scratch.max(gemm::scratch_len(a.width, out.width));
                        Step::Gemm { a, w: gw, b, ep: epilogue, out }
                    }
                }
                Op::Rope { qkv, theta } => {
                    let qkv = loc(qkv);
                    if !qkv.width.is_multiple_of(3 * HEAD) || qkv.rows != Rows::Tokens {
                        return bad(format!("op {i}: rope needs token rows of 3 heads 64"));
                    }
                    let at = match ropes.iter().position(|r| r.0 == theta.to_bits()) {
                        Some(at) => at,
                        None => {
                            ropes.push((theta.to_bits(), Rope::new(theta, HEAD, bucket.tokens)));
                            ropes.len() - 1
                        }
                    };
                    Step::Rope { qkv, rope: at }
                }
                Op::Attention { qkv, window, out } => {
                    let (qkv, out) = (loc(qkv), loc(out));
                    let ok = qkv.width.is_multiple_of(3 * HEAD)
                        && out.width * 3 == qkv.width
                        && qkv.rows == Rows::Tokens
                        && out.rows == Rows::Tokens;
                    if !ok {
                        return bad(format!("op {i}: attention shapes do not match"));
                    }
                    Step::Attention { qkv, window, out }
                }
                Op::GeGlu { x, out } => {
                    let (x, out) = (loc(x), loc(out));
                    if x.width != 2 * out.width || x.rows != out.rows {
                        return bad(format!("op {i}: geglu input is not twice its output"));
                    }
                    Step::GeGlu { x, out }
                }
                Op::AddType { h, table } => {
                    let h = loc(h);
                    if shape(table)?.get(1) != Some(&h.width) || h.rows != Rows::Tokens {
                        return bad(format!("op {i}: type table does not match"));
                    }
                    Step::AddType { h, table }
                }
                Op::GatherMarkers { h, out } => {
                    let (h, out) = (loc(h), loc(out));
                    if h.width != out.width || h.rows != Rows::Tokens || out.rows != Rows::Markers {
                        return bad(format!("op {i}: gather shapes do not match"));
                    }
                    Step::Gather { h, out }
                }
                Op::ActFeatures { h, logits, out } => {
                    let (h, logits, out) = (loc(h), loc(logits), loc(out));
                    let ok = out.width == h.width + 4
                        && logits.width == 1
                        && logits.rows == Rows::Markers
                        && out.rows == Rows::Seqs;
                    if !ok {
                        return bad(format!("op {i}: act feature shapes do not match"));
                    }
                    Step::ActFeatures { h, logits, out }
                }
            };
            steps.push(step);
        }
        let (Some(logits), Some(act)) = (graph.logits, graph.act) else {
            return bad("the graph has no logits or act output".into());
        };
        let (logits, act) = (loc(logits), loc(act));
        if logits.width != 1
            || logits.rows != Rows::Markers
            || act.width != 2
            || act.rows != Rows::Seqs
        {
            return bad(
                "outputs must be one logit per marker and two act logits per sequence".into()
            );
        }
        let threads = self.threads();
        Ok(CpuPlan {
            w: w.clone(),
            bucket,
            steps,
            arena: vec![0.0; lay.len],
            ropes: ropes.into_iter().map(|r| r.1).collect(),
            logits,
            act,
            cu: Vec::with_capacity(bucket.seqs + 1),
            mcu: Vec::with_capacity(bucket.seqs + 1),
            row_seq: Vec::with_capacity(bucket.tokens),
            blocks: Vec::with_capacity(bucket.tokens.div_ceil(QB) + bucket.seqs),
            scratch: PerWorker(
                (0..threads).map(|_| UnsafeCell::new(Vec::with_capacity(scratch))).collect(),
            ),
            profile: None,
            dumps: None,
        })
    }

    fn run(&self, plan: &mut CpuPlan, batch: &Batch<'_>, out: &mut Outputs) -> Result<()> {
        let (t, s, m) = (batch.ids.len(), batch.seqs(), batch.markers.len());
        if !plan.bucket.holds(t, s, m) {
            return Err(Error::Batch(format!("batch does not fit bucket {}", plan.bucket)));
        }
        plan.cu.clear();
        plan.cu.extend(batch.cu.iter().map(|&c| c as usize));
        plan.mcu.clear();
        plan.mcu.extend(batch.mcu.iter().map(|&c| c as usize));
        plan.row_seq.clear();
        plan.blocks.clear();
        for q in 0..s {
            let (lo, hi) = (plan.cu[q], plan.cu[q + 1]);
            plan.row_seq.extend(std::iter::repeat_n(q as u32, hi - lo));
            plan.blocks.extend((lo..hi).step_by(QB).map(|q0| (q as u32, q0 as u32)));
        }
        let ctx = Ctx {
            pool: &self.pool,
            w: &plan.w.0,
            arena: Arena(plan.arena.as_mut_ptr()),
            ropes: &plan.ropes,
            batch,
            cu: &plan.cu,
            mcu: &plan.mcu,
            row_seq: &plan.row_seq,
            blocks: &plan.blocks,
            scratch: &plan.scratch,
            counts: [t, s, m],
        };
        if let Some(d) = plan.dumps.as_mut() {
            d.clear();
        }
        for (i, step) in plan.steps.iter().enumerate() {
            match plan.profile.as_mut() {
                None => ctx.step(step),
                Some(p) => {
                    let at = Instant::now();
                    ctx.step(step);
                    p[i] += u64::try_from(at.elapsed().as_nanos()).unwrap_or(u64::MAX);
                }
            }
            // Copied now, because a later step may reuse the same part of the arena.
            if let Some(d) = plan.dumps.as_mut() {
                let l = step.out();
                // SAFETY: the step is done and the next has not started, so nothing else
                // touches the arena.
                let live = unsafe { ctx.arena.slice(l.off, ctx.rows(l.rows) * l.width) };
                d.push(Dump {
                    name: step.name(),
                    rows: l.rows,
                    width: l.width,
                    data: live.to_vec(),
                });
            }
        }

        // SAFETY: the steps are done, so nothing else touches the arena.
        let logits = unsafe { ctx.arena.slice(plan.logits.off, m) };
        // SAFETY: as above.
        let act = unsafe { ctx.arena.slice(plan.act.off, 2 * s) };
        out.logits.clear();
        out.logits.extend_from_slice(logits);
        out.act.clear();
        out.act.extend_from_slice(act.as_chunks::<2>().0);
        Ok(())
    }
}

/// Everything a step needs for one batch.
struct Ctx<'a> {
    pool: &'a Pool,
    w: &'a [Tensor],
    arena: Arena,
    ropes: &'a [Rope],
    batch: &'a Batch<'a>,
    cu: &'a [usize],
    mcu: &'a [usize],
    row_seq: &'a [u32],
    blocks: &'a [(u32, u32)],
    scratch: &'a PerWorker<Vec<f32>>,
    counts: [usize; 3],
}

impl Ctx<'_> {
    fn rows(&self, r: Rows) -> usize {
        self.counts[r as usize]
    }

    fn w(&self, i: usize) -> &[f32] {
        &self.w[i].data
    }

    /// The live rows of `l`.
    ///
    /// # Safety
    ///
    /// Nobody may write `l` while the slice lives.
    unsafe fn get(&self, l: Loc) -> &[f32] {
        // SAFETY: in the arena by the layout, and by the caller.
        unsafe { self.arena.slice(l.off, self.rows(l.rows) * l.width) }
    }

    /// Runs `f(first row, rows of out)` over the live rows of `out` in blocks of [`ROWS`].
    ///
    /// # Safety
    ///
    /// Nobody else may touch `out` meanwhile, and `f` must not reach `out` any other way.
    unsafe fn rows_of(&self, out: Loc, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
        let n = self.rows(out.rows);
        let arena = self.arena;
        self.pool.run(n.div_ceil(ROWS), &|task, _| {
            let r0 = task * ROWS;
            let r1 = (r0 + ROWS).min(n);
            // SAFETY: rows r0..r1 of out belong to this task alone.
            let rows = unsafe { arena.slice_mut(out.off + r0 * out.width, (r1 - r0) * out.width) };
            f(r0, rows);
        });
    }

    fn step(&self, step: &Step) {
        // The layout gives the inputs and the output of a step disjoint arena ranges unless the
        // step is in place, and an in place step takes one slice only. That is the argument behind
        // every SAFETY comment below.
        match *step {
            Step::Embed { table, out } => {
                let (tab, ids, d) = (self.w(table), self.batch.ids, out.width);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                unsafe {
                    self.rows_of(out, &|r0, rows| {
                        for (i, row) in rows.chunks_exact_mut(d).enumerate() {
                            let id = ids[r0 + i] as usize;
                            row.copy_from_slice(&tab[id * d..(id + 1) * d]);
                        }
                    });
                }
            }
            Step::LayerNorm { x, w, b, eps, out } => {
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let x = unsafe { self.get(x) };
                let (nw, nb, d) = (self.w(w), b.map(|b| self.w(b)), out.width);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                unsafe {
                    self.rows_of(out, &|r0, rows| {
                        layer_norm(&x[r0 * d..r0 * d + rows.len()], d, nw, nb, eps, rows);
                    });
                }
            }
            Step::Gemm { a, w, b, ep, out } => {
                let rows = self.rows(a.rows);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let x = unsafe { self.get(a) };
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let y = unsafe { self.arena.slice_mut(out.off, rows * out.width) };
                let g = Gemm {
                    x,
                    m: rows,
                    k: a.width,
                    w: &self.w[w].packed,
                    n: out.width,
                    b: b.map(|b| self.w(b)),
                    ep,
                };
                let scratch = self.scratch;
                g.run(y, self.pool.threads(), |n, f| {
                    self.pool.run(n, &|i, worker| {
                        // SAFETY: the scratch is this worker's, and lowering reserved enough of it
                        // for every GEMM in the plan, so this does not allocate.
                        let s = unsafe { scratch.get(worker) };
                        s.resize(gemm::scratch_len(g.k, g.n), 0.0);
                        f(i, s);
                    });
                });
            }
            Step::Gemm8 { a, w, b, ep, out } => {
                let rows = self.rows(a.rows);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let x = unsafe { self.get(a) };
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let y = unsafe { self.arena.slice_mut(out.off, rows * out.width) };
                let q = self.w[w]
                    .quant
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("checked when lowered"));
                let g = QGemm { x, m: rows, w: q, b: b.map(|b| self.w(b)), ep };
                let scratch = self.scratch;
                g.run(y, self.pool.threads(), |n, f| {
                    self.pool.run(n, &|i, worker| {
                        // SAFETY: the scratch is this worker's, and lowering reserved enough of it
                        // for every GEMM in the plan, so this does not allocate.
                        let s = unsafe { scratch.get(worker) };
                        s.resize(qgemm::scratch_len(q.k), 0.0);
                        f(i, s);
                    });
                });
            }
            Step::Rope { qkv, rope } => {
                let (rope, d, cu, seq) = (&self.ropes[rope], qkv.width / 3, self.cu, self.row_seq);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                unsafe {
                    self.rows_of(qkv, &|r0, rows| {
                        for (i, row) in rows.chunks_exact_mut(qkv.width).enumerate() {
                            let r = r0 + i;
                            let pos = r - cu[seq[r] as usize];
                            for head in row[..2 * d].as_chunks_mut::<HEAD>().0 {
                                rope.apply(head, pos);
                            }
                        }
                    });
                }
            }
            Step::Attention { qkv, window, out } => {
                let heads = out.width / HEAD;
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let x = unsafe { self.get(qkv) };
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let y = unsafe { self.arena.slice_mut(out.off, self.rows(out.rows) * out.width) };
                let shared = Shared::new(y);
                let (blocks, cu, scratch) = (self.blocks, self.cu, self.scratch);
                self.pool.run(blocks.len() * heads, &|task, worker| {
                    let (s, q0) = blocks[task / heads];
                    let (s, q0, h) = (s as usize, q0 as usize, task % heads);
                    // SAFETY: the scratch is this worker's, and this task owns rows q0 to q0 + QB
                    // of head h.
                    unsafe {
                        let p = scratch.get(worker);
                        // Growing it here would allocate on the warm path.
                        debug_assert!(p.capacity() >= attention::scratch_len(cu[s + 1] - cu[s]));
                        attention::block(x, heads, (cu[s], cu[s + 1]), q0, h, window, p, &shared);
                    }
                });
            }
            Step::GeGlu { x, out } => {
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let (x, d) = (unsafe { self.get(x) }, out.width);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                unsafe {
                    self.rows_of(out, &|r0, rows| {
                        geglu(&x[2 * r0 * d..2 * (r0 * d + rows.len())], d, rows);
                    });
                }
            }
            Step::AddType { h, table } => {
                let (tab, d, seq, qt) = (self.w(table), h.width, self.row_seq, self.batch.qtype);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                unsafe {
                    self.rows_of(h, &|r0, rows| {
                        for (i, row) in rows.chunks_exact_mut(d).enumerate() {
                            let q = usize::from(qt[seq[r0 + i] as usize]);
                            row.iter_mut().zip(&tab[q * d..(q + 1) * d]).for_each(|(a, b)| *a += b);
                        }
                    });
                }
            }
            Step::Gather { h, out } => {
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let (x, d) = (unsafe { self.get(h) }, h.width);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let y = unsafe { self.arena.slice_mut(out.off, self.rows(out.rows) * d) };
                let mut at = 0;
                for s in 0..self.cu.len() - 1 {
                    for &p in &self.batch.markers[self.mcu[s]..self.mcu[s + 1]] {
                        let r = self.cu[s] + p as usize;
                        y[at * d..(at + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
                        at += 1;
                    }
                }
            }
            Step::ActFeatures { h, logits, out } => {
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let (x, d) = (unsafe { self.get(h) }, h.width);
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let l = unsafe { self.get(logits) };
                // SAFETY: the layout keeps the inputs and the output of a step apart.
                let y = unsafe { self.arena.slice_mut(out.off, self.rows(out.rows) * out.width) };
                for (s, row) in y.chunks_exact_mut(out.width).enumerate() {
                    let (lo, hi) = (self.cu[s], self.cu[s + 1]);
                    if hi > lo {
                        row[..d].copy_from_slice(&x[lo * d..(lo + 1) * d]);
                    } else {
                        row[..d].fill(0.0);
                    }
                    row[d..].copy_from_slice(&act_features(&l[self.mcu[s]..self.mcu[s + 1]]));
                }
            }
        }
    }
}

/// `[top1, top1 - top2, entropy / ln k, k / 255]` over the softmax of the logits, with `k` at
/// least 2, the same arithmetic as [`crate::compat::act_features`] without its buffers.
fn act_features(logits: &[f32]) -> [f32; 4] {
    let kf = logits.len().max(2) as f32;
    if logits.is_empty() {
        return [0.0, 0.0, 0.0, kf / 255.0];
    }
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - mx).exp()).sum();
    let (mut top1, mut top2, mut ent) = (f32::NEG_INFINITY, 0f32, 0f32);
    let mut first = true;
    for &l in logits {
        let q = (l - mx).exp() / sum;
        ent += q * q.max(1e-9).ln();
        if q > top1 || first {
            if !first {
                top2 = top1;
            }
            top1 = q;
            first = false;
        } else if q > top2 {
            top2 = q;
        }
    }
    let ent = -ent / kf.ln();
    [top1, top1 - top2, ent, kf / 255.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_features_match_the_reference() {
        let cases: [&[f32]; 6] =
            [&[], &[0.3], &[1.0, 1.0], &[2.0, -1.0, 2.0], &[5.0, 0.1, -3.0, 4.9], &[-1e3, 0.0]];
        for l in cases {
            assert_eq!(
                act_features(l).map(f32::to_bits),
                crate::compat::act_features(l).map(f32::to_bits),
                "{l:?}"
            );
        }
    }
}

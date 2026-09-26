//! Graphs lowered for the GPU: every value gets an element type and a place in one device arena,
//! every GEMM its cuBLASLt setup, and a run is a copy of the batch's index tables, a fixed list of
//! launches and a copy of the outputs back. Lowering captures all three as one CUDA graph, with both
//! copies going through page locked host buffers, so a run is one graph launch and one wait.

use std::sync::OnceLock;

use cudarc::driver::{
    CudaGraph, CudaSlice, DevicePtr, LaunchConfig, PinnedHostSlice, PushKernelArg, result, sys,
};
use kime_tensor::plan::{Epilogue, Graph, Op, Rows, Val, layout};
use kime_tensor::{Backend, Batch, Bucket, Caps, Error, HostTensor, Outputs, Result};

use crate::lt::{self, Ty};
use crate::tune::{self, Key};
use crate::{CudaBackend, Precision, WORKSPACE, dev};

/// Width of one attention head.
const HEAD: usize = 64;

/// Query rows and warps per attention block, `ATT_Q` and `ATT_W` in the kernels.
pub(crate) const ATT_Q: usize = 16;
pub(crate) const ATT_W: u32 = 8;

/// Rows per layer norm block, one warp each, `LN_ROWS` in the kernels.
pub(crate) const LN_ROWS: usize = 4;

/// A weight on the device in FP32, with an FP16 copy made the first time a plan needs one.
struct Tensor {
    shape: Vec<usize>,
    f32: CudaSlice<f32>,
    f16: OnceLock<CudaSlice<u16>>,
}

/// Every weight of a checkpoint on one GPU.
pub struct Weights(Vec<Tensor>);

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights").field("tensors", &self.0.len()).finish_non_exhaustive()
    }
}

/// A value resolved to its type and place.
#[derive(Debug, Clone, Copy)]
struct Loc {
    ptr: u64,
    rows: Rows,
    width: usize,
    ty: Ty,
}

impl Loc {
    fn half(self) -> bool {
        self.ty == Ty::F16
    }
}

fn kind(r: Rows) -> i32 {
    match r {
        Rows::Tokens => 0,
        Rows::Seqs => 1,
        Rows::Markers => 2,
    }
}

#[derive(Debug, Clone, Copy)]
enum Step {
    Embed {
        table: u64,
        out: Loc,
    },
    LayerNorm {
        x: Loc,
        w: u64,
        b: u64,
        eps: f32,
        out: Loc,
    },
    /// `len` FP32 values to FP16 in the staging buffer, for a GEMM with an FP32 input and an FP16
    /// output.
    ToHalf {
        x: u64,
        len: u64,
    },
    /// A GEMM, then the bias and activation when there are any.
    Gemm {
        gemm: usize,
        b: u64,
        act: i32,
        out: Loc,
    },
    Rope {
        qkv: Loc,
        cos: u64,
        sin: u64,
    },
    /// Attention, with rope applied to q and k as they are loaded when the tables are not null.
    Attention {
        qkv: Loc,
        cos: u64,
        sin: u64,
        window: i32,
        out: Loc,
    },
    GeGlu {
        x: Loc,
        out: Loc,
    },
    AddType {
        h: Loc,
        table: u64,
    },
    Gather {
        h: Loc,
        out: Loc,
    },
    ActFeatures {
        h: Loc,
        logits: Loc,
        out: Loc,
    },
}

impl Step {
    fn name(&self) -> &'static str {
        match self {
            Step::Embed { .. } => "embed",
            Step::LayerNorm { .. } => "layer norm",
            Step::ToHalf { .. } => "to half",
            Step::Gemm { .. } => "gemm",
            Step::Rope { .. } => "rope",
            Step::Attention { .. } => "attention",
            Step::GeGlu { .. } => "geglu",
            Step::AddType { .. } => "type embedding",
            Step::Gather { .. } => "gather markers",
            Step::ActFeatures { .. } => "act features",
        }
    }
}

/// Where each index table sits in the staging buffer, in u32 elements.
#[derive(Debug, Clone, Copy)]
struct Index {
    ids: usize,
    pos: usize,
    seq: usize,
    cu: usize,
    mcu: usize,
    mrow: usize,
    qtype: usize,
    len: usize,
}

impl Index {
    fn new(b: Bucket) -> Self {
        let ids = 4;
        let pos = ids + b.tokens;
        let seq = pos + b.tokens;
        let cu = seq + b.tokens;
        let mcu = cu + b.seqs + 1;
        let mrow = mcu + b.seqs + 1;
        let qtype = mrow + b.markers;
        Self { ids, pos, seq, cu, mcu, mrow, qtype, len: qtype + b.seqs }
    }
}

/// Device addresses of the batch's count buffer and index tables.
#[derive(Debug, Clone, Copy)]
struct Ptrs {
    n: u64,
    ids: u64,
    pos: u64,
    seq: u64,
    cu: u64,
    mcu: u64,
    mrow: u64,
    qtype: u64,
}

impl Ptrs {
    fn new(base: u64, at: Index) -> Self {
        let a = |o: usize| base + 4 * o as u64;
        Self {
            n: base,
            ids: a(at.ids),
            pos: a(at.pos),
            seq: a(at.seq),
            cu: a(at.cu),
            mcu: a(at.mcu),
            mrow: a(at.mrow),
            qtype: a(at.qtype),
        }
    }
}

/// A graph lowered for one bucket on one GPU.
pub struct CudaPlan {
    bucket: Bucket,
    steps: Vec<Step>,
    gemms: Vec<lt::Gemm>,
    arena: CudaSlice<u8>,
    /// FP16 copies of FP32 GEMM inputs.
    _stage: CudaSlice<u16>,
    stage: u64,
    /// Held so the rope tables the steps point at stay alive.
    _ropes: Vec<(u64, CudaSlice<f32>, CudaSlice<f32>)>,
    _index: CudaSlice<u32>,
    /// The index tables on the host, page locked so the copy in the graph is asynchronous.
    index_host: PinnedHostSlice<u32>,
    at: Index,
    ptrs: Ptrs,
    logits: Loc,
    act: Loc,
    /// The bucket's logits then its act outputs, page locked.
    host_out: PinnedHostSlice<f32>,
    graph: Option<Captured>,
    profile: Option<Vec<u64>>,
    /// Runs timed since profiling started.
    profiled: u64,
}

/// A captured run.
struct Captured(CudaGraph);

// SAFETY: a CUDA graph exec may be launched from any thread as long as calls on it are serialized.
// The plan is only used through `&mut CudaPlan`, so they are.
unsafe impl Send for Captured {}

impl std::fmt::Debug for CudaPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPlan")
            .field("bucket", &self.bucket)
            .field("steps", &self.steps.len())
            .field("arena", &self.arena.len())
            .finish_non_exhaustive()
    }
}

impl CudaPlan {
    /// The bucket it was built for.
    #[must_use]
    pub fn bucket(&self) -> Bucket {
        self.bucket
    }

    /// Arena size in bytes.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.arena.len()
    }

    /// Times every step from now on. Each step then waits for the GPU and the graph is not used,
    /// so this is for finding where the time goes, not for measuring the total.
    pub fn profile(&mut self) {
        self.profile = Some(vec![0; self.steps.len()]);
        self.profiled = 0;
    }

    /// Time per GEMM shape since [`CudaPlan::profile`]: rows, inner size, columns, calls over all
    /// timed runs and nanoseconds, largest first.
    #[must_use]
    pub fn gemm_timings(&self) -> Vec<((usize, usize, usize), u64, u64)> {
        let mut by: Vec<((usize, usize, usize), u64, u64)> = Vec::new();
        for (s, &ns) in self.steps.iter().zip(self.profile.iter().flatten()) {
            if let Step::Gemm { gemm, .. } = *s {
                let d = self.gemms[gemm].dims;
                match by.iter_mut().find(|b| b.0 == d) {
                    Some(b) => {
                        b.1 += self.profiled;
                        b.2 += ns;
                    }
                    None => by.push((d, self.profiled, ns)),
                }
            }
        }
        by.sort_by_key(|b| std::cmp::Reverse(b.2));
        by
    }

    /// Time per kind of step since [`CudaPlan::profile`], in nanoseconds, largest first.
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

/// The element type of every value.
///
/// With [`Precision::F32`] that is FP32 throughout. With [`Precision::F16`] it is FP16 unless the
/// value is the residual stream (written by the embedding or accumulated into), the type
/// embedding's target, the act features, an output, attention's qkv, or computed from an FP32 value
/// by a gather or by a GEMM whose output is not an attention or GeGLU output. The qkv stays FP32
/// because ModernBERT's attention scores are large enough that FP16 rounding of q and k moves the
/// softmax by more than the probability bound. A GEMM with an FP32 input and an FP16 output
/// converts its input first.
fn types(graph: &Graph, precision: Precision) -> Vec<Ty> {
    let n = graph.vals.len();
    if precision == Precision::F32 {
        return vec![Ty::F32; n];
    }
    let mut ty = vec![Ty::F16; n];
    let mut half = vec![false; n];
    for op in &graph.ops {
        match *op {
            Op::Embed { out, .. } | Op::ActFeatures { out, .. } => ty[out.0 as usize] = Ty::F32,
            Op::Gemm { epilogue: Epilogue::Accumulate, out, .. } => ty[out.0 as usize] = Ty::F32,
            Op::AddType { h, .. } => ty[h.0 as usize] = Ty::F32,
            Op::Attention { qkv, out, .. } => {
                ty[qkv.0 as usize] = Ty::F32;
                half[out.0 as usize] = true;
            }
            Op::GeGlu { out, .. } => half[out.0 as usize] = true,
            _ => {}
        }
    }
    for v in [graph.logits, graph.act].into_iter().flatten() {
        ty[v.0 as usize] = Ty::F32;
    }
    loop {
        let mut changed = false;
        for op in &graph.ops {
            let (a, out) = match *op {
                Op::Gemm { a, out, .. } => (a, out),
                Op::GatherMarkers { h, out } => (h, out),
                _ => continue,
            };
            let o = out.0 as usize;
            if ty[a.0 as usize] == Ty::F32 && ty[o] == Ty::F16 && !half[o] {
                ty[o] = Ty::F32;
                changed = true;
            }
        }
        if !changed {
            return ty;
        }
    }
}

/// Grid for one block per row.
fn rows(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

impl CudaBackend {
    fn tensor<'w>(&self, w: &'w Weights, i: usize) -> Result<&'w Tensor> {
        w.0.get(i).ok_or_else(|| Error::Unsupported(format!("weight {i} is not in the checkpoint")))
    }

    fn ptr32(&self, w: &Weights, i: usize) -> Result<u64> {
        Ok(self.tensor(w, i)?.f32.device_ptr(&self.stream).0)
    }

    /// The FP16 copy of weight `i`, made on first use.
    fn ptr16(&self, w: &Weights, i: usize) -> Result<u64> {
        let t = self.tensor(w, i)?;
        if let Some(h) = t.f16.get() {
            return Ok(h.device_ptr(&self.stream).0);
        }
        let len = t.f32.len();
        // SAFETY: every element is written by the kernel below before anything reads it.
        let mut h = unsafe { self.stream.alloc::<u16>(len) }.map_err(dev)?;
        let blocks = len.div_ceil(256).min(65_535) as u32;
        let cfg =
            LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let n = len as u64;
        let mut l = self.stream.launch_builder(&self.k.to_f16);
        l.arg(&mut h).arg(&t.f32).arg(&n);
        // SAFETY: to_f16 reads len floats and writes len halves, the sizes of both buffers.
        unsafe { l.launch(cfg) }.map_err(dev)?;
        Ok(t.f16.get_or_init(|| h).device_ptr(&self.stream).0)
    }

    fn launch_step(&self, p: &CudaPlan, step: &Step) -> Result<()> {
        let b = p.bucket;
        let Ptrs { n, ids, pos, seq, cu, mcu, mrow, qtype } = p.ptrs;
        let s = &self.stream;
        // SAFETY for every launch below: lowering checked that each value's arena range holds its
        // bucket sized rows at its type, each kernel touches rows below the batch's real count
        // only, and the index tables are sized for the bucket too.
        match *step {
            Step::Embed { table, out } => {
                let d = out.width as i32;
                let mut l = s.launch_builder(&self.k.embed);
                l.arg(&out.ptr).arg(&table).arg(&ids).arg(&n).arg(&d);
                // SAFETY: see above.
                unsafe { l.launch(rows(b.tokens, 256)) }.map_err(dev)?;
            }
            Step::LayerNorm { x, w, b: bias, eps, out } => {
                let f = &self.k.ln[2 * usize::from(x.half()) + usize::from(out.half())];
                let (k, d) = (kind(x.rows), x.width as i32);
                let mut l = s.launch_builder(f);
                l.arg(&out.ptr).arg(&x.ptr).arg(&w).arg(&bias).arg(&n).arg(&k).arg(&d).arg(&eps);
                let cfg = LaunchConfig {
                    grid_dim: (b.rows(x.rows).div_ceil(LN_ROWS) as u32, 1, 1),
                    block_dim: (32 * LN_ROWS as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: see above.
                unsafe { l.launch(cfg) }.map_err(dev)?;
            }
            Step::ToHalf { x, len } => {
                let blocks = len.div_ceil(256).min(65_535) as u32;
                let cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut l = s.launch_builder(&self.k.to_f16);
                l.arg(&p.stage).arg(&x).arg(&len);
                // SAFETY: see above, and lowering sized the staging buffer for the largest one.
                unsafe { l.launch(cfg) }.map_err(dev)?;
            }
            Step::Gemm { gemm, b: bias, act, out } => {
                let ws = self.workspace_ptr();
                // SAFETY: see above, and the workspace is used by nothing else on the stream.
                unsafe { p.gemms[gemm].run(&self.lt, ws, WORKSPACE, s.cu_stream().cast()) }?;
                if bias != 0 || act != 0 {
                    let f = &self.k.bias_act[usize::from(out.half())];
                    let (k, w) = (kind(out.rows), out.width as i32);
                    let mut l = s.launch_builder(f);
                    l.arg(&out.ptr).arg(&bias).arg(&n).arg(&k).arg(&w).arg(&act);
                    // SAFETY: see above.
                    unsafe { l.launch(rows(b.rows(out.rows), 256)) }.map_err(dev)?;
                }
            }
            Step::Rope { qkv, cos, sin } => {
                let heads = (qkv.width / 3 / HEAD) as i32;
                let mut l = s.launch_builder(&self.k.rope[usize::from(qkv.half())]);
                l.arg(&qkv.ptr).arg(&cos).arg(&sin).arg(&pos).arg(&n).arg(&heads);
                // SAFETY: see above.
                unsafe { l.launch(rows(b.tokens, 256)) }.map_err(dev)?;
            }
            Step::Attention { qkv, cos, sin, window, out } => {
                let heads = (out.width / HEAD) as i32;
                let mut l = s.launch_builder(
                    &self.k.attention[2 * usize::from(qkv.half()) + usize::from(out.half())],
                );
                l.arg(&out.ptr).arg(&qkv.ptr).arg(&cos).arg(&sin).arg(&pos);
                l.arg(&seq).arg(&cu).arg(&n);
                l.arg(&heads).arg(&window);
                let cfg = LaunchConfig {
                    grid_dim: (b.tokens.div_ceil(ATT_Q) as u32, heads as u32, 1),
                    block_dim: (32 * ATT_W, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: see above.
                unsafe { l.launch(cfg) }.map_err(dev)?;
            }
            Step::GeGlu { x, out } => {
                let inter = out.width as i32;
                let mut l = s.launch_builder(
                    &self.k.geglu[2 * usize::from(x.half()) + usize::from(out.half())],
                );
                l.arg(&out.ptr).arg(&x.ptr).arg(&n).arg(&inter);
                // SAFETY: see above.
                unsafe { l.launch(rows(b.tokens, 256)) }.map_err(dev)?;
            }
            Step::AddType { h, table } => {
                let d = h.width as i32;
                let mut l = s.launch_builder(&self.k.add_type);
                l.arg(&h.ptr).arg(&table).arg(&seq).arg(&qtype).arg(&n).arg(&d);
                // SAFETY: see above.
                unsafe { l.launch(rows(b.tokens, 256)) }.map_err(dev)?;
            }
            Step::Gather { h, out } => {
                let d = h.width as i32;
                let mut l = s.launch_builder(&self.k.gather[usize::from(h.half())]);
                l.arg(&out.ptr).arg(&h.ptr).arg(&mrow).arg(&n).arg(&d);
                // SAFETY: see above.
                unsafe { l.launch(rows(b.markers, 256)) }.map_err(dev)?;
            }
            Step::ActFeatures { h, logits, out } => {
                let d = h.width as i32;
                let mut l = s.launch_builder(&self.k.act_features);
                l.arg(&out.ptr).arg(&h.ptr).arg(&logits.ptr).arg(&cu).arg(&mcu);
                l.arg(&n).arg(&d);
                // SAFETY: see above.
                unsafe { l.launch(rows(b.seqs, 256)) }.map_err(dev)?;
            }
        }
        Ok(())
    }

    /// Enqueues the copy of the index tables to the device.
    fn copy_in(&self, p: &CudaPlan) -> Result<()> {
        let src = p.index_host.as_slice().map_err(dev)?;
        // SAFETY: the device table holds `at.len` u32 values, as many as the host one, and both
        // live as long as the plan, which outlives every run and the graph.
        unsafe { result::memcpy_htod_async(p.ptrs.n, src, self.stream.cu_stream()) }.map_err(dev)
    }

    /// Enqueues the copy of the bucket's logits and act outputs to the host.
    fn copy_out(&self, p: &mut CudaPlan) -> Result<()> {
        let (lp, ap, m) = (p.logits.ptr, p.act.ptr, p.bucket.markers);
        let cu = self.stream.cu_stream();
        let dst = p.host_out.as_mut_slice().map_err(dev)?;
        let (logits, act) = dst.split_at_mut(m);
        // SAFETY: lowering sized the host buffer for one logit per marker and two act values per
        // sequence of the bucket, the sizes of the two arena values, and it lives as the plan does.
        unsafe {
            result::memcpy_dtoh_async(logits, lp, cu).map_err(dev)?;
            result::memcpy_dtoh_async(act, ap, cu).map_err(dev)
        }
    }

    /// Captures a whole run as one graph.
    fn capture(&self, p: &mut CudaPlan) -> Result<Captured> {
        let s = &self.stream;
        // Relaxed, because the pinned buffers wait on their own event, never recorded, before they
        // hand out their pointers, and a stricter mode refuses any wait during a capture.
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).map_err(dev)?;
        let mut enqueue = || {
            self.copy_in(p)?;
            for step in &p.steps {
                self.launch_step(p, step)?;
            }
            self.copy_out(p)
        };
        let queued = enqueue();
        let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = s.end_capture(flags).map_err(dev)?;
        queued?;
        let graph = graph.ok_or_else(|| Error::Device("graph capture recorded nothing".into()))?;
        graph.upload().map_err(dev)?;
        Ok(Captured(graph))
    }
}

impl Backend for CudaBackend {
    type Weights = Weights;
    type Plan = CudaPlan;

    fn caps(&self) -> Caps {
        Caps { name: "cuda", threads: 1, graphs: true, unified_memory: false }
    }

    fn weight_bytes(&self, w: &Weights) -> usize {
        w.0.iter().map(|t| 4 * t.f32.len() + t.f16.get().map_or(0, |h| 2 * h.len())).sum()
    }

    fn plan_bytes(&self, p: &CudaPlan) -> usize {
        p.arena.len() + 2 * p._stage.len()
    }

    fn upload(&self, tensors: &[HostTensor<'_>], _graph: &Graph) -> Result<Weights> {
        let mut out = Vec::with_capacity(tensors.len());
        let mut host = Vec::new();
        for (i, h) in tensors.iter().enumerate() {
            let n = h.bytes.len() / h.dtype.size();
            if n != h.shape.iter().product::<usize>() {
                return Err(Error::Unsupported(format!(
                    "tensor {i} has {n} values for {:?}",
                    h.shape
                )));
            }
            host.clear();
            host.extend((0..n).map(|j| h.dtype.read_f32(h.bytes, j)));
            let f32 = self.stream.clone_htod(&host).map_err(dev)?;
            out.push(Tensor { shape: h.shape.to_vec(), f32, f16: OnceLock::new() });
        }
        self.stream.synchronize().map_err(dev)?;
        Ok(Weights(out))
    }

    #[allow(clippy::too_many_lines)]
    fn lower(&self, w: &Weights, graph: &Graph, bucket: Bucket) -> Result<CudaPlan> {
        let lay = layout(graph, |r| bucket.rows(r));
        let ty = types(graph, self.precision);
        let arena = self.stream.alloc_zeros::<u8>(4 * lay.len.max(1)).map_err(dev)?;
        let base = arena.device_ptr(&self.stream).0;
        let loc = |v: Val| {
            let s = graph.shape(v);
            let i = v.0 as usize;
            Loc { ptr: base + 4 * lay.offsets[i] as u64, rows: s.rows, width: s.width, ty: ty[i] }
        };
        let bad = |m: String| Err(Error::Unsupported(m));
        let shape = |i: usize| self.tensor(w, i).map(|t| t.shape.as_slice());
        let bias = |b: Option<usize>| b.map_or(Ok(0), |b| self.ptr32(w, b));
        let mut ropes: Vec<(u64, CudaSlice<f32>, CudaSlice<f32>)> = Vec::new();
        let stage_len = graph
            .ops
            .iter()
            .filter_map(|op| match *op {
                Op::Gemm { a, out, .. }
                    if ty[a.0 as usize] == Ty::F32 && ty[out.0 as usize] == Ty::F16 =>
                {
                    let s = graph.shape(a);
                    Some(bucket.rows(s.rows) * s.width)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let stage = self.stream.alloc_zeros::<u16>(stage_len.max(1)).map_err(dev)?;
        let stage_base = stage.device_ptr(&self.stream).0;
        let (mut gemms, mut keys) = (Vec::new(), Vec::new());
        let mut steps = Vec::with_capacity(graph.ops.len());
        let mut fused = None;
        for (i, op) in graph.ops.iter().enumerate() {
            let step = match *op {
                Op::Embed { table, out } => {
                    let out = loc(out);
                    if shape(table)?.get(1) != Some(&out.width) || out.rows != Rows::Tokens {
                        return bad(format!("op {i}: embedding table does not match its output"));
                    }
                    Step::Embed { table: self.ptr32(w, table)?, out }
                }
                Op::LayerNorm { x, w: nw, b, eps, out } => {
                    let (x, out) = (loc(x), loc(out));
                    let ok = shape(nw)? == [x.width]
                        && b.map_or(Ok(true), |b| shape(b).map(|s| s == [x.width]))?
                        && x.width == out.width
                        && x.width <= 1024
                        && x.rows == out.rows;
                    if !ok {
                        return bad(format!(
                            "op {i}: layer norm shapes do not match or are over 1024"
                        ));
                    }
                    let eps = eps as f32;
                    Step::LayerNorm { x, w: self.ptr32(w, nw)?, b: bias(b)?, eps, out }
                }
                Op::Gemm { a, w: gw, b, epilogue, out } => {
                    let (a, out) = (loc(a), loc(out));
                    let ok = shape(gw)? == [out.width, a.width]
                        && b.map_or(Ok(true), |b| shape(b).map(|s| s == [out.width]))?
                        && a.rows == out.rows;
                    if !ok {
                        return bad(format!("op {i}: gemm shapes do not match"));
                    }
                    let m = bucket.rows(a.rows);
                    if m == 0 {
                        continue;
                    }
                    let mut x = a.ptr;
                    let (wp, ab) = match (a.ty, out.ty) {
                        (Ty::F32, Ty::F32) => (self.ptr32(w, gw)?, Ty::F32),
                        (Ty::F32, Ty::F16) => {
                            let len = m * a.width;
                            steps.push(Step::ToHalf { x, len: len as u64 });
                            x = stage_base;
                            (self.ptr16(w, gw)?, Ty::F16)
                        }
                        (Ty::F16, _) => (self.ptr16(w, gw)?, Ty::F16),
                    };
                    let (act, acc) = match epilogue {
                        Epilogue::None => (0, false),
                        Epilogue::Gelu => (1, false),
                        Epilogue::Relu => (2, false),
                        Epilogue::Accumulate => (0, true),
                    };
                    // Picks are per inner size, width and types, not rows, for the reason
                    // RANKED_ROWS gives.
                    let dims = (lt::RANKED_ROWS, a.width, out.width);
                    let key = Key { dims, ab, c: out.ty, acc };
                    let g = lt::Gemm::new(
                        &self.lt,
                        (m, a.width, out.width),
                        ab,
                        out.ty,
                        acc,
                        WORKSPACE,
                        (wp, x, out.ptr),
                        self.picks.get(&key),
                    )?;
                    gemms.push(g);
                    keys.push(key);
                    Step::Gemm { gemm: gemms.len() - 1, b: bias(b)?, act, out }
                }
                Op::Rope { qkv, theta } => {
                    let qkv = loc(qkv);
                    if !qkv.width.is_multiple_of(3 * HEAD) || qkv.rows != Rows::Tokens {
                        return bad(format!("op {i}: rope needs token rows of 3 heads 64"));
                    }
                    let at = match ropes.iter().position(|r| r.0 == theta.to_bits()) {
                        Some(at) => at,
                        None => {
                            let (c, s) = rope_tables(theta, bucket.tokens.max(1));
                            let c = self.stream.clone_htod(&c).map_err(dev)?;
                            let s = self.stream.clone_htod(&s).map_err(dev)?;
                            ropes.push((theta.to_bits(), c, s));
                            ropes.len() - 1
                        }
                    };
                    let cos = ropes[at].1.device_ptr(&self.stream).0;
                    let sin = ropes[at].2.device_ptr(&self.stream).0;
                    // Attention right after on the same rows rotates as it loads, which saves
                    // writing q and k back and reading them again.
                    if matches!(graph.ops.get(i + 1), Some(Op::Attention { qkv: v, .. }) if loc(*v).ptr == qkv.ptr)
                    {
                        fused = Some((qkv.ptr, cos, sin));
                        continue;
                    }
                    Step::Rope { qkv, cos, sin }
                }
                Op::Attention { qkv, window, out } => {
                    let (qkv, out) = (loc(qkv), loc(out));
                    let ok = qkv.width.is_multiple_of(3 * HEAD)
                        && out.width * 3 == qkv.width
                        && qkv.rows == Rows::Tokens
                        && out.rows == Rows::Tokens;
                    if !ok {
                        return bad(format!("op {i}: attention shapes or types do not match"));
                    }
                    let window = window.map_or(Ok(-1), |w| {
                        i32::try_from(w).map_err(|_| Error::Unsupported(format!("op {i}: window")))
                    })?;
                    let (cos, sin) = match fused.take() {
                        Some((p, cos, sin)) if p == qkv.ptr => (cos, sin),
                        _ => (0, 0),
                    };
                    Step::Attention { qkv, cos, sin, window, out }
                }
                Op::GeGlu { x, out } => {
                    let (x, out) = (loc(x), loc(out));
                    if x.width != 2 * out.width || x.rows != out.rows {
                        return bad(format!("op {i}: geglu needs an input twice its output"));
                    }
                    Step::GeGlu { x, out }
                }
                Op::AddType { h, table } => {
                    let h = loc(h);
                    if shape(table)?.get(1) != Some(&h.width) || h.rows != Rows::Tokens {
                        return bad(format!("op {i}: type table does not match"));
                    }
                    Step::AddType { h, table: self.ptr32(w, table)? }
                }
                Op::GatherMarkers { h, out } => {
                    let (h, out) = (loc(h), loc(out));
                    let ok = h.width == out.width
                        && h.rows == Rows::Tokens
                        && out.rows == Rows::Markers
                        && h.ty == out.ty;
                    if !ok {
                        return bad(format!("op {i}: gather shapes do not match"));
                    }
                    Step::Gather { h, out }
                }
                Op::ActFeatures { h, logits, out } => {
                    let (h, logits, out) = (loc(h), loc(logits), loc(out));
                    let ok = out.width == h.width + 4
                        && logits.width == 1
                        && logits.rows == Rows::Markers
                        && out.rows == Rows::Seqs
                        && !h.half()
                        && !logits.half();
                    if !ok {
                        return bad(format!("op {i}: act feature shapes do not match"));
                    }
                    Step::ActFeatures { h, logits, out }
                }
                Op::MeanPool { .. } => return bad(format!("op {i}: mean pool is not on CUDA yet")),
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
            return bad("outputs are not one logit per marker and two per sequence".into());
        }
        let at = Index::new(bucket);
        let index = self.stream.alloc_zeros::<u32>(at.len).map_err(dev)?;
        let ptrs = Ptrs::new(index.device_ptr(&self.stream).0, at);
        let ctx = self.stream.context();
        // SAFETY: both buffers are zeroed below before anything reads them.
        let (mut index_host, mut host_out) = unsafe {
            (
                ctx.alloc_pinned::<u32>(at.len).map_err(dev)?,
                ctx.alloc_pinned::<f32>(bucket.markers + 2 * bucket.seqs).map_err(dev)?,
            )
        };
        index_host.as_mut_slice().map_err(dev)?.fill(0);
        host_out.as_mut_slice().map_err(dev)?.fill(0.0);
        self.stream.synchronize().map_err(dev)?;
        let mut plan = CudaPlan {
            bucket,
            steps,
            gemms,
            arena,
            _stage: stage,
            stage: stage_base,
            _ropes: ropes,
            _index: index,
            index_host,
            at,
            ptrs,
            logits,
            act,
            host_out,
            graph: None,
            profile: None,
            profiled: 0,
        };
        if tune::enabled() {
            let ws = self.workspace_ptr();
            tune::tune(&self.lt, &self.stream, ws, &mut plan.gemms, &keys, &self.name)?;
        }
        plan.graph = Some(self.capture(&mut plan)?);
        Ok(plan)
    }

    fn run(&self, p: &mut CudaPlan, batch: &Batch<'_>, out: &mut Outputs) -> Result<()> {
        let (t, s, m) = (batch.ids.len(), batch.seqs(), batch.markers.len());
        let at = p.at;
        let h = p.index_host.as_mut_slice().map_err(dev)?;
        h[..4].copy_from_slice(&[t as u32, s as u32, m as u32, 0]);
        h[at.ids..at.ids + t].copy_from_slice(batch.ids);
        h[at.cu..=at.cu + s].copy_from_slice(batch.cu);
        h[at.mcu..=at.mcu + s].copy_from_slice(batch.mcu);
        for q in 0..s {
            let (lo, hi) = (batch.cu[q] as usize, batch.cu[q + 1] as usize);
            for r in lo..hi {
                h[at.pos + r] = (r - lo) as u32;
                h[at.seq + r] = q as u32;
            }
            for k in batch.mcu[q] as usize..batch.mcu[q + 1] as usize {
                h[at.mrow + k] = lo as u32 + batch.markers[k];
            }
            h[at.qtype + q] = u32::from(batch.qtype[q]);
        }
        if p.profile.is_some() {
            self.copy_in(p)?;
            self.stream.synchronize().map_err(dev)?;
            for i in 0..p.steps.len() {
                let t = std::time::Instant::now();
                self.launch_step(p, &p.steps[i])?;
                self.stream.synchronize().map_err(dev)?;
                let ns = t.elapsed().as_nanos() as u64;
                if let Some(v) = p.profile.as_mut() {
                    v[i] += ns;
                }
            }
            self.copy_out(p)?;
            p.profiled += 1;
        } else if let Some(g) = &p.graph {
            g.0.launch().map_err(dev)?;
        }
        self.stream.synchronize().map_err(dev)?;
        let host = p.host_out.as_slice().map_err(dev)?;
        let (logits, act) = host.split_at(p.bucket.markers);
        out.logits.clear();
        out.logits.extend_from_slice(&logits[..m]);
        out.act.clear();
        out.act.extend(act[..2 * s].as_chunks::<2>().0.iter().copied());
        Ok(())
    }
}

/// cos and sin tables `[len, 32]` for heads of 64, computed as the CPU backend does.
pub(crate) fn rope_tables(theta: f64, len: usize) -> (Vec<f32>, Vec<f32>) {
    let half = HEAD / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| {
            let e = (2 * i) as f32 / HEAD as f32;
            1.0 / (theta.powf(f64::from(e)) as f32)
        })
        .collect();
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for p in 0..len {
        for &f in &inv {
            let a = f64::from(p as f32 * f);
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (cos, sin)
}

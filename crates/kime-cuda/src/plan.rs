//! Graphs lowered for the GPU: every value gets an element type and a place in one device arena,
//! every GEMM its cuBLASLt setup, and a run is a copy of the batch's index tables, a fixed list of
//! launches and a copy of the outputs back.

use std::sync::OnceLock;

use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use kime_tensor::plan::{Epilogue, Graph, Op, Rows, Val, layout};
use kime_tensor::{Backend, Batch, Bucket, Caps, Error, HostTensor, Outputs, Result};

use crate::lt::{self, Ty};
use crate::{CudaBackend, Precision, WORKSPACE, dev};

/// Width of one attention head.
const HEAD: usize = 64;

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
    Attention {
        qkv: Loc,
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
    base: u64,
    /// FP16 copies of FP32 GEMM inputs.
    _stage: CudaSlice<u16>,
    stage: u64,
    /// Held so the rope tables the steps point at stay alive.
    _ropes: Vec<(u64, CudaSlice<f32>, CudaSlice<f32>)>,
    index: CudaSlice<u32>,
    index_host: Vec<u32>,
    at: Index,
    ptrs: Ptrs,
    logits: Loc,
    act: Loc,
    host_out: Vec<f32>,
}

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
                // SAFETY: see above.
                unsafe { l.launch(rows(b.rows(x.rows).div_ceil(4), 128)) }.map_err(dev)?;
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
            Step::Attention { qkv, window, out } => {
                let heads = (out.width / HEAD) as i32;
                let mut l = s.launch_builder(
                    &self.k.attention[2 * usize::from(qkv.half()) + usize::from(out.half())],
                );
                l.arg(&out.ptr).arg(&qkv.ptr).arg(&seq).arg(&cu).arg(&n);
                l.arg(&heads).arg(&window);
                let cfg = LaunchConfig {
                    grid_dim: (b.tokens.div_ceil(4) as u32, heads as u32, 1),
                    block_dim: (128, 1, 1),
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
}

impl Backend for CudaBackend {
    type Weights = Weights;
    type Plan = CudaPlan;

    fn caps(&self) -> Caps {
        Caps { name: "cuda", threads: 1, graphs: false, unified_memory: false }
    }

    fn upload(&self, tensors: &[HostTensor<'_>]) -> Result<Weights> {
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
        let mut gemms = Vec::new();
        let mut steps = Vec::with_capacity(graph.ops.len());
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
                        && x.rows == out.rows;
                    if !ok {
                        return bad(format!("op {i}: layer norm shapes do not match"));
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
                    let g = lt::Gemm::new(
                        &self.lt,
                        (m, a.width, out.width),
                        ab,
                        out.ty,
                        acc,
                        WORKSPACE,
                        (wp, x, out.ptr),
                    )?;
                    gemms.push(g);
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
                    Step::Attention { qkv, window, out }
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
        self.stream.synchronize().map_err(dev)?;
        Ok(CudaPlan {
            bucket,
            steps,
            gemms,
            arena,
            base,
            _stage: stage,
            stage: stage_base,
            _ropes: ropes,
            index,
            index_host: vec![0; at.len],
            at,
            ptrs,
            logits,
            act,
            host_out: vec![0.0; bucket.markers + 2 * bucket.seqs],
        })
    }

    fn run(&self, p: &mut CudaPlan, batch: &Batch<'_>, out: &mut Outputs) -> Result<()> {
        let (t, s, m) = (batch.ids.len(), batch.seqs(), batch.markers.len());
        let (h, at) = (&mut p.index_host, p.at);
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
        self.stream.memcpy_htod(&p.index_host[..], &mut p.index).map_err(dev)?;
        for step in &p.steps {
            self.launch_step(p, step)?;
        }
        let (lo, ao) = (off(p, p.logits), off(p, p.act));
        let (logits, rest) = p.host_out.split_at_mut(p.bucket.markers);
        let bytes = |o: usize, n: usize| o..o + 4 * n;
        read(&self.stream, &p.arena, bytes(lo, m), &mut logits[..m])?;
        read(&self.stream, &p.arena, bytes(ao, 2 * s), &mut rest[..2 * s])?;
        self.stream.synchronize().map_err(dev)?;
        out.logits.clear();
        out.logits.extend_from_slice(&logits[..m]);
        out.act.clear();
        out.act.extend(rest[..2 * s].as_chunks::<2>().0.iter().copied());
        Ok(())
    }
}

/// Byte offset of a value in the arena.
fn off(p: &CudaPlan, l: Loc) -> usize {
    (l.ptr - p.base) as usize
}

/// Copies FP32 values from a byte range of the arena.
fn read(
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
    arena: &CudaSlice<u8>,
    range: std::ops::Range<usize>,
    dst: &mut [f32],
) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    let view = arena.slice(range);
    // SAFETY: the range holds dst.len() f32 values written by the plan, and u8 has no alignment
    // or validity requirement, so viewing the f32 destination as bytes is sound.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), 4 * dst.len()) };
    s.memcpy_dtoh(&view, bytes).map_err(dev)
}

/// cos and sin tables `[len, 32]` for heads of 64, computed as the CPU backend does.
fn rope_tables(theta: f64, len: usize) -> (Vec<f32>, Vec<f32>) {
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

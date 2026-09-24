//! Graphs lowered for the GPU: every value gets an element type and a place in one shared arena,
//! and a run is the batch's index tables written in place, a fixed list of dispatches in one
//! command buffer, and the outputs read in place.

use std::ptr::NonNull;
use std::sync::OnceLock;

use kime_tensor::plan::{Epilogue, Graph, Op, Rows, Val, layout};
use kime_tensor::{Backend, Batch, Bucket, Caps, Error, HostTensor, Outputs, Result};
use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLSize,
};

use crate::device::{Buffer, Pipeline, dev};
use crate::{MetalBackend, Precision};

/// Width of one attention head.
const HEAD: usize = 64;

/// Query rows and simdgroups per attention threadgroup, `ATT_Q` and `ATT_W` in the kernels.
const ATT_Q: usize = 16;
const ATT_W: usize = 8;

/// Rows per layer norm threadgroup, one simdgroup each, `LN_ROWS` in the kernels.
const LN_ROWS: usize = 4;

/// The GEMM tile, `GM` by `GN` in the kernels, and its threads.
const GM: usize = 32;
const GN: usize = 64;
const GEMM_THREADS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    F32,
    F16,
}

/// A weight in FP32, with an FP16 copy made the first time a plan needs one.
struct Tensor {
    shape: Vec<usize>,
    f32: Buffer,
    f16: OnceLock<Buffer>,
}

/// Every weight of a checkpoint in shared memory.
pub struct Weights(Vec<Tensor>);

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights").field("tensors", &self.0.len()).finish_non_exhaustive()
    }
}

/// A place in one of the plan's buffers: 0 is the arena, 1 the index tables, the rest weights and
/// rope tables.
#[derive(Debug, Clone, Copy)]
struct Ref {
    buf: usize,
    off: usize,
}

/// A value resolved to its type and place in the arena.
#[derive(Debug, Clone, Copy)]
struct Loc {
    off: usize,
    rows: Rows,
    width: usize,
    ty: Ty,
}

impl Loc {
    fn at(self) -> Ref {
        Ref { buf: 0, off: self.off }
    }

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
        table: Ref,
        out: Loc,
    },
    LayerNorm {
        x: Loc,
        w: Ref,
        b: Option<Ref>,
        eps: f32,
        out: Loc,
    },
    /// A GEMM with its bias, activation and residual add.
    Gemm {
        a: Loc,
        w: Ref,
        b: Option<Ref>,
        flags: i32,
        out: Loc,
        k: usize,
    },
    Rope {
        qkv: Loc,
        cos: Ref,
        sin: Ref,
    },
    /// Attention, with rope applied to q and k as they are loaded when there are tables.
    Attention {
        qkv: Loc,
        rope: Option<(Ref, Ref)>,
        window: i32,
        out: Loc,
    },
    GeGlu {
        x: Loc,
        out: Loc,
    },
    AddType {
        h: Loc,
        table: Ref,
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

/// Where each index table sits in the index buffer, in u32 elements, after the four counts.
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

    fn at(o: usize) -> Ref {
        Ref { buf: 1, off: 4 * o }
    }
}

/// A graph lowered for one bucket.
pub struct MetalPlan {
    bucket: Bucket,
    steps: Vec<Step>,
    /// The arena, the index tables, then every weight and rope table a step reads.
    bufs: Vec<Buffer>,
    at: Index,
    logits: Loc,
    act: Loc,
    profile: Option<Vec<u64>>,
}

impl std::fmt::Debug for MetalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalPlan")
            .field("bucket", &self.bucket)
            .field("steps", &self.steps.len())
            .field("arena", &self.bufs[0].len())
            .finish_non_exhaustive()
    }
}

impl MetalPlan {
    /// The bucket it was built for.
    #[must_use]
    pub fn bucket(&self) -> Bucket {
        self.bucket
    }

    /// Arena size in bytes.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.bufs[0].len()
    }

    /// Times every step from now on. Each step then gets its own command buffer and a wait, so
    /// this is for finding where the time goes, not for measuring the total.
    pub fn profile(&mut self) {
        self.profile = Some(vec![0; self.steps.len()]);
    }

    /// Time per kind of step since [`MetalPlan::profile`], in nanoseconds, largest first.
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

/// The element type of every value, placed as the CUDA backend places them: FP32 throughout with
/// [`Precision::F32`], and with [`Precision::F16`] FP16 except the residual stream, the type
/// embedding's target, the act features, the outputs, attention's qkv, and whatever a gather or a
/// GEMM computes from an FP32 value unless it feeds attention or GeGLU.
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
            // Unlike CUDA, the input of GeGLU stays FP32. Rounding it to FP16 put the English
            // model's probabilities 9.6e-3 from Laya's, and the GEMM that writes it is compute
            // bound here, so the wider store costs little.
            Op::GeGlu { x, out } => {
                ty[x.0 as usize] = Ty::F32;
                half[out.0 as usize] = true;
            }
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

/// The slot of `b` in `bufs`, added the first time it is seen.
fn slot(bufs: &mut Vec<Buffer>, b: Buffer) -> Ref {
    let ptr = b.ptr();
    let buf = bufs.iter().position(|x| x.ptr() == ptr).unwrap_or_else(|| {
        bufs.push(b);
        bufs.len() - 1
    });
    Ref { buf, off: 0 }
}

fn size(width: usize, height: usize) -> MTLSize {
    MTLSize { width, height, depth: 1 }
}

type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Binds buffers and small values to one dispatch.
struct Bind<'a> {
    enc: &'a Encoder,
    bufs: &'a [Buffer],
}

impl Bind<'_> {
    fn pipe(&self, p: &Pipeline) -> &Self {
        self.enc.setComputePipelineState(p);
        self
    }

    fn buf(&self, i: usize, r: Ref) -> &Self {
        // SAFETY: every Ref was made by lowering against this plan's buffers, and lowering checked
        // that the range each kernel reads or writes from the offset fits the buffer.
        unsafe { self.enc.setBuffer_offset_atIndex(Some(&self.bufs[r.buf].0), r.off, i) };
        self
    }

    fn val<T: Copy>(&self, i: usize, v: T) -> &Self {
        // SAFETY: Metal copies the bytes before the call returns, and `v` lives until then.
        unsafe {
            self.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), size_of::<T>(), i);
        }
        self
    }

    fn go(&self, groups: MTLSize, threads: usize) {
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(groups, size(threads, 1));
    }
}

impl MetalBackend {
    fn tensor<'w>(&self, w: &'w Weights, i: usize) -> Result<&'w Tensor> {
        w.0.get(i).ok_or_else(|| Error::Unsupported(format!("weight {i} is not in the checkpoint")))
    }

    /// The FP16 copy of weight `i`, made on first use.
    fn half(&self, w: &Weights, i: usize) -> Result<Buffer> {
        let t = self.tensor(w, i)?;
        if let Some(h) = t.f16.get() {
            return Ok(h.clone());
        }
        let n = t.f32.len() / 4;
        let h = self.buffer(2 * n)?;
        // SAFETY: the FP32 buffer holds n floats and the new one room for n halves; the new one is
        // not shared yet and nothing writes the FP32 one after upload.
        unsafe {
            let src = std::slice::from_raw_parts(t.f32.ptr().cast::<f32>(), n);
            let dst = std::slice::from_raw_parts_mut(h.ptr().cast::<u16>(), n);
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = half::f16::from_f32(s).to_bits();
            }
        }
        Ok(t.f16.get_or_init(|| h).clone())
    }

    fn encode(&self, p: &MetalPlan, enc: &Encoder, step: &Step) {
        let b = p.bucket;
        let at = p.at;
        let n = Index::at(0);
        let e = Bind { enc, bufs: &p.bufs };
        let rows = |r: usize| size(r, 1);
        match *step {
            Step::Embed { table, out } => {
                e.pipe(&self.k.embed).buf(0, out.at()).buf(1, table).buf(2, Index::at(at.ids));
                e.buf(3, n).val(4, out.width as i32).go(rows(b.tokens), 256);
            }
            Step::LayerNorm { x, w, b: bias, eps, out } => {
                let f = &self.k.ln[2 * usize::from(x.half()) + usize::from(out.half())];
                e.pipe(f).buf(0, out.at()).buf(1, x.at()).buf(2, w).buf(3, bias.unwrap_or(w));
                e.buf(4, n).val(5, kind(x.rows)).val(6, x.width as i32).val(7, eps);
                e.val(8, i32::from(bias.is_some()));
                e.go(rows(b.rows(x.rows).div_ceil(LN_ROWS)), 32 * LN_ROWS);
            }
            Step::Gemm { a, w, b: bias, flags, out, k } => {
                let f = &self.k.gemm[match (a.ty, out.ty) {
                    (Ty::F32, Ty::F32) => 0,
                    (Ty::F32, Ty::F16) => 1,
                    (Ty::F16, Ty::F32) => 2,
                    (Ty::F16, Ty::F16) => 3,
                }];
                e.pipe(f).buf(0, out.at()).buf(1, a.at()).buf(2, w).buf(3, bias.unwrap_or(w));
                e.buf(4, n).val(5, kind(a.rows)).val(6, k as i32).val(7, out.width as i32);
                e.val(8, flags);
                let m = b.rows(a.rows);
                e.go(size(out.width.div_ceil(GN), m.div_ceil(GM)), GEMM_THREADS);
            }
            Step::Rope { qkv, cos, sin } => {
                let heads = (qkv.width / 3 / HEAD) as i32;
                e.pipe(&self.k.rope[usize::from(qkv.half())]).buf(0, qkv.at()).buf(1, cos);
                e.buf(2, sin).buf(3, Index::at(at.pos)).buf(4, n).val(5, heads);
                e.go(rows(b.tokens), 256);
            }
            Step::Attention { qkv, rope, window, out } => {
                let heads = out.width / HEAD;
                let f = &self.k.attention[2 * usize::from(qkv.half()) + usize::from(out.half())];
                let (cos, sin) = rope.unwrap_or((qkv.at(), qkv.at()));
                e.pipe(f).buf(0, out.at()).buf(1, qkv.at()).buf(2, cos).buf(3, sin);
                e.buf(4, Index::at(at.pos)).buf(5, Index::at(at.seq)).buf(6, Index::at(at.cu));
                e.buf(7, n).val(8, heads as i32).val(9, window).val(10, i32::from(rope.is_some()));
                e.go(size(b.tokens.div_ceil(ATT_Q), heads), 32 * ATT_W);
            }
            Step::GeGlu { x, out } => {
                let f = &self.k.geglu[2 * usize::from(x.half()) + usize::from(out.half())];
                e.pipe(f).buf(0, out.at()).buf(1, x.at()).buf(2, n).val(3, out.width as i32);
                e.go(rows(b.tokens), 256);
            }
            Step::AddType { h, table } => {
                e.pipe(&self.k.add_type).buf(0, h.at()).buf(1, table).buf(2, Index::at(at.seq));
                e.buf(3, Index::at(at.qtype)).buf(4, n).val(5, h.width as i32);
                e.go(rows(b.tokens), 256);
            }
            Step::Gather { h, out } => {
                e.pipe(&self.k.gather[usize::from(h.half())]).buf(0, out.at()).buf(1, h.at());
                e.buf(2, Index::at(at.mrow)).buf(3, n).val(4, h.width as i32);
                e.go(rows(b.markers), 256);
            }
            Step::ActFeatures { h, logits, out } => {
                e.pipe(&self.k.act_features).buf(0, out.at()).buf(1, h.at()).buf(2, logits.at());
                e.buf(3, Index::at(at.cu)).buf(4, Index::at(at.mcu)).buf(5, n);
                e.val(6, h.width as i32).go(rows(b.seqs), 256);
            }
        }
    }

    /// Encodes `steps` into one command buffer, commits it and waits.
    fn submit(&self, p: &MetalPlan, steps: &[Step]) -> Result<()> {
        autoreleasepool(|_| {
            let cb = self.queue.commandBuffer().ok_or_else(|| dev("no command buffer"))?;
            let enc = cb.computeCommandEncoder().ok_or_else(|| dev("no encoder"))?;
            for s in steps {
                self.encode(p, &enc, s);
            }
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error {
                let e = cb.error().map(|e| e.localizedDescription().to_string());
                return Err(dev(format!("command buffer failed: {}", e.unwrap_or_default())));
            }
            Ok(())
        })
    }
}

impl Backend for MetalBackend {
    type Weights = Weights;
    type Plan = MetalPlan;

    fn caps(&self) -> Caps {
        Caps { name: "metal", threads: 1, graphs: false, unified_memory: true }
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
            let f32 = self.buffer_from(&host)?;
            out.push(Tensor { shape: h.shape.to_vec(), f32, f16: OnceLock::new() });
        }
        Ok(Weights(out))
    }

    #[allow(clippy::too_many_lines)]
    fn lower(&self, w: &Weights, graph: &Graph, bucket: Bucket) -> Result<MetalPlan> {
        let lay = layout(graph, |r| bucket.rows(r));
        let ty = types(graph, self.precision);
        let at = Index::new(bucket);
        let mut bufs = vec![self.buffer(4 * lay.len)?, self.buffer(4 * at.len)?];
        let loc = |v: Val| {
            let s = graph.shape(v);
            let i = v.0 as usize;
            Loc { off: 4 * lay.offsets[i], rows: s.rows, width: s.width, ty: ty[i] }
        };
        let bad = |m: String| Err(Error::Unsupported(m));
        let shape = |i: usize| self.tensor(w, i).map(|t| t.shape.as_slice());
        let f32w =
            |bufs: &mut Vec<Buffer>, i: usize| self.tensor(w, i).map(|t| slot(bufs, t.f32.clone()));
        let mut steps = Vec::with_capacity(graph.ops.len());
        let mut ropes: Vec<(u64, Ref, Ref)> = Vec::new();
        let mut fused = None;
        for (i, op) in graph.ops.iter().enumerate() {
            let step = match *op {
                Op::Embed { table, out } => {
                    let out = loc(out);
                    if shape(table)?.get(1) != Some(&out.width) || out.rows != Rows::Tokens {
                        return bad(format!("op {i}: embedding table does not match its output"));
                    }
                    Step::Embed { table: f32w(&mut bufs, table)?, out }
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
                    let b = b.map(|b| f32w(&mut bufs, b)).transpose()?;
                    Step::LayerNorm { x, w: f32w(&mut bufs, nw)?, b, eps: eps as f32, out }
                }
                Op::Gemm { a, w: gw, b, epilogue, out } => {
                    let (a, out) = (loc(a), loc(out));
                    let ok = shape(gw)? == [out.width, a.width]
                        && b.map_or(Ok(true), |b| shape(b).map(|s| s == [out.width]))?
                        && a.rows == out.rows
                        && a.width.is_multiple_of(4);
                    if !ok {
                        return bad(format!(
                            "op {i}: gemm shapes do not match or the inner size is not a multiple of 4"
                        ));
                    }
                    if bucket.rows(a.rows) == 0 {
                        continue;
                    }
                    let wr = match (a.ty, out.ty) {
                        (Ty::F32, Ty::F32) => f32w(&mut bufs, gw)?,
                        _ => slot(&mut bufs, self.half(w, gw)?),
                    };
                    let act = match epilogue {
                        Epilogue::None | Epilogue::Accumulate => 0,
                        Epilogue::Gelu => 1,
                        Epilogue::Relu => 2,
                    };
                    let acc = i32::from(epilogue == Epilogue::Accumulate);
                    let flags = i32::from(b.is_some()) | act << 1 | acc << 3;
                    let b = b.map(|b| f32w(&mut bufs, b)).transpose()?;
                    Step::Gemm { a, w: wr, b, flags, out, k: a.width }
                }
                Op::Rope { qkv, theta } => {
                    let qkv = loc(qkv);
                    if !qkv.width.is_multiple_of(3 * HEAD) || qkv.rows != Rows::Tokens {
                        return bad(format!("op {i}: rope needs token rows of 3 heads 64"));
                    }
                    let (cos, sin) = match ropes.iter().find(|r| r.0 == theta.to_bits()) {
                        Some(r) => (r.1, r.2),
                        None => {
                            let (c, s) = rope_tables(theta, bucket.tokens.max(1));
                            let c = slot(&mut bufs, self.buffer_from(&c)?);
                            let s = slot(&mut bufs, self.buffer_from(&s)?);
                            ropes.push((theta.to_bits(), c, s));
                            (c, s)
                        }
                    };
                    // Attention right after on the same rows rotates as it loads.
                    if matches!(graph.ops.get(i + 1), Some(Op::Attention { qkv: v, .. }) if loc(*v).off == qkv.off)
                    {
                        fused = Some((qkv.off, cos, sin));
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
                    let rope = match fused.take() {
                        Some((off, cos, sin)) if off == qkv.off => Some((cos, sin)),
                        _ => None,
                    };
                    Step::Attention { qkv, rope, window, out }
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
                    Step::AddType { h, table: f32w(&mut bufs, table)? }
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
            || logits.half()
            || act.half()
        {
            return bad("outputs are not one logit per marker and two per sequence".into());
        }
        Ok(MetalPlan { bucket, steps, bufs, at, logits, act, profile: None })
    }

    fn run(&self, p: &mut MetalPlan, batch: &Batch<'_>, out: &mut Outputs) -> Result<()> {
        let (t, s, m) = (batch.ids.len(), batch.seqs(), batch.markers.len());
        let at = p.at;
        // SAFETY: the index buffer holds `at.len` u32 values, the GPU is idle since every run
        // waits for its command buffer, and `&mut` of the plan keeps anyone else out.
        let h = unsafe { std::slice::from_raw_parts_mut(p.bufs[1].ptr().cast::<u32>(), at.len) };
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
        if let Some(mut prof) = p.profile.take() {
            for (i, step) in p.steps.iter().enumerate() {
                let t = std::time::Instant::now();
                self.submit(p, std::slice::from_ref(step))?;
                prof[i] += t.elapsed().as_nanos() as u64;
            }
            p.profile = Some(prof);
        } else {
            self.submit(p, &p.steps)?;
        }
        let arena = p.bufs[0].ptr();
        // SAFETY: lowering placed one FP32 logit per marker and two FP32 act values per sequence of
        // the bucket in the arena at these offsets, and the GPU is done with them.
        let (logits, act) = unsafe {
            (
                std::slice::from_raw_parts(arena.add(p.logits.off).cast::<f32>(), m),
                std::slice::from_raw_parts(arena.add(p.act.off).cast::<f32>(), 2 * s),
            )
        };
        out.logits.clear();
        out.logits.extend_from_slice(logits);
        out.act.clear();
        out.act.extend(act.as_chunks::<2>().0.iter().copied());
        Ok(())
    }
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

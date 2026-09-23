# Engine

This is the core of the implementation: how the Rust workspace is laid out, how a model becomes an execution plan, how memory is managed, how weights are stored and loaded, and how backends plug in. The rule behind every choice here is that the hot path does no allocation, no locking, no dynamic dispatch per op, and no work that depends on the request's JSON shape.

## Toolchain

- Rust stable, edition 2024, MSRV 1.90. No nightly features in the default build. The AMX kernels in 10 use `asm!`, which is stable.
- `#![forbid(unsafe_code)]` in `kime-core`, `kime-route`, `kime-serve` and the SDK crates. `unsafe` is allowed only in `kime-tensor`, `kime-cuda`, `kime-metal`, `kime-ane`, `kime-cpu` and `kime-tok`. Every `unsafe` block has a `// SAFETY:` comment, and `cargo xtask lint` fails without one.
- Release profile: `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` for the server binary (`unwind` for the Python and Node libraries so panics can become exceptions), `opt-level = 3`, `debug = "line-tables-only"` so profiles work.
- The allocator is mimalloc in binaries. Libraries do not set a global allocator.
- Binaries built for distribution use `-C target-cpu` baselines per artifact: `x86-64-v3` and `x86-64-v4` builds for Linux x86, `apple-m1` for macOS, and `neoverse-v1` for Linux ARM. Kernels also dispatch at runtime on CPU features within a build.

## Workspace layout

```
kime/
  Cargo.toml                 workspace, shared deps and lints
  crates/
    kime-core/               request and answer types, validation, rendering, confidence, calibration, email, presets
    kime-tok/                BPE tokenizers, special token handling, parity tests
    kime-model/              ModelSpec, checkpoint formats, weight loading, graph builder for both families
    kime-tensor/             device neutral tensor handles, dtypes, arenas, the Backend trait, the plan executor
    kime-cuda/               NVIDIA backend (cudarc), kernels in .cu compiled to cubin at build time
    kime-metal/              Apple GPU backend (objc2-metal), kernels in .metal compiled to metallib at build time
    kime-ane/                Apple Neural Engine backend (objc2-core-ml), fixed shape buckets
    kime-cpu/                CPU backend, x86 and ARM kernels
    kime-engine/             Session, scheduler, batching, caches, the public in-process API
    kime-route/              script and language detection, the Router
    kime-serve/              axum HTTP server, auth, rate limits, metrics
    kime-cli/                the `kime` binary: serve, predict, bench, eval, convert, train, calibrate
    kime-train/              training and distillation (burn), see 12
    kime-eval/               datasets, metrics, benchmark runners, report generation
    kime-py/                 Python bindings (pyo3, maturin)
    kime-node/               Node bindings (napi-rs)
  kernels/                   shared kernel sources and generators (cuda/, metal/)
  tools/ref/                 Python reference scripts, parity only, never shipped
  xtask/                     build helpers, lint, release, fixture regeneration
```

Main external crates: `tokio` 1 (server only), `axum` 0.8, `hyper` 1, `sonic-rs`, `serde`, `blake3`, `memmap2`, `safetensors`, `half`, `bytemuck`, `parking_lot`, `crossbeam-channel`, `arc-swap`, `tracing`, `metrics` with `metrics-exporter-prometheus`, `clap` 4, `cudarc` 0.19 (features `driver`, `cublaslt`, `nvrtc`), `objc2-metal`, `objc2-core-ml`, `pyo3`, `napi`. We avoid a general tensor framework on the inference path. candle's ModernBERT pads, uses dense masks and does not unpad, and a framework's per-op dispatch costs more than our whole budget at batch 1. We borrow design from TEI's `flash_modernbert.rs` and from candle's kernels where the licenses allow, but the executor is our own.

## Public in-process API

```rust
let kime = kime::Kime::builder()
    .model("kime-v1-s-en")          // id, alias, local path or hf:// ref
    .device(Device::Auto)           // Cuda(0), Metal, Ane, Cpu { threads }, Auto
    .preload(true)
    .build()?;

let req = Request::new(State::json(&value))
    .choice("department", "Which team should handle this", [("billing", "Payment issues"), ("technical", "Bugs")])
    .score("urgency", "How urgent", ["not urgent", "soon", "critical"])
    .noul("churn", "The customer threatens to leave");

let res: Response = kime.decide(&req)?;          // blocking
let res = kime.decide_async(&req).await?;        // async, same engine
let many = kime.decide_batch(&[req1, req2])?;    // batch
```

`Kime` is `Send + Sync + Clone`. Clones share the engine. The request types have `serde` impls that accept and emit the wire format, so `kime-serve` is a thin layer over this API.

## From request to plan

A request goes through four stages. Only the last runs on the device.

1. **Parse and validate** (`kime-core`). This builds a `Validated` request: question list, rendered strings, error list. It runs on the caller's thread.
2. **Route** (`kime-route`). This picks the model. See 11.
3. **Tokenize and lay out** (`kime-tok`, `kime-model`). This produces a `WorkItem` holding state token ids (or segment ids and their hashes), the question tower rows, marker offsets and option counts. Still on the caller's thread.
4. **Execute** (`kime-engine` plus a backend). The scheduler merges WorkItems from many requests into device batches, runs them, and scatters the logits back.

### Plans

A plan is a flat, precomputed list of ops for one model at one shape bucket:

```rust
enum Op {
    Embed { ids: Buf, out: Buf, norm: W },
    Gemm { a: Buf, w: W, out: Buf, epilogue: Epilogue },          // Epilogue: None, Residual(Buf), GeGLU, Gelu
    RopeQkv { qkv: Buf, cos_sin: Table, out_q: Buf, out_k: Buf, out_v: Buf },
    AttnVarlen { q: Buf, k: Buf, v: Buf, cu_seqlens: Buf, window: Option<u32>, segments: Option<Buf>, out: Buf },
    AttnMasked { q: Buf, k: Buf, v: Buf, mask_desc: MaskDesc, out: Buf },   // question tower self attention
    CrossAttn { q: Buf, k_mem: Buf, v_mem: Buf, row_to_state: Buf, out: Buf },
    LayerNorm { x: Buf, w: W, out: Buf },
    GatherMarkers { h: Buf, idx: Buf, out: Buf },
    Scorer { m: Buf, weights: ScorerW, qtype: Buf, out_logits: Buf },
    KvProject { s: Buf, wk: W, wv: W, out: Buf },
}
```

The plan builder in `kime-model` walks the `ModelSpec` and emits ops. Each backend then lowers the op list once per bucket. On CUDA it records a CUDA graph (08). On Metal it encodes an indirect command buffer (09). On CPU it builds a list of function pointers with their arguments resolved. At run time the executor writes the batch's inputs into the bucket's input buffers and launches the lowered plan. There is no per op interpretation.

### Buckets

Unpadded variable length batching means the only shape that matters is the total token count. The number of sequences matters only for `cu_seqlens`. Buckets are:

- State stage: total tokens in {64, 128, 256, 512, 1k, 2k, 4k, 8k, 16k, 32k, 64k}, sequences up to 256. Unused sequence slots get zero length.
- Question stage: total tokens in {32, 64, 128, 256, 512, 1k, 2k, 4k, 8k}, rows up to 1,024, markers up to 8,192.

A batch is padded up to its bucket's token count with a dummy zero-length tail, so the padding costs at most 2x and on average about 1.4x. Buckets are captured lazily on first use and at startup for the common ones (`--warm-buckets`, default the smallest four of each stage). On CUDA a captured graph costs about 50 to 200 microseconds to record and some device memory for its plan. Capturing all buckets for one model takes under 1 second.

## Memory

- **Weights** are uploaded once per device and never move. On Apple unified memory they are wrapped without copying (09).
- **Activations** use one arena per (model, stage, bucket). Buffer offsets are assigned at plan build time by a liveness pass over the op list, which reuses a buffer once its last reader has run. This is a greedy interval colouring, and it cuts peak activation memory to about 3 hidden sized buffers plus attention scratch. Arenas for the largest buckets are allocated on first use, and the operator can cap them with `--max-state-tokens`.
- **State memory cache** (the cross attention K and V per state) lives in a device side slab allocator with 64 KiB pages. Entries are keyed by `blake3(model id, state token ids)` or, in segment mode, per segment. Eviction is LRU with a byte budget (default 25% of free device memory at start). See 11 for the policy.
- **Pinned host staging**: each worker owns two pinned buffers for input ids and output logits, and alternates between them so the next upload overlaps the current run.

## Weight format and loading

The native checkpoint format is a directory with `kime.json`, `tokenizer.json`, `calibration.json` and `weights.safetensors`. `kime convert --pack` produces a single `.kime` file: a 4 KiB header (magic `KIME\x01`, JSON offsets) followed by the same contents, with every tensor aligned to 4 KiB so it can be mmapped straight into device-friendly layouts.

Load path, with a target of 200 ms or less for `kime-v1-s-en` on an L4 or an M3:

1. mmap the file. There is no read and no parse of tensor data.
2. Validate the header, the tensor names and shapes, and the calibration table. Check the file's blake3 against `kime.json` unless `--no-verify`. On a warm page cache this runs at memory bandwidth, about 20 ms for 160 MB.
3. Look for a prepacked cache at `~/.cache/kime/<file hash>/<backend>-<device>.bin`, which holds weights already transposed, quantized or swizzled for this backend. If it exists, mmap it. If not, repack in parallel on all cores and write the cache in the background.
4. Upload. CUDA uses `cuMemcpyHtoDAsync` from the mmapped (registered) pages on 4 streams. At PCIe 4 x16 speeds, 164 MB takes about 12 ms. Metal wraps the mmapped pages with `newBufferWithBytesNoCopy`, so no copy happens.
5. Build the plans for the warm buckets and capture graphs.

There is no random initialization pass, no Python, and no framework graph build. Laya 0.3.7 cut its load from 22 s to about 2 s by skipping random init. We start from zero work.

The compat family loads Laya's `model.safetensors` plus `encoder/config.json` and `tokenizer/*` directly, from a local path or the Hugging Face hub. Downloads use the HF cache layout, so existing downloads are reused. The first load writes a prepacked cache so later loads are as fast as native.

## Backend trait

```rust
pub trait Backend: Send + Sync + 'static {
    type Weights: Send + Sync;
    type Plan: Send;
    fn caps(&self) -> Caps;                                       // dtypes, max buckets, unified memory, graph support
    fn upload(&self, spec: &ModelSpec, packed: &Packed) -> Result<Self::Weights>;
    fn lower(&self, w: &Self::Weights, ops: &[Op], bucket: Bucket) -> Result<Self::Plan>;
    fn run(&self, plan: &mut Self::Plan, inputs: &BatchInputs, out: &mut BatchOutputs) -> Result<()>;
    fn state_mem(&self) -> &dyn StateMemStore;                    // device side slab for cached K and V
}
```

Backends are chosen at compile time with features (`cuda`, `metal`, `ane`, `cpu`) and at run time with `Device`. The engine is generic over `Backend`, so the whole path is monomorphized. There is one `dyn` hop per batch, not per op.

## Determinism

04 promises bit identical logits per (checkpoint, backend, device type, kime version) whatever the batch composition. The engine meets it like this:

- GEMMs use fixed tile configurations per (bucket, shape) that are chosen at plan build time and stored in the plan. The heuristic query result is never re-run. Split-K is disabled when it would make a row's reduction depend on the batch size. On cuBLASLt, the algorithm is pinned by index, and `CUBLASLT_MATMUL_PREF` is fixed.
- Attention kernels reduce each query row in a fixed key order, in blocks that start at the sequence boundary, not the batch buffer offset. Sequences therefore see the same arithmetic wherever they land in the buffer.
- LayerNorm and softmax use a fixed reduction tree per row.
- Padding to a bucket changes nothing about real rows. This is tested by running every fixture alone and inside randomly composed batches, and comparing bits (see 15).

Across backends and device types we promise only the tolerances in 15, not bit equality.

## Errors and failure handling

- Device errors (CUDA launch failure, Metal command buffer error) mark the worker unhealthy. Its batch fails with 500. The engine rebuilds the context and plans in the background. The health endpoint reports degraded until that finishes.
- Out of memory while allocating a large bucket makes the scheduler split the batch and retry at a smaller bucket, and lowers the state cache budget. Unlike Laya, it never silently falls back to CPU. Falling back to CPU turns a 2 ms request into a 2 s request, and the operator should see an error instead.
- NaN or Inf in logits fails the affected question with 500 and a clear message, and increments a metric. We never return non-finite probabilities.

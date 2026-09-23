# Apple backends

Apple Silicon gets two backends. `kime-metal` runs any plan on the GPU with hand written Metal kernels, and it is the default on macOS and iOS. `kime-ane` runs fixed shape buckets of the native models on the Neural Engine through Core ML, for callers who care most about energy and have short inputs. Both use unified memory, so weights are never copied.

The baselines to beat on an M3 Max:

- laya-mlx FP16: 7.39 ms (multilingual) and 13.42 ms (laya) per single question, and 395 questions per second in batches.
- laya-coreml on the ANE: 4.98 ms p50 at 0.154 J per decision, and 4.88 ms at 0.134 J with W8.

## kime-metal

### Setup

- Bindings are `objc2-metal` and `objc2-foundation`. metal-rs is deprecated and not used.
- Kernels are written in Metal Shading Language 3.1 in `kernels/metal/`. At build time, `xcrun metal` compiles them into a `.metallib` that is embedded in the binary. Pipelines are created at load for the function constants each model needs (d, heads, window), and are cached with `MTLBinaryArchive` so the next start skips compilation.
- Weights are wrapped with `newBufferWithBytesNoCopy` over the mmapped prepacked file. This needs page aligned offsets, which the `.kime` format guarantees. Loading is therefore mostly page faults, and for the s tier it takes under 50 ms on a warm cache.

### Kernels

The kernel set mirrors 08, fused the same way:

- GEMMs use `simdgroup_matrix` 8x8 FP16 tiles, with tile sizes chosen per shape instead of a fixed 32x32x16. candle issue #3302 found its fixed tile made Metal GEMMs 2 to 11x slower than MLX. For M under 64 (small buckets) a separate kernel keeps the whole weight panel streaming from memory and uses more threadgroups along N.
- Epilogues fuse RoPE, residual adds, LayerNorm (the whole row is in one threadgroup when d is 768 or less) and GeGLU.
- Attention is FlashAttention 2 style with variable lengths, windows and segments. It is based on the structure of MLX's `sdpa_vector` and `steel_attention` kernels (MIT licensed), extended with our masks.
- Cross attention and the marker scorer work as on CUDA.

### Dispatch overhead

Metal's fixed cost is command buffer commit plus the wait for completion, about 60 to 150 microseconds on an M3. We avoid per op cost like this:

- Each (model, stage, bucket) plan is encoded once into an `MTLIndirectCommandBuffer`, with all buffers bound through argument buffers. At run time only the input buffer contents change. One command buffer per batch runs both stages with `executeCommandsInBuffer`.
- The worker thread waits with a shared event and a spin then yield loop for the first 200 microseconds. That keeps the wake latency of `waitUntilCompleted` off the critical path.
- The CPU writes inputs straight into shared storage buffers. There are no blits.

### Targets on M3 Max (kime-v1-s-en, FP16)

| Workload | laya-mlx best | kime target |
|---|---|---|
| W1, 1 question, short state | 7.39 ms (multilingual) | 0.5 ms or less |
| W3, 10 questions, 512 tokens | about 27 ms (multilingual) | 2.7 ms or less |
| Batched throughput, W1 shaped | 395 q/s | 4,000 q/s or more |
| Peak memory | 687.6 MiB | 200 MiB or less |

## kime-ane

The Neural Engine only runs Core ML models with static or enumerated shapes, needs a specific tensor layout to be efficient, and is faster than the GPU only for short sequences. laya-coreml showed a 1.39x speedup and 2.78x lower energy at L96, and a slowdown at L1024. The split architecture helps a lot here: the state tower and the question tower are separate Core ML models with separate buckets, and the question tower is always short.

### Conversion

The runtime is pure Rust, but building an `.mlpackage` needs coremltools, which is Python. So conversion is an offline step: `kime convert --target ane` runs `tools/ane/export.py` in a managed venv (uv) and generates the packages from the checkpoint. The generator writes the model directly in MIL with coremltools' builder rather than tracing PyTorch, so we control every op. Released native checkpoints include prebuilt ANE packages, so most users never run the conversion.

Layout and op choices, taken from Apple's ANE transformer guidance and laya-coreml's working rewrite:

- Tensors are (B, C, 1, S). Every linear layer is a 1x1 `conv`.
- Attention is split per head with einsum style `matmul`s, softmax over the key axis, and additive masks of -1e4 in FP16. The local window and segment masks are inputs, built on the host, because RoPE positions and masks vary per request.
- RoPE is precomputed on the host as cos and sin inputs for the bucket.
- The embedding lookup and the ordinal and type embeddings run on the host (Rust, a simple gather) and the model starts from embeddings. This keeps the 256k vocab tables of the x tier off the ANE.
- LayerNorm is written as a channel norm that keeps the original affine order.
- Weights are FP16, or W8 with a k-means palette (laya-coreml's W8 palette passed its fidelity checks, while uniform W8, W6 and W4 failed).

Buckets: state tower at L in {32, 64, 96, 128, 192, 256} with B = 1, and question tower at {rows 1, 4, 16} x {tokens 32, 64, 128}. The state memory K and V are an output of the state model and an input of the question model, kept as `MLMultiArray`s backed by our own IOSurface buffers so they are not copied. States longer than 256 tokens go to kime-metal automatically. The rule is set per chip by a measured crossover table in the release.

### Runtime

- Bindings are `objc2-core-ml`. Models are loaded with `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`. At load we check with `MLComputePlan` that every op in the state and question models is placed on the ANE. If more than 1% of ops fall back to the CPU, kime logs a warning and routes that bucket to Metal. laya-coreml saw requests for the ANE silently land on the CPU at 78 to 82 ms, and we must never do that quietly.
- The compiled model cache (`.mlmodelc`) is keyed by the package hash and stored in `~/Library/Caches/kime/`. The first compile of all buckets takes several seconds, and is done once per machine and OS version.
- Predictions use `MLModel.prediction(from:options:)` with output backings, so outputs land in our buffers.

### Targets on M3 Max (kime-v1-s-en on ANE, W8 palette)

| Metric | laya-coreml best | kime target |
|---|---|---|
| W1 latency p50 | 4.88 ms | 1.0 ms or less |
| Energy per decision | 0.1344 J | 0.0134 J or less |
| Power while serving | about 30 W | 8 W or less |

The energy target is the one that needs the ANE. On the GPU a decision takes about 0.5 ms at about 25 W, which is about 0.0125 J. The GPU also meets the target, but only just, and not at all on smaller chips. The ANE should come in around 0.004 to 0.008 J.

## Energy measurement

Energy per decision is measured the way laya-coreml did it, because `powermetrics` and macmon component sums disagreed in their tests. A small Rust helper, `kime-eval/smc`, reads the SMC `PSTR` (total system power) key at 100 Hz through IOKit. We subtract the idle baseline measured for 30 s before the run, integrate over a run of at least 60 s of back to back decisions, and divide by the number of decisions. The report includes the idle baseline and the power trace.

## iOS and visionOS

The same two backends build for `aarch64-apple-ios`. The ANE backend matters most there. `kime-py` does not apply. Swift users call the C ABI exported by `kime-engine` (`kime_decide_json(ctx, request_json, out_buf)`), wrapped in a small Swift package, `KimeKit`. The spec for KimeKit is a v1.1 item in 16.

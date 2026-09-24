# Changelog

Notable changes, newest first. The project is pre-1.0 and makes no compatibility promise until it does. The minor version is the number of milestones finished, per [CONTRIBUTING.md](CONTRIBUTING.md), and the milestones are the issues at https://github.com/tamnd/kime/milestones.

## Unreleased

- `kime-metal` runs the compat graph on Apple GPUs. The kernels are Metal Shading Language compiled when the backend starts, the GEMMs are kime's own on simdgroup matrix tiles with bias, GELU, ReLU and accumulate fused into the store, and a run is one command buffer over buffers the CPU and GPU share, so there is no copy in or out. On a base M4 (10 core GPU) with the English Laya model and the first 40 parity cases (80 questions), one question at a time takes 55.6 ms at p50 in FP16 and 61.0 ms in FP32, and batches of 16 take 38.4 ms per question in FP16 and 41.5 ms in FP32. Laya 0.3.9 on PyTorch MPS on the same Mac takes 74.1 ms at p50 (97.3 ms mean) and 58.1 ms per question with its own batching, and kime's CPU plan 113 ms per question batched. FP32 agrees with Laya on all 625 parity questions for both checkpoints with logits within 4.0e-4. FP16 keeps the GeGLU input in FP32, unlike CUDA, and agrees on all 625 English questions with probabilities within 4.7e-3 and on 624 multilingual ones, the other the same near tie with a margin of 2.2e-4 that CUDA sees.

## 0.0.10

- The CPU thread pool waits only for the tasks of the job it runs, not for every worker to wake. An empty job went from over 100 µs to about 1 µs on the M4 and from milliseconds to a few µs on a loaded EPYC, so single questions are no longer held back by the pool. On the Laya parity questions kime now answers one at a time faster than Laya 0.3.9 on the same Mac.
- Attention runs in f32, taking four queries at a time with register tiles, and uses the vectorized exp. It is 5.6x to 6x faster on x86 with the default target, where the old f64 `mul_add` was a call, and 1.2x to 3x on the M4 depending on load. The worst Laya logit error gate moves from 1.5e-4 to 2.5e-4. The probability gate stays at 2e-5.
- GELU runs in f32 without calls, so the GeGLU loop vectorizes. It was an f64 `erf` call per element. The op is 5.8x to 8x faster on the M4 and 3.2x on an EPYC with the default x86 target, and Laya parity still agrees on all 625 questions for both checkpoints.

## 0.0.9

- `kime-cpu` builds on Linux and Windows again. 0.0.8 sized the GEMM scratch with a name from the macOS only Accelerate module inside `cfg!`, which still type checks on every OS, so 0.0.8 only compiled on macOS. Use 0.0.9 instead of 0.0.8.
- The publish script runs clippy for the other targets rustup has installed before it publishes, because `cargo publish` only builds for the machine it runs on.

## 0.0.8

The CPU GEMM is several times faster, and there is a first INT8 path. On the M4 Mac, 8 threads, the first 40 parity cases (80 questions), kime answers batches of 16 in 122 to 132 ms per question against 263 to 322 ms for Laya 0.3.9 with its own batching, run back to back on a machine with a load average of 20 to 30 from other work. One question at a time kime is still slower, 337 to 350 ms against 211 ms.

- `kime-cpu` packs the weights once into panels of 16 rows and runs a 6 by 16 micro kernel that keeps its outputs in registers, in place of one dot product per output. The sums keep their order, so the bits are the same for any batch, split or thread count. On the M4 at 8 threads, one question at a time went from 869 to 246 ms and batches of 16 from 572 to 175 ms per question on a quiet machine.
- On macOS the GEMM goes to Accelerate, which runs on the AMX units. Every call is 64 rows, at most 256 columns and at most 128 of k, with the k blocks summed in f64, so a question still gets the same bits alone or in any batch. Against the packed kernel it is 1.4x to 1.7x faster at single question shapes and 2.1x to 2.5x on batches of 960 rows. A bias with the Accumulate epilogue was added at the wrong column after the first chunk of 256 columns, which a new test catches.
- `kime-cpu` has an INT8 GEMM: weights rounded per output channel, activations per row, exact sums in i32, with an `smmla` kernel on Arm cores that have i8mm and a scalar kernel elsewhere. `kime predict --precision int8` and `Precision::Int8` turn it on. It is never picked automatically, because on Laya's checkpoint it agrees with FP32 on 95.5% of questions and spec 10 asks for 99.5%. On the M4 it is 1.3x to 1.5x faster than Accelerate FP32 on batches.
- `Device::Cuda` with `Precision::Int8` is an Unsupported error instead of silently running something else.

## 0.0.7

- `kime-cli` builds on Windows again. 0.0.6 imported `PathBuf` for `kime pull` on every OS, but only the unix code path uses it, so the unused import stopped the Windows build under `-D warnings`.

## 0.0.6

kime answers questions end to end. The engine lays out requests as Laya does, runs them on the CPU or CUDA, and builds Laya's answers from the logits, and the CLI can pull a model and predict with it. On the CPU, every parity answer equals Laya's, except a few rounded numbers that are off by 1 in the fourth decimal.

- `kime pull` downloads a model into the Hugging Face cache in the layout `huggingface_hub` uses, blobs named by ETag with relative links from the snapshot, so Laya and kime reuse each other's downloads. It goes through the system `curl`, hands `HF_TOKEN` to curl on stdin rather than on its command line, honours `HF_ENDPOINT`, and checks every LFS file against its SHA-256 before keeping it. Pulling both Laya checkpoints on server3 took 29.6 s for 846 MB and 43.4 s for 678 MB, the files are byte identical to the hub's, and a second pull fetches nothing.
- `kime-engine` has the `decide` API from spec/07-engine.md, re-exported by the `kime` crate: `Kime::builder().model("laya").device(Device::Auto).build()`, then `decide`, `decide_batch` and `decide_async`. It lays out each question as Laya does, packs the questions of all the requests in a call into shared batches on the CPU or a CUDA GPU, and builds Laya's answers from the logits. Models are found by path, by `hf://org/repo/subfolder` or by the aliases `laya` and `laya-multilingual`, in the Hugging Face cache layout so Laya's downloads are reused. `Request` gained a builder (`Request::new(state).choice(..).score(..).noul(..)`). Run through `decide_batch` on the CPU, all 200 parity requests per model come back with every answer equal to Laya's, except 7 English and 4 multilingual requests that differ by 1 in the fourth decimal of a rounded number.
- `kime predict` answers questions from the command line: `--state @file --questions @file` or `--request @file`, `--format table`, and `--batch` for JSONL in and out, where a bad line gets an error line and the rest go on. `--device auto|cpu|cuda:N`, `--threads` and `--precision f16|f32` pick where it runs.
- `kime-core` builds answers from logits the way Laya does, in the module `answer`: the calibrated softmax with the checkpoint's temperatures, entropy confidence, the expected score, the noul probability and the act probability, rounded to 4 places. It uses numpy's float types, numpy's pairwise sums and ports of numpy's AVX2 `exp` and `log`, which are bit exact against numpy over 2 million values each. Given Laya's own logits, all 400 parity responses equal Laya's byte for byte. With libm's `exp`, 2 of them were off in the fourth decimal.

## 0.0.5

The CUDA backend is finished for M0. Rope runs inside attention, every kernel has its own test against a naive version, and a warm plan allocates nothing on the host. The GPU tests skip cleanly on machines without CUDA, which is what broke the 0.0.4 release run.

- `kime-cuda` has a test for each kernel on its own against a naive f64 version on the host, with the real row counts below the launched ones and the padding rows checked to stay untouched. Attention is checked against dense attention with the mask built as a matrix, with and without rope and a window, and agrees to 4.2e-7 in FP32 on an RTX 4090. A second test counts host allocations and finds none once a CUDA plan is warm, in FP32 and FP16.
- `CudaBackend::new` returns an error when the CUDA driver, NVRTC or cuBLASLt is not installed, where before cudarc panicked. This is what failed the 0.0.4 release run, which has no GPU.
- The allocation counting tests run without libtest's harness, whose 60 second warning allocated inside the counted window on slow debug runs.

- `kime-cuda` fuses rope into attention. When attention directly follows rope on the same rows, it rotates q and k in registers as it loads them, so q and k are no longer written back and read again. On an RTX 4090 an 80 token question spends 195 us in attention against 225 us in attention and rope before, and batches of 16 went from 1.23 to 1.20 ms per question in FP16.

## 0.0.4

The CUDA path is 10x Laya. Each plan is captured as one CUDA graph, the kernels were rewritten, and GEMMs run the cuBLASLt algorithm tuned for their shape. On an RTX 4090 one question takes 1.72 ms at p50 in FP16 against 17.2 ms for Laya on the same card, with the same answers on every English parity question.

- `kime-cuda` pins cuBLASLt's algorithm per GEMM shape from a table, `tuned.txt`, keyed by GPU. cuBLASLt's first choice is often not its fastest at the few rows one question has: on an RTX 4090 at 80 rows the 1024 by 1024 projection runs in 10.6 us on its seventh ranked algorithm against 18.8 us on the first. A run with `KIME_CUDA_TUNE=1` times every rank of every GEMM in place and prints the lines to add, and the table ships 73 of them for the RTX 4090. Pinning by rank keeps plans the same on every start, as spec 07 asks.
- `kime-cuda` has faster kernels for attention, layer norm, rope, GEGLU and bias. Attention runs 8 warps of 2 query rows each over 32 key tiles held in padded shared memory, and is 1.5x to 1.9x faster than before (6.7 us instead of 10.8 us for 128 tokens). Layer norm runs one warp per row. Every kernel now loads all it needs into registers before it stores anything, since a store between loads keeps the compiler from overlapping them.
- The compat stage has 20 buckets instead of 9, adding 48, 80, 96, 112, 160, 192, 224, 288, 320 and 384 tokens, so a typical question pads to less.
- Together, on an RTX 4090 with the English model in FP16, one question at a time is 1.72 ms at p50 against 2.08 ms before, 10.0x Laya's 17.2 ms, and batches of 16 are 1.23 ms per question. Parity holds: English agrees with Laya on all 625 questions with probabilities within 5.4e-3, and multilingual on 624, the other a near tie with a margin of 2.2e-4.
- `kime-cuda` has a tiled attention kernel and a new layer norm. Attention stages 32 keys of k and v at a time in shared memory for a block of 16 query rows and prefetches the next tile while it scores the current one, where before every warp read its own keys from memory. Layer norm runs one block per row with the row held in registers. On an RTX 4090 a 128 token question now spends 8.2 us per attention call instead of 28.8 and 3.4 us per layer norm instead of 9.0, and one question at a time is 2.08 ms at p50 in FP16 (8.3x Laya). A new test runs every kernel against the FP32 reference on a small random model, with batches that cross sequence ends and window edges.
- `kime-cuda` captures each plan as one CUDA graph when it is lowered: the copy of the batch's index tables in, every launch, and the copy of the outputs back, both copies through page locked buffers. A run is one graph launch and one wait, with no allocation. On an RTX 4090 with the English model one question at a time went from 4.03 ms to 2.68 ms at p50 in FP16 (6.4x Laya's 17.2 ms) and from 5.57 ms to 4.83 ms in FP32.

## 0.0.3

The engine runs on a plan, and on the GPU. Graphs are lowered once per bucket and replayed, on the CPU with zero allocations once warm and on NVIDIA GPUs through cuBLASLt and kime's own kernels. On an RTX 4090 the FP16 path answers one question in 4.03 ms at p50 against 17.2 ms for Laya on the same card.

- `kime-tensor` has the engine's core: an op graph that model builders emit, the `Backend` trait, arena layout by live ranges, and the bucket table as data (`buckets.txt`, or any table loaded at run time). The `Executor` picks the smallest bucket that holds a batch, builds its plan on first use and replays it after that. `kime-model` emits the whole Laya compat graph as ops.
- `kime-cpu` lowers that graph into a CPU plan run on a persistent worker pool that spins briefly and then sleeps. Once a bucket is warm a batch allocates nothing, which a counting allocator test checks. The plan gives the same bits as the FP32 reference on both Laya checkpoints and all 625 parity questions per model, and a question gives the same bits alone as inside any batch. On an i9-13900K with the English model, batches of 16 take 178.6 ms per question against 227.6 ms for the reference, and one question at a time is 160.5 ms at p50 on 16 threads. Matrix products and attention take 85 to 95% of that time, so they are the next target.
- `kime-cuda` runs the compat graph on NVIDIA GPUs. The GEMMs go through cuBLASLt and the other ops are kime's own kernels, compiled with NVRTC when the backend starts, so building needs no CUDA toolkit and the driver, NVRTC and cuBLASLt are loaded at run time. Every launch is sized for the bucket and reads the batch's real counts from the device, ready for graph capture. In FP32 it agrees with Laya on every parity question, with logits within 2.5e-4, and on an RTX 4090 it runs batches of 16 at 4.07 ms per question, 44x the CPU plan. FP16 (FP16 weights and GEMM inputs, FP32 residual, qkv and heads) runs them at 1.64 ms per question with probabilities within 3.6e-3 of Laya.
- `kime doctor` no longer lists AMX, because std's detection of it is unstable and broke the Linux build.

## 0.0.2

The first model runs. Both Laya checkpoints load, pack into `.kime` and run through the FP32 CPU reference, which agrees with Laya on every parity question. The input side (request validation, Laya exact rendering, both tokenizers and the sequence layout) matches Laya byte for byte and id for id.

- `kime-tok` encodes and decodes with both Laya tokenizers, byte level BPE for ModernBERT and SentencePiece style BPE with byte fallback for mmBERT, with no dependency on Hugging Face tokenizers. Ids match Hugging Face tokenizers 0.23.2 on all 787,214 lines of the MASSIVE train split in 51 languages plus 200,000 fuzz strings. On one core it is about 10x faster than Hugging Face on that corpus and does a 2 KB English request in about 15 microseconds.
- `kime-core` parses and validates `/v1/systemone` requests, reporting every problem at once in FastAPI's shape, and renders compat questions and states the way Laya does, including Python's `json.dumps` spelling of floats and escapes. `kime-tok` lays out Laya's `[CLS] head [SEP] [MASK] option ... [SEP] state [SEP]` sequence with its budgets. On the 200 parity cases all 1,250 questions render to Laya's text and lay out to Laya's ids and markers. The whole input pipeline takes 8.6 microseconds per request at p50 against Laya's 385.
- `kime-model` opens both Laya checkpoints from their directories and binds every tensor to the compat graph, naming each missing, extra or misshapen tensor when one does not fit. The names, shapes and values match what Laya itself builds from the same files. The safetensors reader treats files as hostile and checks every offset before use.
- `kime convert` packs a checkpoint into one `.kime` file with a checked index, a blake3 model hash and every tensor on a 4 KiB boundary, and unpacks it again. The round trip is bit exact, down to the original `model.safetensors`. Opening a `.kime` takes 0.2 ms, and checking its hash takes 26 ms for the 843 MB English model on an i9-13900K. Three fuzz targets cover both parsers.
- `kime-tok` builds the multilingual tokenizer in 179 ms instead of 333 ms, and encodes with it 1.7x faster, after a fix to how its hash spreads string and pair keys.
- `kime-cpu` runs the whole compat graph in FP32: ModernBERT and mmBERT with sliding window and global attention, the two layer decision head, the scorer and the act head. It is the reference the other backends are held to. On all 625 parity questions per model it picks the same option as Laya, with logits within 1.1e-4 for English and 2.9e-5 for multilingual, which is as close as Laya run with 8 threads is to Laya run with 32. The output has the same bits for any thread count.

## 0.0.1

The skeleton. The specification in `spec/`, the workspace with its fifteen crates laid out as `spec/07-engine.md` describes, the confidence formulas and largest remainder rounding in `kime-core` with the worked example from `spec/04-semantics.md` as a test, a `kime` binary that answers `--version` and `doctor`, and the CI, nightly and release workflows. No model runs yet.

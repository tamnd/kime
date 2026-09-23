# Changelog

Notable changes, newest first. The project is pre-1.0 and makes no compatibility promise until it does. The minor version is the number of milestones finished, per [CONTRIBUTING.md](CONTRIBUTING.md), and the milestones are the issues at https://github.com/tamnd/kime/milestones.

## Unreleased

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

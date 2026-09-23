# kime specification

kime (決め, "the decision") is a typed decision engine written in Rust. It takes a state (text, an email, a ticket, a JSON document, an agent's view of a web page) and a set of typed questions, and returns calibrated structured answers without generating any text. It covers every feature of TypeSafe Jev and of Laya, speaks both of their wire formats, and is built to beat both by 10x on every speed and cost benchmark while matching or beating them on accuracy and calibration.

Repository: `github.com/tamnd/kime`. License: Apache 2.0.

## Files

| File | Topic |
|---|---|
| [00-README.md](00-README.md) | This index |
| [01-overview.md](01-overview.md) | Goals, feature parity matrix, the 10x scorecard, non-goals |
| [02-prior-art.md](02-prior-art.md) | What Jev and Laya are, how they work, their numbers and their known defects |
| [03-api.md](03-api.md) | HTTP wire protocol, compatibility with Jev and Laya, extensions, errors, limits |
| [04-semantics.md](04-semantics.md) | Question types, answer fields, confidence formulas, calibration, abstention |
| [05-model.md](05-model.md) | Model architectures: the laya compat graph and the kime-v1 split encoder |
| [06-input.md](06-input.md) | Tokenizers, state rendering, sequence layout, budgets and truncation |
| [07-engine.md](07-engine.md) | Rust workspace, runtime, memory, execution plans, determinism |
| [08-cuda.md](08-cuda.md) | NVIDIA backend: kernels, CUDA graphs, precision |
| [09-apple.md](09-apple.md) | Apple backend: Metal and the Neural Engine |
| [10-cpu.md](10-cpu.md) | CPU backend: x86 and ARM kernels, threading, int8 |
| [11-serving.md](11-serving.md) | Server, scheduler, batching, caches, router, language id |
| [12-training.md](12-training.md) | Data, losses, RLCD, calibration fitting, distillation, quantization |
| [13-benchmarks.md](13-benchmarks.md) | Benchmark suite, workloads, hardware, methodology, targets |
| [14-sdks-cli.md](14-sdks-cli.md) | Rust crate API, Python and TypeScript SDKs, CLI, packaging |
| [15-testing.md](15-testing.md) | Parity tests, fidelity tests, fuzzing, CI |
| [16-roadmap.md](16-roadmap.md) | Milestones, risks, open questions |

## Reading order

Read 01 and 02 first to understand what we are building and why. Then 03 and 04 define what users see. 05 and 06 define the model and its input. 07 to 11 are the implementation. 12 is how the weights get made. 13 and 15 are how we prove it works. 16 is the order we build it in.

## Sources

The research behind this spec was done on 2026-09-23 from these sources.

- TypeSafe docs at https://docs.typesafe.ai (all 111 pages via llms-full.txt), the live OpenAPI spec at https://api.typesafe.ai/openapi.json, the official SDKs `typesafe-sdk` (Python 0.7.1) and `@typesafe-ai/sdk` (JS 0.6.0), `system-one-adapter`, https://evals.typesafe.ai, the launch blog, and https://github.com/browser-use/jev-ultrafast.
- Laya at https://github.com/NandhaKishorM/laya (0.3.7), its Hugging Face checkpoints under `convaiinnovations/`, the dev.to article, the impossibl.com hosted docs, and the Apple ports https://github.com/mizorewww/laya-mlx and https://github.com/mizorewww/laya-coreml.
- Rust and kernel ecosystem: candle, burn, cudarc, ort, text-embeddings-inference, FlashAttention 2/3/4 papers, the ModernBERT, mmBERT and Ettin papers, Apple's ANE transformer work, and the calibration literature (Guo 2017, Nixon 2019, Roelofs 2022).

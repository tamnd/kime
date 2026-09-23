# NVIDIA backend

`kime-cuda` runs plans on NVIDIA GPUs from sm_75 (T4) to sm_120 (Blackwell consumer) and sm_100 (B200). It is built on `cudarc` 0.19 for the driver API, cuBLASLt and NVRTC. Kernels are written in CUDA C++ in `kernels/cuda/`, compiled at build time to cubins for each target architecture plus PTX for forward compatibility, and embedded in the binary. NVRTC is used only in development mode (`KIME_CUDA_JIT=1`), so kernel edits do not need a full rebuild.

## Targets

| Arch | Example GPUs | Tensor core dtypes used | Notes |
|---|---|---|---|
| sm_75 | T4 | FP16 (FP32 accumulate), INT8 | No BF16. This is the baseline Laya numbers were measured on, so it matters. |
| sm_80, sm_86 | A100, A10G, RTX 3090 | FP16, BF16, INT8 | cp.async pipelines |
| sm_89 | L4, RTX 4090, L40S | FP16, BF16, FP8 E4M3, INT8 | The main cost target: cheap, common in clouds |
| sm_90 | H100, H200 | FP16, BF16, FP8 | TMA and wgmma kernels for the large buckets. FA3 style attention. |
| sm_100, sm_120 | B200, RTX 5090 | FP16, BF16, FP8, FP4 (not used in v1) | Uses the sm_90 code path until tuned kernels land |

## Precision

- Default is FP16 weights and activations with FP32 accumulation, on all archs. BF16 is not used for inference: the checkpoints are fine in FP16, and T4 lacks BF16.
- LayerNorm statistics, softmax, the scorer's last layer and the logits are FP32.
- `--precision fp8` (sm_89 and newer) quantizes the linear weights of both towers to E4M3 per output channel, with activation scales per tensor calibrated offline and stored in the prepacked cache. The published results on BERT (SQuAD F1 88.19 in FP16 and 88.09 in E4M3) suggest the accuracy cost is near zero. We still require the release gate in 12: argmax agreement at or above 99.5% and ECE change at or below 0.005 on the calibration split. Embeddings, norms, attention and the scorer stay FP16 or FP32.
- `--precision int8` (all archs, mainly for T4) uses W8A8 with SmoothQuant style per channel migration, with the same gate. On T4, INT8 tensor cores are 2x FP16.

## Kernels

The state tower layer is 7 launches. The question tower layer is 9. Everything else is fused.

| # | Kernel | Fuses | Implementation |
|---|---|---|---|
| 1 | `embed_ln` | token gather, embedding LayerNorm, write FP16 | One warp per token, vectorized 16 byte loads |
| 2 | `ln` | LayerNorm without bias, FP32 stats | Only before the first attention of each layer. Other norms are fused into the previous GEMM's epilogue or the next kernel's prologue. |
| 3 | `gemm_qkv` | Wqkv GEMM plus RoPE plus scatter into Q, K, V head major layout | cuBLASLt for large buckets. Our own CUTLASS 3 kernel with a RoPE epilogue for token counts up to 2k, where the extra launch costs more than cuBLASLt saves. The epilogue reads cos and sin from a table indexed by the position ids buffer, so segment and restart positions come for free. |
| 4 | `attn_varlen` | FlashAttention 2 forward, variable length, optional sliding window, optional segment ids | Our own kernel, head dim 64 only, specialized for window 128 on local layers (each query block reads at most 3 key blocks of 64). Global layers use the full tile loop. Segment ids block cross-segment tiles at the tile level, so block diagonal masks cost nothing on the skipped tiles. On sm_90 the global path uses a wgmma and TMA kernel in FA3 style. |
| 5 | `gemm_o_res_ln` | Wo GEMM plus residual add plus the next LayerNorm | The epilogue writes both the residual stream and the normalized copy. A CTA owns whole rows (d = 512 or 768 fits in one CTA tile width), so the LayerNorm reduction stays in the epilogue with no extra pass. |
| 6 | `gemm_wi_geglu` | Wi GEMM plus GELU(a) times g | The epilogue pairs columns i and I + i, so Wi is repacked at load time to interleave a and g in 64 column groups. Writes only I columns, halving the traffic. |
| 7 | `gemm_wo_res_ln` | Wo of the MLP plus residual plus the next layer's LayerNorm | As kernel 5 |
| 8 | `attn_q_masked` | Question tower self attention with the header, segment and marker mask | The mask is built from three small index arrays in shared memory, never materialized. Rows are short (usually under 256 tokens), so one CTA handles a whole row. |
| 9 | `cross_attn` | Queries from the question row against the cached K and V of its state, GQA 2 kv heads | Split-KV FlashDecoding style when a state has more than 2k tokens: many question rows read the same state, so the kernel groups rows by state and streams K and V once per group through shared memory. The combine step has a fixed number of splits per bucket, so the result stays deterministic. |
| 10 | `kv_project` | Final state LayerNorm plus the K and V projections of all N_q question layers in one GEMM | Concatenates the N_q projection matrices into one [N_q * 256, d] matrix and does a single GEMM, writing straight into cache slab pages through a page table. |
| 11 | `markers_scorer` | Marker gather, ordinal embedding add, scorer LN, Linear, GELU, Linear, type bias, write FP32 logits | One warp per marker. The d x d scorer matrix is read from L2, which is shared across warps. |
| 12 | `softmax_calib` | Per-question temperature, softmax, confidence formulas | Only when the engine is asked to post-process on device (the batch endpoint with many items). Normally this runs on the host in Rust, because it is a few hundred floats. |

Local layers with short sequences: when every sequence in the batch is at most 129 tokens, the window covers the whole sequence and a local layer is exactly a global one. The plan builder then lowers local layers to the global kernel, which has fewer branches.

## Launch overhead and CUDA graphs

At batch 1 with a short state, the arithmetic is tens of microseconds and launch overhead dominates. With 12 state layers of 7 kernels, 4 question layers of 9 kernels, and about 5 extra kernels, the s tier is about 125 launches per request. At 3 to 5 microseconds of CPU launch cost each, that is 0.4 to 0.6 ms before any work. So:

- Every (model, stage, bucket) plan is captured as a CUDA graph and launched with `cuGraphLaunch`, about 8 to 15 microseconds total.
- Input changes between launches (ids, cu_seqlens, positions, marker indexes, row to state map, state memory page table) are written into fixed device buffers with one `cuMemcpyHtoDAsync` of a packed staging struct. The graph is never updated with `cuGraphExecKernelNodeSetParams`, because a copy is faster and simpler.
- Both stages of one batch are enqueued back to back on the same stream, with no host sync in between. The host syncs once, on an event, before reading the logits.
- Programmatic dependent launch (sm_90 and newer) lets each kernel's prologue overlap the previous kernel's tail. It is enabled inside the captured graphs.

## Streams and concurrency

Each GPU gets one worker thread and two streams. Stream A runs batches. Stream B uploads the next batch's inputs and downloads the previous batch's logits. The worker double buffers, so while batch n runs, batch n+1's inputs are uploaded and batch n-1's logits come back. With MIG or MPS, each instance is a separate device to kime.

## Latency budget, kime-v1-s-en, W1 (96 state tokens, 1 question of 24 tokens), L4

| Step | Time |
|---|---|
| HTTP parse and validate (host) | 8 us |
| Tokenize (host) | 6 us |
| Scheduler hop (host) | 5 us |
| Input upload (one 4 KiB copy) | 6 us |
| State stage graph: 12 layers at 128 token bucket | about 190 us |
| KV project | 12 us |
| Question stage graph: 4 layers at 32 token bucket | about 70 us |
| Logit download and host softmax | 10 us |
| Response serialization | 5 us |
| Total in process | about 0.3 ms |
| Total over localhost HTTP | about 0.35 ms |

On a T4 the device steps take about 2.5x longer (fewer SMs, older tensor cores), so the in process total is about 0.7 ms. The target in 13 is 3.3 ms, so there is margin for the first version of the kernels to be slower than planned.

## Laya compat on CUDA

The compat graph uses the same kernels plus:

- Biased linears in the head (cuBLASLt `BIAS` epilogue).
- The head's full attention with the key padding mask. With varlen this is just `attn_varlen` without a window.
- The ReLU FFN (a `RELU_BIAS` epilogue).

One sequence per question, all questions in one varlen batch. There is no padding, so a 10 question call costs the sum of the actual lengths, not 10 times the maximum length.

## Profiling and tuning

- `KIME_PROFILE=1` inserts CUDA events around each op in a non-graph run and prints a per op table. It matches the Laya breakdown format so the two can be compared.
- NVTX ranges per stage are always compiled in and cost nothing when no profiler is attached.
- `kime bench --tune` sweeps GEMM tile configs per bucket on the local GPU and writes the winners into the prepacked cache. Released binaries ship tuned defaults for T4, L4, A10G, A100, H100 and RTX 4090.

# Benchmarks

This file defines how we measure "10x better": the workloads, the quality suites, the hardware, the baselines, the method, and the target for every row of the scorecard. All of it is implemented in `kime-eval` and run by `kime bench` and `kime eval`, so anyone can reproduce the numbers.

## Principles

- **Same hardware, same inputs.** Every speed comparison against Laya runs on the same machine, with the same requests, in the same session. Jev is hosted, so we compare against it only on end to end HTTP latency from a stated location, and also against the numbers TypeSafe publishes.
- **Baselines at their best.** Laya is measured with its best known settings: PyTorch threads tuned on CPU, fp16 autocast on GPU, the Router preloaded, torch.compile where it helps (laya #217 reports 45 to 33 ms on T4), and laya-mlx and laya-coreml on Apple. If a baseline improves, we rerun.
- **Tail latency, not just medians.** Every latency row reports p50, p95 and p99, with at least 2,000 measured calls after 200 warmup calls.
- **Quality on held out test splits only**, with the contamination checks from 12.
- **Every published number comes with the command that produced it**, the git hash, the model hash, the driver and OS versions, and the raw per call data as a Parquet file.

## Workloads

| Id | Description | State | Questions |
|---|---|---|---|
| W1 | Single decision on a short message | 96 tokens, an email body | 1 choice, 4 options |
| W2 | Support ticket triage (Laya `triage` preset) | 300 tokens | 5 mixed: 1 choice of 6, 2 score, 2 noul |
| W3 | Ten questions over a document | 512 tokens | 10 mixed |
| W4 | Fifty questions | 512 tokens | 50 mixed |
| W5 | Browser agent step (jev-ultrafast format) | about 5.3k tokens: page text 6,000 chars plus an element table | operation choice of 8, plus 3 target heads of up to 250 options |
| W6 | Long context | 1,024 tokens (Laya's max) and 8,192 and 32,768 tokens (kime only) | 3 mixed |
| W7 | Multilingual | 200 tokens in each of the 51 MASSIVE languages | 1 choice of 20 |
| W8 | Agent loop | 17 consecutive W5 steps from the recorded Google Flights trace, one segment changing per step | as W5 |
| W9 | Throughput | W2 at saturation with 64 concurrent clients | as W2 |

W5 and W8 inputs are generated from jev-ultrafast's recorded `flights-measurement.json` trace and a replay of its snapshot format. They are checked into `kime-eval/fixtures/`.

### W8 estimate

In the Flights trace, a step usually changes one or two elements' values and appends one history entry. With segment mode, the page text, the unchanged element segments and the old history hit the cache. So a step recomputes the low 9 state layers on about 5 to 10% of the tokens, the top 3 layers on all 5.3k tokens, and the question tower. For kime-v1-s-en that is about 110 GFLOP, where a cold state costs about 480 GFLOP. This is where the target of 17.8 ms per step (cold) and about 6 ms (warm) on an L4 comes from.

## Quality suites

| Suite | Metric(s) | Source |
|---|---|---|
| typed-decisions test | accuracy, soft accuracy, Brier, ECE, score MAE, per workflow, per primitive | LocalLLaMA/typed-decisions, 400 cases, 2,000 decisions |
| TypeSafe workflow evals | accuracy against the LLM consensus labels | evals.typesafe.ai (security, agent trace, invoice, customer service) |
| AG News | accuracy | test split |
| DAIR Emotion | accuracy, zero probability rate on gold | test split |
| Banking77 | accuracy, all 77 labels in one choice | test split |
| CLINC150 | accuracy, out of scope recall | test split |
| SST-2, SST-5 | accuracy | validation, test |
| XNLI | accuracy, 15 languages | test split |
| MASSIVE intent and scenario | accuracy, 51 languages, macro and per language | test split |
| Laya application themes | accuracy and macro F1: spam, phishing, jailbreak, toxicity, RAG relevance, 10-way triage, model routing | the 400 case sets from the Laya repo |
| Laya in-task families | accuracy, ECE, NLL for the 13 families | the 23,024 question set from Laya's HF eval |
| Catalog selection | top-1 over 100 items (Laya #171) | the issue's fixture |
| CLERC rerank | top-1, top-5, top-10 over BM25 top 30 | TypeSafe cookbook setup |
| Order invariance | flip rate under 5 random option permutations | MASSIVE en, Emotion, XNLI |
| Score mirror and noul polarity | agreement rates | 12 release gate sets |
| Calibration | ECE (15 bins and sweep), Brier, NLL, reliability diagrams with bootstrap CIs | every suite above |
| Selective prediction | accuracy at 50% and 80% coverage, AUROC of confidence against correctness | every suite above |

## Hardware

| Class | Machines |
|---|---|
| NVIDIA | T4 (Laya's baseline), L4, A10G, A100 40GB, H100 SXM, RTX 4090 |
| Apple | M3 Max (the laya-mlx and laya-coreml baseline), M4, M2 (floor) |
| CPU x86 | Ryzen 9 6900HX (Laya's CPU baseline), AWS c7i.4xlarge (Sapphire Rapids with AMX), a Zen 4 desktop |
| CPU ARM | AWS c8g.4xlarge (Graviton 4) |

## Measurement

- **In process latency.** Time from calling `decide` to having the `Response`, with the model loaded and warm. For Laya this is `router.predict(state, questions)` or `agent.system_one(...)`. For laya-mlx and laya-coreml, their own `predict`.
- **HTTP latency.** From a Rust load generator on the same host (loopback), with keep-alive and h2, to kime-serve and to laya-serve. Jev is measured from a cloud VM in us-west-2 and one in eu-central-1, at the same time as kime-serve on an L4 in the same region, so network cost is visible separately.
- **Throughput.** Questions per second at saturation, with p99 at or below 5x the unloaded p50. Reported per device, and per dollar using public on-demand hourly prices on the day of the run.
- **Cold load.** From process start to the first answered request, with the page cache dropped (`echo 3 > /proc/sys/vm/drop_caches`, or `purge` on macOS) and with it warm. Reported separately.
- **Memory.** Peak RSS and peak device memory, from `/proc`, NVML or `task_info`.
- **Energy.** On Apple, SMC `PSTR` as described in 09. On NVIDIA, NVML total energy counters. Energy is reported per decision.
- **Cost per million input tokens.** Throughput in tokens per second (counted the Jev way, as `usage.input_tokens`) at saturation, divided into the hourly instance price. Jev's list price is $0.042. The target is $0.0042 or less on an L4 at on-demand prices.

## Speed and cost targets

Baseline is the best of Laya, laya-mlx, laya-coreml and Jev on the same class. The kime target is at most one tenth of the baseline (or ten times, for throughput). Numbers marked hosted are Jev's published or measured end to end numbers.

| Row | Hardware | Baseline | kime target |
|---|---|---|---|
| W1 p50, in process | T4 | 32.8 ms (laya-multilingual) | 3.3 ms |
| W1 p50, in process | L4 | Laya measured on the day | one tenth of it |
| W1 p50, over HTTP | L4, same region | Jev 111 ms (hosted) | 11 ms. Expect under 1 ms plus network. |
| W3 per question | T4 | 7.2 ms (laya-multilingual) | 0.72 ms |
| W4 total | T4 | 337.4 ms | 33.7 ms |
| W2, 13 question GDPR style call | L4 | Jev 0.27 s (hosted) | 27 ms |
| W5 cold state | L4 | Jev 178 ms median (hosted) | 17.8 ms |
| W8 mean step | L4 | Jev 219 ms mean (3,720 ms over 17 requests) | 21.9 ms, expected about 6 ms |
| W1 p50 | M3 Max GPU | 7.39 ms (laya-mlx multilingual) | 0.74 ms |
| W1 p50 | M3 Max ANE | 4.88 ms (laya-coreml W8) | 0.49 ms on GPU, or 1.0 ms on ANE with the energy row met |
| Energy per decision | M3 Max | 0.1344 J (laya-coreml W8) | 0.0134 J |
| Batched throughput | M3 Max | 395 q/s (laya-mlx) | 3,950 q/s |
| W1 p50 | Ryzen 9 6900HX | 329 ms (Laya, tuned threads) | 32.9 ms |
| W9 throughput | L4 | Laya measured on the day | 10x |
| Cold load, warm page cache | any | about 2 s (Laya 0.3.7) | 200 ms |
| Peak memory, English model | M3 Max | 943.6 MiB (laya-mlx laya) | 94 MiB (s-en with W8 or INT8 weights) |
| Peak memory, multilingual model | M3 Max | 687.6 MiB (laya-mlx multilingual) | 170 MiB (s-x with INT8 weights and embeddings, about 145 MB of weights). See the note below. |
| Cost per 1M input tokens | L4 | Jev $0.042 | $0.0042 |

The multilingual memory row cannot reach 10x, because the 256k token embedding table of the mmBERT vocabulary alone is about 98M parameters (196 MB in FP16). Vocabulary trimming (05) brings it close, but we do not claim it. This shows how rows are handled when 10x is not reachable: the row stays in the scorecard with the reason written next to it, and is never quietly dropped.

## Quality targets

| Row | Best baseline | kime-v1-s target | kime-v1-m target |
|---|---|---|---|
| typed-decisions accuracy | 0.766 (laya-typed-decisions, fine tuned on its train split) | 0.78 zero shot, 0.80 after `kime train` on the train split | 0.80 zero shot, 0.82 fine tuned |
| typed-decisions soft accuracy | 0.580 (Jev) | 0.60 | 0.62 |
| typed-decisions Brier | 0.061 (laya-typed-decisions) | 0.055 | 0.050 |
| typed-decisions ECE | 0.144 (Jev raw) | 0.05 | 0.04 |
| typed-decisions score MAE | 0.242 | 0.22 | 0.20 |
| TypeSafe workflow evals | 67.8% (Jev) | 68% | 72% |
| AG News | 0.953 | 0.953 | 0.955 |
| DAIR Emotion | 0.600 | 0.62 | 0.65 |
| Banking77 | 0.870 (Jev) | 0.88 | 0.90 |
| MASSIVE 51 languages, macro accuracy | 0.366 (laya-multilingual) | 0.55 (s-x) | 0.62 (m-x) |
| MASSIVE, languages over 3x random | 45 of 51 | 51 of 51 | 51 of 51 |
| XNLI, 14 non-English languages | 0.731 | 0.74 | 0.78 |
| Catalog selection over 100 items | 92 (Jev) | 92 | 94 |
| Order flip rate, MASSIVE en | 0.13 (Jev) | 0.02 | 0.02 |
| ECE, every suite | best baseline per suite | 0.05 or the baseline, whichever is lower | same |
| AUROC of act_probability | 0.30 (Laya act head), 0.77 (Laya confidence) | 0.80 | 0.82 |

The quality rows use the words of the scorecard in 01. kime must equal or beat the best baseline everywhere, and must clearly beat it where the baseline has a known defect (calibration, order, noul polarity, large label sets, multilingual).

## Reports

`kime bench --suite speed --hw <label>` and `kime eval --suite quality` each write a Markdown report and a Parquet file. `kime report` merges them into the scorecard: one row per target, the measured value, the baseline value, the ratio, pass or fail, and a link to the raw data. The scorecard for each release is committed to `bench/results/<version>/` in the repo, and the README shows the headline table from the latest one.

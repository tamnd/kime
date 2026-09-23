# Training

This file covers how the native weights are made: the data, the teacher, the student distillation, the losses (including what RLCD really needs), the augmentations that fix Laya's known defects, calibration fitting, quantization, the release gate, and the `kime train` command users run for their own fine tunes.

## Tooling

Training lives in `kime-train`, built on Burn 0.21 with the CUDA backend (CubeCL kernels) and autodiff. Keeping training in Rust means the training graph and the inference graph come from the same `ModelSpec`, so there is no export step where the two can drift.

Burn is younger than PyTorch. We manage that risk in two ways:

- `tools/ref/` has a PyTorch implementation of both families, used only to check `kime-train`'s forward and backward on fixed inputs (see 15).
- The data pipeline writes shards in a neutral format (below), so a PyTorch trainer could take over without redoing data work. If Burn blocks us on a feature (for example FlashAttention backward with sliding windows), the fallback is our own CUDA kernel exposed as a custom Burn op, not a switch to Python.

Hardware for the release runs is a single 8x H100 node. Budget estimates are in the table at the end.

## Data

### Format

Every example is one state with one or more questions and soft targets.

```json
{"id": "...", "source": "banking77", "split": "train", "lang": "en",
 "state": "...",
 "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {...}}},
 "targets": {"q": {"probs": [0.0, 0.9, 0.1], "hard": 1, "weight": 1.0}},
 "episode": null}
```

Shards are zstd compressed JSONL of about 256 MB, with a manifest listing each shard's blake3 hash, source, license, split and counts. Every released model points to its data manifest version.

### Sources

1. **Public human labelled datasets, as typed questions.** For each dataset we convert the train split into questions with several instruction phrasings and option descriptions. The datasets are the ones Laya's HF eval file lists (intent, topic, emotion, sentiment, NLI, fact checking, reading comprehension, moderation, email triage, search relevance, instruction following, response quality, conversation outcomes), plus Banking77, CLINC150, MASSIVE (train split, 51 languages), XNLI (train where available, otherwise MultiNLI), AG News, DAIR Emotion, SST-2 and SST-5, GoEmotions, TweetEval, Jigsaw toxicity, deepset prompt-injections, JailbreakBench, MS MARCO and BEIR train splits for relevance nouls, and WANLI. Only splits whose licenses allow training are used. Each is listed in the manifest with its license.
2. **typed-decisions train split** (LocalLLaMA/typed-decisions, 1,200 cases). Its test split (400 cases, 2,000 decisions) is never used in training, calibration or model selection.
3. **LLM labelled synthetic data.** This is the part that makes the model general. A generator writes realistic states (support tickets, emails, invoices, security alerts, agent traces, web page element tables, logs, chat transcripts, JSON records) in 40 languages, and question sets over them in the Jev style: structured instructions, backticked state paths, objects as criteria, and 2 to 255 options. Three frontier LLMs label each question independently with probability outputs, using TypeSafe's own adapter prompt as the template. The soft target is the mean of their distributions. Examples where the three disagree strongly (pairwise total variation over 0.5 on hard targets) are dropped, and moderate disagreement stays as soft probability. Target size is 4M questions over 1M states.
4. **Agent traces.** States and element tables from about 20k browser runs on public sites, with operation and target heads in the jev-ultrafast format. They are labelled by replaying which action advanced the task, plus LLM labels for ambiguous steps. This is the training data for W5 and for large choice sets.
5. **Conversation episodes.** Multi-turn conversations with an outcome label (resolved, churned, escalated), used for the TD(lambda) prefix targets that Laya's RLCD section describes.

### Contamination rules

- No test split of any benchmark in 13 is used for training, calibration, early stopping or prompt selection.
- Synthetic states are checked against every benchmark test set with MinHash (5-gram shingles, Jaccard 0.5 or more) and near duplicates are dropped.
- The benchmark runner in `kime-eval` records the data manifest hash of the model under test and refuses to publish a result if the manifest lists a test split.

## Teacher

`kime-t-en` and `kime-t-x` are full cross encoders in the fixed Laya format: all tokens attend to all tokens from layer 0, which gives the best accuracy per parameter. They start from ModernBERT-large and mmBERT-base and are trained on all sources for 2 epochs with the loss below. They are never served by default. They exist to produce soft labels and hidden state targets for the students. Their fixes over Laya: neutral noul markers, per option budgets, order and label augmentation, and context up to 8,192 tokens.

## Student (kime-v1) distillation

Initialization is described in 05 (tiers table). New weights (cross attention, scorer, ordinal and type embeddings, special token embeddings) are initialized small (std 0.02, and cross attention output projections at zero, so each question layer starts as a plain encoder layer).

Loss per question, with p the student's distribution and t the target:

```
L = w_ps  * ProperScoreLoss(p, t, type)
  + w_kd  * KL(p_teacher_T || p_T)                       # T = 2 distillation temperature, on the teacher's logits
  + w_hid * MSE(proj(Q[markers]), teacher_h[markers])     # marker hidden states, projected to teacher width
  + w_ord * RankedProbabilityScore(p, t)   if score
```

with default weights `w_ps = 1.0`, `w_kd = 1.0`, `w_hid = 0.1` and `w_ord = 1.0`, and w_hid decaying to 0 over the first half of training.

### What RLCD needs, and what it does not

Laya's notebook trains with a Gaussian policy gradient: it perturbs the logits G times, scores each perturbation with a proper scoring rule, and pushes towards the better ones. For the log, spherical and ranked probability scores this estimates a gradient we can compute exactly. All three are differentiable functions of p, so the exact gradient is available at no extra cost and with no variance. kime therefore trains on the proper score directly:

```
ProperScoreLoss(p, t) = -( sum_i t_i log max(p_i, 1e-4)                       # log score, floored like Laya's log_floor
                          + w_sph * (sum_i t_i p_i) / ||p||_2 )                # spherical score, w_sph = 0.5
```

This is a strictly proper objective, so its optimum is calibrated probabilities. It is what "reinforcement learning for calibrated decisions" is after, without the sampling noise.

The policy gradient estimator is kept, as `rlcd_pg`, for the two rewards that are not differentiable in p:

- **Decision cost rewards.** For the correctness head and the `policy` feature in 04, the reward is the cost matrix outcome of acting on the argmax (+1 correct, -3 wrong, -0.5 abstain). The estimator uses G = 8 samples, sigma from 1.0 down to 0.3, and group normalized advantages, as the dev.to article describes.
- **Episode outcomes.** For conversation prefixes the target is a TD(lambda) return: `G_t = (1 - lambda) * p_true(t+1) + lambda * G_(t+1)`, with the last prefix getting the outcome. Up to 6 prefixes per episode, and lambda = 0.9.

### Augmentations

Each one targets a known defect.

| Augmentation | Rate | Fixes |
|---|---|---|
| Shuffle choice option order, permute targets | every example | Order flips (Laya 15%, Jev 13%) |
| Mirror score scales (reverse levels and targets) | 50% of score questions | Multilingual score never picking level 0 (Laya #131). Direction must come from text. |
| Randomize noul criteria wording, including negated phrasing and swapped polarity with flipped target | 30% of noul questions | Noul following label words (Laya #156) |
| Rename choice labels to neutral ids (`A`, `opt_3`, `"12"`) while keeping descriptions | 20% of choice questions | Reliance on label words. Needed for jev-ultrafast style element ids. |
| Random chunking of large choices into groups of 8 to 32 | every choice over 8 options | Logits comparable across chunks (see 04) |
| State as raw text vs JSON vs JSON with extra noise keys | per example | Robustness to state format and context rot |
| Distractor questions in the same request | 30% of requests | Questions staying independent |
| Segment mode on or off | 50% of batches | Segment mode as good as full mode |
| Truncation (head, tail, middle) to random lengths | 10% of examples | Graceful degradation, `[CUT]` token meaning |
| Instructions referencing state paths with backticks | 15% of synthetic questions | Following Jev style path references |
| Empty or null instructions with informative criteria | 5% | Jev allows missing instructions |

### Schedule

AdamW (beta 0.9 and 0.98, weight decay 0.01). Learning rate 1e-4 for new weights and 3e-5 for initialized weights, 2% warmup, cosine decay to 1e-6. Bucketed batches of about 64k state tokens each, with all questions of a state in the same batch (the split architecture makes that cheap). Context curriculum: 1,024 for the first 60% of steps, then 8,192, then a final 5% at 32,768 in segment mode. Gradient clipping at 1.0. BF16 mixed precision with FP32 master weights.

## Correctness head

This is trained after the main training finishes, with the backbone frozen, on a held out 5% slice. It is a binary cross entropy against whether the argmax was correct, plus the `rlcd_pg` cost matrix term. Input features are listed in 04, with a LayerNorm on the input. It ships only if its AUROC against correctness beats `max(p)` by at least 0.01 on the calibration split. Otherwise the checkpoint has `"correctness_head": false` and `act_probability` falls back to `max(p)`.

## Calibration fitting

- **Data.** A calibration split of 50k questions, drawn from the same sources as training but never trained on, stratified by type, k bucket and language group.
- **Method.** Temperature scaling per `(type, k_bucket, lang_group)` bucket. Minimize NLL on soft targets over `log T`, with LBFGS, 200 iterations, clamped to [0.25, 8.0]. Buckets with fewer than 500 examples fall back to the parent bucket (drop lang_group, then k_bucket).
- **Order.** Fit after quantization, on the quantized model for each precision. Each precision gets its own calibration table in the prepacked cache, because FP8 and INT8 shift logits slightly.
- **Output.** `calibration.json` with every temperature, the bucket counts, and the ECE (15 bins and sweep), Brier and NLL before and after, per bucket. `kime calibrate` does all of this on user data.

## Quantization

- **FP8 (GPU).** Post training, per channel weight scales, per tensor activation scales from 2,048 calibration batches (the 99.99th percentile of the absolute value).
- **INT8 (CPU and T4).** SmoothQuant with alpha 0.5, then one epoch of quantization aware fine tuning with straight through estimators at learning rate 5e-6.
- **W8 palette (ANE).** k-means with 256 centroids per output channel group, through coremltools at conversion.
- **Gate for every precision.** Argmax agreement with FP16 at or above 99.5% on the calibration split, per bucket ECE within 0.005 of FP16 after refitting, and no benchmark in 13 dropping more than 0.3 points of accuracy. laya-mlx found q4 flipped 7 of 26 decisions on the multilingual model, so we do not ship 4 bit weights in v1.

## Release gate

A native checkpoint is released only when all of these hold:

1. Every quality target in 13 is met on its tier's benchmarks.
2. The calibration contract in 04 is met.
3. The flip rate is at or below 0.02 on MASSIVE en and Emotion, and the mirrored score test passes: for 1,000 score questions, reversing the levels mirrors the argmax in at least 98% of cases.
4. The noul polarity test passes: swapping true and false criteria flips `noul` to `1 - noul` within 0.05 in at least 97% of cases.
5. No per level bias: every score level is the argmax at least once on each score benchmark where it is the gold at least 1% of the time.
6. Parity between `kime-train`'s forward and the engine's forward on every backend is within the tolerances in 15.

## kime train (user fine tuning)

```
kime train --base kime-v1-s-en --data my.jsonl --eval my_eval.jsonl --out ./my-model \
           [--mode full|question-tower|lora] [--epochs 3] [--calibrate held-out-fraction=0.1]
```

- `--mode question-tower` (default) freezes the state tower and trains the question tower and scorer. It fits in 8 GB, trains fast, and keeps cached state memories valid across base and fine tuned models, so one state encoding can serve both.
- `--mode full` trains everything. `--mode lora` adds rank 16 adapters to every linear layer and merges them at export.
- The input format is the JSONL above. A `--from-laya` flag reads LocalLLaMA/typed-decisions and Laya notebook formats.
- Every run ends with a calibration fit on a held out fraction, never on the training data (the mistake Laya fixed in #186), and prints a report card with the metrics from 13 on the eval file.
- The output is a normal checkpoint directory, loadable by `kime serve --models ./my-model`.

Laya's browser agent fine tune (cklxx/laya-browser: element top-1 from 0.10 to 0.66 at 17 to 23 ms per step) is the reference example for `kime train`, re-run on kime-v1-s-en with the same data.

## Compute budget estimates

| Run | Hardware | Estimate |
|---|---|---|
| Synthetic labelling, 4M questions, 3 LLMs | API | the largest cost item. Budget by tokens, about 6B input tokens |
| Teacher kime-t-en, 2 epochs | 8x H100 | about 40 hours |
| Teacher kime-t-x, 2 epochs | 8x H100 | about 36 hours |
| Student s-en, distillation | 8x H100 | about 12 hours |
| Student s-x, distillation | 8x H100 | about 14 hours |
| m tiers | 8x H100 | about 20 hours each |
| Calibration, quantization, gates, per model | 1x H100 | about 2 hours |

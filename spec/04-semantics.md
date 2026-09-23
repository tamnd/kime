# Decision semantics

This file defines what every answer field means, how it is computed from the model's logits, and how calibration and abstention work. The model produces one real logit per option. Everything in this file is the deterministic math on top of those logits, implemented in `kime-core::answer`, and it is the same on every backend.

## From logits to probabilities

For a question with k options, the question tower produces logits `z[0..k]` in f32. The runtime computes:

```
T = temperature(model, type, k, lang_group)
p = softmax(z / T)
```

The softmax is computed in f32 with max subtraction. Masked or padded slots never enter it, because the kernel only reads the k real logits of each question. We do not add -1e4 to masked slots as Laya does. That trick leaks tiny mass in fp16 and is the source of Laya's warnings about non-finite values.

Temperature lookup order: `(type, k_bucket, lang_group)`, then `(type, k_bucket)`, then `(type)`, then 1.0. The k buckets are `2`, `3-5`, `6-10`, `11-31`, `32-255`. The lang groups are `en`, `latin`, `cjk`, `indic`, `other`, set by the router's detection. Temperatures are clamped to [0.25, 8.0]. The same clamp is used when fitting them (see 12), which removes Laya's mismatch between fitting in [0.1, 10] and running in [0.5, 5]. A checkpoint whose calibration table has any value outside the clamp, or a NaN, fails to load with an error naming the bucket. We never silently repair it.

## Choice

- `probabilities[label] = p[i]` for option i.
- `choice` is the argmax label.
- `confidence` has two formulas, selected by `kime.confidence`:
  - `jev` (default): `clip((max(p) - 1/k) / (1 - 1/k), 0, 1)`. For k = 1 it is 1.0. This matches Jev's numbers exactly.
  - `entropy`: `clip(1 - H(p) / ln k, 0, 1)` with `H(p) = -sum p_i ln p_i`, using `0 ln 0 = 0`. For k = 1 it is 1.0. This matches Laya.
- With extensions on, both are returned (`confidence` plus `kime.answers.<id>.confidence_entropy`).

### Permutation averaging

When `kime.permutations = m > 1`, the question tower is run m times with the option order rotated by `i * k / m` and the option segments renumbered. The final `p` is the mean of the m calibrated distributions. Because the state memory is shared, the cost is m times the question tower cost only, which is about 5 to 15% of a request per extra permutation on the native model. Every native model is trained with order augmentation (see 12) and should have a flip rate at or below 0.02 without averaging. Averaging is for callers who need near zero flip rates for audit reasons.

### More than 255 options

Choices with up to 255 options are scored in one pass. Options are packed into chunks of up to 32 option segments each, and each chunk goes through the question tower with the same question header. All k logits are compared in one softmax. This is valid only because the native model is trained with random chunking, so a logit means the same thing whichever chunk it lands in. Then, if k > 32, the top 16 options by probability are re-scored together in one extra pass. This is the "score independently first, then choose between explicitly" approach Jev describes. The final distribution is:

```
p_final[i] = p_joint[i] * P_top16      for i in top 16
p_final[i] = p_chunked[i]              otherwise
```

where `P_top16 = sum of p_chunked over the top 16`. This keeps the mass of the tail as estimated by the chunked pass and redistributes the head by the joint pass.

With `kime.shortlist = n` and k > 255 (up to 4,096), options are first ranked by cosine similarity between the pooled state embedding and each option's pooled embedding from the question tower's first layer. Only the top n go through the path above. The rest get probability 0. The response lists them anyway with 0.0, so the key set rule in 03 still holds.

## Score

- `probabilities[str(i)] = p[i]` for level i.
- `score = sum(i * p_i)`. It is a real number and can fall between levels.
- `legend[str(i)]` is the criterion as sent.
- `confidence` has the same two formulas:
  - `jev` (default): Jev's mean absolute deviation form. With `m = argmax(p)`, `dist = sum(p_i * |i - m|)`, `c = (k - 1) / 2` and `mad = sum(|i - c| for i in 0..k) / k`, confidence is `max(0, 1 - dist / mad)`. For k = 1 it is 1.0.
  - `entropy`: as for choice.

Jev says each level is evaluated separately and the model does not see level numbers. The native model follows this at the input level: levels are rendered without their index (see 06). Order information enters only through a learned ordinal embedding added to each level's marker, and through the ranked probability score term in training. So a caller who reverses the level list gets a mirrored distribution, not a different judgement. This is tested in CI (see 15).

## Noul

- `noul = p[1]`, the probability that the statement holds.
- There is no `confidence` in the default response, to match Jev. With extensions on, `noul_confidence = max(noul, 1 - noul)` is returned, which matches Laya.
- The model sees two options. In the native model the markers are two learned embeddings, `[NO]` and `[YES]`, with the caller's criteria text (if any) after each. The words "false" and "true" are never inserted, which fixes Laya issue #156 where the model followed label words instead of meaning.

## Act probability and abstention

Laya's act head is unusable. It was never trained, its logits run against correctness, and it reports 1.0 almost everywhere. kime replaces it with two things.

1. `act_probability` in the Laya compat response and in the extension block is the model's calibrated probability that its answer is correct. For choice it is `max(p)`. For score it is the probability of the mode. For noul it is `max(noul, 1 - noul)`. Once temperatures are fitted, this is the best available estimate of accuracy and needs no extra head.
2. Native models also ship a correctness head. It is a two layer MLP over `[pooled question vector, top1, top1 - top2, normalized entropy, k / 256, log T]`, with a LayerNorm on its input so the activation blow-up Laya saw cannot happen. It is trained with binary cross entropy against whether the argmax was right, on held out data (see 12). When the checkpoint has it, `act_probability` comes from this head instead. It must have an AUROC against correctness at least as high as `max(p)` on the calibration split, or the build step drops it.

Callers who want the Laya cost matrix behaviour (correct +1, wrong -3, escalate -0.5, so act when P > 0.625) can pass `kime.policy = {"wrong": -3, "correct": 1, "abstain": -0.5}`. The response then includes `kime.answers.<id>.decision = "act" | "abstain"`. The threshold is computed as `(abstain - wrong) / (correct - wrong)`.

## Calibration contract

Every native model release states, per type and per k bucket, its ECE (15 equal width bins, plus the ECE sweep estimate), Brier score and log loss on the held out calibration split. It also states them on each public benchmark in 13. The release gate is ECE at or below 0.05 on the calibration split for every bucket with at least 500 examples, and on typed-decisions test. Calibration tables are part of the checkpoint (`calibration.json`), are versioned, and their hash appears in `GET /v1/models/{id}`.

## Determinism

For a fixed checkpoint, backend, device type and kime version, the same request produces bit identical logits whatever else is in the batch. This requires three things, and 07 explains how each is met:

- Every reduction (GEMM K loop, softmax, LayerNorm) uses a fixed order that does not depend on batch composition. Rows are never split across CTAs differently depending on batch size.
- Tokens of one sequence never share a reduction with another sequence's tokens.
- The answer cache and the state cache only ever return results that were computed deterministically.

Jev's answers vary slightly between identical calls. kime's do not, so its callers do not need a `uid` nonce or averaging.

## Worked example

State "Hi, we were billed twice for March. Refund the duplicate today or we cancel." One choice question with options billing, technical, sales, other. Suppose the calibrated probabilities come out as [0.91, 0.02, 0.03, 0.04]. Then:

- `choice = "billing"`
- `jev` confidence: (0.91 - 0.25) / 0.75 = 0.88
- `entropy` confidence: H = 0.398 nats, ln 4 = 1.386, so 1 - 0.287 = 0.71
- `act_probability` (no correctness head) = 0.91
- Rounded to 2 places with largest remainder: [0.91, 0.02, 0.03, 0.04], which already sums to 1.

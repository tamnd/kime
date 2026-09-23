# Overview and goals

## What kime is

kime is a "System One" model runtime: a program answers a fixed set of typed questions about some state, and gets probabilities back in a few milliseconds instead of waiting seconds for a language model to write JSON. There are three question types. A `choice` picks one label from a set the caller defines, a `score` places the state on an ordered scale of levels, and a `noul` gives the probability that a statement is true. Every question in a request is answered from one read of the state, and every answer comes with a full probability distribution and a confidence.

kime ships as four things built from one Rust workspace:

1. A library crate, `kime`, that loads a model and answers questions in process.
2. A server, `kime-serve`, that speaks the TypeSafe `POST /v1/systemone` protocol, so any Jev or Laya client works by changing the base URL.
3. A set of model weights. There are two families: the `laya` compat family, which runs the published Laya checkpoints unchanged, and the native `kime-v1` family, which is trained by us with an architecture that encodes the state once for all questions.
4. SDKs for Python and TypeScript and a CLI, `kime`.

The whole runtime path is Rust: tokenization, routing, batching, the forward pass and the kernels. There is no Python, no PyTorch and no ONNX Runtime at inference time. Python appears only in offline tools, such as the Core ML conversion step and the reference scripts we use to check parity.

## Why build it

Jev is fast for a hosted model (about 100 to 280 ms per call, most of it network) and cheap ($0.042 per million input tokens), but it is closed, hosted only, English first, and has no fine tuning. Laya is open and self hostable, but its runtime is PyTorch eager with padded batches and it re-encodes the full state once per question. It takes 33 to 40 ms for one question on a T4 and 7 ms for each extra question. It also has several accuracy defects that come from its input format rather than from model capacity (see 02). Neither one gives you a small, fast, deterministic binary that you can put next to an agent loop or embed in a service with microsecond overheads. kime is that binary.

## Feature parity matrix

Every row is a feature that exists in Jev, Laya or one of the Laya ports. kime must ship all of them. The last column says where the feature is specified.

| Feature | Jev | Laya | kime | Spec |
|---|---|---|---|---|
| `POST /v1/systemone` with `state`, `model`, `questions` | yes | yes (laya-serve) | yes | 03 |
| `choice`, `score`, `noul` question types | yes | yes | yes | 04 |
| State as string, object or array | yes | yes | yes | 06 |
| `instructions` and criteria as string, object, array or null | yes | partial (rendered as JSON) | yes | 06 |
| Choice criteria as label to description map, null descriptions allowed | yes | yes | yes | 04 |
| Choice criteria as a plain list of labels | no | yes | yes | 04 |
| Up to 255 options per choice | yes | degrades after about 20 | yes, with full quality | 05 |
| Score criteria as an ordered list, 2 to 10 levels | yes | yes (any length) | yes, 2 to 32 | 04 |
| Noul with custom true and false criteria | yes | yes | yes | 04 |
| `probabilities`, `confidence`, `legend`, expected `score` | yes | yes | yes | 04 |
| Many questions per call, evaluated independently in parallel | yes | yes (one padded batch) | yes (state encoded once) | 05 |
| 64k request context, 32k state plus longest question | yes | no (512 to 1024, impossibl 8192) | yes | 06 |
| `GET /v1/models`, aliases and pinned versions | yes | no | yes | 03 |
| `GET /health` | yes | yes | yes | 03 |
| `x-typesafe-request-id` style request ids | yes | no | yes | 03 |
| Bearer auth with Jev error shapes (401, 403) | yes | yes (single key) | yes, multi key | 03 |
| 422 validation errors in FastAPI `detail` format | yes | yes (plain string) | yes | 03 |
| 429 rate limit and 529 overloaded with `retry-after` | yes | no | yes | 11 |
| Rate limits by tokens per second and requests per minute | yes | no | yes | 11 |
| `usage.input_tokens`, `usage.output_tokens` | yes | yes | yes | 03 |
| Multilingual input | weak | 100+ languages via routing | 100+ languages | 05, 11 |
| Router that picks a checkpoint per request | no | yes | yes | 11 |
| Router overrides: `model`, `task`, `lang`, `lang_guess`, `default` | no | yes | yes | 11 |
| `routing` block in the response | no | yes | yes, as an extension | 03 |
| `action.act_probability` | no | yes (broken) | yes, fixed | 04 |
| Temperature calibration per type and per option count bucket | internal | yes | yes, plus per language group | 04, 12 |
| Email body cleaning and `email_state` | no | yes (en, pt, es) | yes, 12 languages | 06 |
| Question presets: triage, email, guard, moderation, router | no | yes | yes | 14 |
| Shortlisting for very large label sets | internal | yes | yes, beyond 255 options | 05 |
| Python SDK | yes | yes | yes, both API styles | 14 |
| TypeScript SDK | yes | no | yes | 14 |
| Retries with backoff and `retry-after-ms` | yes (SDK) | no | yes (SDK) | 14 |
| CLI | no | partial (laya-mlx) | yes | 14 |
| Fine tuning on your own data | no | notebook only | yes, `kime train` | 12 |
| Self hosting, Docker, NixOS module | no | yes | yes | 14 |
| Apple Silicon GPU (MLX port) | no | yes | yes, native Metal | 09 |
| Apple Neural Engine (Core ML port) | no | yes | yes | 09 |
| CPU inference | no | yes (slow) | yes, fast | 10 |
| Checkpoint conversion tool | no | yes (laya-mlx convert) | yes, `kime convert` | 14 |
| Hosted gateways (OpenRouter, Vercel) | yes | via impossibl | protocol compatible | 03 |
| Deterministic outputs | no (slightly stochastic) | yes on one device | yes, bit exact per backend | 07 |

## The 10x scorecard

"10x better on every benchmark" needs a definition that can be checked. kime uses one scorecard with two kinds of rows.

Speed and cost rows must improve by at least 10x over the best of Jev and Laya measured on the same hardware class. These are: single question latency, per question cost in a batch, throughput in questions per second, cold load time, resident memory, energy per decision on Apple Silicon, dollars per million decisions, and agent loop step latency.

Quality rows cannot improve 10x, because accuracy is already bounded near 1. For these the rule is: kime must be at least as good as the best baseline on every dataset and every language, and must cut the error of the best baseline by a stated amount on the rows where the baselines have known defects. These rows are: accuracy on each public dataset, macro accuracy over 51 languages, Brier score, ECE, option order flip rate, and AUROC of confidence against correctness.

The concrete numbers are in 13. The short version of the headline targets:

| Metric | Best baseline today | kime target |
|---|---|---|
| 1 question, short state, T4, in process | Laya multilingual 32.8 ms | 3.3 ms or less |
| Per question in a 10 question call, T4 | Laya multilingual 7.2 ms | 0.72 ms or less |
| 1 question, M3 Max | laya-coreml ANE 4.98 ms | 0.5 ms or less (GPU) |
| Energy per decision, M3 Max | laya-coreml ANE W8 0.134 J | 0.0134 J or less |
| Agent step, 5.3k token state, 3 heads | Jev 178 ms median (hosted) | 17.8 ms or less on L4, cold state |
| Cold load | Laya 0.3.7 about 2 s | 200 ms or less |
| CPU, 1 question, Ryzen 9 6900HX | Laya tuned 329 ms | 33 ms or less |
| Cost per million input tokens | Jev $0.042 | $0.0042 or less on a rented L4 at full load |
| typed-decisions accuracy | Laya typed-decisions 0.766 | 0.80 or more |
| ECE, typed-decisions | Jev 0.144 raw | 0.05 or less |
| Option order flip rate, MASSIVE en | Jev 0.13, Laya 0.15 | 0.02 or less |
| MASSIVE 51 language macro accuracy | Laya multilingual 0.366 | 0.55 or more |
| Banking77 | Jev 0.870 | 0.90 or more |

## Design in one paragraph

The main speedup does not come from the kernels. Laya runs a 400M parameter cross encoder over `[question, options, state]` once per question, so ten questions cost ten passes over the state. kime-v1 splits the model into a state tower and a question tower. The state tower runs once per request over the state tokens. It produces keys and values that every question layer can read, and they can be cached across requests. The question tower is small and reads only the question and option tokens (tens of tokens), attending to the cached state with cross attention. The kernels then remove the rest of the overhead: unpadded variable length batches, fused FlashAttention style kernels with sliding window support, CUDA graphs captured per shape bucket, FP8 or INT8 weights, and a scheduler that batches the state stage and the question stage across requests separately. Models are distilled to about 70M to 150M parameters, and a teacher ensemble supplies soft labels, so the smaller model costs no accuracy.

## Non goals

- Text generation of any kind. kime never writes free text. Filling a text box in an agent is the caller's job, as in jev-ultrafast, which hands that to a small LLM.
- A general embedding server. kime exposes state embeddings for shortlisting, but it is not a TEI replacement.
- Training base encoders from scratch. We start from open encoders (ModernBERT, Ettin, mmBERT) and fine tune and distill.
- A hosted multi-tenant SaaS in v1. The server has everything a single operator needs (auth keys, rate limits, metrics), but billing and accounts are out of scope.
- Replicating Jev's internal architecture. Jev appears to use a parallel masked or diffusion style decoder. We do not need it, because the typed answers are fully determined by per-option scores.

## Glossary

| Term | Meaning |
|---|---|
| state | The input being judged. A string, a JSON object or a JSON array. |
| question | A typed request for a decision about the state, keyed by a caller chosen id. |
| option | One element of a question's answer space. A label for choice, a level for score, true or false for noul. |
| marker | A special token placed at the start of each option in the sequence. The hidden state at the marker is scored to produce that option's logit. |
| head | Old Laya term for the layers above the encoder that produce logits. In kime-v1 this is the question tower. |
| state memory | The per layer keys and values produced by the state tower, cached and shared by all questions over that state. |
| bucket | A fixed shape (sequences, total tokens) for which an execution plan and a CUDA graph are prebuilt. |
| compat model | A published Laya checkpoint run by kime with identical math. |
| native model | A kime-v1 checkpoint trained by us. |

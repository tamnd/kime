# Prior art, Jev and Laya

This file records what the two systems we are cloning actually do, with the numbers they publish and the defects we found. Every later design decision refers back to something here.

## Jev (TypeSafe)

### What it is

Jev is TypeSafe's hosted "System One" model, released 2026-09-15. The current version is `jev-1.13.0`, and the alias `jev-latest` resolves to it. TypeSafe describes it as a new architecture with a parallel sampler and a training method called RLCD (Reinforcement Learning for Calibrated Decisions). The architecture itself is not disclosed. Circumstantial hints (their GitHub org forks LLaDA, the blog talks about continuous diffusion, and each answer uses a small fixed number of output tokens: about 18 to 21 per noul or score and about 34 for a 3 option choice) suggest a masked or diffusion decoder that fills in all answer slots in one parallel step. Each score level and each choice option is evaluated separately, and the model does not see level numbers or neighbouring levels.

### Wire protocol

Base URL `https://api.typesafe.ai`. FastAPI behind Envoy and Cloudflare. The OpenAPI title is "TypeSafe", version 0.2.0.

- `POST /v1/systemone` is the only inference endpoint. It takes `state`, `model` and `questions`, all required. Every response has an `x-typesafe-request-id: req_<32 hex>` header.
- `GET /v1/models` returns `{"models":[{"name":"jev-latest","description":"General-purpose system one model.","release_date":"2026-09-15"}]}`. It lists aliases only, and pinned ids such as `jev-1.13.0` are accepted anyway.
- `GET /health` returns `{"status":"ok"}`. `/docs`, `/redoc` and `/openapi.json` are public.
- Auth is `Authorization: Bearer <key>`. A missing key returns 403 with `{"detail":{"error_type":"authentication_error","message":"Must supply an API key! Check your request and try again."}}`. A bad key returns 401 with `{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}`.
- Validation errors are 422 in FastAPI list format: `{"detail":[{"loc":["body","questions","urgency","score","criteria"],"msg":"Field required","type":"missing"}]}`.
- 429 means rate limited and 529 means overloaded. Both come with `retry-after` or `retry-after-ms`.

Request example from the quickstart:

```json
{"state":"Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
 "model":"jev-latest",
 "questions":{
  "department":{"type":"choice","instructions":"Which team should handle this","criteria":{"billing":"Payment or subscription issues","technical":"Bugs or integration problems","sales":"Pricing or account questions"}},
  "frustration":{"type":"score","instructions":"How frustrated the customer appears","criteria":["Calm, just stating facts","Frustrated but civil","Very angry, strong language"]},
  "is_urgent":{"type":"noul","instructions":"The message conveys urgency or time-sensitivity"}}}
```

Response:

```json
{"model":"jev-1.13.0",
 "answers":{
  "department":{"type":"choice","choice":"technical","confidence":0.78,"probabilities":{"technical":0.85,"sales":0.0,"billing":0.15}},
  "frustration":{"type":"score","score":1.0,"confidence":1.0,"legend":{"0":"Calm, just stating facts","1":"Frustrated but civil","2":"Very angry, strong language"},"probabilities":{"0":0.0,"1":1.0,"2":0.0}},
  "is_urgent":{"type":"noul","noul":1.0}},
 "usage":{"input_tokens":392,"output_tokens":65}}
```

Details that matter for compatibility:

- Question keys are never shown to the model. `instructions`, choice descriptions and score levels can each be a string, an object, an array or null. Instructions can refer into the state with backticked paths like `` `ticket.messages[0].text` ``.
- Choice allows up to 255 options. The launch blog says high cardinality choices are scored independently first and then chosen between explicitly.
- Score criteria are an ordered list. The docs say at least 2 levels and the API accepts up to 10. Before Python SDK 0.6.0 the criteria were an int keyed dict.
- Noul answers have no confidence field. The noul value is P(true).
- Probabilities are rounded to 2 decimals, sum to 1, and the map key order is arbitrary. `legend` echoes criteria verbatim.
- Repeated calls are slightly stochastic. TypeSafe's cookbooks add a `uid` nonce to the state, which suggests a response cache.
- Limits: 64k tokens per request, 32k for state plus the longest question, 250,000 tokens per second and 1,200 requests per minute. Price $0.042 per million input tokens, output free. English first. Other languages, including CJK, work at lower accuracy.

### Confidence formulas

These reproduce every example in the docs. They come from TypeSafe's own `system-one-adapter`.

```python
def choice_confidence(p):
    u = 1 / len(p)
    return clip((max(p) - u) / (1 - u), 0, 1)

def score_confidence(p):
    if len(p) == 1: return 1.0
    mode = argmax(p)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2
    mad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / mad)
```

### SDKs

Python `typesafe-sdk` 0.7.1 and TypeScript `@typesafe-ai/sdk` 0.6.0. Both have a client with `system_one` / `systemOne`, typed builders `Noul`, `Choice`, `Score`, `models.list()`, retry policies (Python defaults: 2 retries, 0.5 s initial backoff, 5 s max, 0.25 jitter, retry on 408, 429 and 5xx, respect `retry-after`), and typed exceptions per status. Env vars are `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL` and `TYPESAFE_LOG_LEVEL`. The Python SDK has an `extra_body` escape hatch that merges extra fields into the request body. kime uses this to pass its own options through the official SDK.

### Published numbers

- Latency claims: "about 100 ms" in the docs, "150 ms" on the use case page, 70 to 500 ms end to end in the launch blog. Measured: 111 to 114 ms per single question call (consistency cookbooks), 0.27 s for 13 questions over a 53,777 character article, and 178 ms median for about 5.3k token states with 2 to 4 choice heads (jev-ultrafast).
- Workflow evals (evals.typesafe.ai), accuracy against a GPT-6 Astra plus Fable 5.1 consensus: Jev 67.8% overall, security 61.7%, agent trace 71.6%, invoice 61.8%, customer service 76.0%. At $0.0004 and 0.4 s per case.
- Third party numbers used in the Laya comparison: typed-decisions accuracy 0.727, soft accuracy 0.580, Brier 0.148, ECE 0.144, score MAE 0.391. AG News 0.910, DAIR Emotion 0.480 (with zero probability on the true label in 16% of examples), Banking77 0.870, option order flip rate 0.13.

### jev-ultrafast

This is the browser agent built on Jev, and it is the best example of how a real client uses the API. Each step sends one request with an `operation` choice question (CLICK, TYPE_TEXT, SELECT, SCROLL_UP, SCROLL_DOWN, WAIT, DONE, BLOCKED) and one speculative `<op>_target` choice head per operation that has candidates. Option keys are element indexes such as `"12"`, and option values are objects like `{"element":"[12] Where to?","current_value":"","role":"combobox"}`. The state is `{"page":{url,title,text},"elements":[...],"recent_actions":[...]}`, with page text capped at 6,000 characters and candidates capped at 250. The client checks every response. The choice must be one of the offered keys, the probability keys must equal the offered keys exactly, values must be finite and in [0, 1], they must sum to 1 within 0.02, and the choice must be the argmax. The Google Flights demo ran in 7,073 ms with 17 Jev requests, a median of 178 ms each, and about 5.3k input tokens per request. In that run, 3.7 s of the 7.1 s was spent inside Jev, which makes it the obvious place to win.

## Laya

### What it is

Laya (Convai Innovations, Apache 2.0, package `laya` 0.3.7) is an open reimplementation of the System One idea. It is a ModernBERT or mmBERT encoder plus a 2 layer transformer head and an option scorer. It answers each question with one forward pass over a single sequence that contains the question, its options and the state.

Checkpoints:

| Name | Encoder | Params | max_len / head_max_len |
|---|---|---|---|
| `laya` (english) | ModernBERT-large, 28 layers, d 1024, 16 heads, inter 2624, vocab 50,368 | 421M | 512 / 192 |
| `laya-multilingual` | mmBERT-base, 22 layers, d 768, 12 heads, inter 1152, vocab 256,000 | 322M | 1024 / 256 |
| `laya-typed-decisions` | ModernBERT-large, fine tuned on LocalLLaMA/typed-decisions | 421M | 1024 / 256 |

Both encoders use global attention every third layer and a local window of 128 elsewhere. RoPE theta is 160,000 for global layers and 10,000 for local layers in ModernBERT, and 160,000 for both in mmBERT. There are no biases, GeGLU MLPs, and pre norm LayerNorm without bias. The special ids for ModernBERT are CLS 50281, SEP 50282 and PAD 50283. For mmBERT they are CLS 1, PAD 0 and MASK 4, and SEP is also 1.

### Input format

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
```

Choice options render as `label: description` (or just `label`), score levels as `level i: description`, and noul options as `false: <false criterion or "no, the statement does not hold">` and `true: <true criterion or "yes, the statement holds">`. Each option is capped at 48 tokens. If all options together exceed `head_max_len` minus 16, every option is cut to `max(4, (head_max_len - 16) / k)` tokens. Instructions get the remaining budget, with a floor of 8 tokens. Dict and list states become `json.dumps(ensure_ascii=False)`. The state is truncated from the right, silently, to fill `max_len`.

### Model

The encoder output gets a learned type embedding added, then passes through a 2 layer PyTorch `TransformerEncoderLayer` stack (pre norm, ReLU FFN of 4d, nhead d/64). The hidden states at the marker positions go through a scorer (`LayerNorm, Linear(d,d), GELU, Linear(d,1)`) to give one logit per option. Logits are masked and softmaxed with a temperature. The temperature is chosen per type and per option count bucket (`2`, `3-5`, `6-10`, `11+`) and clamped to [0.5, 5.0] at runtime. An act head reads the CLS vector plus `[top1, top1 - top2, normalized entropy, k/255]` and outputs `act_probability`.

Answer fields: choice gives `choice`, `probabilities` and `confidence = 1 - H(p)/log k`. Score gives `score = sum(i * p_i)`, `legend`, `probabilities` and `confidence`. Noul gives `noul = p[1]` and `confidence = max(p1, 1 - p1)`. Every answer carries `action.act_probability`. Usage is `input_tokens` = number of attention tokens and `output_tokens` = 0.

### Training (RLCD as implemented)

The reward is a mix of strictly proper scoring rules: log score, plus a spherical score weighted by `w_sph`, minus a ranked probability score weighted by `w_rps` for score questions. The fine tuning notebook samples G = 4 Gaussian perturbations of the logits with zero mean noise, computes the reward for each, normalizes advantages, and applies a Gaussian policy gradient. It then adds a soft cross entropy term with weight 1.0. Sigma decays from 0.4 to 0.1. Calibration is one temperature per type, fitted with LBFGS on a held out slice of 400 items. The dev.to article states different hyperparameters (G = 8, sigma 1.0 to 0.3, `w_sph` 0.5, no CE term) than the notebook.

### Serving and tooling

`laya-serve` is FastAPI plus uvicorn. It exposes `GET /health` and `POST /v1/systemone`, answers 400 for a malformed body, 401 for a bad key and 422 for any model or validation error, and blocks the event loop during the forward pass. There is no batching across requests. The `Router` detects the script and language in pure Python in under 0.5 ms. It routes non-Latin scripts to multilingual and scores Latin text against function word lists. It keeps 2 models resident with LRU eviction. The package also has email cleaning (quote headers, signatures and footers in en, pt and es), five question presets, and label shortlisting by embedding similarity.

### Published numbers

T4 latency: laya 39.5 ms for 1 question, 158.6 ms for 10 questions and 771.3 ms for 50. laya-multilingual 32.8 ms for 1, 72.3 ms for 10 (7.2 ms per question) and 337.4 ms for 50. On a Ryzen 9 6900HX CPU the default threads give 9,396 ms, and tuned threads give 329 to 783 ms. On GB10 over HTTP there is about 93 ms of unexplained fixed overhead.

Accuracy:

- typed-decisions: laya-typed-decisions 0.766 (but base laya only 0.361 and multilingual 0.342, below the 0.461 majority class baseline), soft accuracy 0.471 against Jev's 0.580, ECE 0.213.
- Other datasets: AG News 0.950, Emotion 0.595, Banking77 0.425.
- MASSIVE, 51 languages, 20 options: macro accuracy 0.2269 for laya and 0.3661 for multilingual. Khmer on laya is 0.000 accuracy at 0.952 confidence.
- ECE as shipped is 0.466 for laya and 0.314 for multilingual. After refitting the temperature it is 0.081 and 0.106.

### Apple ports

laya-mlx is pure MLX FP16. It runs a single question in 7.39 ms (multilingual) and 13.42 ms (laya) on an M3 Max, reaches 395 questions per second in batches, and matches upstream argmax on 378 out of 378 cases. Its research notes are useful for us:

- q8 and q4 quantization of the current checkpoints gave no speedup and flipped decisions.
- Reusing encoder activations across questions is invalid for this architecture, because the first global layer mixes question and state tokens.
- Only a distilled student or a retrained encoder that reads the state once can give a 10x speedup.

laya-coreml converts the model to Core ML. A plain export is slower than MLX. An ANE rewrite runs at fixed shape B1/L96: BC1S layout, 1x1 convolutions instead of linears, einsum attention per head, and the embedding lookup on the host. It reaches 4.98 ms p50 and 0.154 J per decision, and 4.88 ms and 0.134 J with W8 palette weights. At L1024 the ANE is slower than MLX.

## Defects we must not copy

These are the Laya and Jev problems that kime fixes by design. Each one maps to a later section.

| Defect | Where | kime fix | Spec |
|---|---|---|---|
| State re-encoded once per question | Laya | Split encoder, state encoded once and cached | 05 |
| Act head never trained (`0.0 * act.sum()`), act_probability about 1.0, AUROC 0.30 | Laya #185 | Act derived from calibrated probability, plus a trained correctness head with a norm | 04, 12 |
| Shipped temperature 0.1006 for `choice:11+` | Laya | Fit and runtime use the same clamp, fit only on held out data, validated at load | 04, 12 |
| Noul follows the words "false" and "true" in the labels | Laya #156 | Neutral learned marker embeddings for noul, label words randomized in training | 05, 12 |
| Multilingual score never picks level 0 | Laya #131 | Mirrored scale augmentation, per level bias check in CI | 12, 15 |
| Option order changes the answer (15 to 33% flips) | Laya, Jev 0.13 | Permutation augmentation, order invariant option encoding, optional permutation averaging | 05, 12 |
| Option text cut to 3 or 4 tokens when there are many labels | Laya (Banking77 0.425) | Options encoded in their own segments with their own budget, no shared head budget | 05, 06 |
| Silent state truncation | Laya #174 | Truncation reported in `usage`, configurable strategy | 06 |
| Past about 20 options accuracy drops while confidence stays high | Laya, impossibl docs | Chunked option scoring plus a joint rerank of the top 16 | 05 |
| Event loop blocked during inference, no cross request batching | laya-serve | Dedicated inference threads, two stage batching | 11 |
| Default thread settings make CPU 30x slower | Laya CPU | Explicit core pinning and one thread pool per NUMA node | 10 |
| Router reloads a model on every language switch | Laya #137 | All tiers resident by default, weights mmapped and shared | 11 |
| Mixing question types in one call is 20x slower | laya-coreml #5 | One batch for all types, type is an input not a shape | 07 |
| Slightly stochastic answers | Jev | Deterministic kernels, bit exact per backend and build | 07 |
| No confidence for noul | Jev | Available as an extension field | 04 |

# Serving

This covers `kime-serve` and the parts of `kime-engine` it uses: the HTTP layer, the scheduler and batching, the caches, rate limiting, the router and language detection, and observability. The server is one static binary, `kime serve`.

## Process layout

```
tokio runtime (N_io threads, default 2)            engine
  accept, TLS (rustls, optional), HTTP/1.1 and h2    scheduler thread (1 per device)
  parse (sonic-rs), auth, rate limit                   stage queues, batch former
  validate, route, tokenize  ---- WorkItem ---->       device worker thread(s)
  await oneshot            <---- Answers -----         logits, post-processing
  serialize, respond
```

- Parsing, validation, routing and tokenization run on the tokio threads. These steps are CPU bound but short (tens of microseconds), so a separate blocking pool is not worth the extra hop. A request over 256 KiB is moved to `spawn_blocking` for these steps, so it cannot stall the runtime.
- The forward pass never runs on a tokio thread. That was laya-serve's main flaw: its async handler ran PyTorch inline and blocked the event loop.
- A WorkItem goes to the device's scheduler over a lock free MPSC queue (crossbeam `ArrayQueue` plus an `eventfd` or `futex` wake). The result comes back over a `oneshot`.
- HTTP uses hyper 1 through axum 0.8, with HTTP/1.1 keep-alive and h2 (jev-ultrafast uses `httpx` with `http2=True`). Nagle is off. The response body is written with a single `write` from a buffer reused per connection.

## Scheduler

The scheduler forms batches separately for the two stages of the native model, so each stage gets full batches even when requests are shaped very differently.

1. A new WorkItem first looks up its state in the state memory cache. On a hit (the whole state, or every segment in segment mode), its question rows go straight to the question queue.
2. Otherwise its state (or its missing segments) goes to the state queue. When the state batch that contains it finishes, the K and V are written into the cache and its question rows move to the question queue.
3. The device worker alternates between the two queues. Each turn it takes the queue whose oldest item has waited longest relative to its deadline, and forms the largest batch that fits the biggest bucket not exceeding `max_batch_tokens` (default 16,384 for the state stage and 8,192 for the question stage).

Batch forming:

- **No fixed wait window when idle.** If the device is idle, a single request runs at once, alone. Batching only happens while the device is busy: items that arrive during a run are batched for the next run. This gives the best single request latency and near maximum throughput under load, with no tuning. TEI uses the same continuous batching idea.
- **Token budget, not request count.** A request's state never has to wait for a whole batch of long states. Long states over the budget run alone in their own bucket.
- **Deadline ordering.** Items with `deadline_ms` are ordered earliest deadline first within a queue. Items that cannot make their deadline given the current queue estimate are rejected at enqueue with 504. The estimate is an EWMA of run time per bucket. Nothing runs while every item is refused, so an estimate that a burst left high would never come down. Once the device has sat idle for longer than the estimate, one item goes through whatever its deadline and refreshes it.
- **No head of line blocking by length.** TEI issue #723 (one input longer than the budget hangs the queue) cannot happen, because a state longer than `max_batch_tokens` is split into its own batch at the smallest bucket that fits, up to the model's max.
- **Laya compat model.** It has one stage, and its rows are whole sequences.

Multiple GPUs: each device has its own scheduler. A request goes to the device whose queue has the lowest estimated wait. If one device already holds the state in its cache, the request goes there, unless the queue difference is over 2 ms.

## Caches

### State memory cache

- Key: `blake3(model id, version, state token ids)`, or per segment in segment mode. The key is computed on the tokio thread while tokenizing.
- Value: the K and V pages on the device, plus the pooled state embedding.
- Eviction: LRU with a byte budget. The default is 25% of free device memory at start, or 1 GiB on CPU. It can be set with `--state-cache`. With the s tier's 2 KiB per token, 1 GiB holds about 500k state tokens.
- In agent loops and multi call workflows this is what turns a 5k token state into a question tower only call. The TypeSafe cookbooks' "13 questions in one call is 10x faster than 13 calls" becomes true for separate calls too, as long as they share the state.

### Answer cache

- Key: `blake3(model id, version, state tokens, question tower row tokens, calibration version, kime options)`.
- Value: the final answer JSON fragment for that question.
- Size: `--answer-cache` entries (default 100k). It is only possible because the engine is deterministic.
- `kime.cache = "bypass"` skips both caches. `"refresh"` recomputes and overwrites.

### Tokenization cache

Rendered question headers and options tokenize to the same ids across requests with the same questions, which is the common case (fixed presets, fixed agent heads). A per-worker LRU keyed by the hash of the rendered question text maps to the question tower row ids. This is laya-mlx's prefix cache, applied to the question tower.

## Rate limiting and overload

- Per key limits: requests per minute and input tokens per second, as token buckets in a flat array indexed by key id, with one atomic per bucket and no locks. Each bucket is kept as the time it will be full again (GCRA), so admitting a request is one compare and swap. A key may send a whole minute of requests at once. Tokens are only known after the answer, so they are charged then, and a key more than one second of tokens in debt is refused until it is paid down. The keys file gives each key its limits. Global defaults come from `--rpm` and `--tps`, and with no keys they apply to all requests together. Exceeding a limit returns 429 with `retry-after-ms` computed from the bucket refill time and `retry-after` in whole seconds. A refused request's body is still read, so the connection stays open for the retry.
- Overload: when the estimated queue wait exceeds `--max-queue-ms` (default 500), new requests get 529 with `retry-after-ms` set to the current wait estimate. This is Jev's 529 status. It protects tail latency instead of letting the queue grow without limit.
- Concurrency limit: a semaphore on in-flight requests per server (default 4,096) bounds memory.

## Router

`kime-route` picks the model for each request. It ports Laya's Router precedence and adds a fast language id model.

Precedence, first match wins:

1. `model` in the request, if it names a concrete model or an alias other than `kime-latest` or `jev-*`.
2. `kime.route.model`.
3. `kime.route.task`, mapped through the server's task table (for example `typed-decisions` goes to a fine tuned model if one is loaded).
4. Workflow signature match: if the set of question ids equals a registered workflow signature, and `--auto-task` is on.
5. `kime.route.lang`, a language code.
6. `kime.route.lang_guess`, a code, where `en`, `eng`, `english` and locale strings like `en_US.UTF-8` count as English.
7. Detection, described below.
8. The server default.

Detection runs on up to the first 4,000 characters of the rendered state, walking objects and arrays for string values, as Laya's `state_text` does.

1. **Script.** One pass over the chars with a 64 KiB table from code point block to script. If the dominant non-Common script is not Latin, the request goes to the multilingual tier. About 1 microsecond per KB.
2. **Latin language id.** A linear classifier over hashed character 1 to 4 grams (fastText style, 2^20 buckets, 128 dims, quantized). It is trained on OpenLID-v3 data for the 194 languages it covers, and shipped as a 900 KB file in `kime-route`. It returns a language and a probability in about 5 to 15 microseconds per KB. A language other than English with probability 0.6 or more goes to multilingual.
3. **Short text fallback.** Under 4 words, or when step 2 is unsure, Laya's stopword and diacritic rules apply (ported exactly: URLs, emails and dotted names ignored, a non-English language needs a unique word and a margin of 2, a diacritic rate of 0.02 or more breaks ties). Undecided text goes to the default.

This fixes the routing bugs Laya's issue tracker lists: German (#54, #130, #178), accent stripped Spanish, Italian and Portuguese (#168), Armenian (#20) and unlisted scripts falling through to English (#172). All are in the routing test set (see 15).

Resident models: by default every model named in `--models` is loaded at start and stays resident. Weights are mmapped and shared, and the s tiers are small, so keeping `s-en` and `s-x` both loaded costs about 500 MB of device memory. `--max-loaded` turns on LRU eviction for operators with many fine tuned models. It is off by default, which fixes Laya #137, where a language switch reloaded a model.

The decision is returned in `kime.routing` when extensions are on, and as a top level `routing` block for Laya compat responses.

## Observability

- `tracing` spans per request with the request id. JSON logs to stdout when `--log-format json` is set. Request bodies are never logged. `--log-requests` logs hashes and sizes only.
- Prometheus metrics: request counts by status and model; latency histograms per stage (queue, tokenize, state, question, total) with buckets from 50 us to 5 s; batch size and token histograms per bucket; cache hit ratios; truncations; router decisions by model and reason; device memory; rejected requests by reason.
- `GET /ready` returns JSON with the loaded models, device names, queue depths and the EWMA per bucket.

## Configuration

Every setting has a flag, a field in a TOML file (`--config kime.toml`, or `KIME_CONFIG`) and an env var named after the field in capitals (`max_queue_ms` is `--max-queue-ms` and `KIME_MAX_QUEUE_MS`). Sources override each other in this order, last one wins: the defaults, laya-serve's env vars, the file, the `KIME_*` env vars, the flags. A field the file does not know is an error that names it and its line, as is a value of the wrong type.

The Laya env vars are read the way laya-serve 0.3.9 reads them:

| Var | kime setting | Notes |
|---|---|---|
| `LAYA_HOST` | `host` | |
| `LAYA_PORT` | `port` | |
| `LAYA_DEVICE` | `device` | torch's `cpu`, `cuda` and `cuda:N` are kime's too |
| `LAYA_MODELS` | `models` | `english`, `multilingual`, `typed-decisions` and the aliases Laya's router takes |
| `LAYA_THREADS` | `threads` | ignored unless a whole number above 0, as in laya-serve |
| `LAYA_API_KEY` | one API key | alone, it gives laya-serve's 401 body |
| `LAYA_LOG_LEVEL` | `log_level` | uvicorn's levels, above `info` kime prints nothing at startup |
| `LAYA_PRELOAD` | | kime always loads at startup, and says so when this is off |
| `LAYA_AUTO_TASK` | | kime does not route to typed-decisions by itself yet, and says so when this is on |

Installed or linked under the name `laya-serve`, kime also takes laya-serve's defaults: it listens on `0.0.0.0` rather than `127.0.0.1` and loads all three Laya checkpoints rather than only `laya`. A laya-serve systemd unit, container or Nix service then switches binaries with no config change, as long as the weights are in the Hugging Face cache (`kime pull laya-typed-decisions` and so on), since kime does not download at startup.

```toml
host = "0.0.0.0"
port = 8000
models = ["laya", "laya-multilingual"]
device = "cuda:0"
precision = "f16"
max_batch = 256
max_body = 8388608
max_queue_ms = 500
io_threads = 2
jev_aliases = true
api_keys_file = "/run/secrets/kime-keys"
rpm = 600
tps = 20000
log_level = "info"
```

The state and answer caches, `default_model` and the metrics address get their fields when they arrive.

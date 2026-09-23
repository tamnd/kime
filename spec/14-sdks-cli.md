# SDKs, CLI and packaging

## Rust crate

The `kime` crate on crates.io re-exports `kime-engine`'s public API (see 07) and the request and response types from `kime-core`. Features: `cuda`, `metal`, `ane`, `cpu` (default), `serde` (default), `hub` (download from Hugging Face, default), `presets` (default). There is also a thin HTTP client, `kime::client::Client`, with the same `decide` methods that talks to any System One endpoint (kime-serve, Jev or impossibl). Rust services can then switch between in process and remote without code changes.

## Python: `kime`

Built with pyo3 and maturin as abi3 wheels for CPython 3.10 and newer. Wheels: manylinux x86_64 (CPU and CUDA 12 variants), manylinux aarch64, macOS arm64 (Metal and ANE), Windows x86_64 (CPU and CUDA). The CUDA wheel links the driver API only and loads cuBLASLt from the system or from the `nvidia-cublas-cu12` wheel, so there is no 2 GB PyTorch dependency.

The package offers two API styles, so users of either system can switch with an import change.

Laya style, in process:

```python
import kime
router = kime.Router(preload=True)                 # same arguments as laya.Router
res = router.predict(state, questions)             # same result shape as Laya, including routing
agent = kime.load("kime-v1-s-en")                  # or "convaiinnovations/laya" for the compat model
res = agent.system_one(state, questions)           # alias: predict
from kime import clean_email_body, email_state, triage_questions, guard_questions
from kime import predict_shortlist, detect_language, detect_script
```

TypeSafe style, in process or remote:

```python
from kime import TypeSafeClient, Choice, Score, Noul    # same names and signatures as typesafe_sdk
client = TypeSafeClient(base_url="http://localhost:8000")     # remote kime-serve, Jev or impossibl
client = TypeSafeClient(local="kime-v1-s-en")                  # in process, no HTTP
r = client.system_one(state, {"dept": Choice("Which team", {"billing": "...", "tech": "..."})})
r.choices["dept"].choice, r.request_id, r.usage
```

The TypeSafe style classes mirror `typesafe_sdk` 0.7.1: `TypeSafeClient`, `AsyncTypeSafeClient`, `RetryPolicy` (same defaults), the exception hierarchy (`TypeSafeAPIError`, `RateLimitError` with `retry_after_ms`, and so on), `models.list()`, `extra_body` and `response_model`. The env vars are `KIME_API_KEY`, `KIME_BASE_URL` and `KIME_DEFAULT_MODEL`, with `TYPESAFE_*` read as fallbacks.

The GIL is released for the whole of `decide`. Async methods run on the engine's own threads and complete an asyncio future, so an event loop is never blocked.

## TypeScript: `@kime/sdk`

A pure TypeScript HTTP client for Node 20 and newer, Bun, Deno, edge runtimes and browsers (browser use only with `dangerouslyAllowBrowser`, as in the TypeSafe SDK). It mirrors `@typesafe-ai/sdk` 0.6.0: `new TypeSafeClient({apiKey, baseURL, defaultModel, retry, timeout, fetch})`, `systemOne({state, questions, model}, {signal, timeout})`, `.withResponse()`, the `noul()`, `choice()` and `score()` builders, and typed answers inferred from the question map. The one addition is a typed `kime` options field.

`@kime/node` is an optional native addon (napi-rs) for in process inference in Node, with the same API plus `local: "kime-v1-s-en"`. It ships prebuilt binaries for linux-x64 (CPU and CUDA), linux-arm64, darwin-arm64 and win32-x64.

## CLI: `kime`

One binary, with subcommands:

| Command | What it does |
|---|---|
| `kime serve` | Run the HTTP server (see 11). |
| `kime predict` | Answer questions from the command line: `kime predict --model kime-v1-s-en --state @ticket.json --preset triage`, or `--questions @q.json`. Prints the response JSON, or a table with `--format table`. Reads JSONL on stdin with `--batch` and writes JSONL out, for offline jobs at full device throughput. |
| `kime pull` | Download a model into the cache (`kime pull kime-v1-s-en`, or `kime pull convaiinnovations/laya`). |
| `kime convert` | Convert a Laya checkpoint to kime format, pack to `.kime`, prepack for a backend, build ANE packages, or trim a vocabulary. Refuses to overwrite existing output without `--force`, as laya-mlx's `convert` does. |
| `kime train` | Fine tune (see 12). |
| `kime calibrate` | Fit temperatures on a labelled file and write `calibration.json`. |
| `kime eval` | Run quality suites (see 13). |
| `kime bench` | Run speed suites (see 13). `--tune` tunes GEMM tiles for the local GPU. |
| `kime report` | Build the scorecard from eval and bench outputs. |
| `kime doctor` | Print the detected devices, drivers, CPU features, the cache location and size, and run a 1 second self test per backend. |
| `kime models` | List cached and served models. |

## Presets

`kime-core::presets` ports Laya's five presets with identical question ids, instructions and criteria. That way results from one system can be compared directly with the other:

- `triage`: intent (choice of 6), is_urgent (noul), frustration (score of 4), refund_requested (noul), churn_risk (noul).
- `email`: category (choice of 6), is_spam, is_phishing, urgency (score of 3), needs_reply.
- `guard`: jailbreak, prompt_injection, sensitive_data, harm_severity (score of 4), topic (choice of 6).
- `moderation`: toxic, harassment, threat, spam, severity (score of 4).
- `router`: difficulty (score of 4), domain (choice of 6), needs_tools, is_sensitive.

kime adds `agent_step`, a function that builds jev-ultrafast style operation and target heads from an element table, so browser agents can use kime without writing the question JSON themselves.

## Packaging and deployment

- **Docker images** `ghcr.io/tamnd/kime:<ver>-cpu` (distroless, about 40 MB plus models) and `:<ver>-cuda12` (based on the CUDA runtime image). Models are pulled at start into a volume, or baked in with the `:<ver>-cuda12-s-en` style tags.
- **Homebrew** formula `kime` for macOS, with Metal and ANE.
- **A Nix flake** with a package and a NixOS module, matching Laya's module: a hardened `DynamicUser` systemd unit, `LoadCredential` for the keys file, and options for every config field.
- **A Helm chart** with GPU node selectors, a readiness probe on `/ready`, and a `ServiceMonitor` for Prometheus.
- **Static release binaries** on GitHub for linux x86_64 (v3 and v4), linux aarch64, macOS arm64 and windows x86_64, each with a blake3 checksum and a sigstore signature.
- **Model hosting** on Hugging Face under `tamnd/`, one repo per model, each with prebuilt prepacked caches for the common devices and ANE packages for the native tiers.

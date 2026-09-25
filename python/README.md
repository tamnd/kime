# kime for Python

The kime engine inside your Python process, with Laya's Python API. It is the same Rust code `kime serve` runs, built with pyo3 as one abi3 wheel for CPython 3.10 and newer, and it does not need PyTorch.

```python
import kime

agent = kime.load("convaiinnovations/laya")          # or "laya", "laya-multilingual", a path
agent.system_one("I was charged twice", kime.triage_questions())
agent.predict_batch(["ticket one", "ticket two"], kime.email_questions())
```

Code written for `laya.load(...)` works after changing the import. The answers have Laya's shape and match Laya's to the fourth decimal on the 200 parity cases in `crates/kime-eval/fixtures/parity`.

What is here so far:

- `kime.load` and `kime.Agent` with `system_one`, `predict` and `predict_batch`. `device` is `auto`, `cpu`, `cuda` or `cuda:N`, and kime adds `threads` and `precision` (`f16`, `f32` or `int8`).
- The presets: `triage_questions`, `email_questions`, `guard_questions`, `moderation_questions` and `router_questions`.
- `clean_email_body` and `email_state`, which give the same text as Laya's.
- `agent_step`, which builds a browser agent step the way jev-ultrafast does and reads the answers back into an action.
- `kime.Router` with Laya's arguments, precedence and `RouteDecision`, plus `detect_language`, `detect_script` and `is_english`. It picks the checkpoint the way `kime serve` does, with Laya's word lists and a language identifier on top. `Router(identifier=False)` gives Laya 0.3.20's decisions exactly.
- `kime.TypeSafeClient` and `kime.AsyncTypeSafeClient`, the client from `typesafe_sdk` 0.7.1 with the same names, arguments, answers, errors and retry rules, for `kime serve`, Jev or a model in process. See below.

- `decide` on `kime.decide`, `Agent` and `Router`, which turns a JSON schema or a Pydantic model into questions and gives back the schema's values, with `DecisionResult`, `SchemaError` and the other helpers from `laya.structured`.
- `predict_shortlist` and `shortlist_choice`, which keep the `k` labels of a big choice question closest to the state under an `embed_fn` you pass, as Laya's do.

Not yet: hooks, `lang_temperatures`, `max_len` and `head_max_len` raise `NotImplementedError`, and so does `embed_fn_from_agent`.

kime never downloads anything from Python. Fetch a model first with `kime pull convaiinnovations/laya`, or point `load` at a checkpoint that Laya or `huggingface_hub` already put in the Hugging Face cache.

## The TypeSafe client

Code written for `typesafe_sdk` runs against kime by changing the import:

```python
from kime import TypeSafeClient, Choice, Noul

client = TypeSafeClient(base_url="http://127.0.0.1:8000")   # kime serve, Jev or impossibl
client = TypeSafeClient(local="laya")                        # in process, no HTTP and no server
r = client.system_one("I was charged twice", {"billing": Noul(instructions="Is this about billing?")})
r.nouls["billing"].noul, r.request_id, r.usage
```

`local=` takes what `kime.load` takes, or a `kime.Agent` you already have, and answers the way `kime serve` does, so switching between local and remote changes nothing else. The module uses only the standard library. The differences from `typesafe_sdk`: a key is only required for api.typesafe.ai, the base URL defaults to `http://127.0.0.1:8000`, the `KIME_` environment variables are read first with the `TYPESAFE_` ones as fallbacks, and `http_client` is refused since there is no httpx. Questions and answers are plain classes, and `response_model` still takes a Pydantic model when Pydantic is installed.

On the Mac, 300 English papluca texts, each asked a noul, a choice and a score question, gave the same 300 responses through `typesafe_sdk` 0.7.1 and through `kime.TypeSafeClient` against `kime serve`, and through `local=` with no server. The client itself costs 75 us a call against 211 to 231 us for `typesafe_sdk` on `models.list()` against the same server, imports in 29 to 46 ms against 106 to 158 ms, and the process peaks at 25 MB against 46 MB. `tools/python/typesafe.py` runs the comparison.

## Building

```sh
cd python
uv venv && uv pip install maturin pytest
maturin develop --release
pytest tests
```

The tests that run the model are skipped when the Laya checkpoint is not on disk.

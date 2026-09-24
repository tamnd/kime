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

Not yet: hooks, `lang_temperatures`, `max_len` and `head_max_len` raise `NotImplementedError`. `Router`, `decide` with a schema and the TypeSafe style client come next (see issue #28).

kime never downloads anything from Python. Fetch a model first with `kime pull convaiinnovations/laya`, or point `load` at a checkpoint that Laya or `huggingface_hub` already put in the Hugging Face cache.

## Building

```sh
cd python
uv venv && uv pip install maturin pytest
maturin develop --release
pytest tests
```

The tests that run the model are skipped when the Laya checkpoint is not on disk.

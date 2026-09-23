# kime

kime (決め, "the decision") answers typed questions about text. You give it a state, which can be an email, a support ticket, a JSON document or what an agent sees on a web page, and a set of questions, each one a choice between options, a score on an ordered scale, or a yes or no statement. It returns a calibrated probability for every possible answer. It generates no text, so there is nothing to parse and nothing to hallucinate.

It is written in Rust. It covers every feature of TypeSafe's Jev and of Laya, speaks both of their wire formats, and is built to be ten times faster and cheaper than the best of them on every speed and cost benchmark while matching or beating them on accuracy and calibration.

The design is in [`spec/`](spec/). [`spec/01-overview.md`](spec/01-overview.md) is the place to start.

## Status

Nothing works yet. The specification is written, the workspace, CI and release pipeline are in place, and the first milestone is under way. The table below is what the project has to hit, not what it does today. When a row is measured, the number goes here with the command that produced it.

## What it looks like

```sh
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Hi, we were billed twice for March. Refund the duplicate today or we cancel.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this",
               "options": {"billing": "Payment issues", "technical": "Bugs", "sales": "Plans", "other": "Anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is it",
               "criteria": ["not urgent", "this week", "today", "right now"]},
    "churn":  {"type": "noul", "instructions": "The customer threatens to leave"}
  }
}'
```

The same request works against Jev's API unchanged, and Laya's Python API works against the kime package with an import change. [`spec/03-api.md`](spec/03-api.md) has the full wire format.

## The bar

Ten times the best baseline on the same hardware for every speed and cost row, and equal or better for every quality row. The baselines are Laya 0.3.7 on its own hardware, laya-mlx and laya-coreml on Apple silicon, and Jev as hosted. The full scorecard, with the reason next to the one row that cannot reach 10x, is in [`spec/13-benchmarks.md`](spec/13-benchmarks.md).

| Row | Baseline | Target |
|---|---|---|
| One decision on a short message, T4 | 32.8 ms (Laya) | 3.3 ms |
| Per question, ten questions over a document, T4 | 7.2 ms (Laya) | 0.72 ms |
| Browser agent step, 5.3k tokens, L4 | 178 ms (Jev, hosted) | 17.8 ms |
| One decision, M3 Max | 4.88 ms (laya-coreml) | 0.49 ms |
| Energy per decision, M3 Max | 0.134 J (laya-coreml) | 0.0134 J |
| One decision, Ryzen 9 6900HX | 329 ms (Laya) | 32.9 ms |
| Cost per million input tokens | $0.042 (Jev) | $0.0042 |
| typed-decisions accuracy | 0.766 | 0.78 zero shot |
| MASSIVE, 51 languages | 0.366 (Laya multilingual) | 0.55 |
| Order flip rate, MASSIVE en | 0.13 (Jev) | 0.02 |

## How

The engine alone, running Laya's own weights, is worth three to five times: unpadded variable length attention, fused kernels, CUDA graphs per shape bucket, and no framework dispatch on the hot path. The rest comes from the model. Laya re-encodes the whole state once per question. kime-v1 encodes the state once, caches its keys and values, and answers every question with a small tower that cross attends into that cache, so ten questions cost little more than one. The options are permutation equivariant by construction, which is where the order flip rate comes from. [`spec/05-model.md`](spec/05-model.md) has the architecture and [`spec/07-engine.md`](spec/07-engine.md) the runtime.

## The repositories

| Repository | What it is |
|---|---|
| [tamnd/kime](https://github.com/tamnd/kime) | The engine, the server, the CLI, the training code and the specification |
| [tamnd/kime-bench](https://github.com/tamnd/kime-bench) | The benchmark harness: kime against Laya, laya-mlx, laya-coreml and Jev, on named machines, with the reporting rules |
| [tamnd/kime-compat](https://github.com/tamnd/kime-compat) | The compatibility harness: the TypeSafe API and SDKs, jev-ultrafast and Laya's clients, run against kime |

The Python, TypeScript and Swift SDKs will get repositories of their own under `tamnd/kime-*` when they exist.

## Building

```sh
cargo build --release
./target/release/kime doctor
cargo xtask ci
```

Rust 1.90 or newer. The CUDA, Metal and Neural Engine backends need their toolkits and are off by default. [CONTRIBUTING.md](CONTRIBUTING.md) has the rest.

## License

Apache-2.0. See [LICENSE-APACHE](LICENSE-APACHE).

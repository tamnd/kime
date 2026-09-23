# Testing

Three kinds of tests keep kime honest:

- Correctness tests show the math is right (parity with references, cross backend tolerances, determinism).
- Compatibility tests show that clients of Jev and Laya work unchanged.
- Model behaviour tests show the trained weights behave the way the specs promise.

All of them run in CI. The GPU ones run on self hosted runners.

## Reference parity

`tools/ref/` holds PyTorch implementations used only for tests:

- `laya_ref.py` imports the real `laya` 0.3.7 package and dumps, for a fixture set, the token ids, marker positions and logits of every question, plus every intermediate hidden state when `--dump-hidden` is set.
- `kime_ref.py` is a readable PyTorch implementation of kime-v1 from the same `kime.json`, written separately from `kime-train` so that bugs are not shared.

Fixtures (`kime-eval/fixtures/parity/`): laya-mlx's 16 cases with 63 questions (8 languages, empty and long states, conversations, mask literals, structured criteria, mixed batches, 20 options), extended to 200 cases. The extension adds JSON states with Unicode keys, 255 option choices, 32 level scores, noul with custom criteria, backticked state paths, truncation in each mode, and segment mode states.

Tolerances against the FP32 reference:

| Comparison | Argmax agreement | Max abs probability error | Max abs logit error |
|---|---|---|---|
| Tokenizer ids vs HF tokenizers | exact | n/a | n/a |
| Layout (ids, markers, positions) vs laya_ref | exact | n/a | n/a |
| CPU FP32 vs reference | 100% | 1e-5 | 1e-4 |
| CUDA, Metal FP16 vs reference | 100% on parity fixtures | 6e-3 | 5e-2 |
| ANE FP16 vs reference | 100% on buckets it serves | 1e-2 | 1e-1 |
| FP8, INT8, W8 vs FP16 | 99.5% on the calibration split (release gate) | 5e-2 | n/a |

laya-mlx reached FP16 errors of 0.0054 and laya-coreml an ANE drift of 0.0029 at L96, so these tolerances are achievable.

## Determinism

For every backend in CI, each parity fixture is run:

- alone;
- inside 20 random batches with other fixtures, in random order and at random bucket padding;
- after the state memory cache is warmed by an unrelated request with the same state.

The logits must match bit for bit across all runs. A failure prints the first differing op, found by re-running with per op dumps.

## Kernel tests

Every kernel has a test against a naive Rust implementation over random shapes drawn from the buckets: odd sequence lengths, sequences of length 0 and 1, windows larger than the sequence, all segment layouts, and GQA ratios. The GEMM tests cover every tile config the tuner can pick. Attention kernels are also tested against a dense masked implementation with the mask built explicitly, so the implicit mask logic is checked.

## API compatibility

| Test | What passes |
|---|---|
| TypeSafe OpenAPI | Every kime response validates against api.typesafe.ai's `/openapi.json` snapshot (checked in). Schemathesis generates valid requests from it, and kime must accept every one. |
| TypeSafe Python SDK | The SDK's recorded fixtures replay against kime-serve. The SDK's typed parsing succeeds on every response. |
| TypeSafe JS SDK | The examples run against kime-serve under Node 20 and Bun. |
| jev-ultrafast | `validate_choice` passes on 10,000 generated agent step responses, and the Wikipedia example runs end to end against a local kime-serve with the `jev-latest` model name unchanged. |
| Laya | Laya's own test suite, pointed at the kime Python package's Laya style API (`kime.Router`, `kime.load`, presets, email, shortlist), passes, except for tests that assert Laya's documented bugs. Those are listed in `tests/laya_expected_diffs.md` with the reason. |
| laya-serve clients | The impossibl and laya-serve curl examples give the same response shape, and the same answers within the parity tolerance on compat models. |
| Errors | Snapshot tests for every error status body in 03. |

## Model behaviour tests

These run on every released checkpoint and are part of the release gate in 12:

- **Order invariance.** The flip rate under 5 permutations on MASSIVE en, Emotion and XNLI. The target is 0.02 or less.
- **Score mirror.** Reversing the levels mirrors the argmax in at least 98% of 1,000 cases.
- **Noul polarity.** Swapping the true and false criteria gives `1 - noul` within 0.05 in at least 97% of cases.
- **Level coverage.** Every score level wins somewhere on each suite where it is gold at least 1% of the time. This catches Laya #131.
- **Label neutrality.** Renaming choice labels to neutral ids changes accuracy by at most 1 point when descriptions are present.
- **Chunk consistency.** The same 64 option choice is scored with chunk sizes 8, 16 and 32, and the final argmax agrees in at least 99% of cases.
- **Segment consistency.** Segment mode on and off agree on argmax in at least 99% of cases.
- **Routing.** A labelled set of 5,000 short and long texts across 60 languages, including the cases from Laya issues #20, #54, #130, #168, #172 and #178. Accuracy of the English vs multilingual decision is at least 99%, and no English prose is misrouted, which Laya checked on 20,000 English texts.
- **Calibration regression.** No bucket's ECE increases by more than 0.01 compared with the previous release.

## Tokenizer and input fuzzing

- `cargo fuzz` targets for the JSON request parser, the renderer, both tokenizers and the email cleaner. The invariants: no panics, output length bounded by a linear function of the input, and the tokenizers decode back to the normalized input.
- Property tests: rendering then tokenizing is a pure function of the request (the same bytes give the same ids). Question tower rows depend only on the question, not on the state.

## Server tests

- Load tests in CI (a smaller W9 on a CPU runner) assert no errors, p99 under 5x p50, and bounded memory.
- Chaos tests: kill the device worker mid batch, inject CUDA errors through a test hook, fill the state cache, and send a request larger than the token budget. The server must answer every request with a correct response or a documented error, and must return to healthy.
- Rate limit and overload tests check the 429 and 529 behaviour and the `retry-after-ms` values.

## CI layout

| Job | Runner | Trigger |
|---|---|---|
| fmt, clippy (deny warnings), unsafe audit, `cargo deny` | Linux | every push |
| Unit and property tests, CPU backend, tokenizer parity (sampled) | Linux x86, Linux ARM, macOS | every push |
| Metal and ANE parity, determinism | macOS arm64 self hosted | every push to main, and PRs touching `kime-metal` or `kime-ane` |
| CUDA parity, determinism, kernel tests | T4 and L4 self hosted | every push to main, and PRs touching `kime-cuda`, `kime-model` or `kime-engine` |
| API compatibility | Linux | every push |
| Full tokenizer parity (10M lines), fuzzing (1 hour per target) | Linux | nightly |
| Speed suite on T4, L4, M3 Max and Ryzen 6900HX, with a regression alert over 5% | self hosted | nightly, results to `bench/nightly/` |
| Quality suite on released and candidate checkpoints | H100 | on checkpoint release |

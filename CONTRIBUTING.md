# Contributing

Thanks for looking. This document is short because most of what a contributor needs to know is in [`spec/`](spec/), and this is only the part about how changes get made.

## Before you start

The project is at M0. The design is written down and the code is not, which means one of the most useful contributions right now is reading a specification document and telling us where it is wrong. The open questions are issues labelled `kind/open-question`, and an argument against one of the current answers is worth as much as a patch.

If you want to write code, take an issue from the current milestone and say so on it first. The milestones in [`spec/16-roadmap.md`](spec/16-roadmap.md) are ordered so that each one proves a claim the next one depends on. The compat engine comes before the new model because it measures the engine on weights we did not train, and work on M3 before M0 exists is work that gets thrown away.

## Versions and releases

The minor version is the number of milestones finished. Work inside M0 is tagged 0.0.1, 0.0.2 and so on, the release where M0's exit criterion passes is 0.1.0, work inside M1 is 0.1.1 upwards, and so on. When M5 closes the version is 1.0.0, because that milestone is named 1.0.

Tagging is the whole release process. Push a tag that matches the version in `Cargo.toml`, and the release workflow checks the tag against the manifest, checks that CHANGELOG.md has a section for it, runs the gate, builds the archives for every target, attests them and publishes. If any of those fail there is no release.

Every release that ships weights states the model hashes and the `.kime` format version, because a packed model outlives the build that wrote it.

## Running the checks

```
cargo xtask ci
```

That runs what the per commit CI job runs, cheapest first. The pieces:

```
cargo fmt --all --check
cargo xtask style       # prose against the house rules
cargo xtask unsafe      # crates outside the six allowed ones forbid unsafe
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo xtask msrv        # the workspace still builds on the rust-version in Cargo.toml
```

The MSRV check needs that toolchain, `rustup toolchain install 1.90.0 --profile minimal`. If it is not installed the task says so and passes, because CI runs it either way.

## What a change has to come with

**A change to behavior comes with a test that fails without it.** Not a test that exercises the code, a test that fails. If you cannot write one, say so in the pull request and explain why.

**A kernel comes with a test against the naive implementation.** Over random shapes drawn from the buckets, including sequences of length 0 and 1, per [`spec/15-testing.md`](spec/15-testing.md). A fast kernel that is wrong on a shape nobody tried is the worst kind of bug this project can have, because it returns a plausible probability instead of crashing.

**A change that could move a logit comes with the parity run.** Against the FP32 reference, within the tolerances in [`spec/15-testing.md`](spec/15-testing.md), and bit exact against itself alone and inside a batch. Determinism is a feature here: the same request gives the same bytes whatever else was in the batch.

**A performance claim comes with the command that reproduces it.** p50, p95 and p99 over at least 2,000 calls after warmup, the machine named, and the baseline measured on the same machine in the same session, per [`spec/13-benchmarks.md`](spec/13-benchmarks.md). A number from a shared CI runner is not a number.

**A quality claim comes with the suite, the split and the contamination check.** Held out test splits only, per [`spec/12-training.md`](spec/12-training.md).

**A new dependency is a decision, not a side effect.** Say why in the pull request. `cargo deny` enforces the license and advisory side, and [`spec/07-engine.md`](spec/07-engine.md) says what stays off the inference path.

**A new `#[ignore]` comes with an issue number.** No test is deleted to make CI green.

## Style

**Rust.** `cargo fmt` decides layout. Comments explain why, not what. Public items get documentation, anything that can panic gets a `# Panics` section, and every `unsafe` block gets a `// SAFETY:` comment naming the invariant that makes it sound. Unsafe is allowed in `kime-tensor`, the four backends and `kime-tok`, and forbidden everywhere else, which `cargo xtask unsafe` checks.

Where a decision in the code follows from the specification, cite it. `// per spec/04-semantics.md` costs one line and saves the next person an afternoon.

**Prose.** README, specification, commit messages, issues and pull requests. Plain English, written the way you would explain it to a colleague. No em dashes and no en dashes: a comma, a colon, a period, parentheses or the word "to" always works. No horizontal rules; use a heading. Do not hard wrap sentences, because a one line per paragraph file produces readable diffs. `cargo xtask style` checks all three.

Publish the losses next to the wins. A benchmark table with the regressions removed is not a benchmark table.

## Commits and pull requests

One logical change per commit. The subject line is imperative and under about seventy characters, and the body says why rather than what. Pull requests describe the problem, then the change, then how it was verified. Squash merges keep main bisectable, which is how latency regressions get found.

## Security

See [SECURITY.md](SECURITY.md).

## License

The project is under Apache-2.0 and a contribution is offered under the same terms, which is what section 5 of the license says. There is no separate agreement to sign.

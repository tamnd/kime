## What this changes

<!-- The problem first, then the change. What is in the diff does not need restating. -->

## Why

<!-- If it closes an issue, say `Closes #N`. If it is part of a milestone, say which. -->

## How it was verified

<!-- Which test fails without this change. If none does, say so and say why. -->

## Checklist

- [ ] `cargo xtask ci` passes locally
- [ ] A change to behavior comes with a test that fails without it
- [ ] A new kernel comes with a test against the naive implementation over random bucket shapes, including lengths 0 and 1
- [ ] A change that could move a logit comes with the parity run against the reference, and determinism alone and in a batch still holds
- [ ] A performance claim comes with the command, the machine, p50, p95 and p99, and the baseline measured in the same session
- [ ] A quality claim names the suite and the split, and the split is held out
- [ ] A change to the `.kime` format bumps the format version and comes with a round trip test
- [ ] A new dependency is justified in the description
- [ ] A new `#[ignore]` comes with an issue number
- [ ] Prose follows the house rules: plain English, no em dashes, no horizontal rules, no hard wrapped sentences

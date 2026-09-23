# Changelog

Notable changes, newest first. The project is pre-1.0 and makes no compatibility promise until it does. The minor version is the number of milestones finished, per [CONTRIBUTING.md](CONTRIBUTING.md), and the milestones are the issues at https://github.com/tamnd/kime/milestones.

## Unreleased

- `kime-tok` encodes and decodes with both Laya tokenizers, byte level BPE for ModernBERT and SentencePiece style BPE with byte fallback for mmBERT, with no dependency on Hugging Face tokenizers. Ids match Hugging Face tokenizers 0.23.2 on all 787,214 lines of the MASSIVE train split in 51 languages plus 200,000 fuzz strings. On one core it is about 10x faster than Hugging Face on that corpus and does a 2 KB English request in about 15 microseconds.

## 0.0.1

The skeleton. The specification in `spec/`, the workspace with its fifteen crates laid out as `spec/07-engine.md` describes, the confidence formulas and largest remainder rounding in `kime-core` with the worked example from `spec/04-semantics.md` as a test, a `kime` binary that answers `--version` and `doctor`, and the CI, nightly and release workflows. No model runs yet.

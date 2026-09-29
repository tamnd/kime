# Training data

The public part of kime's training data, per spec/12-training.md: which datasets it comes from, under which licenses, and the converters that turn their train splits into kime's training format.

## Sources

[sources.json](sources.json) lists every public dataset considered, with its license and whether it is used. A source is used only with `"use": "train"`, and only its train split, from the pinned revision. A license that is missing, unknown, non-commercial or research only means the source is excluded, and the reason is written next to it. That is decided here, before training, so a model never has to be retrained to take a source out.

Twelve sources are used: Banking77, CLINC150 with out of scope, MASSIVE intent and scenario in 51 languages, MultiNLI, WANLI, GoEmotions, Civil Comments, deepset's prompt injections, typed-decisions, BoolQ, SQuAD 2.0 and Mind2Web's browser steps. Fifteen are excluded, among them XNLI, MS MARCO, ToxicChat, SST, AG News, Emotion and the Enron spam and phishing email sets that app.email_spam and app.phishing are built from.

## Converting

    python tools/data/convert.py <out dir> [--only name,name] [--max 300000] [--max-options 20] \
        [--kime target/release/kime --tests <test texts dir> --tests <suites dir>] [--reuse]

Each train row becomes one line with the state, its questions and a target per question:

    {"id": "banking77-0", "source": "banking77", "split": "train", "lang": "en",
     "state": {"text": "I am still waiting on my card?"},
     "questions": {"intent": {"type": "choice", "instructions": "Which banking intent does `text` express?", "criteria": {...}}},
     "targets": {"intent": {"probs": [0.0, 1.0, ...], "hard": 1, "weight": 1.0}}, "episode": null}

`probs` follows the order of the criteria, and `hard` is the index of the gold option. A question gets one of a few instruction phrasings that name the state key in backticks, and the state key itself is picked from a few names. A choice over a large label set gets the gold and a random number of other labels, at most 20 in all (`--max-options`) and in a random order, and CLINC150 always offers out of scope. The other augmentations of spec/12-training.md, such as renaming labels, chunking and truncation, are applied by the trainer when it reads the data, so they change from epoch to epoch. Every draw comes from a generator seeded with the source and the row, so converting again gives the same bytes.

What each source becomes:

| Source | Questions |
|---|---|
| Banking77 | intent, choice of 4 to 20 of the 77 intents |
| CLINC150 | intent, choice of 4 to 20 of 151 with out of scope always there |
| MASSIVE | intent and scenario, two choices over the same utterance, 51 languages |
| MultiNLI, WANLI | relation, choice of 3, named as relations or as yes, maybe and no |
| GoEmotions | emotion as a choice for rows with one label, a noul for one present and one absent emotion otherwise |
| Civil Comments | toxic, a noul whose target is the share of raters who called the comment toxic, sampled to `--max` rows |
| Prompt injections | injection, a noul |
| typed-decisions | its own questions, with the soft gold of its three labellers as the target |
| BoolQ | answer, a noul over passage and question |
| SQuAD 2.0 | answerable, a noul over passage and question |
| Mind2Web | operation and the target of that operation, choices over a browser agent step in jev-ultrafast's format |

Mind2Web is people doing 1,009 tasks on real websites, with the page before each action and the element they acted on. tools/data/mind2web.py turns each action into the snapshot jev-ultrafast would send for that page and runs it through kime's `agent_step`, so the questions are the ones a browser agent asks. The operation head's target is what the person did, CLICK, TYPE_TEXT or SELECT, and the target head of that operation gets the element they acted on. A page offers a window of up to 100 elements that holds the target, since Laya's head fits about 126 target options. The converter needs `ijson` and kime's Python package, and streams the 5.9 GB train split one file at a time.

With `--kime` and `--tests`, each converted source goes through `kime contam` against the test texts before it is sharded, and lines within Jaccard 0.5 of a test text are dropped. The test texts are the quality suites and the whole of every split they are drawn from, as tools/eval/split_texts.py writes them. `--reuse` starts from the raw files already in `<out dir>/raw`, so the conversion can run where the datasets are cached and the check where the test texts are.

The output is:

- `raw/<source>.jsonl`, the converted lines.
- `clean/<source>.jsonl` and `<source>.contam.json`, the lines kept and the `kime contam` report.
- `shards/<source>-00000.jsonl.zst`, zstd shards of about 256 MB of JSON lines each.
- `manifest.json`, with each source's dataset, revision, license, rows, near duplicates dropped and questions, and each shard's file, blake3, source, dataset, config, license, split, rows and questions. `kime eval --data-manifest manifest.json` reads it and refuses to score a model trained on a test split.

## The committed manifest

[manifest.json](manifest.json) is the manifest of the conversion below, and [results/2026-09-28-convert](results/2026-09-28-convert) holds the `kime contam` report of each source. It was converted on an M4 MacBook in 21 minutes, most of it downloading, with 3.1 GB peak memory, and checked against 383,311 test texts, the 62 suites and the 59 whole splits they are drawn from. Converting again on server3 gave raw files of the same bytes.

| Source | Rows | Near a test text | Kept | Questions | License |
|---|---:|---:|---:|---:|---|
| Banking77 | 9,993 | 318 | 9,675 | 9,675 | CC-BY-4.0 |
| CLINC150 | 15,250 | 383 | 14,867 | 14,867 | CC-BY-3.0 |
| MASSIVE | 587,214 | 6,659 | 580,555 | 1,161,110 | CC-BY-4.0 |
| MultiNLI | 392,702 | 38 | 392,664 | 392,664 | OANC, CC-BY-SA-3.0, CC-BY-3.0 |
| WANLI | 102,885 | 0 | 102,885 | 102,885 | CC-BY-4.0 |
| GoEmotions | 43,410 | 16 | 43,394 | 50,496 | Apache-2.0 |
| Civil Comments | 299,850 | 148 | 299,702 | 299,702 | CC0-1.0 |
| Prompt injections | 546 | 12 | 534 | 534 | Apache-2.0 |
| typed-decisions | 1,200 | 96 | 1,104 | 5,520 | Apache-2.0 |
| BoolQ | 9,427 | 1,655 | 7,772 | 7,772 | CC-BY-SA-3.0 |
| SQuAD 2.0 | 130,319 | 84 | 130,235 | 130,235 | CC-BY-SA-4.0 |
| Total | 1,592,796 | 9,409 | 1,583,387 | 2,175,460 | |

The 14 shards are 1,295 MB of JSON lines and 123 MB compressed. Most of what is dropped is short utterances that are also in a test split, such as MASSIVE's Japanese and Chinese commands (1,218 and 703 lines) and Banking77's questions (307 lines near its test split), and BoolQ passages that recur in its validation split (1,647 lines). `kime eval --data-manifest tools/data/manifest.json` over all 62 suites finds no conflict.

## Mind2Web

Mind2Web is converted on its own, since its pages are checked against the Mind2Web suites and not the text suites:

    python tools/data/convert.py <out dir> --only mind2web --kime target/release/kime --tests <mind2web suites dir>

On the M4 it took about 25 minutes, most of it parsing pages. The 1,009 train tasks give 7,036 usable steps. 290 of them are within Jaccard 0.5 of a suite case and are dropped, all against mind2web.task, whose tasks are on the same websites as the train split, and 6 are the same page and step. The website and domain suites have none. That leaves 6,746 steps and 13,197 questions, 5,661 CLICK, 875 TYPE_TEXT and 210 SELECT, in one shard of 9.8 MB. [results/2026-09-29-mind2web](results/2026-09-29-mind2web) has the manifest and the `kime contam` report. The check is against the 1,500 suite cases, not every step of the test splits.

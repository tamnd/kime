# Quality suites

`kime eval` scores an engine on suites of real questions with known answers. These scripts build the suites and answer them with Laya, so the harness can be checked against Laya's published numbers before it is used on kime.

## What is here

`build_suites.py` builds the suites of Laya's benchmark notebook (research/scripts/build_benchmark_nb.py in NandhaKishorM/laya) from the public datasets, with the same seed, option counts and sizes, so the questions are the ones behind Laya's published results in research/results/t4_colab_benchmark.json. That is typed-decisions (2,000 questions), MASSIVE intent and scenario in 14 languages, XNLI in 15, and SST-5, Emotion, prompt injections, Banking77, AG News and BoolQ in English, 51 suites and 17,916 questions. It also builds Laya's seven application themes from research/scripts/bench_apps.py (support triage, email spam, phishing, jailbreak, toxicity, RAG relevance and model routing, 400 cases each, drawn in the script's order from its seed), SST-2 and CLINC150, and the order suites. It writes one `<suite>.jsonl` per suite and a manifest.json with each dataset's revision. PolyAI/banking77 is a loading script the current `datasets` refuses, so Banking77 comes from mteb/banking77, the same test split as parquet.

CLINC150 has 151 labels, too many option names for one sequence, so each of its 1,000 questions gets 20 options like MASSIVE: the gold, out of scope and 18 others at random. Its report groups by `scope`, and the accuracy of the `out` group is out of scope recall.

The order suites, order.massive_intent.en, order.en.emotion and order.xnli.en, hold each line of the base suite and 5 copies with the options in a random order. A copy has `perm_of` set to the id of its original, and the report gives the flip rate, the share of copies whose answer is a different option from the original's, and the share of questions where any copy flips.

en.clerc_rerank and en.clerc_rerank_more are the CLERC rerank setup of TypeSafe's re-ranking cookbook (docs.typesafe.ai/cookbooks/rerank_typesafe): 170 rows of the CLERC training file pooled into one corpus of 3,565 court opinion passages, the BM25 top 30 of that corpus for each query with bm25s and English stopwords, and one noul question for each query and candidate with the cookbook's wording. The rows are drawn as leepokai/llm-prompt-techniques-on-jev draws them, which gets the cookbook's BM25 numbers back. en.clerc_rerank is the cookbook's 40 queries, 1,200 questions, and en.clerc_rerank_more the other 110 of the pool, 3,300 questions. Each line has `rank`, the query and the candidate's BM25 place, and the report ranks each query's candidates by the probability of `true` and gives top 1, top 5, top 10 and MRR for BM25 and for the model. On en.clerc_rerank BM25 alone is 0.050, 0.150 and 0.375, as the cookbook has it. The corpus comes from streaming the first 1,000 usable rows of a 4 GB file, and it needs `bm25s` as well.

mind2web.website, mind2web.task and mind2web.domain are browser agent steps from Mind2Web's three test splits (osunlp/Mind2Web, CC BY 4.0): new websites in the domains of the train split, new tasks on the train split's websites, and domains the train split does not have. Each is 500 steps drawn with the seed from the 1,212, 1,882 and 5,375 usable steps of its split. A step goes through tools/data/mind2web.py and kime's `agent_step`, so its questions are the operation head and the target head of the recorded operation, as jev-ultrafast sends them, and the gold is what the person did. A page offers up to 100 elements, and a target head has a median of 98 options. The test zip is password protected and its authors ask that it is not shared unzipped, so it is only read as a stream and nothing from it is kept but the suites. `baseline_mind2web.py` answers them with no model, the train split's share of each operation and the goal words each element's label holds.

`run_laya.py` answers the suites with Laya the way the notebook does, in length sorted fp16 autocast batches with the calibrated temperature of each question type and option count, and writes `<suite>.answers.jsonl`. `kime eval --answers` scores those files with the same code it scores kime with.

## Running

```
python tools/eval/build_suites.py suites
python tools/eval/run_laya.py <laya model dir> laya-ans suites/*.jsonl --device cuda
kime eval suites --answers laya-ans --out eval-laya
kime eval suites --device cuda --precision f16 --out eval-kime
kime eval suites --suite quality --device metal --out eval-quality
```

`--suite` keeps the suites of one group: `quality` is every suite of spec/13-benchmarks.md, `order` the order suites and `agent` the Mind2Web steps. It can be given more than once, and without it every suite named runs.

`build_suites.py` needs `datasets`, `bm25s` for the CLERC suites, `ijson`, `unzip` and kime's Python package for the Mind2Web suites, and `laya` for the email suite, whose states go through Laya's `email_state` as in bench_apps.py (`import laya` does not need torch). It pulls about 3 GB of datasets into the Hugging Face cache. `run_laya.py` needs the `laya` package and PyTorch.

## Results

results/2026-09-28-rtx4090 has both runs on the RTX 4090, Laya 0.3.20 in PyTorch and kime 0.1.0 in FP16, with against-t4.tsv lining up the correct answers of each suite against Laya's published T4 run. Banking77 is not in Laya's published file, so 49 suites and 17,416 questions compare.

kime gets the same number of questions right as the published run on 40 of the 49 suites and is 9 questions off in all, 3 fewer net. AG News is 0.9467, Emotion 0.5733, SST-5 0.3717, BoolQ 0.8300, prompt injections 0.6983, XNLI English 0.8600 and MASSIVE intent English 0.7833, all as published, and typed-decisions is 0.3615 against 0.362. Laya itself on the 4090 is 47 questions off its own T4 run, 11 fewer net, since fp16 on a different GPU flips close questions. ECE matches the published run to about 1e-3 everywhere except the MASSIVE suites, which have 20 options: Laya 0.3.20 raised the lowest calibrated temperature from 0.1 to 0.5 after the T4 run, which moves ECE and NLL on those suites and not accuracy.

results/2026-09-29-quality-metal is the whole quality group on an M4 with Metal, within one question per suite of the 4090 run.

results/2026-09-29-mind2web has the Mind2Web suites answered by Laya, by cklxx/laya-browser and by the baselines.

`chunk_consistency.py` is the chunk consistency test of spec/15-testing.md: the same 64 option choices from banking77 and the Mind2Web target heads, scored in one pass and in chunks of 8, 16 and 32 with `kime.chunk`, and how often the sizes pick the same option. results/2026-09-30-chunking has Laya's run.

`kime eval --gate` runs the model behaviour tests of spec/15-testing.md on the suites it was given, after the quality run, and exits non-zero when one misses its threshold. The flip rate comes from the order suites, the score mirror and noul polarity tests ask the model again with the levels reversed or the true and false criteria swapped, label neutrality renames the options of English choices whose every option has a description to `option 1` and up, and chunk consistency cuts 300 en.banking77_full questions to 64 options and scores them with `kime.chunk` 8, 16 and 32. `--baseline` names the results.json of the previous release, and each suite's ECE may rise by at most 0.01 against it. results/2026-09-30-gate has Laya's run, which fails 14 of 27 checks.

against-apps.tsv does the same for the application themes against research/results/app_benchmark_results.json, which Laya 0.2.1 wrote on a CPU. kime matches it on five of the seven, ECE included to 1e-4, is one question off on model routing, and gets two more emails right on spam, as Laya 0.3.20 on the 4090 does, so the spam difference is Laya's email cleaning since 0.2.1 and not the engine.

| Order suite | kime flip rate | Laya 0.3.20 flip rate | Laya's notebook, 1 order of 200 |
|---|---|---|---|
| MASSIVE intent English | 0.160 (any of 5: 0.297) | 0.157 (0.303) | 0.15 |
| Emotion | 0.048 (0.110) | 0.048 (0.113) | 0.04 |
| XNLI English | 0.001 (0.003) | 0.001 (0.003) | 0.0 |

The compatible model moves its answer on 16 percent of reordered MASSIVE questions, more than the 0.13 measured for Jev, and spec/13-benchmarks.md asks 0.02 of kime's own model. CLINC150 is 0.913 with out of scope recall 0.659, SST-2 is 0.914, and 1.2 percent of Emotion questions give the gold emotion under 5e-5.

## Contamination

spec/12-training.md says no test split of a benchmark may be trained on, and that training text within Jaccard 0.5 of a test text, over 5-gram word shingles, is dropped. `kime contam` does that check and `kime eval --data-manifest` refuses to score a model whose data manifest lists a test split, or the split a suite was drawn from, as the manifest.json next to the suites records it. The blake3 of the manifest goes into results.json, and report.md says when no manifest was given.

The suites hold samples, so `split_texts.py` writes the whole of every split they are drawn from, 355,124 texts in 59 files, and the check indexes those and the suites together:

```
python tools/eval/split_texts.py suites tests
kime contam --tests tests --tests suites train/*.jsonl --out clean --report report.json
```

Words are lowercased runs of letters and digits, and in Chinese, Japanese, Thai and the other scripts written without spaces each character is a word. Candidates come from MinHash with 128 hash functions in 64 bands of 2, and every candidate is checked with the exact Jaccard of the two shingle sets, so nothing under 0.5 is dropped. On the train and test splits of MASSIVE English, Banking77, Emotion, CLINC150, AG News and BoolQ it finds all 5,725 lines an exhaustive search over every pair finds, where 32 bands of 4 found 5,539. Against the whole splits, 28,146 of the 28,187 suite lines are found; the 41 left are emails the suites cut to 3,000 characters or ran through Laya's email cleaning, and those are in the index as the suites have them.

results/2026-09-28-contamination has the check of the train split of every dataset behind the suites against all of the test texts, 1,462,616 lines in 9.2 seconds on an M4 after 3.5 seconds of indexing, with 2.8 GB resident. 71,614 lines are near duplicates, and train-vs-test.tsv has them by split. suites-vs-train.tsv turns it around and indexes the train lines, 12.3 seconds for 1.46 million, to count the suite lines that are near a train line, which is what a score on the suite would have seen in training. Some are worth knowing before any of these sets is trained on:

- 48 percent of Enron spam train is within 0.5 of a test text, 14,481 lines of it of the phishing email set app.phishing is drawn from, which holds Enron mail. 208 of the 400 app.phishing emails and 165 of the 400 app.email_spam emails are near an Enron spam train email.
- 17.6 percent of BoolQ train shares its Wikipedia passage with a validation question, and 204 of the 600 en.boolq questions have a passage from train.
- 6.7 percent of MS MARCO train lines share a passage with validation, and 93 of the 400 app.rag_relevance pairs are near a train pair.
- MASSIVE train and test share 2 to 5 percent of their utterances in most languages and 10.6 percent in Japanese, where short commands that differ in one word, like 寝室の電気を消す and 浴室の電気を消す, reach 0.5 as characters. 35 of the 300 MASSIVE English questions and 107 of the Japanese ones are near a train utterance.
- 88 of the 600 AG News questions, 67 of the 400 typed-decisions lines and 41 of the 500 Banking77 questions are near a train line. MNLI train, which is XNLI's English train split, shares 38 premises with XNLI English test, and no XNLI English question is within 0.5 of it once the hypothesis is counted.

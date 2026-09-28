# Quality suites

`kime eval` scores an engine on suites of real questions with known answers. These scripts build the suites and answer them with Laya, so the harness can be checked against Laya's published numbers before it is used on kime.

## What is here

`build_suites.py` builds the suites of Laya's benchmark notebook (research/scripts/build_benchmark_nb.py in NandhaKishorM/laya) from the public datasets, with the same seed, option counts and sizes, so the questions are the ones behind Laya's published results in research/results/t4_colab_benchmark.json. That is typed-decisions (2,000 questions), MASSIVE intent and scenario in 14 languages, XNLI in 15, and SST-5, Emotion, prompt injections, Banking77, AG News and BoolQ in English, 51 suites and 17,916 questions. It also builds Laya's seven application themes from research/scripts/bench_apps.py (support triage, email spam, phishing, jailbreak, toxicity, RAG relevance and model routing, 400 cases each, drawn in the script's order from its seed), SST-2 and CLINC150, and the order suites. It writes one `<suite>.jsonl` per suite and a manifest.json with each dataset's revision. PolyAI/banking77 is a loading script the current `datasets` refuses, so Banking77 comes from mteb/banking77, the same test split as parquet.

CLINC150 has 151 labels, too many option names for one sequence, so each of its 1,000 questions gets 20 options like MASSIVE: the gold, out of scope and 18 others at random. Its report groups by `scope`, and the accuracy of the `out` group is out of scope recall.

The order suites, order.massive_intent.en, order.en.emotion and order.xnli.en, hold each line of the base suite and 5 copies with the options in a random order. A copy has `perm_of` set to the id of its original, and the report gives the flip rate, the share of copies whose answer is a different option from the original's, and the share of questions where any copy flips.

`run_laya.py` answers the suites with Laya the way the notebook does, in length sorted fp16 autocast batches with the calibrated temperature of each question type and option count, and writes `<suite>.answers.jsonl`. `kime eval --answers` scores those files with the same code it scores kime with.

## Running

```
python tools/eval/build_suites.py suites
python tools/eval/run_laya.py <laya model dir> laya-ans suites/*.jsonl --device cuda
kime eval suites --answers laya-ans --out eval-laya
kime eval suites --device cuda --precision f16 --out eval-kime
```

`build_suites.py` needs `datasets`, and `laya` for the email suite, whose states go through Laya's `email_state` as in bench_apps.py (`import laya` does not need torch). It pulls about 3 GB of datasets into the Hugging Face cache. `run_laya.py` needs the `laya` package and PyTorch.

## Results

results/2026-09-28-rtx4090 has both runs on the RTX 4090, Laya 0.3.20 in PyTorch and kime 0.1.0 in FP16, with against-t4.tsv lining up the correct answers of each suite against Laya's published T4 run. Banking77 is not in Laya's published file, so 49 suites and 17,416 questions compare.

kime gets the same number of questions right as the published run on 40 of the 49 suites and is 9 questions off in all, 3 fewer net. AG News is 0.9467, Emotion 0.5733, SST-5 0.3717, BoolQ 0.8300, prompt injections 0.6983, XNLI English 0.8600 and MASSIVE intent English 0.7833, all as published, and typed-decisions is 0.3615 against 0.362. Laya itself on the 4090 is 47 questions off its own T4 run, 11 fewer net, since fp16 on a different GPU flips close questions. ECE matches the published run to about 1e-3 everywhere except the MASSIVE suites, which have 20 options: Laya 0.3.20 raised the lowest calibrated temperature from 0.1 to 0.5 after the T4 run, which moves ECE and NLL on those suites and not accuracy.

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

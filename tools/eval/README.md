# Quality suites

`kime eval` scores an engine on suites of real questions with known answers. These scripts build the suites and answer them with Laya, so the harness can be checked against Laya's published numbers before it is used on kime.

## What is here

`build_suites.py` builds the suites of Laya's benchmark notebook (research/scripts/build_benchmark_nb.py in NandhaKishorM/laya) from the public datasets, with the same seed, option counts and sizes, so the questions are the ones behind Laya's published results in research/results/t4_colab_benchmark.json. That is typed-decisions (2,000 questions), MASSIVE intent and scenario in 14 languages, XNLI in 15, and SST-5, Emotion, prompt injections, Banking77, AG News and BoolQ in English, 51 suites and 17,916 questions. It writes one `<suite>.jsonl` per suite and a manifest.json with each dataset's revision. PolyAI/banking77 is a loading script the current `datasets` refuses, so Banking77 comes from mteb/banking77, the same test split as parquet.

`run_laya.py` answers the suites with Laya the way the notebook does, in length sorted fp16 autocast batches with the calibrated temperature of each question type and option count, and writes `<suite>.answers.jsonl`. `kime eval --answers` scores those files with the same code it scores kime with.

## Running

```
python tools/eval/build_suites.py suites
python tools/eval/run_laya.py <laya model dir> laya-ans suites/*.jsonl --device cuda
kime eval suites --answers laya-ans --out eval-laya
kime eval suites --device cuda --precision f16 --out eval-kime
```

`build_suites.py` needs `datasets` and pulls about 2 GB of datasets into the Hugging Face cache. `run_laya.py` needs the `laya` package and PyTorch.

## Results

results/2026-09-28-rtx4090 has both runs on the RTX 4090, Laya 0.3.20 in PyTorch and kime 0.1.0 in FP16, with against-t4.tsv lining up the correct answers of each suite against Laya's published T4 run. Banking77 is not in Laya's published file, so 49 suites and 17,416 questions compare.

kime gets the same number of questions right as the published run on 40 of the 49 suites and is 9 questions off in all, 3 fewer net. AG News is 0.9467, Emotion 0.5733, SST-5 0.3717, BoolQ 0.8300, prompt injections 0.6983, XNLI English 0.8600 and MASSIVE intent English 0.7833, all as published, and typed-decisions is 0.3615 against 0.362. Laya itself on the 4090 is 47 questions off its own T4 run, 11 fewer net, since fp16 on a different GPU flips close questions. ECE matches the published run to about 1e-3 everywhere except the MASSIVE suites, which have 20 options: Laya 0.3.20 raised the lowest calibrated temperature from 0.1 to 0.5 after the T4 run, which moves ECE and NLL on those suites and not accuracy.

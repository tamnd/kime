# Training objectives: proper score, cross entropy and rlcd_pg

The ablation of #37. The same fine tune of Laya's English checkpoint is run three times with only the loss changed, and each model is scored on held out lines and on test suites. All runs start from Laya, train the top 4 of 28 encoder layers and the heads for one epoch at 2,048 tokens a batch, with seed 13, on an M4 with 24 GB and Burn's Metal backend.

- `proper` is the proper score loss of spec/12-training.md, the default. It is the exact gradient of the log, spherical and ranked probability scores.
- `ce` is cross entropy against the soft target, with no floor, no spherical term and no ranked probability term.
- `pg` is rlcd_pg as spec/12-training.md describes it: 8 Gaussian perturbations of the logits a question, sigma from 1.0 down to 0.3, the proper score of each as its reward, and advantages normalized within each question's group. It estimates the gradient `proper` has exactly.

```
kime-train --base laya --data shards --only <sources> --out <dir> --tokens 2048 --train-layers 4 --eval-max 500 --eval-every <200 or 50> --seed 13 --loss <proper, ce or pg>
kime eval <suites> --model <dir> --device metal --precision f32
```

## Choice questions: Banking77 and CLINC150

The data is the 12,000 lines of the Banking77 and CLINC150 shards from the side by side run (tools/eval/results/2026-09-29-train-side-by-side), with the same 491 held out. The `proper` column is kime seed 13 from that run, and seed 14 there shows how far a run moves by chance.

| Suite | Metric | Laya | proper | ce | pg |
|---|---|---|---|---|---|
| Held out, 491 | Proper score loss | -0.2203 | -0.2596 | -0.2608 | -0.0013 |
| | Accuracy | 0.9369 | 0.9409 | 0.9409 | 0.9369 |
| | NLL | 0.2491 | 0.2130 | 0.2119 | 0.4685 |
| en.banking77_full, 500 | Accuracy | 0.4600 | 0.4620 | 0.4600 | 0.4560 |
| | NLL | 3.9999 | 2.8861 | 2.8747 | 12.2394 |
| | ECE | 0.3211 | 0.2142 | 0.2244 | 0.4912 |
| | Brier | 0.8372 | 0.7470 | 0.7516 | 1.0276 |
| | AUROC | 0.7660 | 0.7758 | 0.7820 | 0.6826 |
| en.clinc150, 1000 | Accuracy | 0.9120 | 0.9130 | 0.9140 | 0.9130 |
| | NLL | 0.5315 | 0.4476 | 0.4541 | 1.8263 |
| | ECE | 0.0665 | 0.0572 | 0.0583 | 0.0836 |
| | Brier | 0.1477 | 0.1459 | 0.1469 | 0.1689 |
| | AUROC | 0.8914 | 0.8973 | 0.8938 | 0.6538 |

`proper` and `ce` give the same model to within the seed to seed spread of the side by side run, where two proper seeds are 0.16 apart in Banking77 NLL and 0.029 in ECE. Here they are 0.011 and 0.010 apart.

`pg` makes the model worse. Accuracy stays where it was, but the held out NLL nearly doubles and Banking77 NLL goes from 4.0 to 12.2, which is a model that is confidently wrong on the questions it misses. The likely cause is the group normalization. Advantages are divided by the spread of the rewards within the group, so a question the model already gets right with high confidence, whose 8 rewards differ only a little, still gets a push of full size, and that push is towards an even sharper distribution. The median gradient norm before clipping is 11.3 for `pg` against 1.2 for `proper` and `ce`, so every `pg` step is clipped to 1.0 and moves the weights by about as much whether there was anything to learn or not. 29 of the 649 steps had a gradient of exactly zero, batches where every question was so confident that no perturbation changed its f32 probabilities.

## Score and noul questions: typed_decisions

The rlcd_pg of #37 is named for the score and noul families, which Banking77 and CLINC150 do not have. So the three losses also train on the typed_decisions shard of tools/data, 1,082 lines with 5 questions each (choice, score and noul), with 22 lines held out, and are scored on the 2,000 question typed_decisions suite from the dataset's test split. Laya starts far from these targets, with a proper score loss of 1.43 on the held out questions.

| Suite | Metric | Laya | proper | ce | pg |
|---|---|---|---|---|---|
| Held out, 110 | Proper score loss | 1.4316 | 0.7758 | 0.7766 | 0.8207 |
| | Accuracy | 0.3273 | 0.5091 | 0.5091 | 0.5545 |
| | NLL | 1.5953 | 0.9586 | 0.9565 | 1.0187 |
| typed_decisions, 2000 | Accuracy | 0.3615 | 0.5410 | 0.5330 | 0.5030 |
| | NLL | 1.3227 | 1.0258 | 1.0240 | 1.0873 |
| | ECE | 0.1747 | 0.1139 | 0.1055 | 0.0830 |
| | Brier | 0.7497 | 0.5885 | 0.5874 | 0.6162 |
| | AUROC | 0.5818 | 0.6334 | 0.6366 | 0.6201 |
| | Score MAE | 0.6937 | 0.5595 | 0.5597 | 0.5775 |
| | Within 1 | 0.7525 | 0.8600 | 0.8575 | 0.8225 |

| typed_decisions accuracy by type | Laya | proper | ce | pg |
|---|---|---|---|---|
| choice, 600 | 0.2883 | 0.5333 | 0.5233 | 0.4700 |
| noul, 600 | 0.4867 | 0.6500 | 0.6400 | 0.6483 |
| score, 800 | 0.3225 | 0.4650 | 0.4600 | 0.4188 |

Here `pg` does learn, since there is a lot to learn, but it trails the exact gradient on accuracy (0.503 against 0.541, below the 95% interval of `proper`), NLL, Brier and score MAE. It has the lowest ECE, 0.083 against 0.114, likely because it moved the model less far from Laya's probabilities. The 110 held out questions are too few to rank the three on accuracy.

`proper` and `ce` again land within noise of each other, score questions included. The ranked probability term in `proper` does not show up in score MAE in one epoch of 1,082 lines.

The data manifest of these shards passes the contamination gate for the typed_decisions suite ("no test split listed", typed-gate.report.md).

## What this means for kime

The spec's choice holds: train on the exact proper score and keep rlcd_pg only for rewards that have no gradient in p, the decision cost of the correctness head and the TD(lambda) episode returns. Those two wait for the correctness head and the conversation data. As a replacement for the proper score, rlcd_pg with group normalized advantages is worse in every run here and harmful once the model is already good.

Cross entropy against the soft target is as good as the proper score on these runs. The proper score is still the default because its floor and spherical term bound the loss of a confidently wrong answer and keep the objective strictly proper when targets are soft, which matters more on the teacher labels of M2 than on the mostly one hot targets here.

## Cost

The runs shared the machine with evaluations and a crates.io publish, so their times are not comparable with each other or with the side by side run. For the record, `ce` took 3,909 seconds on the choice data and `pg` 3,543, and on typed_decisions `proper` 5,987, `ce` 7,764 and `pg` 4,949. The perturbations of `pg` act on the logits only, so they add nothing measurable to the cost of a step.

## Files

`choice-*` and `typed-*` are the reports and per step training logs of each run, and `typed-laya.report.md` is Laya's typed_decisions report. Laya's and `proper`'s choice reports are in the side by side folder.

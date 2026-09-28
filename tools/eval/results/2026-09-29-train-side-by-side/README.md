# kime-train against the PyTorch reference

The done check of #36: a model trained with kime-train against the same model trained with tools/ref/train_ref.py, on two suites. Both start from Laya's English checkpoint, train the top 4 of 28 encoder layers and the heads for one epoch at 2,048 tokens a batch, on an M4 with 24 GB. kime-train runs on Burn's Metal backend and the reference on PyTorch 2.14 with MPS.

The data is 12,000 lines of the Banking77 and CLINC150 shards of tools/data, with 491 held out. `kime-train --dump` wrote the laid out questions of seed 13's first epoch and the reference trained on exactly those, so kime seed 13 against the reference differs only in the trainer. kime seed 14 draws another 12,000 lines and another batch order, and shows how far a run moves by chance.

```
kime-train --base laya --data shards --only banking77,clinc150 --max-lines 12000 --out <dir> --tokens 2048 --train-layers 4 --eval-max 500 --eval-every 200 --seed <13 or 14>
kime-train <same flags, seed 13> --dump data
python tools/ref/train_ref.py --base laya --data data --out ref --train-layers 4 --tokens 2048 --eval-every 200 --seed 13 --device mps
kime eval en.banking77_full.jsonl en.clinc150.jsonl --model <dir> --device metal --precision f32
```

## Suites

| Suite | Metric | Laya | kime seed 13 | kime seed 14 | Reference seed 13 |
|---|---|---|---|---|---|
| en.banking77_full, 500 | Accuracy | 0.4600 | 0.4620 | 0.4580 | 0.4600 |
| | NLL | 3.9999 | 2.8861 | 3.0464 | 2.9356 |
| | ECE | 0.3211 | 0.2142 | 0.2435 | 0.2346 |
| | Brier | 0.8372 | 0.7470 | 0.7669 | 0.7605 |
| | AUROC | 0.7660 | 0.7758 | 0.7797 | 0.7846 |
| en.clinc150, 1000 | Accuracy | 0.9120 | 0.9130 | 0.9130 | 0.9130 |
| | NLL | 0.5315 | 0.4476 | 0.4685 | 0.4451 |
| | ECE | 0.0665 | 0.0572 | 0.0559 | 0.0557 |
| | Brier | 0.1477 | 0.1459 | 0.1436 | 0.1465 |
| | AUROC | 0.8914 | 0.8973 | 0.8898 | 0.8923 |

Every gap between kime seed 13 and the reference is about the size of the gap between kime's two seeds or smaller. On Banking77 NLL the trainers are 0.05 apart and the seeds 0.16, on ECE 0.020 and 0.029, and on CLINC150 NLL 0.003 and 0.021. Accuracy is the same to 0.002 everywhere, far inside the 95% intervals in the reports. So the two trainers give the same model to within noise.

Scoring the same answers again with `--answers` and `--data-manifest` passes the contamination gate ("no test split listed"), but gives a higher NLL (4.02 on Banking77 for kime seed 13) because the answer files round probabilities, so the NLL here comes from the model runs.

## Held out lines

| | Before | kime seed 13 | Reference seed 13 |
|---|---|---|---|
| Loss | -0.2203 | -0.2596 | -0.2631 |
| Accuracy | 0.9369 | 0.9409 | 0.9409 |
| NLL | 0.2491 | 0.2130 | 0.2098 |

Before training both trainers score the 491 held out questions the same to four places, which checks kime-train's loss and forward pass against Laya's own `DecisionModel` and `proper_reward`.

## Cost

| | Seconds | Tokens a second | Peak memory |
|---|---|---|---|
| kime-train, Burn Metal | 2,175 | about 450 | 12.3 GB |
| Reference, PyTorch MPS | 1,354 | about 700 | 5.7 GB |

kime-train is 1.6 times slower than PyTorch here and needs twice the memory. That is a follow up, not a blocker for #36.

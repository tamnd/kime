# Fine tuning Laya on Banking77 with kime-train

The first real run of kime-train (#36), on an M4 with 24 GB and Burn's Metal backend. It starts from Laya's English checkpoint, trains on the Banking77 shard of tools/data (9,481 training lines, 194 held out) for one epoch, and trains only the top 4 of the 28 encoder layers and the heads.

```
kime-train --base laya --data shards --only banking77 --out ktrain-b77e --tokens 2048 --train-layers 4 --eval-max 500 --eval-every 100 --log train.log.jsonl
kime eval en.banking77_full.jsonl --model <laya or ktrain-b77e> --device metal --precision f32
```

## Held out training lines

| | Loss | Accuracy | NLL |
|---|---|---|---|
| Laya | 0.1218 | 0.8660 | 0.5610 |
| Step 100 | -0.0280 | 0.8763 | 0.4173 |
| Step 200 | -0.0741 | 0.8866 | 0.3754 |
| Step 400 | -0.1152 | 0.8866 | 0.3379 |
| Step 547, the end | -0.1143 | 0.8866 | 0.3391 |

The loss is the proper score of spec/12-training.md, which goes below zero because the spherical score is subtracted.

## The en.banking77_full suite, 500 questions of 77 intents

| | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | AUROC |
|---|---|---|---|---|---|---|---|
| Laya | 0.4600 | 0.4180 to 0.5060 | 0.1543 | 0.3211 | 0.8372 | 3.9999 | 0.7660 |
| Fine tuned | 0.4580 | 0.4140 to 0.5040 | 0.1550 | 0.2215 | 0.7489 | 2.6718 | 0.7910 |

Accuracy does not move, but calibration does: NLL drops by a third and ECE from 0.32 to 0.22. The Laya row is the same to four places on the CPU in FP32, which took 30 minutes on the loaded machine against 103 seconds on Metal.

## Cost

547 steps in 1,961 seconds, about 420 real tokens a second, with a peak memory footprint of 12.3 GB. Training every layer at 4,096 tokens a batch needed about 27 GB, which on this machine meant swapping and 5 minutes a step, which is why `--train-layers` exists. Writing the checkpoint takes 5 seconds.

train.log.jsonl has every step, run.json is the note kime-train writes into rl_agent_config.json, and the two reports are what `kime eval` wrote.

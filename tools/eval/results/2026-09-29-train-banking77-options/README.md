# Banking77 with every option in training

The Banking77 run of tools/eval/results/2026-09-29-train-banking77 learned the held out training lines (accuracy 0.866 to 0.887) but left the en.banking77_full suite where Laya had it, 0.458 against 0.460. The reason is in tools/data: a choice over a large label set gets the gold and a random number of other labels, at most 20 in all, so the model never trained on a 77 way question, and the suite asks nothing else. This run converts Banking77 with `--max-options 255`, which gives every question between 4 and 77 options, and trains again with the same settings: Laya's English checkpoint, the top 4 of 28 layers and the heads, one epoch, 2,048 tokens a batch, seed 13, on an M4 with Metal.

```
python tools/data/convert.py <out> --only banking77 --max-options 255 --kime target/release/kime --tests <test texts> --tests <suites>
kime-train --base laya --data <out>/shards --only banking77 --out b77-all --tokens 2048 --train-layers 4 --eval-max 500 --eval-every 200 --seed 13
kime eval en.banking77_full.jsonl --model b77-all --device metal --precision f32
```

## en.banking77_full, 500 questions, 77 options each

| | Laya | Trained, at most 20 options | Trained, up to 77 options |
|---|---|---|---|
| Accuracy | 0.4600 | 0.4580 | 0.4900 |
| 95% interval | 0.418 to 0.506 | 0.414 to 0.504 | 0.444 to 0.536 |
| Macro F1 | 0.1543 | 0.1550 | 0.1730 |
| NLL | 3.9999 | 2.6718 | 2.5161 |
| ECE | 0.3211 | 0.2215 | 0.1750 |
| Brier | 0.8372 | 0.7489 | 0.7101 |
| Acc at 50% coverage | 0.6640 | 0.6560 | 0.6760 |
| AUROC | 0.7660 | 0.7910 | 0.7563 |

Training on the full label set moves accuracy for the first time, 0.458 to 0.490, and improves NLL, ECE and Brier further. The accuracy gain is inside the interval of 500 questions, so it is a direction, not a result, and it is far from the 0.870 spec/13-benchmarks.md lists as the best baseline. AUROC went down, 0.791 to 0.756.

## Held out lines

The 194 held out lines now have between 4 and 77 options, so they are harder than the held out lines of the 20 option run and the two are not comparable.

| | Loss | Accuracy | NLL |
|---|---|---|---|
| Laya | 1.2721 | 0.6701 | 1.6177 |
| Step 400 | 0.7469 | 0.6856 | 1.1101 |
| Step 800 | 0.6866 | 0.6959 | 1.0544 |
| Step 1074, the end | 0.6708 | 0.6959 | 1.0396 |

The loss keeps falling while accuracy barely moves, so with 4 layers and one epoch the model mostly learns to spread its probability better rather than to rank the right intent first. The run took 5,329 seconds, against 1,961 for the 20 option run, because the sequences are about twice as long.

## What it means for the teachers

The option cap is one of the data problems M2 has to fix before the teachers, and `--max-options` makes it a choice rather than a constant. The default stays 20, so the committed manifest and shards do not change. The bigger gap is capacity: training 4 of 28 layers for one epoch does not get Laya's checkpoint anywhere near 0.87 on 77 way Banking77, and training every layer needs about 27 GB in kime-train today (#181), more than this machine has.

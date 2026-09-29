# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.banking77_full | 500 | 0 | 0.4560 | 0.4120 to 0.5020 | 0.1484 | 0.4912 | 1.0276 | 12.2394 | 0.4920 | 0.6826 |
| en.clinc150 | 1000 | 0 | 0.9130 | 0.8950 to 0.9290 | 0.9122 | 0.0836 | 0.1689 | 1.8263 | 0.9780 | 0.6538 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9652 | 0.0339 | 0.0688 | 0.7219 |
| scope out | 167 | 0.6527 | 0.3317 | 0.6684 | 7.3352 |

No data manifest was given, so the training data of the model was not checked against the suites.

# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.banking77_full | 500 | 0 | 0.4620 | 0.4180 to 0.5060 | 0.1462 | 0.2142 | 0.7470 | 2.8861 | 0.6520 | 0.7758 |
| en.clinc150 | 1000 | 0 | 0.9130 | 0.8950 to 0.9290 | 0.9125 | 0.0572 | 0.1459 | 0.4476 | 0.9940 | 0.8973 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9676 | 0.0218 | 0.0593 | 0.2032 |
| scope out | 167 | 0.6407 | 0.2382 | 0.5779 | 1.6666 |

No data manifest was given, so the training data of the model was not checked against the suites.

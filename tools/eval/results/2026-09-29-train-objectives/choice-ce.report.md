# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.banking77_full | 500 | 0 | 0.4600 | 0.4160 to 0.5060 | 0.1428 | 0.2244 | 0.7516 | 2.8747 | 0.6480 | 0.7820 |
| en.clinc150 | 1000 | 0 | 0.9140 | 0.8960 to 0.9300 | 0.9134 | 0.0583 | 0.1469 | 0.4541 | 0.9940 | 0.8938 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9676 | 0.0250 | 0.0595 | 0.2013 |
| scope out | 167 | 0.6467 | 0.2392 | 0.5831 | 1.7153 |

No data manifest was given, so the training data of the model was not checked against the suites.

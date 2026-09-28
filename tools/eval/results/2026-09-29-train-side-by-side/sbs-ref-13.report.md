# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.banking77_full | 500 | 0 | 0.4600 | 0.4160 to 0.5060 | 0.1423 | 0.2346 | 0.7605 | 2.9356 | 0.6560 | 0.7846 |
| en.clinc150 | 1000 | 0 | 0.9130 | 0.8940 to 0.9290 | 0.9124 | 0.0557 | 0.1465 | 0.4451 | 0.9940 | 0.8923 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9664 | 0.0254 | 0.0596 | 0.1982 |
| scope out | 167 | 0.6467 | 0.2555 | 0.5803 | 1.6765 |

No data manifest was given, so the training data of the model was not checked against the suites.

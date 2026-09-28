# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.banking77_full | 500 | 0 | 0.4580 | 0.4120 to 0.5040 | 0.1445 | 0.2435 | 0.7669 | 3.0464 | 0.6520 | 0.7797 |
| en.clinc150 | 1000 | 0 | 0.9130 | 0.8940 to 0.9290 | 0.9124 | 0.0559 | 0.1436 | 0.4685 | 0.9900 | 0.8898 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9664 | 0.0244 | 0.0599 | 0.2443 |
| scope out | 167 | 0.6467 | 0.2326 | 0.5611 | 1.5871 |

No data manifest was given, so the training data of the model was not checked against the suites.

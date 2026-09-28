# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.banking77_full | 500 | 0 | 0.4600 | 0.4180 to 0.5060 | 0.1543 | 0.3211 | 0.8372 | 3.9999 | 0.6640 | 0.7660 |
| en.clinc150 | 1000 | 0 | 0.9120 | 0.8940 to 0.9280 | 0.9113 | 0.0665 | 0.1477 | 0.5315 | 0.9920 | 0.8914 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9628 | 0.0295 | 0.0635 | 0.2788 |
| scope out | 167 | 0.6587 | 0.2513 | 0.5675 | 1.7920 |

No data manifest was given, so the training data of the model was not checked against the suites.

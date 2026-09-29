# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| en.clerc_rerank | 1200 | 0 | 0.3408 | 0.3133 to 0.3667 | 0.2807 | 0.2691 | 0.6359 | 0.8381 | 0.2383 | 0.3550 |
| en.clerc_rerank_more | 3300 | 0 | 0.3776 | 0.3615 to 0.3952 | 0.2985 | 0.2505 | 0.6111 | 0.8108 | 0.2715 | 0.3577 |

| Suite | Queries | Candidates | Order | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|---|---|---|
| en.clerc_rerank | 40 | 1200 | first stage | 0.0500 | 0.1500 | 0.3750 | 0.1552 |
| en.clerc_rerank | 40 | 1200 | model | 0.0250 | 0.1250 | 0.2500 | 0.1179 |
| en.clerc_rerank_more | 110 | 3300 | first stage | 0.0455 | 0.2091 | 0.2545 | 0.1366 |
| en.clerc_rerank_more | 110 | 3300 | model | 0.0273 | 0.1000 | 0.3000 | 0.1153 |

No data manifest was given, so the training data of the model was not checked against the suites.

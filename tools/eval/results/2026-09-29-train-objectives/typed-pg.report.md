# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| typed_decisions | 2000 | 0 | 0.5030 | 0.4785 to 0.5245 | 0.4766 | 0.0830 | 0.6162 | 1.0873 | 0.5830 | 0.6201 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| typed_decisions | 0.3638 | 0.1900 | 0.5775 | 0.8225 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.4700 | 0.1666 | 0.7044 | 1.3273 |
| type noul | 600 | 0.6483 | 0.0693 | 0.4350 | 0.6224 |
| type score | 800 | 0.4188 | 0.0485 | 0.6859 | 1.2560 |
| workflow agent_trace_observability | 500 | 0.3560 | 0.0476 | 0.7048 | 1.2699 |
| workflow customer_service | 500 | 0.5460 | 0.1461 | 0.6096 | 1.0998 |
| workflow invoice_processing | 500 | 0.5220 | 0.0576 | 0.5578 | 0.9710 |
| workflow security_incidents | 500 | 0.5880 | 0.1623 | 0.5925 | 1.0086 |

No data manifest was given, so the training data of the model was not checked against the suites.

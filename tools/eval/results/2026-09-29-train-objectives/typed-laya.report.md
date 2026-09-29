# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| typed_decisions | 2000 | 0 | 0.3615 | 0.3400 to 0.3835 | 0.3266 | 0.1747 | 0.7497 | 1.3227 | 0.4280 | 0.5818 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| typed_decisions | 0.3315 | 0.3155 | 0.6937 | 0.7525 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.2883 | 0.1681 | 0.8079 | 1.5603 |
| type noul | 600 | 0.4867 | 0.1747 | 0.6068 | 0.8473 |
| type score | 800 | 0.3225 | 0.2065 | 0.8131 | 1.5011 |
| workflow agent_trace_observability | 500 | 0.3880 | 0.1999 | 0.7477 | 1.3386 |
| workflow customer_service | 500 | 0.3820 | 0.1067 | 0.7051 | 1.2950 |
| workflow invoice_processing | 500 | 0.3600 | 0.2281 | 0.7889 | 1.3281 |
| workflow security_incidents | 500 | 0.3160 | 0.2287 | 0.7570 | 1.3292 |

No data manifest was given, so the training data of the model was not checked against the suites.

# kime eval, laya on metal, Apple M4, F32

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| typed_decisions | 2000 | 0 | 0.5330 | 0.5110 to 0.5550 | 0.5083 | 0.1055 | 0.5874 | 1.0240 | 0.6270 | 0.6366 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| typed_decisions | 0.3720 | 0.1704 | 0.5597 | 0.8575 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.5233 | 0.1729 | 0.6513 | 1.2118 |
| type noul | 600 | 0.6400 | 0.0932 | 0.4282 | 0.6151 |
| type score | 800 | 0.4600 | 0.0871 | 0.6588 | 1.1900 |
| workflow agent_trace_observability | 500 | 0.3720 | 0.0405 | 0.6774 | 1.1935 |
| workflow customer_service | 500 | 0.5880 | 0.1678 | 0.5838 | 1.0365 |
| workflow invoice_processing | 500 | 0.5560 | 0.1118 | 0.5287 | 0.9090 |
| workflow security_incidents | 500 | 0.6160 | 0.1729 | 0.5596 | 0.9572 |

No data manifest was given, so the training data of the model was not checked against the suites.

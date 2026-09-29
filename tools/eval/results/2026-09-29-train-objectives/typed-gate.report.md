# kime eval

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| typed_decisions | 2000 | 0 | 0.5410 | 0.5195 to 0.5630 | 0.5147 | 0.1139 | 0.5885 | 1.0258 | 0.6350 | 0.6334 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| typed_decisions | 0.3716 | 0.1711 | 0.5596 | 0.8600 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.5333 | 0.1845 | 0.6515 | 1.2126 |
| type noul | 600 | 0.6500 | 0.0803 | 0.4312 | 0.6184 |
| type score | 800 | 0.4650 | 0.0900 | 0.6591 | 1.1913 |
| workflow agent_trace_observability | 500 | 0.3880 | 0.0396 | 0.6794 | 1.1981 |
| workflow customer_service | 500 | 0.5920 | 0.1731 | 0.5838 | 1.0355 |
| workflow invoice_processing | 500 | 0.5760 | 0.1282 | 0.5290 | 0.9096 |
| workflow security_incidents | 500 | 0.6080 | 0.1641 | 0.5618 | 0.9599 |

Data manifest blake3 55f7e531682dea98f2ca6aeb5ad506fbed80d0a80e14c7c0229741857d160051, no test split listed.

# kime eval, laya on metal, Apple M4, F16

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| order.en.emotion | 3600 | 0 | 0.5708 | 0.5556 to 0.5867 | 0.5694 | 0.3302 | 0.7355 | 2.1824 | 0.6922 | 0.6737 |
| order.massive_intent.en | 1800 | 0 | 0.7750 | 0.7561 to 0.7944 | 0.7733 | 0.1796 | 0.3904 | 1.8874 | 0.9400 | 0.8232 |
| order.xnli.en | 1800 | 0 | 0.8611 | 0.8444 to 0.8750 | 0.8611 | 0.0658 | 0.2179 | 0.3956 | 0.9800 | 0.8388 |
| en.sst5 | 600 | 0 | 0.3717 | 0.3333 to 0.4133 | 0.3288 | 0.2478 | 0.8172 | 1.7291 | 0.4567 | 0.6224 |
| typed_decisions | 2000 | 0 | 0.3620 | 0.3405 to 0.3840 | 0.3274 | 0.1742 | 0.7497 | 1.3227 | 0.4280 | 0.5808 |
| app.phishing | 400 | 0 | 0.9800 | 0.9650 to 0.9925 | 0.9789 | 0.0121 | 0.0292 | 0.0479 | 1.0000 | 0.9807 |
| massive_intent.en | 300 | 0 | 0.7833 | 0.7367 to 0.8300 | 0.7860 | 0.1810 | 0.3772 | 1.8364 | 0.9400 | 0.8285 |
| massive_scenario.en | 300 | 0 | 0.6033 | 0.5467 to 0.6600 | 0.5944 | 0.3628 | 0.7482 | 3.8157 | 0.7933 | 0.7421 |
| xnli.en | 300 | 0 | 0.8600 | 0.8233 to 0.8933 | 0.8612 | 0.0685 | 0.2204 | 0.3987 | 0.9800 | 0.8372 |
| en.ag_news | 600 | 0 | 0.9483 | 0.9300 to 0.9650 | 0.9456 | 0.0410 | 0.0869 | 0.1731 | 1.0000 | 0.9052 |
| app.support_triage | 400 | 0 | 0.5025 | 0.4525 to 0.5475 | 0.4802 | 0.0933 | 0.6438 | 1.4237 | 0.5950 | 0.6740 |
| app.model_routing_domain | 399 | 0 | 0.6391 | 0.5940 to 0.6842 | 0.3199 | 0.0888 | 0.4939 | 1.0890 | 0.8744 | 0.7987 |
| en.banking77_full | 500 | 0 | 0.4620 | 0.4180 to 0.5100 | 0.1548 | 0.3189 | 0.8372 | 3.9997 | 0.6640 | 0.7634 |

| Suite | Questions | Reordered copies | Flip rate | Questions with any flip |
|---|---|---|---|---|
| order.en.emotion | 600 | 3000 | 0.0477 | 0.1100 |
| order.massive_intent.en | 300 | 1500 | 0.1600 | 0.2967 |
| order.xnli.en | 300 | 1500 | 0.0013 | 0.0033 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| en.sst5 |  |  | 0.9045 | 0.6333 |
| typed_decisions | 0.3315 | 0.3155 | 0.6937 | 0.7525 |

## order.massive_intent.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 1800 | 0.7750 | 0.1796 | 0.3904 | 1.8874 |

## order.xnli.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 1800 | 0.8611 | 0.0658 | 0.2179 | 0.3956 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.2900 | 0.1664 | 0.8079 | 1.5603 |
| type noul | 600 | 0.4867 | 0.1751 | 0.6068 | 0.8474 |
| type score | 800 | 0.3225 | 0.2065 | 0.8131 | 1.5011 |
| workflow agent_trace_observability | 500 | 0.3900 | 0.2019 | 0.7477 | 1.3386 |
| workflow customer_service | 500 | 0.3820 | 0.1049 | 0.7052 | 1.2950 |
| workflow invoice_processing | 500 | 0.3600 | 0.2281 | 0.7889 | 1.3281 |
| workflow security_incidents | 500 | 0.3160 | 0.2287 | 0.7570 | 1.3292 |

## massive_intent.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.7833 | 0.1810 | 0.3772 | 1.8364 |

## massive_scenario.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.6033 | 0.3628 | 0.7482 | 3.8157 |

## xnli.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.8600 | 0.0685 | 0.2204 | 0.3987 |

No data manifest was given, so the training data of the model was not checked against the suites.

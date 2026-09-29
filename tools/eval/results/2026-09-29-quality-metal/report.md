# Laya through kime on Metal, quality group

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| app.email_spam | 400 | 0 | 0.9975 | 0.9925 to 1.0000 | 0.9975 | 0.0109 | 0.0052 | 0.0126 | 1.0000 | 1.0000 |
| app.guardrails_jailbreak | 400 | 0 | 0.7075 | 0.6650 to 0.7500 | 0.6840 | 0.2591 | 0.5389 | 1.8574 | 0.8300 | 0.6477 |
| app.model_routing_domain | 399 | 0 | 0.6391 | 0.5940 to 0.6842 | 0.3199 | 0.0888 | 0.4939 | 1.0890 | 0.8744 | 0.7987 |
| app.moderation_toxicity | 400 | 0 | 0.5300 | 0.4800 to 0.5775 | 0.4003 | 0.2961 | 0.6015 | 0.8862 | 0.7350 | 0.7510 |
| app.phishing | 400 | 0 | 0.9800 | 0.9650 to 0.9925 | 0.9789 | 0.0121 | 0.0292 | 0.0479 | 1.0000 | 0.9807 |
| app.rag_relevance | 400 | 0 | 0.6225 | 0.5750 to 0.6700 | 0.6210 | 0.1200 | 0.4657 | 0.6710 | 0.7200 | 0.6574 |
| app.support_triage | 400 | 0 | 0.5025 | 0.4525 to 0.5475 | 0.4802 | 0.0933 | 0.6438 | 1.4237 | 0.5950 | 0.6740 |
| en.ag_news | 600 | 0 | 0.9483 | 0.9300 to 0.9650 | 0.9456 | 0.0410 | 0.0869 | 0.1731 | 1.0000 | 0.9052 |
| en.banking77_full | 500 | 0 | 0.4620 | 0.4180 to 0.5100 | 0.1548 | 0.3189 | 0.8372 | 3.9997 | 0.6640 | 0.7634 |
| en.boolq | 600 | 0 | 0.8300 | 0.7983 to 0.8583 | 0.8196 | 0.0961 | 0.2732 | 0.4658 | 0.9367 | 0.7480 |
| en.clinc150 | 1000 | 0 | 0.9120 | 0.8940 to 0.9280 | 0.9113 | 0.0655 | 0.1477 | 0.5315 | 0.9920 | 0.8914 |
| en.emotion | 600 | 0 | 0.5717 | 0.5333 to 0.6100 | 0.4732 | 0.3254 | 0.7306 | 2.1453 | 0.6867 | 0.6747 |
| en.prompt_injections | 116 | 0 | 0.6983 | 0.6121 to 0.7845 | 0.6751 | 0.2621 | 0.5261 | 1.6818 | 0.9310 | 0.8346 |
| en.sst2 | 872 | 0 | 0.9128 | 0.8945 to 0.9312 | 0.9126 | 0.0168 | 0.1392 | 0.2440 | 0.9839 | 0.8089 |
| en.sst5 | 600 | 0 | 0.3717 | 0.3333 to 0.4133 | 0.3288 | 0.2478 | 0.8172 | 1.7291 | 0.4567 | 0.6224 |
| massive_intent.ar | 300 | 0 | 0.1267 | 0.0867 to 0.1667 | 0.1256 | 0.6039 | 1.3966 | 6.9918 | 0.1733 | 0.6324 |
| massive_intent.de | 300 | 0 | 0.4167 | 0.3633 to 0.4733 | 0.4171 | 0.4593 | 0.9926 | 4.8812 | 0.6200 | 0.7736 |
| massive_intent.en | 300 | 0 | 0.7833 | 0.7367 to 0.8300 | 0.7860 | 0.1810 | 0.3772 | 1.8364 | 0.9400 | 0.8285 |
| massive_intent.es | 300 | 0 | 0.4767 | 0.4233 to 0.5333 | 0.4745 | 0.4181 | 0.9226 | 4.5424 | 0.6400 | 0.7160 |
| massive_intent.fr | 300 | 0 | 0.4867 | 0.4300 to 0.5433 | 0.4903 | 0.4005 | 0.8660 | 3.6282 | 0.6800 | 0.7803 |
| massive_intent.hi | 300 | 0 | 0.1000 | 0.0667 to 0.1367 | 0.0954 | 0.6493 | 1.4656 | 7.3677 | 0.1133 | 0.5227 |
| massive_intent.ja | 300 | 0 | 0.5467 | 0.4900 to 0.6033 | 0.5505 | 0.3555 | 0.8033 | 3.7497 | 0.7533 | 0.7519 |
| massive_intent.ko | 300 | 0 | 0.1033 | 0.0700 to 0.1400 | 0.0957 | 0.6629 | 1.4765 | 7.4428 | 0.1200 | 0.6185 |
| massive_intent.pt | 300 | 0 | 0.4567 | 0.4067 to 0.5133 | 0.4503 | 0.4503 | 0.9488 | 5.0778 | 0.6533 | 0.7573 |
| massive_intent.ru | 300 | 0 | 0.2933 | 0.2433 to 0.3400 | 0.2931 | 0.5856 | 1.2348 | 6.5569 | 0.4133 | 0.7108 |
| massive_intent.sw | 300 | 0 | 0.1033 | 0.0733 to 0.1367 | 0.0957 | 0.6296 | 1.4364 | 6.9002 | 0.1267 | 0.6142 |
| massive_intent.ta | 300 | 0 | 0.1100 | 0.0800 to 0.1467 | 0.1009 | 0.5582 | 1.3528 | 5.9480 | 0.1467 | 0.5813 |
| massive_intent.tr | 300 | 0 | 0.1600 | 0.1200 to 0.2000 | 0.1485 | 0.6047 | 1.3688 | 7.1391 | 0.2333 | 0.7412 |
| massive_intent.zh-CN | 300 | 0 | 0.5933 | 0.5400 to 0.6467 | 0.5972 | 0.3357 | 0.7365 | 3.7481 | 0.7600 | 0.7523 |
| massive_scenario.ar | 300 | 0 | 0.1400 | 0.1000 to 0.1800 | 0.1395 | 0.6370 | 1.4493 | 7.1094 | 0.1133 | 0.4240 |
| massive_scenario.de | 300 | 0 | 0.2967 | 0.2467 to 0.3533 | 0.2877 | 0.5601 | 1.2144 | 5.7213 | 0.4133 | 0.7197 |
| massive_scenario.en | 300 | 0 | 0.6033 | 0.5467 to 0.6600 | 0.5944 | 0.3628 | 0.7482 | 3.8157 | 0.7933 | 0.7421 |
| massive_scenario.es | 300 | 0 | 0.3867 | 0.3333 to 0.4400 | 0.3836 | 0.5131 | 1.1040 | 5.5759 | 0.5067 | 0.6739 |
| massive_scenario.fr | 300 | 0 | 0.4600 | 0.4000 to 0.5167 | 0.4519 | 0.4448 | 0.9454 | 4.0679 | 0.5800 | 0.7108 |
| massive_scenario.hi | 300 | 0 | 0.0967 | 0.0633 to 0.1333 | 0.0922 | 0.6845 | 1.5049 | 7.6734 | 0.1067 | 0.5422 |
| massive_scenario.ja | 300 | 0 | 0.4667 | 0.4100 to 0.5200 | 0.4639 | 0.4289 | 0.9329 | 4.1323 | 0.6467 | 0.7666 |
| massive_scenario.ko | 300 | 0 | 0.1267 | 0.0900 to 0.1667 | 0.1211 | 0.6885 | 1.5033 | 7.8318 | 0.1267 | 0.5375 |
| massive_scenario.pt | 300 | 0 | 0.4200 | 0.3600 to 0.4767 | 0.4151 | 0.4834 | 1.0406 | 5.4988 | 0.5800 | 0.7134 |
| massive_scenario.ru | 300 | 0 | 0.3367 | 0.2833 to 0.3900 | 0.3341 | 0.5374 | 1.1675 | 6.1558 | 0.4067 | 0.6326 |
| massive_scenario.sw | 300 | 0 | 0.1000 | 0.0667 to 0.1367 | 0.0955 | 0.6775 | 1.4699 | 7.6408 | 0.1000 | 0.5099 |
| massive_scenario.ta | 300 | 0 | 0.1033 | 0.0700 to 0.1400 | 0.1012 | 0.7699 | 1.6051 | 8.1601 | 0.0667 | 0.4226 |
| massive_scenario.tr | 300 | 0 | 0.2267 | 0.1800 to 0.2800 | 0.2297 | 0.5625 | 1.2780 | 6.6417 | 0.3267 | 0.7245 |
| massive_scenario.zh-CN | 300 | 0 | 0.4867 | 0.4333 to 0.5433 | 0.4818 | 0.4245 | 0.9187 | 4.3661 | 0.6200 | 0.7087 |
| typed_decisions | 2000 | 0 | 0.3620 | 0.3405 to 0.3840 | 0.3274 | 0.1742 | 0.7497 | 1.3227 | 0.4280 | 0.5808 |
| xnli.ar | 300 | 0 | 0.4433 | 0.3867 to 0.5033 | 0.4171 | 0.2880 | 0.7813 | 1.3378 | 0.4800 | 0.5550 |
| xnli.bg | 300 | 0 | 0.5433 | 0.4867 to 0.6000 | 0.5329 | 0.2334 | 0.6456 | 1.1370 | 0.6867 | 0.6931 |
| xnli.de | 300 | 0 | 0.6400 | 0.5833 to 0.6933 | 0.6326 | 0.2109 | 0.5390 | 0.9683 | 0.8000 | 0.7335 |
| xnli.el | 300 | 0 | 0.5133 | 0.4567 to 0.5733 | 0.5044 | 0.2225 | 0.6728 | 1.1809 | 0.6200 | 0.6498 |
| xnli.en | 300 | 0 | 0.8600 | 0.8233 to 0.8933 | 0.8612 | 0.0685 | 0.2204 | 0.3987 | 0.9800 | 0.8372 |
| xnli.es | 300 | 0 | 0.6867 | 0.6333 to 0.7400 | 0.6854 | 0.1733 | 0.4741 | 0.8471 | 0.8667 | 0.7688 |
| xnli.fr | 300 | 0 | 0.6800 | 0.6300 to 0.7333 | 0.6767 | 0.1878 | 0.4913 | 0.8926 | 0.8267 | 0.7358 |
| xnli.hi | 300 | 0 | 0.4000 | 0.3467 to 0.4600 | 0.3925 | 0.2873 | 0.8294 | 1.3949 | 0.3667 | 0.4672 |
| xnli.ru | 300 | 0 | 0.6100 | 0.5567 to 0.6633 | 0.6006 | 0.2161 | 0.6065 | 1.1247 | 0.7067 | 0.6535 |
| xnli.sw | 300 | 0 | 0.4233 | 0.3667 to 0.4800 | 0.4209 | 0.2924 | 0.7785 | 1.3181 | 0.4667 | 0.5727 |
| xnli.th | 300 | 0 | 0.4133 | 0.3600 to 0.4700 | 0.4127 | 0.1888 | 0.7152 | 1.1854 | 0.4667 | 0.5846 |
| xnli.tr | 300 | 0 | 0.3833 | 0.3300 to 0.4400 | 0.3659 | 0.3563 | 0.8457 | 1.4683 | 0.4667 | 0.6090 |
| xnli.ur | 300 | 0 | 0.3733 | 0.3200 to 0.4267 | 0.3357 | 0.3221 | 0.8426 | 1.4337 | 0.4000 | 0.5553 |
| xnli.vi | 300 | 0 | 0.5100 | 0.4533 to 0.5633 | 0.5048 | 0.1963 | 0.6320 | 1.0815 | 0.6267 | 0.6763 |
| xnli.zh | 300 | 0 | 0.6767 | 0.6267 to 0.7233 | 0.6656 | 0.2122 | 0.5226 | 1.0025 | 0.8000 | 0.6733 |
| en.clerc_rerank | 1200 | 0 | 0.3408 | 0.3133 to 0.3667 | 0.2807 | 0.2691 | 0.6359 | 0.8381 | 0.2383 | 0.3551 |
| en.clerc_rerank_more | 3300 | 0 | 0.3785 | 0.3624 to 0.3961 | 0.2990 | 0.2511 | 0.6111 | 0.8108 | 0.2715 | 0.3555 |

| Suite | Queries | Candidates | Order | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|---|---|---|
| en.clerc_rerank | 40 | 1200 | first stage | 0.0500 | 0.1500 | 0.3750 | 0.1552 |
| en.clerc_rerank | 40 | 1200 | model | 0.0250 | 0.1250 | 0.2500 | 0.1174 |
| en.clerc_rerank_more | 110 | 3300 | first stage | 0.0455 | 0.2091 | 0.2545 | 0.1366 |
| en.clerc_rerank_more | 110 | 3300 | model | 0.0273 | 0.1000 | 0.3000 | 0.1151 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| en.sst5 |  |  | 0.9045 | 0.6333 |
| typed_decisions | 0.3315 | 0.3155 | 0.6937 | 0.7525 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9628 | 0.0295 | 0.0635 | 0.2787 |
| scope out | 167 | 0.6587 | 0.2448 | 0.5676 | 1.7924 |

## massive_intent.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.1267 | 0.6039 | 1.3966 | 6.9918 |

## massive_intent.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.4167 | 0.4593 | 0.9926 | 4.8812 |

## massive_intent.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.7833 | 0.1810 | 0.3772 | 1.8364 |

## massive_intent.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.4767 | 0.4181 | 0.9226 | 4.5424 |

## massive_intent.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.4867 | 0.4005 | 0.8660 | 3.6282 |

## massive_intent.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.1000 | 0.6493 | 1.4656 | 7.3677 |

## massive_intent.ja

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ja | 300 | 0.5467 | 0.3555 | 0.8033 | 3.7497 |

## massive_intent.ko

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ko | 300 | 0.1033 | 0.6629 | 1.4765 | 7.4428 |

## massive_intent.pt

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang pt | 300 | 0.4567 | 0.4503 | 0.9488 | 5.0778 |

## massive_intent.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.2933 | 0.5856 | 1.2348 | 6.5569 |

## massive_intent.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.1033 | 0.6296 | 1.4364 | 6.9002 |

## massive_intent.ta

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ta | 300 | 0.1100 | 0.5582 | 1.3528 | 5.9480 |

## massive_intent.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.1600 | 0.6047 | 1.3688 | 7.1391 |

## massive_intent.zh-CN

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh-CN | 300 | 0.5933 | 0.3357 | 0.7365 | 3.7481 |

## massive_scenario.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.1400 | 0.6370 | 1.4493 | 7.1094 |

## massive_scenario.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.2967 | 0.5601 | 1.2144 | 5.7213 |

## massive_scenario.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.6033 | 0.3628 | 0.7482 | 3.8157 |

## massive_scenario.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.3867 | 0.5131 | 1.1040 | 5.5759 |

## massive_scenario.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.4600 | 0.4448 | 0.9454 | 4.0679 |

## massive_scenario.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.0967 | 0.6845 | 1.5049 | 7.6734 |

## massive_scenario.ja

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ja | 300 | 0.4667 | 0.4289 | 0.9329 | 4.1323 |

## massive_scenario.ko

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ko | 300 | 0.1267 | 0.6885 | 1.5033 | 7.8318 |

## massive_scenario.pt

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang pt | 300 | 0.4200 | 0.4834 | 1.0406 | 5.4988 |

## massive_scenario.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.3367 | 0.5374 | 1.1675 | 6.1558 |

## massive_scenario.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.1000 | 0.6775 | 1.4699 | 7.6408 |

## massive_scenario.ta

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ta | 300 | 0.1033 | 0.7699 | 1.6051 | 8.1601 |

## massive_scenario.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.2267 | 0.5625 | 1.2780 | 6.6417 |

## massive_scenario.zh-CN

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh-CN | 300 | 0.4867 | 0.4245 | 0.9187 | 4.3661 |

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

## xnli.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.4433 | 0.2880 | 0.7813 | 1.3378 |

## xnli.bg

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang bg | 300 | 0.5433 | 0.2334 | 0.6456 | 1.1370 |

## xnli.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.6400 | 0.2109 | 0.5390 | 0.9683 |

## xnli.el

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang el | 300 | 0.5133 | 0.2225 | 0.6728 | 1.1809 |

## xnli.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.8600 | 0.0685 | 0.2204 | 0.3987 |

## xnli.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.6867 | 0.1733 | 0.4741 | 0.8471 |

## xnli.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.6800 | 0.1878 | 0.4913 | 0.8926 |

## xnli.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.4000 | 0.2873 | 0.8294 | 1.3949 |

## xnli.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.6100 | 0.2161 | 0.6065 | 1.1247 |

## xnli.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.4233 | 0.2924 | 0.7785 | 1.3181 |

## xnli.th

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang th | 300 | 0.4133 | 0.1888 | 0.7152 | 1.1854 |

## xnli.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.3833 | 0.3563 | 0.8457 | 1.4683 |

## xnli.ur

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ur | 300 | 0.3733 | 0.3221 | 0.8426 | 1.4337 |

## xnli.vi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang vi | 300 | 0.5100 | 0.1963 | 0.6320 | 1.0815 |

## xnli.zh

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh | 300 | 0.6767 | 0.2122 | 0.5226 | 1.0025 |

No data manifest was given, so the training data of the model was not checked against the suites.

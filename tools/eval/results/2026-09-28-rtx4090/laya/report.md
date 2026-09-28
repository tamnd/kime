# Laya 0.3.20, PyTorch fp16 autocast, RTX 4090

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| app.email_spam | 400 | 0 | 0.9975 | 0.9925 to 1.0000 | 0.9975 | 0.0108 | 0.0052 | 0.0125 | 1.0000 | 1.0000 |
| app.guardrails_jailbreak | 400 | 0 | 0.7125 | 0.6700 to 0.7550 | 0.6885 | 0.2648 | 0.5377 | 1.8484 | 0.8350 | 0.6440 |
| app.model_routing_domain | 399 | 0 | 0.6391 | 0.5940 to 0.6842 | 0.3198 | 0.0789 | 0.4926 | 1.0871 | 0.8744 | 0.8001 |
| app.moderation_toxicity | 400 | 0 | 0.5300 | 0.4800 to 0.5775 | 0.4003 | 0.2986 | 0.6017 | 0.8869 | 0.7350 | 0.7505 |
| app.phishing | 400 | 0 | 0.9800 | 0.9650 to 0.9925 | 0.9789 | 0.0169 | 0.0293 | 0.0481 | 1.0000 | 0.9774 |
| app.rag_relevance | 400 | 0 | 0.6250 | 0.5775 to 0.6700 | 0.6236 | 0.1172 | 0.4652 | 0.6700 | 0.7200 | 0.6532 |
| app.support_triage | 400 | 0 | 0.5025 | 0.4525 to 0.5475 | 0.4802 | 0.0901 | 0.6435 | 1.4230 | 0.5950 | 0.6741 |
| en.ag_news | 600 | 0 | 0.9500 | 0.9333 to 0.9667 | 0.9474 | 0.0347 | 0.0866 | 0.1726 | 1.0000 | 0.9014 |
| en.banking77_full | 500 | 0 | 0.4620 | 0.4180 to 0.5080 | 0.1548 | 0.3168 | 0.8363 | 4.0141 | 0.6600 | 0.7673 |
| en.boolq | 600 | 0 | 0.8300 | 0.7983 to 0.8583 | 0.8196 | 0.0955 | 0.2732 | 0.4652 | 0.9367 | 0.7484 |
| en.clinc150 | 1000 | 0 | 0.9130 | 0.8940 to 0.9290 | 0.9123 | 0.0645 | 0.1471 | 0.5312 | 0.9900 | 0.8969 |
| en.emotion | 600 | 0 | 0.5750 | 0.5367 to 0.6133 | 0.4775 | 0.3227 | 0.7320 | 2.1476 | 0.6800 | 0.6673 |
| en.prompt_injections | 116 | 0 | 0.7069 | 0.6207 to 0.7931 | 0.6859 | 0.2728 | 0.5263 | 1.6927 | 0.9310 | 0.8187 |
| en.sst2 | 872 | 0 | 0.9140 | 0.8956 to 0.9312 | 0.9138 | 0.0176 | 0.1391 | 0.2439 | 0.9839 | 0.8054 |
| en.sst5 | 600 | 0 | 0.3683 | 0.3300 to 0.4083 | 0.3266 | 0.2510 | 0.8176 | 1.7307 | 0.4533 | 0.6272 |
| massive_intent.ar | 300 | 0 | 0.1300 | 0.0933 to 0.1700 | 0.1284 | 0.6042 | 1.3905 | 6.9716 | 0.1733 | 0.6266 |
| massive_intent.de | 300 | 0 | 0.4167 | 0.3633 to 0.4733 | 0.4162 | 0.4632 | 0.9952 | 4.8886 | 0.6267 | 0.7758 |
| massive_intent.en | 300 | 0 | 0.7800 | 0.7300 to 0.8267 | 0.7831 | 0.1807 | 0.3784 | 1.8414 | 0.9400 | 0.8317 |
| massive_intent.es | 300 | 0 | 0.4833 | 0.4267 to 0.5400 | 0.4818 | 0.4149 | 0.9191 | 4.5402 | 0.6400 | 0.7018 |
| massive_intent.fr | 300 | 0 | 0.4900 | 0.4333 to 0.5433 | 0.4944 | 0.3989 | 0.8702 | 3.6352 | 0.6867 | 0.7752 |
| massive_intent.hi | 300 | 0 | 0.0967 | 0.0633 to 0.1333 | 0.0924 | 0.6486 | 1.4647 | 7.3588 | 0.1067 | 0.5326 |
| massive_intent.ja | 300 | 0 | 0.5467 | 0.4900 to 0.6033 | 0.5505 | 0.3556 | 0.8026 | 3.7613 | 0.7533 | 0.7502 |
| massive_intent.ko | 300 | 0 | 0.1033 | 0.0700 to 0.1400 | 0.0957 | 0.6592 | 1.4778 | 7.4524 | 0.1133 | 0.5968 |
| massive_intent.pt | 300 | 0 | 0.4567 | 0.4067 to 0.5133 | 0.4501 | 0.4482 | 0.9464 | 5.0876 | 0.6467 | 0.7543 |
| massive_intent.ru | 300 | 0 | 0.2933 | 0.2433 to 0.3433 | 0.2932 | 0.5842 | 1.2369 | 6.5679 | 0.3933 | 0.7017 |
| massive_intent.sw | 300 | 0 | 0.1000 | 0.0700 to 0.1333 | 0.0934 | 0.6333 | 1.4380 | 6.9063 | 0.1267 | 0.6374 |
| massive_intent.ta | 300 | 0 | 0.1133 | 0.0800 to 0.1500 | 0.1054 | 0.5580 | 1.3546 | 5.9587 | 0.1333 | 0.5659 |
| massive_intent.tr | 300 | 0 | 0.1667 | 0.1233 to 0.2100 | 0.1542 | 0.5998 | 1.3693 | 7.1588 | 0.2400 | 0.7155 |
| massive_intent.zh-CN | 300 | 0 | 0.5900 | 0.5333 to 0.6433 | 0.5949 | 0.3407 | 0.7364 | 3.7388 | 0.7600 | 0.7590 |
| massive_scenario.ar | 300 | 0 | 0.1400 | 0.1000 to 0.1800 | 0.1399 | 0.6407 | 1.4484 | 7.0933 | 0.1133 | 0.4315 |
| massive_scenario.de | 300 | 0 | 0.2900 | 0.2367 to 0.3467 | 0.2825 | 0.5699 | 1.2156 | 5.7073 | 0.4200 | 0.7364 |
| massive_scenario.en | 300 | 0 | 0.6067 | 0.5500 to 0.6633 | 0.5981 | 0.3658 | 0.7459 | 3.8127 | 0.7933 | 0.7447 |
| massive_scenario.es | 300 | 0 | 0.3700 | 0.3167 to 0.4200 | 0.3662 | 0.5294 | 1.1060 | 5.5704 | 0.5067 | 0.7092 |
| massive_scenario.fr | 300 | 0 | 0.4600 | 0.4000 to 0.5167 | 0.4523 | 0.4397 | 0.9455 | 4.0670 | 0.5867 | 0.7114 |
| massive_scenario.hi | 300 | 0 | 0.0967 | 0.0633 to 0.1333 | 0.0925 | 0.6845 | 1.5015 | 7.6537 | 0.1000 | 0.5533 |
| massive_scenario.ja | 300 | 0 | 0.4700 | 0.4167 to 0.5233 | 0.4686 | 0.4257 | 0.9301 | 4.1114 | 0.6400 | 0.7619 |
| massive_scenario.ko | 300 | 0 | 0.1267 | 0.0933 to 0.1667 | 0.1220 | 0.6907 | 1.5074 | 7.8302 | 0.1333 | 0.5389 |
| massive_scenario.pt | 300 | 0 | 0.4200 | 0.3600 to 0.4767 | 0.4152 | 0.4825 | 1.0383 | 5.4846 | 0.5800 | 0.7177 |
| massive_scenario.ru | 300 | 0 | 0.3433 | 0.2900 to 0.3967 | 0.3408 | 0.5352 | 1.1657 | 6.1530 | 0.4067 | 0.6160 |
| massive_scenario.sw | 300 | 0 | 0.0967 | 0.0633 to 0.1333 | 0.0925 | 0.6821 | 1.4677 | 7.6216 | 0.1000 | 0.5415 |
| massive_scenario.ta | 300 | 0 | 0.1033 | 0.0700 to 0.1400 | 0.1010 | 0.7675 | 1.6011 | 8.1286 | 0.0667 | 0.4258 |
| massive_scenario.tr | 300 | 0 | 0.2233 | 0.1767 to 0.2733 | 0.2252 | 0.5668 | 1.2784 | 6.6262 | 0.3267 | 0.7364 |
| massive_scenario.zh-CN | 300 | 0 | 0.4833 | 0.4300 to 0.5400 | 0.4769 | 0.4283 | 0.9192 | 4.3517 | 0.6200 | 0.7168 |
| order.en.emotion | 3600 | 0 | 0.5708 | 0.5556 to 0.5869 | 0.5694 | 0.3300 | 0.7356 | 2.1821 | 0.6933 | 0.6738 |
| order.massive_intent.en | 1800 | 0 | 0.7744 | 0.7550 to 0.7939 | 0.7727 | 0.1797 | 0.3911 | 1.8892 | 0.9400 | 0.8253 |
| order.xnli.en | 1800 | 0 | 0.8611 | 0.8444 to 0.8750 | 0.8611 | 0.0665 | 0.2182 | 0.3960 | 0.9800 | 0.8375 |
| typed_decisions | 2000 | 0 | 0.3595 | 0.3380 to 0.3820 | 0.3207 | 0.1769 | 0.7498 | 1.3232 | 0.4290 | 0.5841 |
| xnli.ar | 300 | 0 | 0.4400 | 0.3800 to 0.5000 | 0.4134 | 0.2769 | 0.7819 | 1.3390 | 0.4800 | 0.5583 |
| xnli.bg | 300 | 0 | 0.5467 | 0.4867 to 0.6033 | 0.5370 | 0.2307 | 0.6433 | 1.1365 | 0.6867 | 0.6873 |
| xnli.de | 300 | 0 | 0.6333 | 0.5800 to 0.6867 | 0.6256 | 0.2090 | 0.5409 | 0.9709 | 0.8000 | 0.7425 |
| xnli.el | 300 | 0 | 0.5167 | 0.4600 to 0.5767 | 0.5086 | 0.2155 | 0.6751 | 1.1838 | 0.6267 | 0.6396 |
| xnli.en | 300 | 0 | 0.8600 | 0.8233 to 0.8933 | 0.8612 | 0.0699 | 0.2207 | 0.3992 | 0.9800 | 0.8373 |
| xnli.es | 300 | 0 | 0.6833 | 0.6300 to 0.7367 | 0.6819 | 0.1837 | 0.4733 | 0.8441 | 0.8667 | 0.7769 |
| xnli.fr | 300 | 0 | 0.6800 | 0.6267 to 0.7333 | 0.6758 | 0.1908 | 0.4917 | 0.8930 | 0.8200 | 0.7352 |
| xnli.hi | 300 | 0 | 0.3867 | 0.3333 to 0.4433 | 0.3780 | 0.2942 | 0.8294 | 1.3954 | 0.3733 | 0.4909 |
| xnli.ru | 300 | 0 | 0.6133 | 0.5600 to 0.6667 | 0.6060 | 0.2273 | 0.6077 | 1.1245 | 0.7067 | 0.6469 |
| xnli.sw | 300 | 0 | 0.4233 | 0.3667 to 0.4800 | 0.4199 | 0.2830 | 0.7788 | 1.3189 | 0.4733 | 0.5747 |
| xnli.th | 300 | 0 | 0.4233 | 0.3700 to 0.4767 | 0.4225 | 0.1782 | 0.7147 | 1.1848 | 0.4733 | 0.5666 |
| xnli.tr | 300 | 0 | 0.3900 | 0.3333 to 0.4467 | 0.3710 | 0.3528 | 0.8457 | 1.4682 | 0.4667 | 0.5933 |
| xnli.ur | 300 | 0 | 0.3700 | 0.3200 to 0.4233 | 0.3333 | 0.3237 | 0.8432 | 1.4335 | 0.4000 | 0.5547 |
| xnli.vi | 300 | 0 | 0.5100 | 0.4533 to 0.5633 | 0.5062 | 0.1944 | 0.6318 | 1.0806 | 0.6333 | 0.6708 |
| xnli.zh | 300 | 0 | 0.6700 | 0.6200 to 0.7167 | 0.6585 | 0.2073 | 0.5213 | 0.9992 | 0.8000 | 0.6886 |

| Suite | Questions | Reordered copies | Flip rate | Questions with any flip |
|---|---|---|---|---|
| order.en.emotion | 600 | 3000 | 0.0480 | 0.1133 |
| order.massive_intent.en | 300 | 1500 | 0.1573 | 0.3033 |
| order.xnli.en | 300 | 1500 | 0.0013 | 0.0033 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| en.sst5 |  |  | 0.9052 | 0.6300 |
| typed_decisions | 0.3315 | 0.3156 | 0.6943 | 0.7525 |

## en.clinc150

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| scope in | 833 | 0.9640 | 0.0272 | 0.0632 | 0.2792 |
| scope out | 167 | 0.6587 | 0.2503 | 0.5656 | 1.7882 |

## massive_intent.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.1300 | 0.6042 | 1.3905 | 6.9716 |

## massive_intent.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.4167 | 0.4632 | 0.9952 | 4.8886 |

## massive_intent.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.7800 | 0.1807 | 0.3784 | 1.8414 |

## massive_intent.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.4833 | 0.4149 | 0.9191 | 4.5402 |

## massive_intent.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.4900 | 0.3989 | 0.8702 | 3.6352 |

## massive_intent.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.0967 | 0.6486 | 1.4647 | 7.3588 |

## massive_intent.ja

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ja | 300 | 0.5467 | 0.3556 | 0.8026 | 3.7613 |

## massive_intent.ko

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ko | 300 | 0.1033 | 0.6592 | 1.4778 | 7.4524 |

## massive_intent.pt

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang pt | 300 | 0.4567 | 0.4482 | 0.9464 | 5.0876 |

## massive_intent.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.2933 | 0.5842 | 1.2369 | 6.5679 |

## massive_intent.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.1000 | 0.6333 | 1.4380 | 6.9063 |

## massive_intent.ta

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ta | 300 | 0.1133 | 0.5580 | 1.3546 | 5.9587 |

## massive_intent.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.1667 | 0.5998 | 1.3693 | 7.1588 |

## massive_intent.zh-CN

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh-CN | 300 | 0.5900 | 0.3407 | 0.7364 | 3.7388 |

## massive_scenario.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.1400 | 0.6407 | 1.4484 | 7.0933 |

## massive_scenario.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.2900 | 0.5699 | 1.2156 | 5.7073 |

## massive_scenario.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.6067 | 0.3658 | 0.7459 | 3.8127 |

## massive_scenario.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.3700 | 0.5294 | 1.1060 | 5.5704 |

## massive_scenario.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.4600 | 0.4397 | 0.9455 | 4.0670 |

## massive_scenario.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.0967 | 0.6845 | 1.5015 | 7.6537 |

## massive_scenario.ja

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ja | 300 | 0.4700 | 0.4257 | 0.9301 | 4.1114 |

## massive_scenario.ko

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ko | 300 | 0.1267 | 0.6907 | 1.5074 | 7.8302 |

## massive_scenario.pt

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang pt | 300 | 0.4200 | 0.4825 | 1.0383 | 5.4846 |

## massive_scenario.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.3433 | 0.5352 | 1.1657 | 6.1530 |

## massive_scenario.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.0967 | 0.6821 | 1.4677 | 7.6216 |

## massive_scenario.ta

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ta | 300 | 0.1033 | 0.7675 | 1.6011 | 8.1286 |

## massive_scenario.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.2233 | 0.5668 | 1.2784 | 6.6262 |

## massive_scenario.zh-CN

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh-CN | 300 | 0.4833 | 0.4283 | 0.9192 | 4.3517 |

## order.massive_intent.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 1800 | 0.7744 | 0.1797 | 0.3911 | 1.8892 |

## order.xnli.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 1800 | 0.8611 | 0.0665 | 0.2182 | 0.3960 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.2850 | 0.1718 | 0.8085 | 1.5610 |
| type noul | 600 | 0.4833 | 0.1730 | 0.6066 | 0.8479 |
| type score | 800 | 0.3225 | 0.2054 | 0.8131 | 1.5014 |
| workflow agent_trace_observability | 500 | 0.3880 | 0.1943 | 0.7474 | 1.3377 |
| workflow customer_service | 500 | 0.3780 | 0.1122 | 0.7059 | 1.2961 |
| workflow invoice_processing | 500 | 0.3600 | 0.2282 | 0.7886 | 1.3288 |
| workflow security_incidents | 500 | 0.3120 | 0.2326 | 0.7573 | 1.3303 |

## xnli.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.4400 | 0.2769 | 0.7819 | 1.3390 |

## xnli.bg

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang bg | 300 | 0.5467 | 0.2307 | 0.6433 | 1.1365 |

## xnli.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.6333 | 0.2090 | 0.5409 | 0.9709 |

## xnli.el

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang el | 300 | 0.5167 | 0.2155 | 0.6751 | 1.1838 |

## xnli.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.8600 | 0.0699 | 0.2207 | 0.3992 |

## xnli.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.6833 | 0.1837 | 0.4733 | 0.8441 |

## xnli.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.6800 | 0.1908 | 0.4917 | 0.8930 |

## xnli.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.3867 | 0.2942 | 0.8294 | 1.3954 |

## xnli.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.6133 | 0.2273 | 0.6077 | 1.1245 |

## xnli.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.4233 | 0.2830 | 0.7788 | 1.3189 |

## xnli.th

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang th | 300 | 0.4233 | 0.1782 | 0.7147 | 1.1848 |

## xnli.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.3900 | 0.3528 | 0.8457 | 1.4682 |

## xnli.ur

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ur | 300 | 0.3700 | 0.3237 | 0.8432 | 1.4335 |

## xnli.vi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang vi | 300 | 0.5100 | 0.1944 | 0.6318 | 1.0806 |

## xnli.zh

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh | 300 | 0.6700 | 0.2073 | 0.5213 | 0.9992 |

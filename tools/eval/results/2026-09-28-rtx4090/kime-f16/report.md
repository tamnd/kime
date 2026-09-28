# kime eval, laya on cuda, NVIDIA GeForce RTX 4090, F16

| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage |
|---|---|---|---|---|---|---|---|---|---|
| en.ag_news | 600 | 0 | 0.9467 | 0.9283 to 0.9633 | 0.9438 | 0.0393 | 0.0869 | 0.1731 | 1.0000 |
| en.banking77_full | 500 | 0 | 0.4600 | 0.4180 to 0.5060 | 0.1543 | 0.3210 | 0.8374 | 4.0007 | 0.6640 |
| en.boolq | 600 | 0 | 0.8300 | 0.7983 to 0.8583 | 0.8196 | 0.0961 | 0.2732 | 0.4658 | 0.9367 |
| en.emotion | 600 | 0 | 0.5733 | 0.5350 to 0.6117 | 0.4763 | 0.3237 | 0.7306 | 2.1453 | 0.6833 |
| en.prompt_injections | 116 | 0 | 0.6983 | 0.6121 to 0.7845 | 0.6751 | 0.2621 | 0.5259 | 1.6817 | 0.9310 |
| en.sst5 | 600 | 0 | 0.3717 | 0.3333 to 0.4133 | 0.3288 | 0.2478 | 0.8172 | 1.7292 | 0.4567 |
| massive_intent.ar | 300 | 0 | 0.1267 | 0.0867 to 0.1667 | 0.1256 | 0.6012 | 1.3971 | 6.9925 | 0.1733 |
| massive_intent.de | 300 | 0 | 0.4200 | 0.3667 to 0.4733 | 0.4194 | 0.4559 | 0.9924 | 4.8814 | 0.6200 |
| massive_intent.en | 300 | 0 | 0.7833 | 0.7367 to 0.8300 | 0.7860 | 0.1810 | 0.3770 | 1.8363 | 0.9400 |
| massive_intent.es | 300 | 0 | 0.4767 | 0.4233 to 0.5333 | 0.4745 | 0.4180 | 0.9225 | 4.5426 | 0.6400 |
| massive_intent.fr | 300 | 0 | 0.4867 | 0.4300 to 0.5433 | 0.4903 | 0.4005 | 0.8661 | 3.6281 | 0.6800 |
| massive_intent.hi | 300 | 0 | 0.1000 | 0.0667 to 0.1367 | 0.0954 | 0.6480 | 1.4654 | 7.3669 | 0.1133 |
| massive_intent.ja | 300 | 0 | 0.5467 | 0.4900 to 0.6033 | 0.5505 | 0.3556 | 0.8034 | 3.7504 | 0.7533 |
| massive_intent.ko | 300 | 0 | 0.1033 | 0.0700 to 0.1400 | 0.0957 | 0.6628 | 1.4765 | 7.4425 | 0.1200 |
| massive_intent.pt | 300 | 0 | 0.4567 | 0.4067 to 0.5133 | 0.4503 | 0.4503 | 0.9487 | 5.0787 | 0.6533 |
| massive_intent.ru | 300 | 0 | 0.2933 | 0.2433 to 0.3400 | 0.2931 | 0.5854 | 1.2343 | 6.5555 | 0.4133 |
| massive_intent.sw | 300 | 0 | 0.1033 | 0.0733 to 0.1367 | 0.0957 | 0.6296 | 1.4365 | 6.9003 | 0.1267 |
| massive_intent.ta | 300 | 0 | 0.1133 | 0.0800 to 0.1500 | 0.1052 | 0.5603 | 1.3529 | 5.9484 | 0.1400 |
| massive_intent.tr | 300 | 0 | 0.1600 | 0.1200 to 0.2000 | 0.1487 | 0.6047 | 1.3688 | 7.1393 | 0.2333 |
| massive_intent.zh-CN | 300 | 0 | 0.5933 | 0.5400 to 0.6467 | 0.5966 | 0.3356 | 0.7364 | 3.7488 | 0.7600 |
| massive_scenario.ar | 300 | 0 | 0.1400 | 0.1000 to 0.1800 | 0.1395 | 0.6372 | 1.4496 | 7.1111 | 0.1133 |
| massive_scenario.de | 300 | 0 | 0.2933 | 0.2433 to 0.3500 | 0.2836 | 0.5632 | 1.2141 | 5.7217 | 0.4133 |
| massive_scenario.en | 300 | 0 | 0.6033 | 0.5467 to 0.6600 | 0.5944 | 0.3628 | 0.7481 | 3.8150 | 0.7933 |
| massive_scenario.es | 300 | 0 | 0.3867 | 0.3333 to 0.4400 | 0.3836 | 0.5132 | 1.1037 | 5.5754 | 0.5067 |
| massive_scenario.fr | 300 | 0 | 0.4600 | 0.4000 to 0.5167 | 0.4519 | 0.4448 | 0.9452 | 4.0666 | 0.5800 |
| massive_scenario.hi | 300 | 0 | 0.0967 | 0.0633 to 0.1333 | 0.0922 | 0.6846 | 1.5050 | 7.6748 | 0.1067 |
| massive_scenario.ja | 300 | 0 | 0.4667 | 0.4100 to 0.5200 | 0.4639 | 0.4290 | 0.9326 | 4.1331 | 0.6467 |
| massive_scenario.ko | 300 | 0 | 0.1267 | 0.0900 to 0.1667 | 0.1211 | 0.6886 | 1.5036 | 7.8330 | 0.1267 |
| massive_scenario.pt | 300 | 0 | 0.4200 | 0.3600 to 0.4767 | 0.4151 | 0.4835 | 1.0407 | 5.4989 | 0.5800 |
| massive_scenario.ru | 300 | 0 | 0.3367 | 0.2833 to 0.3900 | 0.3341 | 0.5375 | 1.1671 | 6.1547 | 0.4067 |
| massive_scenario.sw | 300 | 0 | 0.1000 | 0.0667 to 0.1367 | 0.0957 | 0.6775 | 1.4701 | 7.6414 | 0.1000 |
| massive_scenario.ta | 300 | 0 | 0.1033 | 0.0700 to 0.1400 | 0.1012 | 0.7701 | 1.6052 | 8.1614 | 0.0667 |
| massive_scenario.tr | 300 | 0 | 0.2267 | 0.1800 to 0.2800 | 0.2297 | 0.5625 | 1.2781 | 6.6421 | 0.3267 |
| massive_scenario.zh-CN | 300 | 0 | 0.4867 | 0.4333 to 0.5433 | 0.4818 | 0.4210 | 0.9187 | 4.3659 | 0.6200 |
| typed_decisions | 2000 | 0 | 0.3615 | 0.3395 to 0.3835 | 0.3262 | 0.1747 | 0.7497 | 1.3227 | 0.4280 |
| xnli.ar | 300 | 0 | 0.4400 | 0.3800 to 0.5000 | 0.4134 | 0.2847 | 0.7814 | 1.3380 | 0.4800 |
| xnli.bg | 300 | 0 | 0.5433 | 0.4867 to 0.6000 | 0.5329 | 0.2334 | 0.6455 | 1.1370 | 0.6867 |
| xnli.de | 300 | 0 | 0.6367 | 0.5800 to 0.6933 | 0.6286 | 0.2077 | 0.5391 | 0.9684 | 0.8067 |
| xnli.el | 300 | 0 | 0.5133 | 0.4567 to 0.5733 | 0.5044 | 0.2226 | 0.6730 | 1.1811 | 0.6200 |
| xnli.en | 300 | 0 | 0.8600 | 0.8233 to 0.8933 | 0.8612 | 0.0686 | 0.2205 | 0.3988 | 0.9800 |
| xnli.es | 300 | 0 | 0.6867 | 0.6333 to 0.7400 | 0.6854 | 0.1734 | 0.4741 | 0.8473 | 0.8667 |
| xnli.fr | 300 | 0 | 0.6800 | 0.6300 to 0.7333 | 0.6767 | 0.1878 | 0.4911 | 0.8924 | 0.8267 |
| xnli.hi | 300 | 0 | 0.4000 | 0.3467 to 0.4600 | 0.3925 | 0.2873 | 0.8294 | 1.3950 | 0.3667 |
| xnli.ru | 300 | 0 | 0.6100 | 0.5567 to 0.6633 | 0.6006 | 0.2162 | 0.6067 | 1.1250 | 0.7067 |
| xnli.sw | 300 | 0 | 0.4233 | 0.3667 to 0.4800 | 0.4208 | 0.2924 | 0.7786 | 1.3182 | 0.4733 |
| xnli.th | 300 | 0 | 0.4167 | 0.3633 to 0.4700 | 0.4160 | 0.1855 | 0.7152 | 1.1853 | 0.4667 |
| xnli.tr | 300 | 0 | 0.3833 | 0.3300 to 0.4400 | 0.3659 | 0.3563 | 0.8457 | 1.4684 | 0.4667 |
| xnli.ur | 300 | 0 | 0.3733 | 0.3200 to 0.4267 | 0.3357 | 0.3222 | 0.8427 | 1.4337 | 0.4000 |
| xnli.vi | 300 | 0 | 0.5100 | 0.4533 to 0.5633 | 0.5048 | 0.1963 | 0.6320 | 1.0815 | 0.6267 |
| xnli.zh | 300 | 0 | 0.6767 | 0.6267 to 0.7233 | 0.6656 | 0.2123 | 0.5227 | 1.0024 | 0.8000 |

| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |
|---|---|---|---|---|
| en.sst5 |  |  | 0.9045 | 0.6333 |
| typed_decisions | 0.3315 | 0.3155 | 0.6938 | 0.7525 |

## massive_intent.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.1267 | 0.6012 | 1.3971 | 6.9925 |

## massive_intent.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.4200 | 0.4559 | 0.9924 | 4.8814 |

## massive_intent.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.7833 | 0.1810 | 0.3770 | 1.8363 |

## massive_intent.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.4767 | 0.4180 | 0.9225 | 4.5426 |

## massive_intent.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.4867 | 0.4005 | 0.8661 | 3.6281 |

## massive_intent.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.1000 | 0.6480 | 1.4654 | 7.3669 |

## massive_intent.ja

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ja | 300 | 0.5467 | 0.3556 | 0.8034 | 3.7504 |

## massive_intent.ko

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ko | 300 | 0.1033 | 0.6628 | 1.4765 | 7.4425 |

## massive_intent.pt

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang pt | 300 | 0.4567 | 0.4503 | 0.9487 | 5.0787 |

## massive_intent.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.2933 | 0.5854 | 1.2343 | 6.5555 |

## massive_intent.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.1033 | 0.6296 | 1.4365 | 6.9003 |

## massive_intent.ta

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ta | 300 | 0.1133 | 0.5603 | 1.3529 | 5.9484 |

## massive_intent.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.1600 | 0.6047 | 1.3688 | 7.1393 |

## massive_intent.zh-CN

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh-CN | 300 | 0.5933 | 0.3356 | 0.7364 | 3.7488 |

## massive_scenario.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.1400 | 0.6372 | 1.4496 | 7.1111 |

## massive_scenario.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.2933 | 0.5632 | 1.2141 | 5.7217 |

## massive_scenario.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.6033 | 0.3628 | 0.7481 | 3.8150 |

## massive_scenario.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.3867 | 0.5132 | 1.1037 | 5.5754 |

## massive_scenario.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.4600 | 0.4448 | 0.9452 | 4.0666 |

## massive_scenario.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.0967 | 0.6846 | 1.5050 | 7.6748 |

## massive_scenario.ja

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ja | 300 | 0.4667 | 0.4290 | 0.9326 | 4.1331 |

## massive_scenario.ko

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ko | 300 | 0.1267 | 0.6886 | 1.5036 | 7.8330 |

## massive_scenario.pt

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang pt | 300 | 0.4200 | 0.4835 | 1.0407 | 5.4989 |

## massive_scenario.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.3367 | 0.5375 | 1.1671 | 6.1547 |

## massive_scenario.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.1000 | 0.6775 | 1.4701 | 7.6414 |

## massive_scenario.ta

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ta | 300 | 0.1033 | 0.7701 | 1.6052 | 8.1614 |

## massive_scenario.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.2267 | 0.5625 | 1.2781 | 6.6421 |

## massive_scenario.zh-CN

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh-CN | 300 | 0.4867 | 0.4210 | 0.9187 | 4.3659 |

## typed_decisions

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| type choice | 600 | 0.2883 | 0.1681 | 0.8079 | 1.5603 |
| type noul | 600 | 0.4867 | 0.1731 | 0.6068 | 0.8473 |
| type score | 800 | 0.3225 | 0.2065 | 0.8132 | 1.5011 |
| workflow agent_trace_observability | 500 | 0.3860 | 0.1955 | 0.7477 | 1.3386 |
| workflow customer_service | 500 | 0.3840 | 0.1029 | 0.7051 | 1.2950 |
| workflow invoice_processing | 500 | 0.3600 | 0.2258 | 0.7889 | 1.3281 |
| workflow security_incidents | 500 | 0.3160 | 0.2288 | 0.7571 | 1.3293 |

## xnli.ar

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ar | 300 | 0.4400 | 0.2847 | 0.7814 | 1.3380 |

## xnli.bg

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang bg | 300 | 0.5433 | 0.2334 | 0.6455 | 1.1370 |

## xnli.de

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang de | 300 | 0.6367 | 0.2077 | 0.5391 | 0.9684 |

## xnli.el

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang el | 300 | 0.5133 | 0.2226 | 0.6730 | 1.1811 |

## xnli.en

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang en | 300 | 0.8600 | 0.0686 | 0.2205 | 0.3988 |

## xnli.es

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang es | 300 | 0.6867 | 0.1734 | 0.4741 | 0.8473 |

## xnli.fr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang fr | 300 | 0.6800 | 0.1878 | 0.4911 | 0.8924 |

## xnli.hi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang hi | 300 | 0.4000 | 0.2873 | 0.8294 | 1.3950 |

## xnli.ru

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ru | 300 | 0.6100 | 0.2162 | 0.6067 | 1.1250 |

## xnli.sw

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang sw | 300 | 0.4233 | 0.2924 | 0.7786 | 1.3182 |

## xnli.th

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang th | 300 | 0.4167 | 0.1855 | 0.7152 | 1.1853 |

## xnli.tr

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang tr | 300 | 0.3833 | 0.3563 | 0.8457 | 1.4684 |

## xnli.ur

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang ur | 300 | 0.3733 | 0.3222 | 0.8427 | 1.4337 |

## xnli.vi

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang vi | 300 | 0.5100 | 0.1963 | 0.6320 | 1.0815 |

## xnli.zh

| Group | Questions | Accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|
| lang zh | 300 | 0.6767 | 0.2123 | 0.5227 | 1.0024 |

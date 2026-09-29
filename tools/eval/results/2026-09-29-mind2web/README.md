# Mind2Web agent steps

The three Mind2Web suites answered by Laya's English checkpoint, by cklxx/laya-browser (Laya's browser fine tune, at 645cf366), and by two baselines with no model, on an M4 with Metal through kime 0.1.4. Each suite is 500 steps, one operation question and one target question per step, 976 to 987 questions.

```
MIND2WEB_TEST_ZIP=test.zip python tools/eval/build_suites.py suites --only mind2web
kime eval suites --model laya --device metal
kime pull cklxx/laya-browser --revision 645cf366a2ae35f1086e8c20eff48f909bb49206
kime eval suites --model hf://cklxx/laya-browser --device metal
python tools/eval/baseline_mind2web.py suites lexical-ans
kime eval suites --answers lexical-ans
```

The uniform run gives every option the same probability. Its accuracy is only where ties land, on the first option, which is CLICK for the operation and the first element on the page for the target.

## Operation and target

| Answerer | Suite | Operation accuracy | Operation NLL | Target top 1 | Target NLL |
|---|---|---:|---:|---:|---:|
| Uniform | website | 0.782 | 1.91 | 0.061 | 4.21 |
| | task | 0.862 | 1.92 | 0.078 | 4.22 |
| | domain | 0.842 | 1.91 | 0.087 | 4.19 |
| Train share and goal words | website | 0.818 | 0.60 | 0.200 | 4.38 |
| | task | 0.870 | 0.45 | 0.189 | 4.24 |
| | domain | 0.868 | 0.45 | 0.188 | 4.42 |
| Laya | website | 0.042 | 2.22 | 0.048 | 4.49 |
| | task | 0.038 | 2.18 | 0.057 | 4.44 |
| | domain | 0.060 | 2.20 | 0.054 | 4.32 |
| laya-browser | website | 0.640 | 1.26 | 0.080 | 4.16 |
| | task | 0.564 | 1.44 | 0.086 | 4.15 |
| | domain | 0.616 | 1.33 | 0.083 | 4.16 |

A target head has a median of 98 options, so chance is about 0.01 and its NLL about 4.6.

Laya was not trained for this and says DONE to 1,121 of the 1,500 steps, WAIT to 168 and BLOCKED to 120. Its target pick is no better than taking the first element. laya-browser knows the operation is CLICK, TYPE_TEXT or SELECT on almost every step, but splits them 764, 658 and 76 when the gold is 1,278, 170 and 52, so it is below the train share on the operation. Its target is a little above the first element and far below matching the goal's words against the labels, which gets 1 step in 5.

laya-browser was trained on Mind2Web's train split, but in the request format of its own jev-ultrafast patch, and it reports 0.48 target top 1 on held out WebChain steps in that format. These suites use the format of stock jev-ultrafast as kime's `agent_step` builds it, with windows of up to 100 elements. So the gap most likely says the fine tune does not carry over to the stock format, not that it is weak in its own. Both models are well below the lexical floor on targets, which is what training on the Mind2Web train steps in kime's own format is for.

kime names a checkpoint by its encoder, and laya-browser has laya-multilingual's mmBERT encoder, so laya-browser.results.json says laya-multilingual. Laya took 487 to 501 seconds per suite and laya-browser 460 to 470, with a llama.cpp server running on the same GPU.

## Files

`<answerer>.report.md` is each run's report with groups by website and domain, `laya.results.json` and `laya-browser.results.json` the full metrics, and `suites.manifest.json` the split and usable step count behind each suite.

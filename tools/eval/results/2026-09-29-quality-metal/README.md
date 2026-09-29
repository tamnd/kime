# The quality group on Metal

`kime eval --suite quality` over every suite `build_suites.py` writes, with Laya's English checkpoint through kime 0.1.5 on an M4 with Metal in f16. The order and Mind2Web suites were in the same directories and were left out by `--suite`, so the run is the 61 suites of spec/13-benchmarks.md that have public data, 27,087 questions.

```
kime eval suites clerc-suites m2w-suites --suite quality --model laya --device metal
```

It took 98 minutes, 39 of them the two CLERC suites, whose court opinions fill the whole 512 tokens, with a llama.cpp server running on the same GPU.

[against-4090.tsv](against-4090.tsv) lines up each suite's correct answers with kime in f16 on the RTX 4090 (results/2026-09-28-rtx4090) and with Laya's published T4 run. Of the 59 suites both runs have, 47 get the same number right and the other 12 differ by one question each, 2 more right on Metal net. ECE is within 0.004 on every suite. Against the T4 run, 37 of 49 suites match and 12 are one question off. f16 on a different GPU flips questions whose top two options are close, and on the 4090 Laya's own PyTorch code was 47 questions off its T4 run, so this is within noise. The CLERC numbers are the ones in results/2026-09-29-clerc-rerank.

`report.md` and `results.json` are the run.

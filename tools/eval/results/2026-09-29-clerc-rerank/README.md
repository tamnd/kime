# CLERC rerank with Laya

The CLERC rerank row of spec/13-benchmarks.md, built as TypeSafe's re-ranking cookbook builds it and run with Laya's English checkpoint through kime on an M4 with Metal in f32, and through Laya 0.3.20 in PyTorch on MPS as a check of the harness.

```
python tools/eval/build_suites.py suites --only clerc
kime eval suites/en.clerc_rerank.jsonl suites/en.clerc_rerank_more.jsonl --model laya --device metal --precision f32
python tools/eval/run_laya.py <laya model dir> laya-ans suites/en.clerc_rerank.jsonl --device mps
kime eval suites/en.clerc_rerank.jsonl --answers laya-ans
```

The pool is 170 rows of the CLERC training file, 3,565 passages, and each query gets the BM25 top 30 of it. Every query has its gold passage somewhere in its 30. A candidate's question is the cookbook's noul, whether the passage could be the precedent the excerpt cites, and the candidates of a query are ranked by the probability of `true`. Ties keep BM25 order.

## Ranking

| Suite | Queries | Order | Top 1 | Top 5 | Top 10 | MRR |
|---|---|---|---|---|---|---|
| en.clerc_rerank, the cookbook's 40 | 40 | BM25 | 0.050 | 0.150 | 0.375 | 0.155 |
| | | Laya | 0.025 | 0.125 | 0.250 | 0.118 |
| | | Jev 1.12, from the cookbook | 0.18 | | 0.62 | |
| en.clerc_rerank_more, the other 110 | 110 | BM25 | 0.045 | 0.209 | 0.255 | 0.137 |
| | | Laya | 0.027 | 0.100 | 0.300 | 0.115 |

The BM25 row of the 40 is the cookbook's own, 5, 15 and 38 percent, so the slice is the same one. Laya does not rerank these passages. It is below BM25 on the cookbook's 40 and within a few queries of it on the other 110, where 1 query is 0.009. Its probability of `true` does not tell the gold passage from the others: the AUROC of the gold against the other candidates is 0.498 on the 40 and 0.489 on the 110, and the mean probability of `true` is 0.556 for gold passages against 0.554 for the rest. It says `true` to 68 and 63 percent of all candidates, which is why the accuracy of the noul questions is 0.34 and 0.38 when 1 candidate in 30 is the gold. The opinion excerpts and passages are about 3,600 characters each, so the pair is cut to Laya's 512 tokens and much of the passage is never seen.

## kime against Laya's own code

On the 1,200 questions of en.clerc_rerank, kime in f32 on Metal and Laya 0.3.20 in PyTorch under fp16 autocast on MPS differ by at most 5.5e-5 in the probability of `true`, 2.6e-5 on average, with no answer on the other side of 0.5. Both give the same ranking numbers to every digit. kime took 468 seconds for the 1,200 and 1,650 for the 3,300, and Laya in PyTorch 555 for the 1,200.

## Files

`kime.report.md` and `kime.results.json` are the kime run over both suites, and `laya-pytorch.*` the scores of Laya's PyTorch answers on the 40.

# Choices in chunks

Laya English through kime on an M4 with Metal, answering the same 64 option choices in one pass and in chunks of 8, 16 and 32 options with the rerank of the best 16, the chunk consistency test of spec/15-testing.md (#46). A llama.cpp server was labelling on the same GPU the whole time, so the seconds below compare the sizes with each other and say nothing about kime's speed.

```
python tools/eval/chunk_consistency.py build suites/en.banking77_full.jsonl --out b77 --n 300
python tools/eval/chunk_consistency.py build suites/mind2web.*.jsonl --out m2w --n 300
for f in b77/requests.*.jsonl m2w/requests.*.jsonl; do kime predict --model laya --device metal --batch < $f > ${f/requests/responses}; done
python tools/eval/chunk_consistency.py compare b77
```

Each question keeps its gold option and 63 others drawn with seed 13, in their order. banking77 gives 300 of its 500 lines with 64 of the 77 intents. Mind2Web gives 300 target heads of 64 of their elements from the three test suites.

## Results

| Suite | Chunk size | Top 1 | Seconds |
|---|---|---:|---:|
| banking77 | one pass | 0.523 | 228 |
| | 8 | 0.607 | 432 |
| | 16 | 0.623 | 367 |
| | 32 | 0.607 | 369 |
| Mind2Web target | one pass | 0.020 | 402 |
| | 8 | 0.043 | 3,708 |
| | 16 | 0.033 | 1,921 |
| | 32 | 0.047 | 1,203 |

The three chunk sizes pick the same option on 237 of the 300 banking77 questions (0.790) and on 105 of the 300 Mind2Web ones (0.350). Pairs of sizes agree on 0.82 to 0.88 of banking77. `banking77.md` and `mind2web.md` have every pair.

The spec asks for 99%, and Laya is far from it. That is expected: a logit means the same thing in every chunk only for a model trained with random chunking, which Laya was not, so the gate is for kime-v1 (#44) and this run is its baseline. On Mind2Web Laya is near chance (1 in 64 is 0.016) whatever the chunks, and noise does not agree with itself.

Chunks still help Laya on banking77, by 8 to 10 points. In one pass Laya's crowding rule cuts each of the 64 options to 4 tokens to fit its 192 token head, and a chunk of 8 leaves each option 22. The price is one sequence per chunk plus the rerank, each reading the whole state again, which is why Mind2Web with its long pages takes 9 times as long in chunks of 8. kime-v1 reads the state once for all of them.

kime scores a choice in one pass whenever it fits, so none of this changes an answer Laya would give, and chunks are only used for a choice that does not fit or when a request sets `kime.chunk`.

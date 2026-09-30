# Release gate on Laya

`kime eval --gate` run on Laya's English checkpoint on an M4 with Metal through kime 0.1.6 plus the gate, over the 13 suites the behaviour tests read. The baseline is the 0.1.5 Metal run in results/2026-09-29-quality-metal, so the calibration check compares Laya with itself and is only a check that the plumbing works.

```
S=kime-suites
kime eval $S/order.en.emotion.jsonl $S/order.massive_intent.en.jsonl $S/order.xnli.en.jsonl \
  $S/en.sst5.jsonl $S/typed_decisions.jsonl $S/app.phishing.jsonl $S/massive_intent.en.jsonl \
  $S/massive_scenario.en.jsonl $S/xnli.en.jsonl $S/en.ag_news.jsonl $S/app.support_triage.jsonl \
  $S/app.model_routing_domain.jsonl $S/en.banking77_full.jsonl --model laya --device metal \
  --out out-laya --gate --baseline 2026-09-29-quality-metal/results.json
```

The quality run took 27 minutes and the gate 21 more, with the llama.cpp server of the #34 labelling on the same GPU. 27 of 30 checks ran and 14 failed, so Laya would not pass the gate kime-v1 has to pass, which is what the gate is for: every failure is a defect spec/12-training.md already names and trains against.

| Test | Laya | Target |
|---|---|---|
| Order flips | 0.048 Emotion, 0.160 MASSIVE intent, 0.001 XNLI | 0.02 or less |
| Score mirror | 0.200 SST-5, 0.731 typed decisions | 0.98 or more |
| Noul polarity | 0.000 phishing, 0.263 typed decisions | 0.97 or more within 0.05 |
| Level coverage | SST-5 passes, typed decisions has 3 levels that never win | none |
| Label neutrality | from 0.033 to -0.447, only AG News within 0.01 | within 0.01 |
| Chunk consistency | 0.803 on 300 Banking77 questions of 64 options | 0.99 or more |
| Calibration regression | 0 on every suite, as the baseline is Laya | 0.01 or less |

Score mirror at 0.200 on SST-5 means that with the levels reversed Laya mostly keeps picking the same position rather than the same meaning, so its sense of which end is positive comes from the level's place and not its text. Noul polarity at 0 on phishing is not a harness fault: the swapped question keeps its instructions, "Is this email a phishing or scam attempt", and Laya answers the instructions and ignores that the criteria now say true means a legitimate email. Five spot checks give 0.0166 and 0.0326, 0.7098 and 0.5102, 0.9999 and 1.0 for the original and swapped question. Label neutrality costs Laya 0.447 on XNLI English, where the labels entailment, neutral and contradiction carry most of the meaning and the descriptions do not make up for them. Chunk consistency is 0.803 here and 0.790 in results/2026-09-30-chunking on a different draw of questions.

Segment consistency and routing are reported as not run, since segment mode and the routing set do not exist yet, and the quality targets of spec/13-benchmarks.md are left to report.md.

## Files

`gate.md` and `gate.json` are what the command wrote, and `report.md` the quality run before it.

# Release gate, kime eval, laya on metal, Apple M4, F16

| Test | Suite | Value | Target | n | Result | Note |
|---|---|---:|---|---:|---|---|
| Order flips | order.en.emotion | 0.0477 | <= 0.02 | 3000 | fail |  |
| Order flips | order.massive_intent.en | 0.1600 | <= 0.02 | 1500 | fail |  |
| Order flips | order.xnli.en | 0.0013 | <= 0.02 | 1500 | pass |  |
| Score mirror | en.sst5 | 0.2000 | >= 0.98 | 600 | fail |  |
| Score mirror | typed_decisions | 0.7312 | >= 0.98 | 800 | fail |  |
| Noul polarity | typed_decisions | 0.2625 | >= 0.97 within 0.05 | 400 | fail |  |
| Noul polarity | app.phishing | 0.0000 | >= 0.97 within 0.05 | 400 | fail |  |
| Level coverage | en.sst5 | 0.0000 | 0 questions | 600 | pass |  |
| Level coverage | typed_decisions | 3.0000 | 0 questions | 800 | fail | discrepancy_severity of 4 levels never picks [1, 3], severity of 5 levels never picks [4], urgency of 4 levels never picks [0] |
| Label neutrality | typed_decisions | 0.0333 | within 0.01 | 600 | fail |  |
| Label neutrality | massive_intent.en | 0.0133 | within 0.01 | 300 | fail |  |
| Label neutrality | massive_scenario.en | -0.0533 | within 0.01 | 300 | fail |  |
| Label neutrality | xnli.en | -0.4467 | within 0.01 | 300 | fail |  |
| Label neutrality | en.ag_news | -0.0067 | within 0.01 | 600 | pass |  |
| Label neutrality | app.support_triage | -0.0275 | within 0.01 | 400 | fail |  |
| Label neutrality | app.model_routing_domain | -0.0576 | within 0.01 | 399 | fail |  |
| Chunk consistency | en.banking77_full | 0.8033 | >= 0.99 | 300 | fail |  |
| Segment consistency |  |  | >= 0.99 | 0 | not run | segment mode is not built yet (M4) |
| Routing |  |  | >= 0.99 | 0 | not run | the routing set is not built yet |
| Calibration regression | en.sst5 | -0.0000 | ECE up by <= 0.01 | 600 | pass | 0.2478 to 0.2478 |
| Calibration regression | typed_decisions | 0.0000 | ECE up by <= 0.01 | 2000 | pass | 0.1742 to 0.1742 |
| Calibration regression | app.phishing | -0.0000 | ECE up by <= 0.01 | 400 | pass | 0.0121 to 0.0121 |
| Calibration regression | massive_intent.en | -0.0000 | ECE up by <= 0.01 | 300 | pass | 0.1810 to 0.1810 |
| Calibration regression | massive_scenario.en | 0.0000 | ECE up by <= 0.01 | 300 | pass | 0.3628 to 0.3628 |
| Calibration regression | xnli.en | 0.0000 | ECE up by <= 0.01 | 300 | pass | 0.0685 to 0.0685 |
| Calibration regression | en.ag_news | 0.0000 | ECE up by <= 0.01 | 600 | pass | 0.0410 to 0.0410 |
| Calibration regression | app.support_triage | -0.0000 | ECE up by <= 0.01 | 400 | pass | 0.0933 to 0.0933 |
| Calibration regression | app.model_routing_domain | 0.0000 | ECE up by <= 0.01 | 399 | pass | 0.0888 to 0.0888 |
| Calibration regression | en.banking77_full | 0.0000 | ECE up by <= 0.01 | 500 | pass | 0.3189 to 0.3189 |
| Quality targets |  |  | spec/13-benchmarks.md | 0 | not run | read report.md against the targets of the model's tier |

27 of 30 checks ran and 14 failed, so the checkpoint does not pass.

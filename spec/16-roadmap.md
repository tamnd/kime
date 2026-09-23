# Roadmap

The order is chosen so that each milestone ships something usable and proves one claim before the next one depends on it. The compat engine comes first, because it lets us measure engine gains on known weights before any training money is spent.

## M0: compat engine (weeks 1 to 5)

- `kime-core`, `kime-tok`, `kime-model` (laya family), `kime-tensor`, and `kime-cpu` with FP32 kernels.
- `kime-cuda` with the kernel set in 08 for the compat graph, CUDA graphs and buckets.
- `kime predict` and the Rust API.
- Parity with `laya_ref.py` on all fixtures, on CPU and CUDA.

Exit criteria: compat W1 on T4 at 12 ms or less (from 32.8 to 39.5 ms), compat W3 per question at 2 ms or less, and bit exact determinism tests passing.

## M1: API and server (weeks 4 to 8)

- `kime-serve` with the full API in 03, auth, rate limits, the scheduler (single stage for compat), metrics.
- `kime-route` with script detection, the language id model and Laya's rules.
- The email cleaner and the presets.
- The Python package with both API styles, and the TypeScript SDK.
- The compatibility test suite in 15.

Exit criteria: all compatibility tests pass. jev-ultrafast runs against kime-serve with only the base URL changed, and laya-serve users can switch binaries with no config change.

## M2: data and teacher (weeks 3 to 10, in parallel)

- Dataset conversion for the public sources, synthetic data generation and labelling, agent trace collection, the contamination checks.
- `kime-train` on Burn for the cross encoder family, and checks against the PyTorch reference.
- Train `kime-t-en` and `kime-t-x`.

Exit criteria: the teachers beat Laya on every quality suite in 13. The teacher is the upper bound for the students, so if the teacher does not beat Laya, the students will not either, and we stop to fix the data before going on.

## M3: kime-v1 students (weeks 9 to 14)

- Split architecture in `kime-model`, `kime-train` and `kime-cuda` (cross attention, the question mask, segment mode, KV projection, the state cache).
- Distil `kime-v1-s-en` and `kime-v1-s-x`, fit calibration, train the correctness head.
- Two stage scheduler, state cache and answer cache in the server.

Exit criteria: the release gate in 12 passes for both s tiers, and every CUDA speed row in 13 meets its target on T4 and L4.

## M4: Apple and CPU (weeks 12 to 18)

- `kime-metal` with indirect command buffers. `kime-ane` with the MIL exporter and the ANE packages.
- `kime-cpu` INT8 kernels for AVX2, AVX-512 VNNI, AMX and NEON i8mm, and quantization aware fine tuning for INT8.
- Energy measurement tooling.

Exit criteria: the Apple and CPU rows in 13 meet their targets.

## M5: 1.0 (weeks 18 to 20)

- m tiers, FP8 on sm_89 and newer, `kime train` for users, Docker, Homebrew, Nix, Helm, and the full published scorecard.

Exit criteria: every row of the scorecard passes, or has a written reason like the memory row in 13. The README headline table is generated from the scorecard.

## After 1.0

- KimeKit (Swift) on the C ABI.
- 4 bit weights, if a checkpoint trained for them passes the gate.
- A 32k context state tower trained natively rather than through segment mode only.
- More question types. Jev says more are coming. Likely candidates are multi label choice (independent probability per option), ranking (a distribution over orderings of a few items, via Plackett-Luce over option logits) and span pointing (a probability per state segment). The split architecture supports all three with new scorers only.
- ROCm backend, if there is demand. The vLLM semantic router's work on MI300X shows ModernBERT style encoders run well there with custom attention.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| The split architecture loses accuracy against a cross encoder | Quality targets missed | Distillation from a cross encoder teacher with hidden state targets, 3 to 4 question layers with cross attention, the m tier as a fallback. If the gap stays above 1 point on typed-decisions, add one joint layer on top (question and state tokens together), which costs one layer per question instead of all of them. |
| The synthetic labels carry the LLMs' biases | Calibration looks good on synthetic data but not on real data | Calibration is fitted and reported on human labelled splits only. Synthetic data is capped at 50% of each batch. |
| Burn is missing training features | Training schedule slips | Custom CUDA ops exposed to Burn. The data format stays neutral so a PyTorch trainer could take over. |
| Kernel launch overhead dominates on T4 even with graphs | W1 target missed on T4 | Fold more layers into persistent kernels (one CTA per sequence runs several layers for tiny buckets). There is 4x margin in the budget in 08. |
| Jev improves fast | Targets move | The scorecard is re-measured against the current Jev on every release. Targets are ratios to the best baseline, not fixed numbers. |
| ANE op placement changes between macOS versions | Silent CPU fallback | The placement check at load in 09, and CI on the two latest macOS versions. |
| Legal: dataset licenses, LLM terms for labelling | Cannot release weights | The license of every source is tracked in the manifest. Labelling uses providers whose terms allow training classifiers on outputs. Anything unclear is excluded before training. |

## Open questions

1. Should `kime-latest` be the s tier or the m tier? The s tier meets every speed target, and the m tier has more quality margin. The current plan is s, with m available by name. We decide after M3, using the measured accuracy gap.
2. Should `jev-*` aliases be on by default? It makes drop-in use trivial, but some operators may find it confusing. The plan is on by default, logged at start, and switched off with a flag.
3. Do we publish the synthetic dataset? It would help others reproduce the models. The deciding factor is the labelling providers' terms.
4. Should the compat family be in the default build? It adds about 1,500 lines and the ReLU head kernels. The plan is yes, because the same-weights comparison is the most convincing evidence we can give that the engine is fast.

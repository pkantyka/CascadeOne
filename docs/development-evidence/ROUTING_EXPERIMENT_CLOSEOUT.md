# CascadeOne routing experiment closeout (2026-10-13)

**Decision: CLOSED / NO-GO for tested Laya -> Clef routing policies.**
Automated benchmark verdict: `CLOSE_CURRENT_ROUTING_EXPERIMENTS_NO_GO`.
This closes the current R0/R2/R3a/R4 sweep, not System One model
selection, Fusion, the V1 release, or System One/System Two in general.

## Frozen scope and execution

- Runner/code baseline: Git `466ee80` on `experiment/system1-routing`.
- GPU: NVIDIA GB10 on DGX Spark; CUDA 13.0, live CUDA inference.
- Timestamp: 2026-10-13; run directory `routing-dgx-real-20261013T063706Z`.
- Calibration: 5,099 requests, 3,812 Laya-eligible. Timed subset:
  256 independent state groups, 194 eligible (75.78%).
- Nine variants: Clef-only; R0/R2/R3a/R4 for question-only and
  both-metric gates. Three repetitions per arm/request = 6,912
  individual end-to-end trials, following real-model warmup.
- The timed path includes eligibility, tokenization, pre-router, live
  Laya/Clef forwards, R3a pooling and CPU transfer. Each measurement
  uses CUDA synchronization; paired arm order is randomized.
- Routing weights and thresholds are frozen numeric-only NPZ from
  five grouped cross-fitted folds, not retrained on DGX.

## Measured end-to-end latency

Routing research target: >=30% reduction in mean latency versus
Clef-only, subject to an exploratory <=5% relative eligible-only
quality-loss budget for the selected question/both gate.

| Policy | Mean ms | p95 ms | Mean reduction | Paired 95% CI |
|---|---:|---:|---:|---|
| clef_only | 441.16 | 915.21 | 0.00% | [0.00%, 0.00%] |
| r0_question | 462.20 | 923.53 | -4.77% | [-6.95%, -2.37%] |
| r2_question | 461.80 | 919.48 | -4.68% | [-6.91%, -2.33%] |
| r3a_question | 442.82 | 921.21 | -0.38% | [-3.41%, +2.96%] |
| r4_question | 432.07 | 921.88 | +2.06% | [-0.84%, +5.33%] |
| r0_both | 468.50 | 919.20 | -6.20% | [-8.28%, -4.02%] |
| r2_both | 470.33 | 919.66 | -6.61% | [-8.50%, -4.71%] |
| r3a_both | 459.51 | 922.44 | -4.16% | [-6.58%, -1.55%] |
| r4_both | 444.35 | 923.32 | -0.72% | [-2.98%, +1.76%] |

No arm approaches the 30% mean-latency target, and every routing
arm has worse observed raw p95 than Clef-only. Best point estimate:
R4 question saves 9.09 ms (2.06%) with a CI that includes zero;
its exploratory eligible question/exact losses are 4.85% / 8.29%.
R4 both has eligible losses 3.07% / 4.86% but is slower by 0.72%.
R4 improves over R3a alone by 10.75 ms (question) or 15.16 ms
(both); this does not rescue total cascade efficiency.

## Integrity and methodological boundaries

- The five files in the results checksum manifest pass SHA-256.
  The run-local manifest independently passes for the three outputs.
- 256 unique request IDs, nine arms x 768 trials, and three repeats
  were counted from `trials.jsonl`; no missing or partial measurement.
- `router_drift_events` and `pinned_clef_parity_drift` are empty.
  These implemented parity checks pass; the runner does NOT directly
  check full Laya probability or latent-vector parity.
- The quality gate is based on exploratory grouped calibration
  cross-validation, with R4 candidate selection on related data.
  Actual measured-subset quality is also recorded in `report.json`,
  but this is NOT an independent production-quality holdout.
- Timings measure one sequential worker on one GB10, not
  saturated throughput or real production traffic. The routing
  objective here (30%) is distinct from broader V1 release gates.

## Preserved evidence and reproducibility

Under `reports/runs/routing-dgx-real-20261013T063706Z/`:

- `report.md`: automatic decision and human-readable timing matrix.
- `report.json`: full statistics, model/GPU metadata, paths, drift.
- `trials.jsonl`: all 6,912 live measured trials.
- `SHA256SUMS`: checksums for the three run outputs.
- `RESULTS_SHA256SUMS`: checksums for results AND frozen router files.

Under `artifacts/routing-policy-frozen/`: `weights.npz` and
`manifest.json`, the exact policy bank used by the DGX run.

From the repository root, verify the five archived evidence files:

```bash
sha256sum -c reports/runs/routing-dgx-real-20261013T063706Z/RESULTS_SHA256SUMS
```

Input artifacts needed to fully reproduce the bank/run, including
the original latent archive and R4 reference, are stored in the
original self-contained handoff:
`artifacts/cascadeone-routing-measure-466ee80.tar.gz`.
SHA-256: `c53f6f05f63f5116ba2bff1876882483f233b9babcbb561af2275d2e6f52630e`.
The source bundle and its SHA-256 sidecar are now stored as regular Git
blobs. See `docs/REPRODUCIBILITY_ARCHIVES.md` for verification and
restore instructions. Upstream Hugging Face model snapshots are still
external to Git. The measured results and frozen weights are unchanged.

## Decision and next experiment

Stop further tuning of these frozen R0/R2/R3a/R4 policies as a path
to 30% improvement on the current Laya/Clef pair. Do not promote
this routing implementation to `master` or
`experiment/system1-model-selection` from this negative result.
Explore a genuinely faster System One model, cheaper encoder early
exit, or altered execution strategy only in new separately frozen
experiments. Retain this branch as the audited negative result.

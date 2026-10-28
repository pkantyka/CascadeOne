# CascadeOne research hypotheses

The framework measures these questions; they are not assertions of universal performance.

| ID | Hypothesis | Current scope |
| --- | --- | --- |
| H-C | Confidence-gated cascades reduce end-to-end latency with acceptable Gold-quality loss | Testable with included reference pairs and three-arm benchmarks; no universal claim |
| H-R | More selective routing beats confidence-only at a matched escalation budget | Research question; no learned gain router implemented here |
| H-P | Pico -> Ultra improves the quality/latency tradeoff versus either model alone | Requires live hardware-matched measurements on frozen datasets |
| H-D | Distilling a small model from the strong model improves the cascade tradeoff | Controlled distillation experiment not included |
| H-G | Predicting expected Gold improvement outperforms confidence as a routing signal | Concept only; not an available policy |
| H-F | A separate fidelity-risk head can improve selective agreement | Design only |

## Suggested controlled distillation experiment

With identical source requests, model revisions and hardware, compare the unchanged fast model, the strong model, their original cascade, a separately distilled fast model and its cascade. Measure independent Gold correctness, teacher fidelity, escalation, mean/p95 latency and resource usage. Calibrate routing only on a dedicated calibration split; keep protected test cases untouched.

A difference between model families, sizes, GPU residency or workloads is not evidence for or against distillation. Each benchmark must preserve its own input and runtime provenance.

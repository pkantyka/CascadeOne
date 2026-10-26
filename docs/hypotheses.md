# Engineering hypotheses and status

| ID | Hypothesis | Status |
| --- | --- | --- |
| H-C | Confidence-gated cascades save latency with bounded quality loss | Measured in two distinct pairs; not generally established |
| H-R | More elaborate routing improves E2E tradeoff over simple confidence | Laya/Clef R0–R4 negative on recorded 2026 DGX run |
| H-P | Pico→Ultra provides useful mean-latency vs Gold-quality tradeoffs | H1 measured, but production SLA unproven |
| H-D | A small model **distilled from the actual strong fallback** improves cascade tradeoffs vs its pre-distillation version | OPEN: controlled Clef→Laya comparison required |
| H-G | Predicting expected benefit of escalation beats confidence-only at the same escalation budget | OPEN: G1 code prepared; no verified G1 result in this release |
| H-F | Separate fidelity head improves agreement-constrained routing | DESIGN ONLY |

**Crucial caveat**: The two historical model pairs differ in model sizes, families, runtime versions, data and GPU cost. Existing Laya/Clef vs Pico/Ultra comparisons are **not a controlled test of distillation**.

## Suggested controlled distillation comparison

On a matched frozen dataset and hardware, compare Clef-only, unchanged Laya-only, unchanged Laya→Clef, newly distilled Laya-only, and newly distilled Laya→Clef. Preserve the backbone/architecture where feasible. Match calibrated escalation budgets, report Gold correctness AND teacher fidelity, and measure real E2E and resource consumption. Do not tune on protected test splits.

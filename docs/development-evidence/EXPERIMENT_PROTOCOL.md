# Experiment protocol

configs/experiment.yaml is the machine-readable protocol. This document explains the operational
interpretation.

## Primary hypothesis

CascadeOne is useful only if it preserves Clef decision quality while reducing real end-to-end request
latency. A lower strong-execution fraction is not sufficient evidence.

## Fixed V1 choices

- Modality: text.
- Routing unit: request.
- Confidence aggregation: minimum question answer confidence.
- Quality margin: zero.
- One-sided confidence level: 95 percent.
- Practical mean latency reduction target: at least 20 percent.
- Practical p95 ratio limit: at most 1.10.
- Training partition: train.
- Checkpoint selection partition: validation.
- Threshold selection partition: calibration.
- Protected results may not select checkpoints or thresholds.

## Data roles

Training data may come only from approved pinned dataset revisions. Validation selects adapter
checkpoints. Calibration selects routing thresholds. Protected benchmark suites are evaluation-only.

Splits are grouped by normalized state so questions from the same state cannot cross fitting roles.
Exact and advisory near-overlap checks must be completed before production training.

Dataset preparation does not discard rows by option count. Core and Fusion are evaluated on the same frozen population; requests outside Laya fast-path eligibility execute plain Clef in both modes. Fusion adapter fitting is restricted to fast-path-eligible train/validation rows because the adapter is never executed on direct-Clef requests.

## Compute-budget gate

The real-device compute profile is operational evidence, not a scientific result. It measures 100 fast requests, 100 strong requests and 200 GatedResidualFusion training steps on CUDA. A production Fusion pilot may run for at most 500 steps without a profile report. Any longer production run must use a passing profile bound to the same source fingerprint, exact Fusion configuration and training-data fingerprint.

Pilot latency uses a deterministic frozen validation subset (default 256 independent states) with a manifest binding ordered IDs and content. The final confirmatory sample size is selected and frozen separately from pilot evidence.

## GatedResidualFusion controls

GatedResidualFusion is the only Fusion architecture in the current V1 protocol. Cross-attention is outside the planned experiment and would require a separately revised protocol before use. A production GatedResidualFusion claim requires three inference controls on the same cases:

- real latent;
- zero latent;
- shuffled latent with the donor mask moved together with the donor representation.

A separately trained lower-information control should be added before a strong causal claim that gains
come specifically from Laya information rather than adapter capacity.

## Confirmatory freeze

Before protected evaluation, record suite membership and weights, sample size, selected checkpoint,
routing threshold, architecture, seeds and all benchmark execution settings. Protected results may be
reported descriptively but must not trigger a new configuration selection.


## Git branches and reproducibility

Use Git branches to separate experimental implementation paths and commit SHAs to identify the exact source snapshot used by a run. A branch name is descriptive and movable; the commit SHA is the immutable code identifier. Reports record both when Git is available.

Generated experiment outputs remain ignored by Git by default. Selected immutable evidence is deliberately tracked in ordinary Git: the real DGX routing run, 16 historical reports, frozen routing heads, two checksum-bound input archives and the Tiny Fusion CPU smoke evidence. See `docs/REPRODUCIBILITY_ARCHIVES.md` and `docs/ROUTING_EXPERIMENT_CLOSEOUT.md`. Reusing an output directory that already contains `report.json` is rejected; each new attempt requires a new path. Dataset/split manifests, SHA-256 artifacts and pinned model revisions identify the data/model inputs.

The runtime and DGX handoff source fingerprints include `src/`, `configs/` and `scripts/`, so changing an evaluation or preprocessing script changes the fingerprint.

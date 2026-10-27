# Laya → Clef (reference cascade)

Pinned Laya (fast) and Clef-Flash (strong) run in **separate, independently measurable HTTP services**. The fast server uses the audited lossless eligibility checks. Ineligible cases skip Laya execution and escalate. Threshold 0.90 is an illustrative original configuration, **not** a newly calibrated optimum.

## Requirements

ARM64 NVIDIA CUDA system, Docker Compose with GPU support, an authorized immutable NGC CUDA base digest compatible with Dockerfile, access to the pinned Laya/Clef Hugging Face revisions, and enough GPU RAM for both models concurrently. This original stack is DGX Spark-oriented. It has not been rebuilt from the Community tree yet.

Pin a compatible CUDA base:

    export CASCADEONE_CUDA_BASE='nvcr.io/...@sha256:YOUR_VERIFIED_DIGEST'

From Community root:

    docker compose -f cascades/laya-clef/compose.yaml build fast strong
    docker compose -f cascades/laya-clef/compose.yaml up -d fast strong

Wait for /health to become ready for both models (first load can be long). Run a tiny smoke test:

    docker compose -f cascades/laya-clef/compose.yaml run --rm benchmark python benchmarks/runner.py --setup benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json --output output/laya-smoke

Run a genuine benchmark with another unique output path and frozen dataset. The benchmark measures all three arms: "fast_only", "strong_only" and "cascade" while **both models are resident**. For stand-alone residency, stop fast, keep strong, use --phase strong-alone and a new output path. Compare with tools/compare_runs.py.

Do not submit smoke results as real quality benchmarks.

Historical R0–R4 experiments are documented in docs/development-evidence/ROUTING_EXPERIMENT_CLOSEOUT.md, but the confidence threshold is defined by the cascade profile and not the runner; R0–R4 historical policies are not automatically reimplemented. The strong-only baseline in this wrapper is a separate Clef HTTP server, not the exact old experimental runner; new results must be compared to old numbers with care.

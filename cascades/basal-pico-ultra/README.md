# Basal Omni Pico → Ultra (reference cascade)

This kascade uses **two independent pinned Basal versions**: Pico engine 2.3.2, Ultra engine 2.1.0. Do not silently upgrade either. The models are pinned to the revisions declared in benchmark.json and Compose.

## Requirements

ARM64 NVIDIA GPU with NGC vLLM 26.07 compatible PyTorch/CUDA environment, Docker Compose GPU support, HF snapshot access, enough GPU RAM for Pico and Ultra together. Base image tag in these experimental Dockerfiles has not yet been converted to an immutable digest: **not release-qualified**. Do not claim environment-level reproduction yet.

From Community root:

    docker compose -f cascades/basal-pico-ultra/compose.yaml build fast strong
    docker compose -f cascades/basal-pico-ultra/compose.yaml up -d fast strong

Wait for service health/model readiness. Run:

    docker compose -f cascades/basal-pico-ultra/compose.yaml run --rm benchmark python benchmarks/runner.py --setup benchmarks/smoke-general__basal-pico-ultra__confidence-min-t0p656352/benchmark.json --output output/pico-smoke

The default threshold 0.6563522219657898 comes from the **historic H1 calibration**; it is frozen in this configuration, not recalibrated using the test population. To make a new experiment, create a new profile with explicit provenance.

To assess memory residency, first run strong-only with Pico stopped using --phase strong-alone, then run both servers and the colocated phase. Output paths must differ. Compare with tools/compare_runs.py. The runner supports simple hard-label and categorical-distribution Gold; more complex Gold schemas require a separately validated adapter.

Historical benchmark: docs/development-evidence/PICO_ULTRA_H1_EXPERIMENT_SUMMARY_2026-10-23.md. The prior live results are **not** Community v0.1 measurements. Performance/quality on a new dataset or hardware may be different.

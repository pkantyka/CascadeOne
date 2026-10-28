# Laya → Clef (reference cascade)

Pinned Laya (fast) and Clef-Flash (strong) run in **separate, independently measurable HTTP services**. The fast server uses the audited lossless eligibility checks. Ineligible cases skip Laya execution and escalate. Threshold 0.90 is an illustrative original configuration, **not** a newly calibrated optimum.

## Requirements

ARM64 NVIDIA CUDA system, Docker Compose with GPU support, an authorized immutable NGC CUDA base digest compatible with Dockerfile, access to the pinned Laya/Clef Hugging Face revisions, and enough GPU RAM for both models concurrently. The stack targets DGX Spark-class hardware; a clean GPU deployment has not yet been validated for release.

Pin a compatible CUDA base:

    export CASCADEONE_CUDA_BASE='nvcr.io/...@sha256:YOUR_VERIFIED_DIGEST'

From the repository root:

    docker compose -f cascades/laya-clef/compose.yaml build fast strong
    docker compose -f cascades/laya-clef/compose.yaml up -d fast strong

Wait for /health to become ready for both models (first load can be long). Run a tiny smoke test:

    docker compose -f cascades/laya-clef/compose.yaml run --rm benchmark python benchmarks/runner.py --setup benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json --output output/laya-smoke

Run a genuine benchmark with another unique output path and frozen dataset. The benchmark measures all three arms: "fast_only", "strong_only" and "cascade" while **both models are resident**. For stand-alone residency, stop fast, keep strong, use --phase strong-alone and a new output path. Compare with tools/compare_runs.py.

Do not submit smoke results as real quality benchmarks.

The confidence threshold is defined by the cascade profile, not the benchmark runner. Only the policies implemented in cascades/policies.py are available. The strong-only baseline measures the configured Clef HTTP service; compare latency and quality only under matching workload and hardware conditions.

## Serve the complete cascade through one API

The optional Docker Compose `inference` profile provides a host-local System One endpoint on `127.0.0.1:18358`. Start fast, strong and inference together:

```bash
docker compose -f cascades/laya-clef/compose.yaml --profile inference up --build fast strong inference
```

See [local inference](../../inference/README.md) for the API, validated benchmark startup comparisons and alert limits.

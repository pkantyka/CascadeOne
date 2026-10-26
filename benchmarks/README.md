# Shared benchmark runner

    python benchmarks/runner.py --cascade cascades/<name>/benchmark.json \
      --dataset-id smoke-general-v1 --dataset-split test --output output/unique-run

A benchmark can also take --dataset /path/to/your.jsonl. For a downloadable dataset run python datasets/catalog.py prepare ID beforehand. The runner verifies registered data, records its manifest hash and split, and never downloads implicitly during performance timing.

Runner v0.1 uses standard-library HTTP POST, two independent model endpoints, one sequential client, deterministic shuffled request/arm order and live execution. "colocated" measures strong_only and cascade **with both servers resident**; "strong-alone" measures strong_only with no fast endpoint expected. Records are written at every completed trial and output directories are immutable (no overwrite).

Each report includes relative quality on the same requests when complete Gold exists, request latency (mean,p50,p95,p99), escalation, CPU/host RAM/GPU telemetry if available, model declarations, dataset provenance/hashes, HTTP identity and integrity checks. It does not measure throughput at concurrency >1 or infer per-model VRAM ownership. Host telemetry can be missing due to device/driver/toolkit constraints.

    python tools/validate_result.py output/unique-run
    python tools/compare_runs.py --standalone output/solo --colocated output/paired

A real contributor submission should use a frozen, properly licensed dataset, independent calibration and quality evaluation, enough independent requests for stable tail statistics, and multiple repeats. Bundled smoke data are not meaningful for performance claims. Dataset *domain*, question-type mix and context lengths can materially change both quality and economics of the cascade.

The HTTP /v1/models name check does not cryptographically attest weights; immutability relies on audited deployment/model pins. Publish base image digests and any custom build decisions.

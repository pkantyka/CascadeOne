# Shared benchmark runner

    python benchmarks/runner.py --cascade cascades/<name>/benchmark.json \
      --dataset path/to/frozen.jsonl --output output/unique-run

Runner v0.1 uses standard-library HTTP POST, two independent model endpoints, one sequential client, deterministic shuffled request/arm order and live execution. "colocated" measures strong_only and cascade **with both servers resident**; "strong-alone" measures strong_only with no fast endpoint expected. Records are written at every completed trial and output directories are immutable (no overwrite).

Each report includes relative quality on same requests when *full* Gold exists, request latency (mean,p50,p95,p99), escalation, CPU/host RAM/GPU telemetry if available, model declarations, HTTP identity, hashes and integrity checks. It does not measure throughput at concurrency >1 or infer per-model VRAM ownership. Host telemetry can be missing due to device/driver/toolkit constraints.

    python tools/validate_result.py output/unique-run
    python tools/compare_runs.py --standalone output/solo --colocated output/paired

A real contributor submission should use a broad frozen dataset, independent calibration and quality evaluation, enough independent requests for stable tail statistics, and multiple repeats. The provided smoke data are not meaningful for performance claims.

The HTTP /v1/models name check does not cryptographically attest weights; immutability relies on audited deployment/model pins. Publish base image digests and any custom build decisions.

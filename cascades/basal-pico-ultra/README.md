# Basal Omni Pico → Ultra (reference cascade)

This cascade uses **two independent pinned Basal versions**: Pico engine 2.3.2, Ultra engine 2.1.0. Do not silently upgrade either. The models are pinned to the revisions declared in benchmark.json and Compose.

## Requirements

ARM64 NVIDIA GPU with NGC vLLM 26.07 compatible PyTorch/CUDA environment, Docker Compose GPU support, HF snapshot access, enough GPU RAM for Pico and Ultra together. Base image tag in these experimental Dockerfiles has not yet been converted to an immutable digest: **not release-qualified**. Do not claim environment-level reproduction yet.

From the repository root, run one of the two usage modes.

### Benchmark mode

All three arms — fast model only, strong model only, and cascade — use
the same inference service. For a synthetic smoke integration test:

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
docker compose -f cascades/basal-pico-ultra/compose.yaml --profile inference up -d --build fast strong inference
docker compose -f cascades/basal-pico-ultra/compose.yaml --profile inference --profile benchmark run --rm benchmark \
  python benchmarks/smoke-general__basal-pico-ultra__confidence-min-t0p656352/run.py --repeats 2
python tools/validate_result.py output/benchmarks/smoke-general__basal-pico-ultra__confidence-min-t0p656352
~~~

The configured confidence threshold is `0.6563522219657898`. It is
frozen for these benchmark setups, not calibrated on their evaluation
populations. Other datasets require acquisition and setup-specific
adaptation; a new threshold must be documented as a distinct profile.
Use a fresh output directory for every execution. Synthetic smoke
results cannot establish model quality or real-world speed.

For a strong-alone residency control, launch inference with
`--strong-alone` and **only the strong backend**; the standard
Compose stack has both models resident. See
[benchmark mode](../../benchmarks/README.md).

### Inference mode

Serve Pico → Ultra as a single logical System One model:

~~~bash
docker compose -f cascades/basal-pico-ultra/compose.yaml --profile inference up -d --build fast strong inference
curl http://127.0.0.1:18358/v1/models
~~~

The public API does not require a benchmark token.
See [local inference](../../inference/README.md) for the request
contract and optional startup warnings based on matching benchmark
results. Performance and decision quality depend on workload and GPU
configuration; no speedup is assumed.

# Laya → Clef (reference cascade)

Pinned Laya (fast) and Clef-Flash (strong) run in **separate, independently measurable HTTP services**. The fast server uses the audited lossless eligibility checks. Ineligible cases skip Laya execution and escalate. Threshold 0.90 is an illustrative original configuration, **not** a newly calibrated optimum.

## Requirements

ARM64 NVIDIA CUDA system, Docker Compose with GPU support, an authorized immutable NGC CUDA base digest compatible with Dockerfile, access to the pinned Laya/Clef Hugging Face revisions, and enough GPU RAM for both models concurrently. The stack targets DGX Spark-class hardware; a clean GPU deployment has not yet been validated for release.

Pin a compatible CUDA base:

    export CASCADEONE_CUDA_BASE='nvcr.io/...@sha256:YOUR_VERIFIED_DIGEST'

From the repository root, run one of the two usage modes.

### Benchmark mode

All three arms run through the shared inference service: fast model only,
strong model only and the complete cascade. For a smoke integration test:

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
docker compose -f cascades/laya-clef/compose.yaml --profile inference up -d --build fast strong inference
docker compose -f cascades/laya-clef/compose.yaml --profile inference --profile benchmark run --rm benchmark \
  python benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/run.py --repeats 2
python tools/validate_result.py output/benchmarks/smoke-general__laya-clef__confidence-min-t0p900000
~~~

Choose a fresh output directory for subsequent runs. Synthetic smoke
results test integration only. The strong model baseline is the configured
Clef service under **the same residency conditions**. For a genuine
strong-alone residency control, start inference in `--strong-alone` mode
with only the strong backend available and use `--phase strong-alone`;
the standard Compose setup keeps both models resident and is not such
a control. See the [benchmark guide](../../benchmarks/README.md).

### Inference mode

To use Laya → Clef as one logical System One model without running a
benchmark, start the same inference service (the benchmark token is
optional and not needed for public requests):

~~~bash
docker compose -f cascades/laya-clef/compose.yaml --profile inference up -d --build fast strong inference
curl http://127.0.0.1:18358/v1/models
~~~

The configured routing policy and models remain unchanged.
See [local inference](../../inference/README.md) for an API request,
startup comparisons with verified benchmark results, and alert limits.

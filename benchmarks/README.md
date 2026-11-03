# Benchmark mode: evaluate fast, strong and cascade

**Benchmark** measures a configured cascade against two single-model
baselines on the same requests: `fast_only` (fast model), `strong_only`
(strong model) and `cascade` (routing and conditional escalation).
All three arms use the **same System One inference service**. The runner
does not implement a second inference engine.

## Benchmark definitions

Each named benchmark has one directory:

~~~text
benchmarks/<dataset-and-split>__<models>__<strategy>/
  benchmark.json    dataset and split, cascade profile, adapter and selection
  run.py            invokes the common three-arm runner
  README.md         benchmark-specific context
~~~

The strategy name includes the confidence aggregation and shortened
threshold where relevant. The exact threshold, **model revisions and
per-model backend profiles** live in the referenced cascade JSON.
Dataset download and checksum verification belong in `datasets/`;
selection, conversion, normalization and Gold interpretation belong
to the benchmark setup, not the runtime.

**Datasets are model-independent.** The same pinned source can be evaluated
with Laya → Clef and with Basal Pico → Ultra, provided both models can handle
the normalized request types. A *named benchmark setup* is a specific
dataset–adapter–cascade combination; the absence of such a setup for one
model pair does not imply an unavailable or model-specific dataset. The
shared runner also accepts `--cascade`, `--dataset-id` and `--adapter`
directly without a named setup. See the two cascade guides for a paired
example using the same held-out source.

## Run a benchmark

The root-level **`./cascadeone` launcher** is the recommended entry point. It automatically supplies the benchmark-only token to both the inference and benchmark containers, builds the Python runner (including Parquet support), acquires and verifies source files, waits for model readiness and validates results. Run from the `dist/CascadeOne-Community` repository root on a supported NVIDIA ARM64 GPU host.

~~~bash
# Show registered benchmark configurations (ID, dataset ID and optional group); no GPU.
./cascadeone benchmark list --cascade basal-pico-ultra

# Run a three-case smoke benchmark, using an explicit example CUDA base image.
./cascadeone benchmark smoke --cascade basal-pico-ultra \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Run ONE registered dataset/adapter/cascade experiment; --suite is optional.
./cascadeone benchmark evaluate --cascade basal-pico-ultra \
  --benchmark banking77-test__basal-pico-ultra__confidence-min-t0p656352

# OPTIONAL: run every compatible registered benchmark in the external group.
./cascadeone benchmark evaluate --cascade basal-pico-ultra --suite external

# Instead of a named setup, use ANY registered dataset and compatible adapter.
./cascadeone benchmark evaluate --cascade laya-clef \
  --dataset-id bjb-civil-comments-heldout-v1 --adapter better_jev_heldout_gold_v1
~~~

**`--suite` is never required for an individual benchmark.** Named setups are independent from their source datasets: `--benchmark` selects a saved configuration, `--dataset-id` and `--adapter` build a direct experiment without creating a named setup, and `--suite` batches multiple matching named setups for one cascade. Use `./cascadeone --help` for `--output`, `--repeats` and `--cuda-base`. A tag like `nvcr.io/nvidia/vllm:26.07-py3` is not an immutable digest.

The launcher uses the same System One inference HTTP engine for all three arms and leaves it running after a benchmark so further runs can reuse the models. Stop it with `./cascadeone inference stop --cascade basal-pico-ultra` (or the selected cascade name). It does not silently exclude unsupported requests; model eligibility, Gold provenance and possible model pretraining overlap remain important caveats.

The older shell helpers `benchmarks/run_confidence_wide.sh` and `benchmarks/run_confidence_external.sh` remain available for direct Compose workflows. Prefer the root launcher for new interactive use; advanced details are in [dataset acquisition](../datasets/README.md) and [inference internals](../inference/README.md).

## Two timing paths and backend provenance

Each measurement contains:

| Field | Scope |
| --- | --- |
| `latency_ms` | Client-observed HTTP end-to-end: request to complete response |
| `engine_ms` | Inside the shared inference engine: routing plus backend HTTP requests |
| `proxy_overhead_ms` | Difference: `latency_ms - engine_ms`, including client/proxy HTTP handling |
| `fast_ms`, `strong_ms` | Each model backend's HTTP round-trip, when invoked |

Both end-to-end and engine-internal latency have **mean and p95**
summaries for each arm. The backend HTTP times are **not GPU-only
forward-pass measurements**. Any extra backend-level profiling must be
explicitly instrumented, labelled and measured separately.

The run manifest records the cascade configuration SHA-256, declared
model/backend implementations, versions, revisions, execution modes,
container image references, inference service identity and an engine
version (`inference-systemone-v1`). Declared backend metadata is
distinct from fields attested by a running model server. A tag is not
an immutable container digest.

The execution field `systemone_inference_proxy_http` distinguishes new
runs from the former `single_client_sequential_http` path. **Re-run
historical performance benchmarks** before comparing their latency
with the new measurement protocol; do not silently mix the two.
Validated runs include paired input provenance, Gold quality where
available, escalation, host-level telemetry and paired bootstrap
quality/speed comparisons. Telemetry is host-wide, not per-model.

A `strong-alone` residency control is supported by launching the
inference service with `--strong-alone` and running the benchmark with
`--phase strong-alone`. This is an auxiliary single-model measurement,
not a complete three-arm benchmark. It needs a dedicated deployment
with only the strong backend running; the standard Compose inference
profile starts both models and is therefore **not** a strong-alone
control.

See [confidence-wide sources](CONFIDENCE-WIDE.md),
[external sources](EXTERNAL-SOURCES.md) and
[run output format](../results/schema/README.md).
GPU validation of this refactored deployment is still pending.

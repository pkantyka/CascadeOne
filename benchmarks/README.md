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

## Run a benchmark

From the repository root on a compatible NVIDIA ARM64 GPU host, use the
chosen cascade's Docker Compose file and a named matching benchmark.
For Laya → Clef, first configure a verified `CASCADEONE_CUDA_BASE`
as described in [the cascade guide](../cascades/laya-clef/README.md).

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
docker compose -f cascades/laya-clef/compose.yaml --profile inference up -d --build fast strong inference
docker compose -f cascades/laya-clef/compose.yaml --profile inference --profile benchmark run --rm benchmark \
  python benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/run.py --repeats 2
python tools/validate_result.py output/benchmarks/smoke-general__laya-clef__confidence-min-t0p900000
~~~

The same token must be present in the inference and benchmark services;
it authorizes benchmark-only modes, not ordinary `POST /v1/systemone`.
Do not store or publish the token. Each run requires a new output directory
(the runner refuses to overwrite existing measurements).

For Basal Pico → Ultra, use
`cascades/basal-pico-ultra/compose.yaml` and a matching named setup.
The benchmark's runtime URL inside Compose is
`http://inference:18358`. Running a named benchmark locally outside
Compose requires an accessible inference URL and
`CASCADEONE_BENCHMARK_TOKEN`; use the runner's
`--inference-url` option if needed.

List or dry-run suites without starting models:

~~~bash
python benchmarks/run_confidence_wide.py --suite smoke --list
python benchmarks/run_confidence_wide.py --suite wide --dry-run
python benchmarks/run_confidence_wide.py --suite external --dry-run
python benchmarks/run_confidence_wide.py --suite source --dry-run
~~~

These commands validate configurations and available data only; a
`--dry-run` never establishes model speed or quality. For the Basal
Pico → Ultra suites, the ready-to-run shell launchers download and
verify sources, run a dry-run **before starting GPU models**, start
the shared inference service and then run the benchmarks:

~~~bash
sh benchmarks/run_confidence_wide.sh
sh benchmarks/run_confidence_external.sh
~~~

Run them **separately**, from the repository root on compatible GPU
hardware. Each accepts an optional, unique output-root path as its first
argument, for example `sh benchmarks/run_confidence_wide.sh output/my-wide-run`.
By default the launchers create timestamped output paths. They generate
a random benchmark token using `openssl` unless
`CASCADEONE_BENCHMARK_TOKEN` is already set. This token is not written
to the repository. The running model services are left up for reuse.
See [dataset acquisition](../datasets/README.md) for source licenses
and download requirements.

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

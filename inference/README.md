# Local inference: a whole cascade as one System One model

Serve an existing CascadeOne profile through a **single System One HTTP API**.
This is a **proxy/orchestration service**, not a new GPU backend: the fast
model and strong model run in their own backend services. CascadeOne has
two main usage modes: [Benchmark](../benchmarks/README.md) measures three
arms through this engine; **Inference** exposes the cascade as one model.

## Start

From the Community repository root on a compatible NVIDIA ARM64 GPU host, use the `./cascadeone` launcher. The optional `--cuda-base` flag sets the NVIDIA CUDA/PyTorch image for **both** model backends. Otherwise the Compose default is used.

~~~bash
# Start Laya and Clef, wait until the HTTP API is ready.
./cascadeone inference start --cascade laya-clef --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Or start Basal Pico and Ultra (after stopping the other stack).
./cascadeone inference start --cascade basal-pico-ultra

# Inspect the inference service logs for the selected cascade.
./cascadeone inference logs --cascade basal-pico-ultra
~~~

The image shown is an example **tag**, not an immutable digest; for strict reproduction use an approved, compatible image digest. The public endpoint is host-local at `http://127.0.0.1:18358`. Do not run both reference stacks simultaneously: both use that port. See the [Laya → Clef](../cascades/laya-clef/README.md) and [Basal Pico → Ultra](../cascades/basal-pico-ultra/README.md) guides for prerequisites.

The same service can run without Compose if the model-backend URLs in the selected cascade YAML resolve from that process:

~~~bash
# Advanced: launch the Python proxy after starting both backends manually.
# Install PyYAML==6.0.2 and export CASCADEONE_API_TOKEN first.
# CASCADEONE_API_TOKEN must be exported; standalone mode does not generate it.
python inference/server.py --cascade cascades/laya-clef/cascade.yaml --host 127.0.0.1 --port 18358
~~~

The standalone CLI requires **Python 3.12** but no extra API framework packages. The server waits up to 600 seconds for model services and verifies declared backend identity where supported.

## Authentication

The same `CASCADEONE_API_TOKEN` authorizes both `POST /v1/systemone` and benchmark-only requests. **When this environment variable is set, the launcher does not read, create or modify `.cascadeone/token`.** Otherwise it creates the file once (Git-ignored, permissions `0600`) and reuses it across runs. A token must have at least 16 characters. The bearer token is required on every POST; `/health` and `/v1/models` remain unauthenticated for readiness and discovery.

~~~bash
# Optional: use a token from your environment or secret manager; no token file is written.
export CASCADEONE_API_TOKEN="$(openssl rand -hex 32)"

# Launch using that token instead of a local file.
./cascadeone inference start --cascade laya-clef
~~~

The benchmark endpoints `/internal/benchmark/*` additionally require the launcher to start inference with benchmark mode enabled. They return `404` in normal inference mode, even with a valid token. The local reference deployment has no TLS; use trusted local networking only.

## Client API

```bash
# Discover the public logical cascade model.
curl http://127.0.0.1:18358/v1/models

# Ask for a typed decision, not only model discovery.
curl -s http://127.0.0.1:18358/v1/systemone \
  -H "Authorization: Bearer ${CASCADEONE_API_TOKEN:-$(cat .cascadeone/token)}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya-clef-core",
    "state": "Customer needs a password reset",
    "questions": {
      "request_type": {
        "type": "choice",
        "criteria": {
          "account": "Account support",
          "billing": "Billing support"
        }
      }
    }
  }'
```

`POST /v1/systemone` returns the chosen backend's System One `answers`
with the original question IDs and probability distributions, not the
internal fast/strong decision transcript. The cascade model name is the
`id` in the profile; `GET /v1/models` lists this single logical model.
Use `GET /health` to check the local proxy. Optional response headers
`X-CascadeOne-Route`, `X-CascadeOne-Policy` and
`X-CascadeOne-Escalated` expose its route without changing the response
body. Requests are routed according to the profile, including pre-routing.

Use the endpoint only on trusted machines/networks: Bearer authorization is enabled, but there is **no TLS** in the reference server. Compose exposes the port on host `127.0.0.1`; backend services are not exposed through that port.

## Startup benchmark comparison

Supply an **existing completed benchmark run directory** in the
same filesystem visible to the inference service. In Compose, set
`CASCADEONE_BENCHMARK_RUN` when starting, for example:

```bash
# Advanced manual Compose start: export CASCADEONE_API_TOKEN first.
CASCADEONE_BENCHMARK_RUN=output/benchmarks/MY-VERIFIED-RUN \
docker compose -f cascades/laya-clef/compose.yaml --profile inference up -d --build --wait --wait-timeout 660 fast strong inference
```

Or outside Compose:

```bash
python inference/server.py \
  --cascade cascades/laya-clef/cascade.yaml \
  --benchmark output/benchmarks/MY-VERIFIED-RUN
```

`MY-VERIFIED-RUN` is a **placeholder** for an actual run directory,
not a pre-shipped performance result. The startup report uses only a
structurally and checksum-validated **colocated**, three-arm run whose
config file SHA-256, model metadata and policy match the serving
cascade *exactly*. A mismatched or tampered supplied report blocks
startup instead of producing a misleading comparison.

The startup banner compares the cascade with the **strong model**
(`strong_only`) using:
- **Client end-to-end mean latency**, percentage reduction and
  times-faster ratio (negative reduction means the cascade is slower).
- **Engine-internal mean latency**, excluding client-to-proxy HTTP but
  including backend HTTP.
- **Client end-to-end p95 latency** and its ratio to `strong_only`.
- **Gold-based question accuracy** and the difference in percentage
  points, when both arms have Gold.
- Warnings for insufficient mean speedup, excessive quality loss, p95
  regression, small/synthetic samples and workload limitations.

Default **configurable warning thresholds** (pragmatic alert limits, not
claims of statistical significance):

| Metric | Default warning |
| --- | --- |
| Mean latency improvement vs strong-only | Less than 10% |
| Gold question accuracy loss vs strong-only | More than 2 percentage points |
| p95 latency vs strong-only | More than 1.10× |

Change the thresholds with `--min-speedup-pct`,
`--max-quality-loss-pp` and `--max-p95-ratio` when using the
standalone CLI. If no benchmark was supplied, the server starts with
**explicit UNKNOWN speedup and quality warnings**, never invented
numbers.

**Important:** A current benchmark uses the same System One inference
proxy as the public API. Its client-measured end-to-end latency
**includes** the proxy; its separately recorded engine-internal latency
excludes client-to-proxy HTTP overhead but still includes upstream model
HTTP. The service rejects historical results measured by the legacy
direct-backend path as startup evidence. A matching benchmark is not
a performance guarantee for different hardware, workloads or concurrency.
Do not draw representative conclusions from smoke fixtures,
training-source data or host-level telemetry without the corresponding
caveats.

## Unified benchmark execution and timing

The benchmark runner uses the same HTTP inference service for `cascade`, `fast_only` and `strong_only`. Both the public inference endpoint and benchmark-only endpoints use `CASCADEONE_API_TOKEN`; the latter are available only with `CASCADEONE_BENCHMARK_MODE=1` (set by the launcher for benchmark runs). The public `/v1/systemone` endpoint cannot select a baseline.

Every trial records `latency_ms` (client-measured full HTTP end-to-end), `engine_ms` (server-measured routing plus backend HTTP), `proxy_overhead_ms` (their difference), `fast_ms`, and `strong_ms`. Summaries report mean and p95 for both latency paths. **Engine-internal timing includes model-service HTTP overhead and is not a pure CUDA or neural network forward-pass timer.** GPU-only timing requires opt-in, backend-specific instrumentation with CUDA synchronization; it is not collected by this reference proxy.

`--strong-alone` starts a special single-backend service to preserve model-residency controls, and accepts only `strong_only` runs. Legacy benchmark reports remain historical artifacts, but cannot be used as evidence for the new proxy startup comparison. Re-run benchmarks after migration.

The cascade definition now records declared backend implementation, package version when known, source revision, container image tag, execution mode and platform for each model. The benchmark manifest retains these fields and the service reports observed identity fields separately. Image tags are not immutable digests; availability of runtime attestation depends on the upstream model server.

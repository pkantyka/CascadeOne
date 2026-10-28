# Local inference: a whole cascade as one System One model

Serve an existing CascadeOne profile through a **single System One HTTP API**.
This is a **proxy/orchestration service**, not a new GPU engine: fast and strong
models run in their original separately managed backend services.

## Start

Both reference cascades have an optional Compose `inference` profile.
From the repository root on a compatible NVIDIA ARM64 GPU machine:

```bash
docker compose -f cascades/laya-clef/compose.yaml --profile inference up --build fast strong inference
```

For Pico/Ultra use `cascades/basal-pico-ultra/compose.yaml` instead.
The inference endpoint is **host-local** at `http://127.0.0.1:18358`.
Backend model weights remain in their separately started services.
Starting both Compose projects on the same host requires distinct local
ports / project names.

The same service can run without Compose whenever the URLs in the
chosen cascade JSON resolve from that process and both backends are running:

```bash
python inference/server.py --cascade cascades/laya-clef/benchmark.json \
  --host 127.0.0.1 --port 18358
```

This standalone CLI requires **Python 3.12** but no extra API framework packages.
The server waits up to 600 seconds for the model services to come online and
checks their declared identity. A backend that does not expose revision
information cannot provide a revision attestation: the server warns.

## Client API

```bash
curl http://127.0.0.1:18358/v1/models

curl -s http://127.0.0.1:18358/v1/systemone \
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

Only run this endpoint on trusted machines/networks: there is **no
authentication or TLS** in this small reference server. Compose exposes
the host port on `127.0.0.1` only. Backend services are not exposed
through the Compose host port.

## Startup benchmark comparison

Supply an **existing completed benchmark run directory** in the
same filesystem visible to the inference service. In Compose, set
`CASCADEONE_BENCHMARK_RUN` when starting, for example:

```bash
CASCADEONE_BENCHMARK_RUN=output/benchmarks/MY-VERIFIED-RUN \
docker compose -f cascades/laya-clef/compose.yaml --profile inference up --build fast strong inference
```

Or outside Compose:

```bash
python inference/server.py \
  --cascade cascades/laya-clef/benchmark.json \
  --benchmark output/benchmarks/MY-VERIFIED-RUN
```

`MY-VERIFIED-RUN` is a **placeholder** for an actual run directory,
not a pre-shipped performance result. The startup report uses only a
structurally and checksum-validated **colocated**, three-arm run whose
config file SHA-256, model metadata and policy match the serving
cascade *exactly*. A mismatched or tampered supplied report blocks
startup instead of producing a misleading comparison.

The banner displays:
- Cascade and strong-only **mean latency**, percent reduction and
  times-faster ratio; a negative reduction means slower.
- Cascade and strong-only **p95 latency** and their ratio.
- Paired **Gold-based question accuracy** and loss in percentage points
  where both arms have Gold. Missing Gold is explicitly flagged.
- Warnings for underwhelming speedup, excessive quality loss, p95
  regression, very small/synthetic samples, and workload limitations.

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

**Important:** These are historical *sequential benchmark-runner*
measurements on its recorded hardware and requests. In particular,
the extra HTTP proxy introduced here adds overhead **not included**
in the historical benchmark. A valid record does not guarantee that
the deployed facade will achieve the same speedup. Do not draw
representative conclusions from smoke fixtures, training-source
data, or host-level telemetry without the corresponding caveats.

## Unified benchmark execution and timing

The benchmark runner uses this same HTTP inference service for `cascade`, `fast_only` and `strong_only`. Baseline variants require a separate benchmark-only endpoint protected by `CASCADEONE_BENCHMARK_TOKEN`; public `/v1/systemone` cannot select a baseline. Use a randomly generated token of at least 16 characters, available to both the inference service and benchmark runner (for example `openssl rand -hex 32`).

Every trial records `latency_ms` (client-measured full HTTP end-to-end), `engine_ms` (server-measured routing plus backend HTTP), `proxy_overhead_ms` (their difference), `fast_ms`, and `strong_ms`. Summaries report mean and p95 for both latency paths. **Engine-internal timing includes model-service HTTP overhead and is not a pure CUDA or neural network forward-pass timer.** GPU-only timing requires opt-in, backend-specific instrumentation with CUDA synchronization; it is not collected by this reference proxy.

`--strong-alone` starts a special single-backend service to preserve model-residency controls, and accepts only `strong_only` runs. Legacy benchmark reports remain historical artifacts, but cannot be used as evidence for the new proxy startup comparison. Re-run benchmarks after migration.

The cascade definition now records declared backend implementation, package version when known, source revision, container image tag, execution mode and platform for each model. The benchmark manifest retains these fields and the service reports observed identity fields separately. Image tags are not immutable digests; availability of runtime attestation depends on the upstream model server.

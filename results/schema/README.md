# Run output contract

Every run contains manifest.json, environment.json, measurements.jsonl, telemetry.jsonl, summary.json and SHA256SUMS.

The manifest identifies a cascade and its routing policy/parameters, the source dataset's original SHA-256 and identity, the benchmark setup and adapter/selection, plus SHA-256 of the normalized selected requests. A model's confidence is not mandatory. Raw private source inputs are not copied to run output.

Each measurement contains id, repeat, arm, latency_ms, fast_ms, strong_ms, escalated, domain, optional quality and a routing object. The routing object records policy type, reason, used signals if present, and actual fast/strong execution. Confidence-based policies may record confidence; pre-routers and other policies need not.

The summary provides all-arm and per-domain mean and tail latency, quality where available, escalation rates, and resource telemetry. strong_only and cascade are measured on the same normalized requests in colocated mode.

Offline dataset adaptation is part of the benchmark setup but excluded from request latency. Per-request runtime preprocessing/transport and routing overhead are included. Structural validation and hashes do not prove input legality or measurement honesty.

## Unified inference timing (current execution protocol)

New benchmark manifests declare `execution: systemone_inference_proxy_http`
and `engine_version: inference-systemone-v1`. They also preserve both
models' declared backend implementation, version, source revision, image,
mode and observed service identity/attestation where available.

Each trial has:
- `latency_ms` — **client-observed end-to-end** HTTP round-trip to the
  shared inference service;
- `engine_ms` — **engine-internal** time for routing and backend HTTP;
- `proxy_overhead_ms` — difference of those two measurements;
- `fast_ms` and `strong_ms` — HTTP round-trip to each model's backend.

Summaries include mean and p95 for both primary timing paths, plus
the additional proxy overhead. No field claims to measure pure GPU
forward-pass time. All three arms (`fast_only`, `strong_only` and
`cascade`) use the same service; public callers use `POST /v1/systemone`,
while benchmark-only controls are token-protected.

Historical `single_client_sequential_http` runs remain valid historical
artifacts but do **not** measure the same execution path. Do not merge
their latency statistics with results from the current engine.

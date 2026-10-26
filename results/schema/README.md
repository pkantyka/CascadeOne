# Run output contract

Every run contains manifest.json, environment.json, measurements.jsonl, telemetry.jsonl, summary.json and SHA256SUMS.

The manifest identifies a cascade and its routing policy/parameters, the source dataset's original SHA-256 and identity, the benchmark setup and adapter/selection, plus SHA-256 of the normalized selected requests. A model's confidence is not mandatory. Raw private source inputs are not copied to run output.

Each measurement contains id, repeat, arm, latency_ms, fast_ms, strong_ms, escalated, domain, optional quality and a routing object. The routing object records policy type, reason, used signals if present, and actual fast/strong execution. Confidence-based policies may record confidence; pre-routers and other policies need not.

The summary provides all-arm and per-domain mean and tail latency, quality where available, escalation rates, and resource telemetry. strong_only and cascade are measured on the same normalized requests in colocated mode.

Offline dataset adaptation is part of the benchmark setup but excluded from request latency. Per-request runtime preprocessing/transport and routing overhead are included. Structural validation and hashes do not prove input legality or measurement honesty.

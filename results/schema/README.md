# Result format v0.1

Every run contains manifest.json (schema in manifest.schema.json), environment.json, measurements.jsonl, telemetry.jsonl, summary.json and SHA256SUMS. The validator also checks SHA-256, complete arm counts, unique (id, repeat, arm) tuples, and phase-compatible arm names.

Measurements use id, repeat, arm, latency_ms, fast_ms, strong_ms, fast_min_confidence, escalated and quality (null if no Gold; otherwise per-request expected question accuracy, optional hard request-exact, question count).

Summary has schema cascadeone-summary-v0.1, phase, arms and telemetry. All reported mean/p95 comparisons for colocated use the same input IDs; requests and arm order are shuffled with a recorded seed.

Result validation does not prove honest telemetry or proper dataset licensing. Contributor review must check methods and permissions separately.

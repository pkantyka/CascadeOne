# CascadeOne run registry

All contributors — maintainers included — submit results under results/runs/<run-id>/. There is no special "reference" class.

Runs carry manifest.json, environment.json, measurements.jsonl, telemetry.jsonl, summary.json and SHA256SUMS. The manifest identifies dataset mode (bundled/downloadable/local), source dataset ID and exact raw hash, and the separate benchmark setup adapter, selection and normalized request hash. Use tools/validate_result.py before a PR. Results are immutable; revisions require a new run ID and documented supersession.

**Never aggregate runs from different data populations as if they were a controlled comparison.** Domain (e.g., medical vs legal), source, Gold provenance, language, request distribution and difficulty are experimental variables. Within-run strong-only vs cascade requires identical requests and hardware. Comparing separate runs should additionally require matched model and backend versions, dataset hashes, policy, residency, workload protocol and inference execution version. Legacy direct-backend latency is not directly interchangeable with shared-inference-proxy latency.

For private data, submit only non-sensitive aggregates and provenance that the owner permits; identifiers in per-request measurements may themselves be sensitive. Reduced reproduction claims must be clearly labelled.

No validated GPU runs are bundled. A results browser is not part of this release.

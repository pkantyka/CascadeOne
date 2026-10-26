# Dataset contract (Community v0.1)

A UTF-8 JSONL file with one object per line:

    {"id":"unique-1","request":{"state":"...", "questions":{"q":{"type":"choice","criteria":{"A":"...","B":"..."}}}},"gold":{"q":"A"}}

- Request shape follows SystemOne typed decisions (choice, noul, score).
- gold is optional; if supplied, it must be complete for each request. A label string counts top-1 correctness. A full normalized map of option IDs to probabilities yields expected probability assigned by Gold to the predicted top-1 option.
- If Gold is omitted, latency is measurable, **accuracy is not**. No auto-generated pseudo-labels are accepted as independent ground truth.
- Confidence threshold must be frozen on separate calibration data before a held-out performance evaluation. Run records include dataset SHA-256, but **do not** themselves establish train/test disjointness.
- datasets/examples/smoke.jsonl is synthetic, tiny and for integration testing only — never use it to claim real performance or model quality.
- Do not commit sensitive inputs or unlicensed dataset rows. For private evaluations, publish only aggregate summaries and sufficient protocol metadata after reviewing privacy rules.

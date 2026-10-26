# n4ze3m typed decisions — downloadable benchmark candidate

Upstream: n4ze3m/typed-decisions-synth, revision 7b86c5eda358b0d3bf4a45d503675ea1c16cfa00, file data/train.jsonl.
The exact upstream file SHA-256 was captured in Development's frozen 2026-10-17 source manifest. The raw file is **never included in this Git repository**.

Run python datasets/catalog.py prepare n4ze3m-typed-decisions-v1 to fetch and verify. The converter parses state/questions/Gold, retains declared domain, and deterministically partitions by **state fingerprint**, with 20% calibration and 30% test and remaining entries reserved.

Caution: these calibration/test partitions are derived from an upstream **train** split, not an independent official holdout. Model training overlap and original label provenance are not guaranteed. Source license is marked *claimed MIT, pending independent review*. The downloader is not a license bypass; gated/authenticated access and all use restrictions still apply.

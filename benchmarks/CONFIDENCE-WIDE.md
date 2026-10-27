# Wide-source confidence baseline (no G1)

**Research question:** Does a frozen `Pico -> min-confidence -> Ultra` cascade
offer a useful quality-latency tradeoff on a wider System One workload than
the original H1 evaluation? No latent routing, G0, fine-tuning or Gold-assisted
routing is used. The configuration keeps the original H1 threshold
`0.6563522219657898`, chosen on the old calibration split only.

The suite measures, on every *identical* sampled request, three real live
HTTP paths: **Pico-only**, **Ultra-only**, and **Pico -> confidence -> Ultra**.
It records mean/p50/p95/p99, expected Gold question accuracy, escalation,
per-domain results, source SHA-256, selected-input SHA-256 and paired
request-level bootstrap intervals for both quality differences and speedup.
There is no answer cache in timed inference. Run with a single client and
both GPU model services co-resident. Model revisions/engines are frozen in
`cascades/basal-pico-ultra/`.

## Sources

The datasets registry downloads/validates the **exact raw upstream files**.
Adapters/selection belong exclusively to benchmark setups:

| Setup | Raw dataset | Selection |
| --- | --- | --- |
| pico-ultra-wide-procedural-0 | procedural shard 0 | 0.3% by state hash |
| pico-ultra-wide-procedural-1 | procedural shard 1 | 0.3% by state hash |
| pico-ultra-wide-helmo | helmo synthetic train | 10% by state hash |
| pico-ultra-wide-n4ze3m | n4ze3m synthetic train | 15% by state hash |
| pico-ultra-wide-hellaswag | HellaSwag train | 2.5% by state hash |

The hash selection uses seed 20261027 and no model score, Gold or confidence.
Changing any fraction creates a new protocol/setup. All Gold comes from
the *source*, not Pico or Ultra. Hard labels and complete probability
distributions are scored; ambiguous score mean/variance and missing
probability mass are excluded and counted in `normalization_audit`.
The registry downloads raw files; it **never** creates model-ready datasets.

**Methodological limit:** All five files are upstream *train* splits; the
four source families were already represented in the earlier CascadeOne
study, and Pico/Ultra pretraining contamination is not ruled out. This
is a wider within-family replication, **not** a fresh independent
held-out-source or Jev benchmark. We do not claim general OOD validity.
Some labels are synthetic and not independently human-audited.

## GPU execution (from Community repository root)

Run CPU-only checks first:

    python -m unittest discover -s tests -v

On a compatible ARM64 DGX Spark with Docker, NGC images and Hugging Face access:

    docker compose -f cascades/basal-pico-ultra/compose.yaml build benchmark fast strong
    docker compose -f cascades/basal-pico-ultra/compose.yaml run --rm benchmark python datasets/download_wide.py
    docker compose -f cascades/basal-pico-ultra/compose.yaml run --rm benchmark python datasets/download_wide.py --verify
    docker compose -f cascades/basal-pico-ultra/compose.yaml run --rm benchmark python benchmarks/run_confidence_wide.py --dry-run
    docker compose -f cascades/basal-pico-ultra/compose.yaml up -d fast strong

Wait for both `/health` endpoints and model identity. Then:

    docker compose -f cascades/basal-pico-ultra/compose.yaml run --rm benchmark python benchmarks/run_confidence_wide.py --output-root output/confidence-wide-20261027 --repeats 2

Alternatively, `sh benchmarks/run_confidence_wide.sh` builds, downloads,
verifies, starts the models, waits for readiness and runs the complete suite.
The runner refuses to overwrite existing results. If a run fails,
retain its `failure.json` and choose a **new output directory** for rerun.

Use the complete per-source `summary.json` and `suite.json`. Compare
speedup **and** signed quality delta vs Ultra-only; a faster cascade with
materially lower Gold quality is a tradeoff, not an unconditional win.
A lower p95 or strong-only co-residency comparison requires separate
evidence. Thresholds must not be re-fitted against these samples.

For a true external-family test, add distinct pinned sources and audit
their training overlap first; do not relabel this train-source suite
as an independent Jev benchmark.

# Benchmark: dataset/split + model pair + escalation strategy

A benchmark lives in its own clearly named folder:

    benchmarks/<dataset-and-split>__<models>__<strategy>/
      benchmark.json   frozen source, model configuration and strategy
      run.py           runs all three arms
      README.md        documents the benchmark

Every benchmark runs fast_only (small-only), strong_only (large-only)
and cascade (small model with routing and optional escalation) on
identical requests. Baselines are not separate benchmarks.
Output contains latency percentiles, Gold accuracy, escalation,
provenance and paired bootstrap quality/speed comparisons.

The strategy tag includes confidence aggregation and a shortened
threshold. Exact threshold/model revisions and deterministic input
selection are in benchmark.json and its referenced cascade config.

One named benchmark:

    python benchmarks/banking77-test__basal-pico-ultra__confidence-min-t0p656352/run.py --dry-run
    python benchmarks/banking77-test__basal-pico-ultra__confidence-min-t0p656352/run.py --repeats 2

Group launchers reuse the same 19 definitions:

    python benchmarks/run_confidence_wide.py --suite external --dry-run
    python benchmarks/run_confidence_wide.py --suite wide --dry-run
    python benchmarks/run_confidence_wide.py --suite source --dry-run
    python benchmarks/run_confidence_wide.py --suite smoke --dry-run

Download original data first using datasets/ source downloaders.
Selection, conversion and optional deduplication occur only inside
benchmark processing. The shared execution engine is inference/server.py. benchmarks/runner.py is the measurement client.
An auxiliary single-model strong-alone residency control is exposed
by runner.py, but is NOT a complete three-arm benchmark.

See EXTERNAL-SOURCES.md and CONFIDENCE-WIDE.md. GPU validation pending.

The smoke suite contains only synthetic bundled integration inputs (3 rows). The source suite contains full n4ze3m train data (not an independent test).

## Timing and backend provenance

The three arms now use the same System One inference HTTP service, including the public cascade execution path. Set CASCADEONE_BENCHMARK_TOKEN (random 16+ characters) in both the inference service and runner; internal benchmark-only baseline endpoints are token-protected. The runner stores client end-to-end latency_ms, server engine_ms (includes backend HTTP), proxy_overhead_ms, fast_ms and strong_ms for each request, plus mean/p95 summaries. This does not measure pure GPU compute time. The cascade JSON pins the declared backend implementation, source revision, software version and execution mode per model; service identity and missing attestations are reported separately.

The manifest labels the new execution path as systemone_inference_proxy_http and records engine version. Legacy timing reports must not be combined with these runs. Re-run performance experiments to establish updated results.

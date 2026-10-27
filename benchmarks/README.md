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
    python benchmarks/run_confidence_wide.py --suite smoke --dry-run

Download original data first using datasets/ source downloaders.
Selection, conversion and optional deduplication occur only inside
benchmark processing. The shared engine is benchmarks/runner.py.
An auxiliary single-model strong-alone residency control is exposed
by runner.py, but is NOT a complete three-arm benchmark.

See EXTERNAL-SOURCES.md and CONFIDENCE-WIDE.md. GPU validation pending.

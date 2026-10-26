# Benchmark setup owns data adaptation and routing configuration

A benchmark setup specifies a cascade, source dataset, versioned adapter and optional selection rule. A dataset definition stores only source identity and delivery mode.

    python datasets/catalog.py list
    python datasets/catalog.py acquire n4ze3m-typed-decisions-v1

    python benchmarks/runner.py \
      --setup benchmarks/setups/laya-clef-smoke.json \
      --output output/laya-smoke

    python benchmarks/runner.py \
      --setup benchmarks/setups/pico-ultra-n4ze3m.json \
      --output output/pico-source-run

The examples above are different workloads; their absolute quality/latency should not be combined as controlled comparisons. Source-to-request conversion and Gold interpretation reside in benchmarks/adapters.py; benchmark setup JSON selects the adapter. selection.type=all is the default. Optional hash_range selection is explicit, reproducible and grouped by state, not silently imposed by dataset acquisition.

A custom local typed-decision JSONL can still be evaluated directly:

    python benchmarks/runner.py --cascade cascades/laya-clef/benchmark.json \
      --dataset /path/to/your.jsonl --output output/local-run

Routing is part of the **cascade runtime** (cascades/runtime.py), using the configured policy (cascades/policies.py). The benchmark runner has no confidence threshold logic. Built-in policies include confidence_threshold, always_fast, always_strong and max_questions_pre_router; a trusted module:factory plugin may implement other policies. Any policy must decide before/after fast inference and emit a route decision, which is logged for every request.

In colocated mode, both servers are resident and requests are measured for strong_only and cascade. In strong-alone mode, only strong is resident. Repeats and arm order are randomized and the client is sequential. The runner measures HTTP wall time and request-time routing overhead, not offline dataset conversion or dataset download. Reports contain mean/p50/p95/p99, Gold where present, escalation, routing metadata, source and normalized input SHA-256, and host resource telemetry if available.

    python tools/validate_result.py output/laya-smoke
    python tools/compare_runs.py --standalone output/solo --colocated output/paired

The smoke data are not a real performance benchmark. No concurrency/energy or per-process VRAM accuracy is claimed in this preview.

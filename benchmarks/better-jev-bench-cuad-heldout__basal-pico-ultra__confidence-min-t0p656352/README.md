# better-jev-bench-cuad-heldout__basal-pico-ultra__confidence-min-t0p656352

One benchmark = source/split + model pair + routing policy.

- Original dataset ID: bjb-cuad-heldout-v1.
- Cascade configuration: cascades/basal-pico-ultra/benchmark.json.
- Routing policy: confidence-min-t0p656352.
- Dataset selection: {"type":"all"}.
- Every execution compares fast_only, strong_only and cascade on identical requests.

From Community root with pinned inference servers running:

    python benchmarks/better-jev-bench-cuad-heldout__basal-pico-ultra__confidence-min-t0p656352/run.py --dry-run
    python benchmarks/better-jev-bench-cuad-heldout__basal-pico-ultra__confidence-min-t0p656352/run.py --repeats 2

All source files are independently downloaded through datasets/sources/.

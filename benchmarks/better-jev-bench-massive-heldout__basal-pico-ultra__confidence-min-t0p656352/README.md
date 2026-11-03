# better-jev-bench-massive-heldout__basal-pico-ultra__confidence-min-t0p656352

One benchmark = source/split + model pair + routing policy.

- Original dataset ID: bjb-massive-heldout-v1.
- Cascade configuration: cascades/basal-pico-ultra/cascade.yaml.
- Routing policy: confidence-min-t0p656352.
- Dataset selection: {"type":"all"}.
- Every execution compares fast_only, strong_only and cascade on identical requests.

From the repository root with pinned inference servers running:

    python benchmarks/better-jev-bench-massive-heldout__basal-pico-ultra__confidence-min-t0p656352/run.py --dry-run
    python benchmarks/better-jev-bench-massive-heldout__basal-pico-ultra__confidence-min-t0p656352/run.py --repeats 2

Source acquisition and verification are documented in datasets/README.md. Synthetic smoke data is bundled; other sources are pinned.

# helmo-synthetic-train-hash000000-100000-seed20261027__basal-pico-ultra__confidence-min-t0p656352

One benchmark = source/split + model pair + routing policy.

- Original dataset ID: helmo-synthetic-v1.
- Cascade configuration: cascades/basal-pico-ultra/benchmark.json.
- Routing policy: confidence-min-t0p656352.
- Dataset selection: {"type":"hash_range","start":0.0,"end":0.1,"seed":20261027}.
- Every execution compares fast_only, strong_only and cascade on identical requests.

From the repository root with pinned inference servers running:

    python benchmarks/helmo-synthetic-train-hash000000-100000-seed20261027__basal-pico-ultra__confidence-min-t0p656352/run.py --dry-run
    python benchmarks/helmo-synthetic-train-hash000000-100000-seed20261027__basal-pico-ultra__confidence-min-t0p656352/run.py --repeats 2

Source acquisition and verification are documented in datasets/README.md. Synthetic smoke data is bundled; other sources are pinned.

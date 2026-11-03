# smoke-general__laya-clef__pre-router-max-questions-1

One benchmark = source/split + model pair + routing policy.

- Original dataset ID: smoke-general-v1.
- Cascade configuration: cascades/laya-clef/cascade-pre-router.yaml.
- Routing policy: pre-router-max-questions-1.
- Dataset selection: {"type":"all"}.
- Every execution compares fast_only, strong_only and cascade on identical requests.

From the repository root with pinned inference servers running:

    python benchmarks/smoke-general__laya-clef__pre-router-max-questions-1/run.py --dry-run
    python benchmarks/smoke-general__laya-clef__pre-router-max-questions-1/run.py --repeats 2

Source acquisition and verification are documented in datasets/README.md. Synthetic smoke data is bundled; other sources are pinned.

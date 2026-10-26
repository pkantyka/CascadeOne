# CascadeOne Community

**Engineering preview** of conditional cascades for typed decisions. Two reference pairs: **Laya → Clef** and **Basal Pico → Ultra**. This is not yet a GPU-validated public release.

## Quick start

Run CPU-only contract tests:

```bash
python -m unittest discover -s tests -v
```

List and verify source datasets (bundled or pinned/downloadable):

```bash
python datasets/catalog.py list
python datasets/catalog.py verify smoke-general-v1
python datasets/catalog.py acquire n4ze3m-typed-decisions-v1
```

On a compatible ARM64 NVIDIA GPU host, follow `cascades/laya-clef/README.md` or `cascades/basal-pico-ultra/README.md` to start model services. Then run a benchmark using one explicit setup:

```bash
python benchmarks/runner.py --setup benchmarks/setups/laya-clef-smoke.json --output output/laya-smoke
python tools/validate_result.py output/laya-smoke
```

The bundled smoke dataset only tests integration. No production-quality or performance claim follows from it.

## One responsibility per component

- `datasets/`: original source acquisition and SHA-256 verification, **no implicit preparation or splits**.
- `benchmarks/setups/` and `benchmarks/adapters.py`: which cascade/data to test and how to interpret source and Gold.
- `cascades/`: model configuration, policy and execution; **confidence is optional**.
- `benchmarks/runner.py`: strong-only versus cascade measurements and provenance; no routing policy.
- `results/runs/`: identical submission format for maintainers and contributors.

For details, see `benchmarks/README.md`, `datasets/README.md`, `CONTRIBUTING.md` and `docs/`. Historic measurements are contextual evidence in `docs/development-evidence/`, not new reference runs.

**Release gates:** GPU/Compose installation and live measurements, model/source licensing and third-party notices are not yet approved; see `docs/RELEASE-READINESS.md`. No website, model weights or redistributable third-party raw datasets are included.

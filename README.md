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

On a compatible ARM64 NVIDIA GPU host, follow `cascades/laya-clef/README.md` or `cascades/basal-pico-ultra/README.md` to start model services. Then run one named benchmark with its three mandatory arms:

```bash
python benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/run.py
python tools/validate_result.py output/benchmarks/smoke-general__laya-clef__confidence-min-t0p900000
```

The bundled smoke dataset only tests integration. No production-quality or performance claim follows from it.

## Repository lifecycle

The Development script `publishing/build_community.py` was used only to **bootstrap** this repository. Community now has its own Git history and is maintained directly; it is **not** rebuilt or synchronized from Development automatically. A new build from that script is not a replacement for this checkout.

The original distribution-manifest.json is a historical bootstrap snapshot, not the current list of benchmark configurations. The current Git revision and per-benchmark SHA-256 manifests identify reproducible execution inputs.

## One responsibility per component

- `datasets/`: original source acquisition and SHA-256 verification, **no implicit preparation or splits**.
- `benchmarks/<dataset>__<models>__<strategy>/`: one complete three-arm benchmark per dataset, model pair and routing strategy.
- `cascades/`: model configuration, policy and execution; **confidence is optional**.
- `benchmarks/runner.py`: small-only, large-only and cascade measurements and provenance; routing is separate.
- `results/runs/`: identical submission format for maintainers and contributors.

For details, see `benchmarks/README.md`, `datasets/README.md`, `CONTRIBUTING.md` and `docs/`. Historic measurements are contextual evidence in `docs/development-evidence/`, not new reference runs.

**Release gates:** GPU/Compose installation and live measurements, model/source licensing and third-party notices are not yet approved; see `docs/RELEASE-READINESS.md`. No website, model weights or redistributable third-party raw datasets are included.

## Wide-source confidence benchmark

To compare **Pico-only, Ultra-only and frozen-confidence Pico -> Ultra**
on multiple pinned original-source datasets, see
[the wide benchmark protocol](benchmarks/CONFIDENCE-WIDE.md).
No G1 model or retraining is involved.

## External confidence benchmark

Download/test BANKING77, CLINC150, and pinned better-jev-bench held-out
sources using [external source protocol](benchmarks/EXTERNAL-SOURCES.md).
No G1 or retraining is required.

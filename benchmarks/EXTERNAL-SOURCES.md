# External System One benchmark sources (confidence H1, no G1)

This suite tests the **unchanged** Pico -> minimum-confidence -> Ultra
cascade with frozen threshold 0.6563522219657898, against Pico-only
and Ultra-only. Unlike the earlier four-family train suite, these are
original public **test** files or the pinned better-jev-bench **heldout**
collection. Do not tune thresholds on these results.

## Frozen raw sources

Each downloadable source has an immutable 40-character revision and the
SHA-256 of the **exact original bytes** in `datasets/registry.json`.
Files are cached under `datasets/cache/raw/<dataset_id>/`, never committed.

| Dataset ID | Original file | Source split / options |
| --- | --- | --- |
| banking77-test-v1 | PolyAI/banking77 | original test, 3080 rows, 77 intents |
| clinc150-plus-test-v1 | clinc/clinc_oos | plus/test, 5500 rows, 151 options including OOS |
| bjb-banking77-heldout-v1 | better-jev-bench | heldout curated mixture, 2000 rows |
| bjb-clinc150-heldout-v1 | better-jev-bench | heldout curated mixture, 2000 rows |
| bjb-cfpb-heldout-v1 | better-jev-bench | CFPB product routing, 2000 rows |
| bjb-cuad-heldout-v1 | better-jev-bench | CUAD contract classification, 3174 rows |
| bjb-civil-comments-heldout-v1 | better-jev-bench | Civil Comments toxicity decisions, 4000 rows |

The two original-source datasets use the ClassLabel names embedded in their
Parquet metadata and full original taxonomies. The BJB adapter translates
all original options and the source Gold from its JSONL.GZ requests.
Adapters live in `benchmarks/adapters.py`: the dataset catalog only
acquires/verifies exact raw files. PyArrow 21.0.0 is needed **only** to
normalize Parquet benchmark sources.

**Important nonindependence:** better-jev-bench includes original source
train/validation/test examples and **overlaps** the standalone BANKING77
and CLINC150 test files. In a local check there were 468 normalized
BANKING77 matching texts and 479 CLINC150 test matching texts.
Each setup must be reported **separately**. Never pool these seven
runs as independent requests, and do not sum the source-family metrics
as seven independent benchmark families.

Training exposure of the frozen Pico/Ultra checkpoints to these public
datasets is unknown. Dataset licensing and public redistributability
are separate from download permission; raw files stay in cache.

## Local setup and verification

Run from the Community repository root (Python 3.12):

    python datasets/download_external.py --list
    python datasets/download_external.py
    python datasets/download_external.py --verify
    python -m unittest discover -s tests -v

A single source can be acquired without other downloads:

    python datasets/download_external.py --dataset-id banking77-test-v1

CPU dataset preflight (does not invoke any inference):

    python benchmarks/run_confidence_wide.py --suite external --dry-run

One setup (both Pico and Ultra servers must be running):

    python benchmarks/run_confidence_wide.py --suite external --setup-id pico-ultra-external-banking77 --output-root output/external-banking77-20261027

To run on ARM64 NVIDIA DGX Spark with Docker Compose, from Community root:

    sh benchmarks/run_confidence_external.sh

This builds the existing model containers, downloads/verifies sources,
starts the models, runs preflight, and executes all external setups.
Results are under `output/confidence-external-20261027/`, with one
manifest/summary/paired bootstrap CI per setup and a suite index.

**Model limitation:** The original choice width is never silently
truncated. BANKING77 has 77 options and CLINC150 has 151. If the
pinned Basal service cannot handle a request, the run must fail visibly
rather than silently turning the benchmark into a smaller task.
High-width model-serving support is **not** GPU-validated here.

Do not interpret a positive speedup alone as a win. Compare quality
loss/gain to Ultra-only and p95, and keep uncertainty and per-domain
differences visible.

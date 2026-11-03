# External benchmark sources and named cascade comparisons

The raw data and benchmark definitions are separate. Raw download
always preserves all original records; no filtering or deduplication.

| Source download directory | Pinned original files |
| --- | ---: |
| datasets/sources/banking77/ | 2 (train/test) |
| datasets/sources/clinc150/ | 9 (three variants x three splits) |
| datasets/sources/better-jev-bench/ | 11 (all held-out task files) |

    python datasets/download_external.py
    python datasets/download_external.py --verify

Exact upstream revisions and SHA-256 are listed in datasets/registry.json.
Downloaded raw files stay in ignored datasets/cache/raw/; no datasets
are included in Git.

There are ten named text benchmarks for the frozen Pico/Ultra models
and the pinned minimum-confidence strategy. Every one runs Pico-only,
Ultra-only and the full Pico -> confidence -> Ultra cascade.

Examples:

    python benchmarks/banking77-test__basal-pico-ultra__confidence-min-t0p656352/run.py --dry-run
    python benchmarks/clinc150-plus-test__basal-pico-ultra__confidence-min-t0p656352/run.py --dry-run

To run all external benchmarks on a compatible GPU host, use
the updated launcher from the repository root:

    sh benchmarks/run_confidence_external.sh

Optionally pass a unique output root:

    sh benchmarks/run_confidence_external.sh output/my-external-run

The launcher downloads and verifies original sources, performs the
dry-run before starting GPU models, then starts fast, strong and the
shared inference service and executes all three-arm benchmarks. It
generates a temporary `CASCADEONE_API_TOKEN` using `openssl` unless it is already supplied by the environment. The recommended `./cascadeone` launcher uses `.cascadeone/token` when the environment variable is absent. See the
[benchmark guide](README.md) for manual execution.

All eleven better-jev-bench held-out JSONL files are downloadable.
Three are multimodal (Atari Head, OS-Atlas, ScreenSpot v2), and their
image payloads reside in external archives; text-only benchmarking of
these tasks would be invalid, so no inference setup was invented.

BANKING77/CLINC150 source records overlap versions in better-jev-bench.
Preserve source duplicates; any deduplication must happen within a
specific benchmark evaluation. Pretraining overlap is not yet ruled out.
GPU serving with 77/151 choices remains unverified.

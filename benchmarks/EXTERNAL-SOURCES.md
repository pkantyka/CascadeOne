# External confidence benchmark: complete original sources

Raw source collections and benchmarks are independent. Acquiring a source
never filters, samples, re-labels or removes duplicate rows. Benchmarks
may select original splits or transform raw requests to System One.

## Complete source access

All source files are pinned to immutable revisions and SHA-256 in
datasets/registry.json, and grouped in immutable manifests.

| Source directory | Files | Scope |
| --- | ---: | --- |
| datasets/sources/banking77/ | 2 | original train + test |
| datasets/sources/clinc150/ | 9 | plus/small/imbalanced x train/validation/test |
| datasets/sources/better-jev-bench/ | 11 | every published bench/heldout/*.jsonl.gz |

Run from Community root with Python 3.12:

    python datasets/sources/banking77/download.py
    python datasets/sources/clinc150/download.py
    python datasets/sources/better-jev-bench/download.py

Alternatively, python datasets/download_external.py downloads all three
complete sources. The --verify option only checks existing original
bytes, and --list describes all source members. No raw source data is
committed: files download on demand into gitignored datasets/cache/raw/.

To fetch a single original file rather than a complete source:

    python datasets/download_external.py --dataset-id banking77-test-v1

The dataset acquisition process never deduplicates or converts records.

**Multimodal limit:** Atari Head, OS-Atlas and ScreenSpot v2 held-out
JSONL.GZ datasets refer to externally stored images by URI and SHA-256.
Their complete original source records are downloadable, but the actual
image binaries are not included in this source snapshot. Their complete
multimodal use requires additional image retrieval and validation.

## Separate benchmark directories

| Benchmark directory | Default evaluated inputs |
| --- | --- |
| benchmarks/banking77/ | original BANKING77 test (3080 requests, 77 choices) |
| benchmarks/clinc150/ | original CLINC150 plus/test (5500 requests, 151 choices including OOS) |
| benchmarks/better-jev-bench/ | eight individually selected text-only BJB held-out tasks |

Each benchmark has its own script and setup; BJB has one setup per text
task. All eleven BJB held-out source files remain available unchanged,
including three multimodal tasks not executed by the text-only runner.

    sh benchmarks/banking77/run.sh --dry-run
    sh benchmarks/clinc150/run.sh --dry-run
    python benchmarks/better-jev-bench/run.py --dry-run

To run GPU inference (on supported architecture):

    sh benchmarks/run_confidence_external.sh

It installs required container images, downloads sources, checks the
normalization, starts Pico and Ultra and evaluates all text benchmarks.
The shared H1 runner compares Pico-only, Ultra-only and frozen
confidence gating (threshold 0.6563522219657898). No G1 or training.

BJB BANKING77 and CLINC150 overlap source test sets and include train
examples. Preserve all original data. If needed, a particular benchmark
may deduplicate its evaluation subset without touching source files.

Actual GPU service operation with 77/151 option questions has not yet
been validated. The server must reject unsupported choice widths, not
silently truncate them. Training-data leakage with original model
pretraining is unknown.

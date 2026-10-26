# CascadeOne datasets: two delivery modes

Every registered dataset has a unique ID, a versioned registry entry, source/provenance/license details and pinned file hashes. It is equally valid to test on general, medical, legal or other workloads. Results from different datasets are not interchangeable.

## A. Download & Prepare (source data stays outside the Git repo)

    python datasets/catalog.py list
    python datasets/catalog.py prepare n4ze3m-typed-decisions-v1
    python datasets/catalog.py verify n4ze3m-typed-decisions-v1
    python datasets/catalog.py path n4ze3m-typed-decisions-v1 --split test

The downloader contacts the **specific pinned Hugging Face dataset revision**, verifies the original raw-file SHA-256, transforms request fields and Gold using the declared adapter, and writes deterministic calibration/test partitions and their hashes to datasets/cache/prepared/. The original bytes are cached in datasets/cache/raw/. Both directories are ignored by Git and Docker builds.

The source is not shipped with this repo. Access may require HF_TOKEN or acceptance of upstream terms. Scripts do **not** override usage restrictions or grant redistribution rights. If a publisher changes a pinned file, preparation fails closed. A stale/corrupt local cache also fails closed, never silently overwrites.

The current n4ze3m example is an *experimental benchmark candidate* derived from an upstream TRAIN split. It has genuine source labels but **is not a certified independent benchmark**: model pretraining overlap, original Gold provenance and final licensing need review. Fixed state-level partition: 20% calibration, 30% test, 50% reserved, from SHA-256 seed 20261026. The converter does not consume predicted confidences to assign splits.

## B. Bundled Dataset (actual dataset files in this Git repo)

    python datasets/catalog.py verify smoke-general-v1
    python datasets/catalog.py path smoke-general-v1 --split test

The synthetic project-authored smoke sample lives in datasets/bundled/smoke/{calibration,test}.jsonl; registry.json pins row counts and checksums. Preparation for bundled data performs integrity verification only; it never downloads. This tiny sample tests integration, not production quality or performance.

**Bundling third-party data requires a verified right to redistribute the exact transformed files** (attribution, license notices and derivative terms as applicable). Open access on Hugging Face is not sufficient.

## Benchmark with a registered dataset

    python benchmarks/runner.py \
      --cascade cascades/laya-clef/benchmark.json \
      --dataset-id n4ze3m-typed-decisions-v1 \
      --dataset-split test \
      --output output/my-unique-run

The runner verifies the registered dataset before inference and writes dataset ID, delivery mode, dataset manifest hash, split, dataset-file hash and domain to its result manifest. A custom local/private dataset can still be used with --dataset /path/file.jsonl; it is labelled local_unregistered, is never uploaded automatically, and still records a file SHA-256.

## Prepared JSONL contract

Each row must carry a stable id, typed decision request and optionally complete independent gold labels. Optional domain supports per-domain workload analysis. Supported request question types: choice, noul, score; Gold is a hard option ID ("true"/"false" for noul, decimal index for score) or a normalized categorical distribution in the runner. All inputs are UTF-8 JSONL.

Use only calibration partitions to tune a routing threshold. Do not use the same rows for training, calibrating and final testing, and do not regard synthetic teacher labels as independent source Gold. Do not upload sensitive inputs or unpublished dataset rows in a PR.

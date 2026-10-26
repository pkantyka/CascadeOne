# n4ze3m — immutable downloadable source

Hugging Face repository n4ze3m/typed-decisions-synth pinned to commit 7b86c5eda358b0d3bf4a45d503675ea1c16cfa00 and file data/train.jsonl. Raw SHA-256 is in datasets/registry.json. The source is never redistributed in this repository.

    python datasets/catalog.py acquire n4ze3m-typed-decisions-v1

No conversion, Gold transformation or train/test/calibration partition takes place during acquisition.

Example benchmark setup: benchmarks/setups/pico-ultra-n4ze3m.json selects n4ze3m_gold_v1, mapping booleans to true/false option IDs and numeric score labels to string option IDs, and keeps all source rows. A different cascade setup can select another adapter or explicit subset without editing this dataset.

The upstream input was a TRAIN split. Rights to redistribute, potential model training exposure and the provenance of source labels require independent review before describing results as an unbiased benchmark.

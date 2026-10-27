#!/bin/sh
# GPU execution from Community root. Data acquisition never trains models.
set -eu
COMPOSE=cascades/basal-pico-ultra/compose.yaml
DEST=${1:-output/confidence-external-20261027}
docker compose -f "$COMPOSE" build fast strong benchmark
docker compose -f "$COMPOSE" run --rm benchmark python datasets/download_external.py
docker compose -f "$COMPOSE" run --rm benchmark python datasets/download_external.py --verify
docker compose -f "$COMPOSE" run --rm benchmark python benchmarks/run_confidence_wide.py --suite external --dry-run
docker compose -f "$COMPOSE" up -d fast strong
docker compose -f "$COMPOSE" run --rm benchmark python benchmarks/wait_models.py
docker compose -f "$COMPOSE" run --rm benchmark python benchmarks/run_confidence_wide.py --suite external --output-root "$DEST" --repeats 2

#!/bin/sh
# Run from Community root on compatible ARM64 NVIDIA GPU host.
set -eu
COMPOSE=cascades/basal-pico-ultra/compose.yaml
DEST=${1:-output/confidence-wide-20261027}
docker compose -f "$COMPOSE" build fast strong benchmark
docker compose -f "$COMPOSE" run --rm benchmark python datasets/download_wide.py
docker compose -f "$COMPOSE" run --rm benchmark python datasets/download_wide.py --verify
docker compose -f "$COMPOSE" run --rm benchmark python benchmarks/run_confidence_wide.py --dry-run
docker compose -f "$COMPOSE" up -d fast strong
docker compose -f "$COMPOSE" run --rm benchmark python benchmarks/wait_models.py
docker compose -f "$COMPOSE" run --rm benchmark python benchmarks/run_confidence_wide.py --output-root "$DEST" --repeats 2

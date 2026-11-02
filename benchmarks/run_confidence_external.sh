#!/bin/sh
# Run from the CascadeOne Community repository root on an ARM64 NVIDIA GPU host.
set -eu
COMPOSE=cascades/basal-pico-ultra/compose.yaml
DEST=${1:-output/confidence-external-$(date -u +%Y%m%dT%H%M%SZ)-$$}
if [ ! -f "$COMPOSE" ]; then
  echo "Run this script from the CascadeOne Community repository root." >&2
  exit 2
fi
if [ -e "$DEST" ]; then
  echo "Output path already exists: $DEST" >&2
  exit 2
fi
if [ -z "${CASCADEONE_BENCHMARK_TOKEN:-}" ]; then
  if ! command -v openssl >/dev/null 2>&1; then
    echo "Set CASCADEONE_BENCHMARK_TOKEN (16+ chars) or install openssl." >&2
    exit 2
  fi
  CASCADEONE_BENCHMARK_TOKEN=$(openssl rand -hex 32)
fi
if [ "${#CASCADEONE_BENCHMARK_TOKEN}" -lt 16 ]; then
  echo "CASCADEONE_BENCHMARK_TOKEN must be at least 16 characters." >&2
  exit 2
fi
export CASCADEONE_BENCHMARK_TOKEN

# Build the data/benchmark image without starting either GPU model.
docker compose -f "$COMPOSE" --profile inference --profile benchmark build benchmark
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark python datasets/download_external.py
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark python datasets/download_external.py --verify
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark python benchmarks/run_confidence_wide.py --suite external --dry-run

# Bring up the SAME inference service used by applications and benchmarks.
docker compose -f "$COMPOSE" --profile inference up -d --build --wait --wait-timeout 660 fast strong inference
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark python benchmarks/wait_models.py --inference-url http://inference:18358
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark python benchmarks/run_confidence_wide.py --suite external --output-root "$DEST" --repeats 2
echo "Completed benchmark suite in $DEST"

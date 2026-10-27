#!/bin/sh
# Invoke from Community repository root; model servers must already be running.
set -eu
exec python benchmarks/run_confidence_wide.py --suite external --setup-id pico-ultra-external-clinc150 --output-root output/benchmarks/clinc150 "$@"

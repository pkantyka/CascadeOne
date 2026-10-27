# Within-family confidence benchmark group

Five named benchmark definitions compare Pico-only, Ultra-only and
the Pico -> minimum-confidence -> Ultra cascade using frozen
threshold 0.6563522219657898.

| Benchmark directory | Original source | Selected fraction |
| --- | --- | ---: |
| hellaswag-train-hash000000-025000-seed20261027__basal-pico-ultra__confidence-min-t0p656352 | hellaswag-train-v1 | 2.5% |
| helmo-synthetic-train-hash000000-100000-seed20261027__basal-pico-ultra__confidence-min-t0p656352 | helmo-synthetic-v1 | 10.0% |
| n4ze3m-typed-decisions-train-hash000000-150000-seed20261027__basal-pico-ultra__confidence-min-t0p656352 | n4ze3m-typed-decisions-v1 | 15.0% |
| procedural-typed-train-shard0-hash000000-003000-seed20261027__basal-pico-ultra__confidence-min-t0p656352 | procedural-typed-0-v1 | 0.3% |
| procedural-typed-train-shard1-hash000000-003000-seed20261027__basal-pico-ultra__confidence-min-t0p656352 | procedural-typed-1-v1 | 0.3% |

All original upstream source files remain complete; the hash sample
with seed 20261027 is selected only for these benchmark evaluations.
These are within-family upstream train datasets, not independent OOD.

    python datasets/download_wide.py
    python benchmarks/run_confidence_wide.py --suite wide --dry-run
    sh benchmarks/run_confidence_wide.sh

Each benchmark records the three arms with paired uncertainty estimates.

# CascadeOne Community — engineering preview v0.1

CascadeOne is an engineering workbench for conditional cascades of **typed decision models**. It is not tied to Basal, Laya, or any particular inference engine.

## Included

- **Laya → Clef**: pinned CUDA backends, confidence gating, bypass for ineligible Laya requests.
- **Basal Omni Pico → Ultra**: pinned model revisions, independent Basal 2.3.2 (Pico) and 2.1.0 (Ultra) services, confidence gating.
- **One benchmarking protocol**: real HTTP strong-only baseline and cascade on the same request set, sequential randomized orders, latency, Gold evaluation where supplied, best-effort host telemetry, immutable output files.
- CPU-only unit/contract tests, run validation, comparisons, architecture/hypothesis documents.
- Historical experiment conclusions in docs/development-evidence, **not** re-labeled as newly reproduced Community runs.

## Status and limits

This is a **first engineering preview, not a verified public release**. GPU container builds, model downloads, live reference runs and independent installation have not yet been exercised from this distributed tree. CPU-only tests validate the tooling. No empirical performance advantage is claimed for v0.1.

No model weights or third-party dataset rows are redistributed. Check docs/RELEASE-READINESS.md before publishing publicly; the project license and dependencies need explicit approval.

## Run CPU-only tests

    python -m unittest discover -s tests -v

## Run a cascade

Both configurations currently target an ARM64 NVIDIA GPU (DGX Spark / NGC 26.07 family). See the cascade READMEs for machine-specific prerequisites, image pinning, and commands:

- cascades/laya-clef/README.md
- cascades/basal-pico-ultra/README.md

On a suitable host, start the model services with Compose and run the benchmark container. The included datasets/examples/smoke.jsonl checks the end-to-end path only; it **is not a representative quality or latency benchmark**. For performance submissions, use a publicly shareable frozen dataset with documented licensing, IDs and SHA-256.

## Run artifacts

Each run creates a new directory with manifest.json, environment.json, measurements.jsonl, telemetry.jsonl, summary.json and SHA256SUMS. Run IDs must be unique. For example:

    python tools/validate_result.py output/my-run
    python tools/compare_runs.py --standalone output/my-solo --colocated output/my-paired

**Never compare stand-alone GPU memory with colocated latency without labelling the phase.** Host-level telemetry includes other processes and does not establish per-model resource consumption.

## Contribute

See CONTRIBUTING.md. All community runs use the same format; original maintainers receive no special status. Results are only comparable within a declared dataset/protocol/model/precision envelope. No web application is included in this release.

# Contributing to CascadeOne

Contribute a new cascade definition, a benchmark improvement, a useful failure report, or a measured run.

## Submitting a run

1. Use a documented cascade and a frozen dataset. Select a bundled dataset only when redistribution is permitted, or use a pinned downloadable source with rights/terms respected. A private local dataset must never be committed with its raw rows.
2. Record actual model revisions, runtime commits, inference settings, CPU/GPU and driver versions, precision, and model residency.
3. Execute fast-only, strong-only and cascade with the **same input requests and repetitions**, and store all raw measurements. No answer-cache timing or proxy/summed offline latencies.
4. Validate the generated directory: python tools/validate_result.py PATH.
5. Submit the run directory in results/runs/ with a PR. In the PR describe hardware, any experimental changes, and limitations. Do not upload private prompts, private telemetry, model weights, access tokens, or third-party data without permission.

No contributor needs to reproduce the maintainer's latency. Different hardware may legitimately yield different conclusions. Report failures as well as improvements.

## Acceptance criteria

A run must pass the result validator, declare model/config and dataset hashes, include all three comparison arms, be clearly labelled by phase, and be free of unlicensed raw data. The first runner measures single-client sequential HTTP only. Do not claim concurrency performance, statistical p95 significance, or per-process GPU memory from these reports.

An approved PR is not a scientific endorsement. The validator checks structural integrity; reviewers assess licensing and methodology.

## Code contribution

Keep backends and router logic separate; do not hardcode vendor-specific behavior in the generic runner. Include deterministic CPU unit tests for routing, errors and report schemas. Additional GPU cases require opt-in tests and must not be CI requirements for ordinary contributors.

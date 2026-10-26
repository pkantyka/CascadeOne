# Source datasets are independent of cascade benchmark setups

A source dataset is not a benchmark setup. The source registry does not map labels, construct System One requests, select questions, filter domains, tune routing or create calibration/test splits. Those decisions belong to a versioned benchmark setup under benchmarks/setups/ and its named adapter.

Two delivery modes share the same contract:

- **downloadable**: a pinned raw upstream file is acquired on the user's machine from Hugging Face (respecting upstream access and licensing); exact source bytes and SHA-256 are verified.
- **bundled**: a repository-owned raw source file is committed with an explicit checksum; redistributable only where permitted.

Use:

    python datasets/catalog.py list
    python datasets/catalog.py verify smoke-general-v1
    python datasets/catalog.py acquire n4ze3m-typed-decisions-v1
    python datasets/catalog.py verify n4ze3m-typed-decisions-v1
    python datasets/catalog.py path n4ze3m-typed-decisions-v1

The downloadable file is placed under datasets/cache/raw/ and ignored by Git. The catalog never creates prepared datasets or splits. It never alters source content.

Run a specific benchmark setup:

    python benchmarks/runner.py \
      --setup benchmarks/setups/pico-ultra-n4ze3m.json \
      --output output/pico-n4ze3m-run

That setup selects the cascade and its routing policy, a source dataset, an adapter and a selection rule. The adapter parses upstream n4ze3m state/questions/Gold into typed requests, entirely outside the source registry. All rows are included when selection.type=all. Example optional per-setup selection: {"type":"hash_range","start":0.0,"end":0.2,"seed":20261026}. No such selection is applied by default.

**Timing**: a batch's offline source normalization is part of the benchmark setup/provenance, but is intentionally excluded from timed per-request inference. Model-side preprocessing, routing, HTTP and fallback occur within the measured request path. This distinction is recorded explicitly so users do not mistake offline conversion for runtime costs.

Every run records exact original source SHA-256, benchmark setup configuration hash, adapter ID, selection rule and normalized-request SHA-256. Domains and Gold provenance remain workload-specific; a label originating from a dataset or teacher does not automatically constitute independent human Gold.

The bundled smoke-general-v1 has three synthetic rows and is only for integration testing. The pinned upstream n4ze3m file comes from a train split; exposure during model pretraining and label provenance may be unknown, so it is not certified as an independent test benchmark. Private local data can be supplied via --dataset FILE with --cascade, without uploading the original inputs.

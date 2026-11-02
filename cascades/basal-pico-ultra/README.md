# Basal Omni Pico → Ultra (reference cascade)

**Basal Pico (fast model)** and **Basal Ultra (strong model)** run as separately pinned HTTP backends. Pico uses `basal-serve` 2.3.2 and Ultra uses 2.1.0, both in `eager` mode. Do not silently upgrade either backend or its pinned model revision. The confidence threshold **0.6563522219657898** is frozen for the reference setup, not tuned on evaluation data.

## Requirements

An NVIDIA ARM64 GPU host with Docker Compose GPU support, a compatible NGC vLLM 26.07 / PyTorch–CUDA environment, access to the pinned Hugging Face snapshots and sufficient GPU RAM for both models concurrently. The experimental Dockerfiles currently use an image tag rather than an immutable base digest; this stack is **not yet release-qualified**. GPU validation remains pending.

Run all examples from the **Community repository root**. Benchmark and Inference use the **same cascade definition and inference engine**. Datasets are reusable independently of the models; a benchmark setup selects the dataset, adapter and cascade.

## Benchmark mode

Start the shared inference service with a benchmark token. The token enables the internal baseline endpoints (`fast_only` and `strong_only`); ordinary public inference does not require it.

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
COMPOSE=cascades/basal-pico-ultra/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build fast strong inference
~~~

The following benchmark commands execute Python **inside the Compose benchmark container**. Results are saved under the ignored `output/` directory, using a fresh directory for every run.

### A. Smoke benchmark

The bundled `smoke-general-v1` contains **three synthetic decision requests**. Its actual latency, quality and routing measurements are useful for diagnosing behavior on those inputs, not for generalizing to an unseen workload.

~~~bash
SMOKE_OUT="output/benchmarks/basal-pico-ultra-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
    --setup benchmarks/smoke-general__basal-pico-ultra__confidence-min-t0p656352/benchmark.json \
    --output "$SMOKE_OUT" --repeats 2
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$SMOKE_OUT"
~~~

### B. Evaluation benchmark

Use the **same model-independent Civil Comments held-out dataset** as in the Laya → Clef evaluation example. The existing **named Basal benchmark setup** links that dataset, its `better_jev_heldout_gold_v1` adapter and the Pico → Ultra cascade. All 4,000 Gold-scored rows are included.

~~~bash
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python datasets/catalog.py acquire bjb-civil-comments-heldout-v1
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python datasets/catalog.py verify bjb-civil-comments-heldout-v1

EVAL_OUT="output/benchmarks/basal-pico-ultra-civil-comments-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
    --setup benchmarks/better-jev-bench-civil-comments-heldout__basal-pico-ultra__confidence-min-t0p656352/benchmark.json \
    --output "$EVAL_OUT" --repeats 2
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$EVAL_OUT"
~~~

For the **full external evaluation suite**, including the other pinned test and held-out sources, run the maintained launcher **instead of** the single-dataset commands:

~~~bash
sh benchmarks/run_confidence_external.sh
~~~

The launcher downloads and checks the sources, dry-runs their adapters, starts both backends and the shared inference service, and records all three benchmark arms. These are measured results, which may demonstrate that the cascade is **slower or less accurate** than `strong_only`. Held-out source labels alone do not guarantee absence of model pretraining overlap. The standard three-arm comparison measures with both model backends resident; `strong-alone` requires a separate deployment. See the [benchmark guide](../../benchmarks/README.md).

## Inference mode

Expose the Pico → Ultra cascade as **one System One model**. A benchmark token is not needed for standard public API calls.

~~~bash
COMPOSE=cascades/basal-pico-ultra/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build fast strong inference
curl http://127.0.0.1:18358/v1/models
~~~

For example, ask the model to **make a decision under a stated return policy**:

~~~bash
curl -i -sS http://127.0.0.1:18358/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "basal-pico-ultra-h1",
    "state": "A customer purchased an unopened item yesterday. Store policy allows refunds of unopened items within 14 days.",
    "questions": {
      "refund": {
        "type": "choice",
        "instructions": "Decide whether to approve the refund under the stated policy.",
        "criteria": {
          "approve": "Approve the refund",
          "reject": "Reject the refund"
        }
      }
    }
  }'
~~~

The response has an `answers.refund.probabilities` distribution; inspect `X-CascadeOne-Escalated` to see whether Ultra was invoked. The API exposes model predictions, not a guarantee of the expected policy decision. The two reference Compose projects both bind host port `127.0.0.1:18358`, so run them separately or change one port mapping. See [inference documentation](../../inference/README.md) for benchmark-based startup warnings, HTTP headers and security limitations.

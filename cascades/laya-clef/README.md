# Laya → Clef (reference cascade)

Pinned **Laya (fast model)** and **Clef-Flash (strong model)** run as separate HTTP backends. Laya performs audited eligibility checks; ineligible requests bypass the fast model and escalate to Clef. The confidence threshold **0.90** is an illustrative frozen configuration, not a newly calibrated optimum.

## Requirements

A compatible NVIDIA ARM64 GPU host with Docker Compose GPU support, the pinned Laya/Clef Hugging Face revisions, sufficient VRAM to keep both models resident, and an authorized, immutable NVIDIA CUDA base image for the Dockerfile. The configuration targets DGX Spark-class hardware; clean GPU deployment remains to be validated for release.

Run the commands below **from the Community repository root**. Configure a real, compatible image digest (the example is only a placeholder):

~~~bash
export CASCADEONE_CUDA_BASE='nvcr.io/REPLACE_WITH_VERIFIED_IMAGE@sha256:REPLACE_WITH_VERIFIED_DIGEST'
~~~

The two primary use cases share the **same inference engine**. Choose **Benchmark** to measure the cascade against individual models, or **Inference** to expose it as a single System One API.

## Benchmark mode

Start the fast and strong backends with the shared inference service. The benchmark token enables internal `fast_only` and `strong_only` measurement endpoints; it is not required by ordinary API clients.

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
COMPOSE=cascades/laya-clef/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build fast strong inference
~~~

All benchmark commands below run Python **inside the Compose benchmark container**. Outputs are written under the repository's ignored `output/` directory.

### A. Smoke benchmark

The bundled `smoke-general-v1` dataset contains **three synthetic requests**. The benchmark executes `fast_only`, `strong_only` and `cascade` on the same inputs and checks that results and timing are recorded. Its measured latencies and Gold quality describe these inputs only.

~~~bash
SMOKE_OUT="output/benchmarks/laya-clef-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
    --setup benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json \
    --output "$SMOKE_OUT" --repeats 2
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$SMOKE_OUT"
~~~

### B. Evaluation benchmark

Community does **not yet contain a named Laya–Clef external benchmark setup**. For a broader, Gold-scored evaluation you can nevertheless use the shared runner directly with the pinned **better-jev-bench Civil Comments held-out** source. Its request adapter produces System One decision questions and Gold answers; the 4,000 records have two- or five-option decisions.

Download and verify the unmodified source, then run the three arms using the **Laya–Clef cascade profile**:

~~~bash
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python datasets/catalog.py acquire bjb-civil-comments-heldout-v1
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python datasets/catalog.py verify bjb-civil-comments-heldout-v1

EVAL_OUT="output/benchmarks/laya-clef-civil-comments-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
    --cascade cascades/laya-clef/benchmark.json \
    --dataset-id bjb-civil-comments-heldout-v1 \
    --adapter better_jev_heldout_gold_v1 \
    --output "$EVAL_OUT" --repeats 2
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$EVAL_OUT"
~~~

This evaluates the configured Laya → Clef cascade on a real, pinned held-out source, **not** a hypothetical speedup. The dataset may still overlap unknown model pretraining, and Laya's eligibility checks may reject specific requests in the `fast_only` arm; the runner stops rather than inventing a fast response for them. An independent, fully qualified Laya evaluation requires auditing eligibility, Gold provenance and potential training overlap. The strong-only baseline uses Clef with **both backends resident**; a separate strong-alone residency experiment is described in the [benchmark guide](../../benchmarks/README.md).

## Inference mode

Start the same service without a benchmark token if you only want to serve decisions:

~~~bash
COMPOSE=cascades/laya-clef/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build fast strong inference
curl http://127.0.0.1:18358/v1/models
~~~

Send a **decision request**, not just a model-discovery request:

~~~bash
curl -i -sS http://127.0.0.1:18358/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya-clef-core",
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

The response contains the selected backend's `answers.refund.probabilities` distribution. HTTP headers such as `X-CascadeOne-Escalated` indicate whether the request was routed to Clef. Neither model is guaranteed to choose the expected answer. See [inference documentation](../../inference/README.md) for startup warnings based on validated benchmarks, network security and API details.

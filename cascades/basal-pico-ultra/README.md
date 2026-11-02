# Basal Pico → Ultra

Cascade: **Basal Pico (fast model) → Basal Ultra (strong model)**, confidence routing at **0.6563522219657898**. The pinned model backends use `basal-serve` 2.3.2 (Pico) and 2.1.0 (Ultra), both in `eager` mode.

## Before you start

Run all commands **from the `dist/CascadeOne-Community` repository root** (containing `cascades/` and `benchmarks/`).

You need an **ARM64 NVIDIA GPU host** (such as DGX Spark), Docker Compose v2 with GPU access and `--wait` support, `curl` and `openssl`, enough VRAM for both models and access to their pinned Hugging Face snapshots. Dockerfiles use the default NVIDIA image `nvcr.io/nvidia/vllm:26.07-py3`. This image is referenced by tag rather than immutable digest; exact environment reproduction and end-to-end GPU execution still require validation.

## 1. Benchmark

The benchmark uses the **same inference engine** as the public API for `fast_only`, `strong_only` and `cascade`. The random token enables benchmark-only endpoints.

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
COMPOSE=cascades/basal-pico-ultra/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build --wait --wait-timeout 660 fast strong inference
docker compose -f "$COMPOSE" --profile inference logs --tail=30 inference
~~~

### Smoke — three synthetic decisions

~~~bash
OUT="output/basal-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
  --setup benchmarks/smoke-general__basal-pico-ultra__confidence-min-t0p656352/benchmark.json \
  --output "$OUT" --repeats 2

docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$OUT"
~~~

Smoke reports actual timing and quality for **three synthetic cases**; it is not a representative performance estimate.

### Evaluation — a pinned held-out dataset

The `bjb-civil-comments-heldout-v1` dataset is **independent of models** and is used in both cascade guides. Basal already has a named benchmark setup combining this source, its adapter and the Pico → Ultra cascade. It normalizes 4,000 Gold-scored requests.

With the benchmark stack above running:

~~~bash
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python datasets/catalog.py acquire bjb-civil-comments-heldout-v1

OUT="output/basal-eval-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
  --setup benchmarks/better-jev-bench-civil-comments-heldout__basal-pico-ultra__confidence-min-t0p656352/benchmark.json \
  --output "$OUT" --repeats 2

docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$OUT"
~~~

To run **all** available external evaluation setups, use the complete launcher *instead of* the single-dataset commands:

~~~bash
sh benchmarks/run_confidence_external.sh
~~~

The measured result may be **negative**: a cascade can be slower than the strong model or lose too much quality. Public held-out sources can overlap unknown model pretraining. Both model backends remain resident during the three-arm comparison; a strong-alone experiment needs a different deployment.

## 2. Inference service

To serve the cascade as a **single System One model**, no benchmark token is needed:

~~~bash
COMPOSE=cascades/basal-pico-ultra/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build --wait --wait-timeout 660 fast strong inference
curl -f http://127.0.0.1:18358/health
~~~

Ask the API for a decision:

~~~bash
curl -i -sS http://127.0.0.1:18358/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "basal-pico-ultra-h1",
    "state": "An unopened product was purchased yesterday. Refunds are allowed within 14 days.",
    "questions": {
      "refund": {
        "type": "choice",
        "instructions": "Should the refund be approved under the policy?",
        "criteria": {"approve": "Approve refund", "reject": "Reject refund"}
      }
    }
  }'
~~~

The response includes `answers.refund.probabilities`. Header `X-CascadeOne-Escalated` shows whether Ultra was invoked. Startup warnings show speed/quality as **unknown** until a matching benchmark is supplied (see [Inference API](../../inference/README.md)).

To stop the services:

~~~bash
docker compose -f "$COMPOSE" --profile inference down
~~~

Run **one reference Compose project at a time**: both expose host port `127.0.0.1:18358`. See [benchmark methodology](../../benchmarks/README.md) for metric definitions and provenance.

# Laya → Clef

Cascade: **Laya (fast model) → Clef-Flash (strong model)**, with confidence routing at **0.90**.

## Before you start

Run all commands **from the `dist/CascadeOne-Community` repository root** (the directory containing `cascades/` and `benchmarks/`).

You need an **ARM64 NVIDIA GPU host** (such as DGX Spark), Docker Compose v2 with GPU access and `--wait` support, `curl` and `openssl`, sufficient VRAM for both models, and access to their pinned Hugging Face revisions. A real GPU end-to-end deployment has not yet been release-validated.

**No `CASCADEONE_CUDA_BASE` export is required for the quickstart.** Compose defaults to the same NVIDIA image used by the Basal reference:
`nvcr.io/nvidia/vllm:26.07-py3`. It is a **tag, not an immutable digest**. For strictly reproducible experiments, you can optionally override `CASCADEONE_CUDA_BASE` with a verified, compatible image digest; the quickstart itself needs no extra configuration.

If you previously exported a placeholder or invalid `CASCADEONE_CUDA_BASE`, run `unset CASCADEONE_CUDA_BASE` to restore the default. If the NVIDIA image is inaccessible, configure registry access and compatible NVIDIA drivers first.

## 1. Benchmark

The same inference engine runs **`fast_only`, `strong_only` and `cascade`** on identical requests. Set a random token to authorize benchmark-only endpoints, then start all three services:

~~~bash
export CASCADEONE_BENCHMARK_TOKEN="$(openssl rand -hex 32)"
COMPOSE=cascades/laya-clef/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build --wait --wait-timeout 660 fast strong inference
docker compose -f "$COMPOSE" --profile inference logs --tail=30 inference
~~~

### Smoke — three synthetic decisions

~~~bash
OUT="output/laya-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
  --setup benchmarks/smoke-general__laya-clef__confidence-min-t0p900000/benchmark.json \
  --output "$OUT" --repeats 2

docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$OUT"
~~~

Smoke produces **actual measurements for three cases**, not representative estimates of real-world quality and speed.

### Evaluation — a pinned held-out dataset

Datasets are **independent of models**. This example reuses the `bjb-civil-comments-heldout-v1` source and `better_jev_heldout_gold_v1` adapter. The same dataset is used in the Basal guide. It contains 4,000 normalized Gold-scored requests.

With the benchmark stack above still running:

~~~bash
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python datasets/catalog.py acquire bjb-civil-comments-heldout-v1

OUT="output/laya-eval-$(date -u +%Y%m%dT%H%M%SZ)"
docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python benchmarks/runner.py \
  --cascade cascades/laya-clef/benchmark.json \
  --dataset-id bjb-civil-comments-heldout-v1 \
  --adapter better_jev_heldout_gold_v1 \
  --output "$OUT" --repeats 2

docker compose -f "$COMPOSE" --profile inference --profile benchmark run --rm --no-deps benchmark \
  python tools/validate_result.py "$OUT"
~~~

**Limitations:** A held-out *source* may still overlap unknown model pretraining. Laya can mark requests ineligible; if an ineligible request reaches the mandatory `fast_only` arm, the runner fails instead of inventing a prediction. This is a compatibility limitation to audit, not a reason to silently exclude data. Both backends are resident during the three-arm comparison.

## 2. Inference service

If you only want one System One HTTP endpoint, **no benchmark token is needed**:

~~~bash
COMPOSE=cascades/laya-clef/compose.yaml
docker compose -f "$COMPOSE" --profile inference up -d --build --wait --wait-timeout 660 fast strong inference
curl -f http://127.0.0.1:18358/health
~~~

Ask for a **decision**:

~~~bash
curl -i -sS http://127.0.0.1:18358/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya-clef-core",
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

The response includes `answers.refund.probabilities`; `X-CascadeOne-Escalated` indicates whether the strong model ran. Startup warnings report speed and quality as **unknown** unless you supply a verified benchmark result (see [Inference API](../../inference/README.md)).

To stop the stack:

~~~bash
docker compose -f "$COMPOSE" --profile inference down
~~~

See [benchmark methodology](../../benchmarks/README.md) for metric definitions, provenance and evaluation caveats.

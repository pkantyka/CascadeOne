# Laya → Clef

The Laya → Clef pair runs **Laya (fast model)** and **Clef-Flash (strong model)**. Its confidence variant uses a frozen minimum-confidence threshold of **0.90**; its pre-router variant escalates requests with more than one question before running Laya. Both are research configurations, not demonstrated optima.

There are **two independently launchable variants** of the Laya → Clef pair. The [confidence variant](confidence/cascade.yaml) uses a post-inference minimum-confidence threshold; the [pre-router variant](pre-router/cascade.yaml) uses a question-count rule before the fast model. Each variant owns its full [`compose.yaml`](confidence/compose.yaml) ([pre-router Compose](pre-router/compose.yaml)); no Compose inheritance is required. Both reuse the pair's model backend implementation.

## Requirements

Run the commands from the `dist/CascadeOne-Community` repository root on an **ARM64 NVIDIA GPU host** (for example DGX Spark). Docker Compose v2 with `--wait` support, sufficient GPU RAM for both models, pinned Hugging Face model access and a compatible NVIDIA/PyTorch base are required. This deployment has **not** been fully validated end-to-end on a clean GPU host.

The launcher supplies `nvcr.io/nvidia/vllm:26.07-py3` as the default CUDA base. The optional `--cuda-base` flag overrides it for **both** backends. The example image is a tag, not an immutable digest.

## Benchmark

A benchmark compares `fast_only`, `strong_only` and `cascade` on the same requests. The `./cascadeone` launcher prepares the runner, uses the shared API token, starts the services in benchmark mode, waits for readiness, runs the test and validates the output.

~~~bash
# Measure the three-arm cascade on three bundled synthetic decisions.
./cascadeone benchmark smoke --cascade laya-clef/confidence \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Evaluate the same cascade on the pinned Civil Comments held-out dataset.
# This uses a model-independent dataset; --suite is NOT required.
./cascadeone benchmark evaluate --cascade laya-clef/confidence

# Example of evaluating another source without changing the cascade.
# HellaSwag is a training-source dataset, NOT an independent held-out test.
./cascadeone benchmark evaluate --cascade laya-clef/confidence \
  --dataset-id hellaswag-train-v1 --adapter hellaswag_gold_v1

# List named setups for the confidence variant.
./cascadeone benchmark list --cascade laya-clef/confidence

# Run the pre-router variant with its own Compose stack and smoke setup.
./cascadeone benchmark smoke --cascade laya-clef/pre-router

# List benchmarks that use the pre-router variant.
./cascadeone benchmark list --cascade laya-clef/pre-router
~~~

Datasets are independent of models, but Laya's eligibility logic can reject unsupported or oversized inputs. A mandatory `fast_only` baseline may then fail rather than silently skip requests. The Civil Comments example contains 4,000 Gold-scored normalized records; unknown pretraining overlap remains a limitation. The synthetic smoke benchmark produces actual diagnostic measurements, not representative workload estimates. Currently there is no named Laya evaluation *suite* for batching; `--suite` is therefore unnecessary for these examples.

## Inference service

Start the same cascade as one HTTP System One decision model. The **same API token** secures both modes: if `CASCADEONE_API_TOKEN` is exported, the launcher uses it and **does not read or write `.cascadeone/token`**; otherwise it creates/reuses that private file (`0600`).

~~~bash
# Start both model services and the public inference API; wait until healthy.
./cascadeone inference start --cascade laya-clef/confidence \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Ask for a decision, not only a list of installed models.
curl -i -sS http://127.0.0.1:18358/v1/systemone \
  -H "Authorization: Bearer ${CASCADEONE_API_TOKEN:-$(cat .cascadeone/token)}" \
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

# Stop the Laya → Clef containers.
./cascadeone inference stop --cascade laya-clef/confidence
~~~

The API returns `answers.refund.probabilities`. `X-CascadeOne-Escalated` indicates whether Clef ran; no particular decision is guaranteed. To inspect the service, run `./cascadeone inference logs --cascade laya-clef/confidence`. See [inference API](../../inference/README.md) and [benchmark methodology](../../benchmarks/README.md) for detailed contracts, provenance and limitations.

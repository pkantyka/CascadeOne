# Laya → Clef

The reference cascade runs **Laya (fast model)** and **Clef-Flash (strong model)**, using a frozen minimum-confidence threshold of **0.90**. It is a research configuration, not a demonstrated optimum.

## Requirements

Run the commands from the `dist/CascadeOne-Community` repository root on an **ARM64 NVIDIA GPU host** (for example DGX Spark). Docker Compose v2 with `--wait` support, sufficient GPU RAM for both models, pinned Hugging Face model access and a compatible NVIDIA/PyTorch base are required. This deployment has **not** been fully validated end-to-end on a clean GPU host.

The launcher supplies `nvcr.io/nvidia/vllm:26.07-py3` as the default CUDA base. The optional `--cuda-base` flag overrides it for **both** backends. The example image is a tag, not an immutable digest.

## Benchmark

A benchmark compares `fast_only`, `strong_only` and `cascade` on the same requests. The `./cascadeone` launcher prepares the runner, creates the private benchmark token, starts the services, waits for readiness, runs the test and validates the output.

~~~bash
# Measure the three-arm cascade on three bundled synthetic decisions.
./cascadeone benchmark smoke --cascade laya-clef \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Evaluate the same cascade on the pinned Civil Comments held-out dataset.
# This uses a model-independent dataset; --suite is NOT required.
./cascadeone benchmark evaluate --cascade laya-clef

# Example of evaluating another source without changing the cascade.
# HellaSwag is a training-source dataset, NOT an independent held-out test.
./cascadeone benchmark evaluate --cascade laya-clef \
  --dataset-id hellaswag-train-v1 --adapter hellaswag_gold_v1

# List currently registered named setups for the default Laya → Clef profile.
./cascadeone benchmark list --cascade laya-clef
~~~

Datasets are independent of models, but Laya's eligibility logic can reject unsupported or oversized inputs. A mandatory `fast_only` baseline may then fail rather than silently skip requests. The Civil Comments example contains 4,000 Gold-scored normalized records; unknown pretraining overlap remains a limitation. The synthetic smoke benchmark produces actual diagnostic measurements, not representative workload estimates. Currently there is no named Laya evaluation *suite* for batching; `--suite` is therefore unnecessary for these examples.

## Inference service

Start the same cascade as one HTTP System One decision model, without a benchmark token:

~~~bash
# Start both model services and the public inference API; wait until healthy.
./cascadeone inference start --cascade laya-clef \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Ask for a decision, not only a list of installed models.
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

# Stop the Laya → Clef containers.
./cascadeone inference stop --cascade laya-clef
~~~

The API returns `answers.refund.probabilities`. `X-CascadeOne-Escalated` indicates whether Clef ran; no particular decision is guaranteed. To inspect the service, run `./cascadeone inference logs --cascade laya-clef`. See [inference API](../../inference/README.md) and [benchmark methodology](../../benchmarks/README.md) for detailed contracts, provenance and limitations.

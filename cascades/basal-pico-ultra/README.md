# Basal Pico → Ultra

The reference cascade runs **Basal Pico (fast model)** and **Basal Ultra (strong model)** with the frozen minimum-confidence threshold **0.6563522219657898**. The independently pinned `basal-serve` versions are 2.3.2 (Pico) and 2.1.0 (Ultra), in `eager` mode.

## Requirements

Run from the `dist/CascadeOne-Community` repository root on an **ARM64 NVIDIA GPU host** (for example DGX Spark). You need Docker Compose v2 with `--wait` support, sufficient RAM/VRAM for both models and access to their pinned Hugging Face revisions. The default CUDA/PyTorch image is `nvcr.io/nvidia/vllm:26.07-py3`. It is a **tag**, not an immutable digest, and full clean-host GPU validation is still pending.

The launcher takes an optional `--cuda-base IMAGE` flag and applies the image to **both** model builds. A tested compatible digest is preferable for precisely reproduced experiments.

## Benchmark

All benchmark configurations use the **same shared inference service** for the three arms (`fast_only`, `strong_only`, `cascade`). The launcher generates the benchmark token, prepares and verifies data, starts the backend services and validates the results.

~~~bash
# Smoke: measure routing, latency and decision quality on three synthetic cases.
./cascadeone benchmark smoke --cascade basal-pico-ultra \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# See the datasets and named benchmark setups for this model pair.
./cascadeone benchmark list --cascade basal-pico-ultra

# Evaluate the cascade on ONE named Civil Comments held-out benchmark.
# No --suite needed: --benchmark selects precisely one configured experiment.
./cascadeone benchmark evaluate --cascade basal-pico-ultra \
  --benchmark better-jev-bench-civil-comments-heldout__basal-pico-ultra__confidence-min-t0p656352

# Evaluate the SAME cascade on a DIFFERENT named dataset (BANKING77 test).
./cascadeone benchmark evaluate --cascade basal-pico-ultra \
  --benchmark banking77-test__basal-pico-ultra__confidence-min-t0p656352

# OPTIONAL: run the entire external group of registered benchmark setups.
./cascadeone benchmark evaluate --cascade basal-pico-ultra --suite external

# Alternative: supply a dataset and adapter directly, without a named setup.
./cascadeone benchmark evaluate --cascade basal-pico-ultra \
  --dataset-id bjb-civil-comments-heldout-v1 --adapter better_jev_heldout_gold_v1
~~~

Datasets are **model-independent**. A named benchmark defines the dataset, adapter, selection and reference cascade; `--suite` is only an optional grouping filter for batch execution. The runner refuses to overwrite results and reports both negative and positive speedups. Source-held-out labels do not exclude possible pretraining overlap; smoke performance on three examples is not representative.

## Inference service

Run Pico → Ultra as one public System One decision API without manually creating a benchmark token:

~~~bash
# Build/start the models and wait for the inference API to be healthy.
./cascadeone inference start --cascade basal-pico-ultra \
  --cuda-base nvcr.io/nvidia/vllm:26.07-py3

# Send one policy decision to the cascade.
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

# Stop the Basal containers.
./cascadeone inference stop --cascade basal-pico-ultra
~~~

The HTTP response includes `answers.refund.probabilities` and an `X-CascadeOne-Escalated` header indicating Ultra execution. Only one reference stack should bind `127.0.0.1:18358` at a time. For logs, run `./cascadeone inference logs --cascade basal-pico-ultra`. See [inference API](../../inference/README.md) and [benchmark methodology](../../benchmarks/README.md) for detailed contracts and limitations.

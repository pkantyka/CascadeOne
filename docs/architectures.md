# CascadeOne architectures

## Implemented: conditional decision cascade

The runtime in `cascades/runtime.py` executes a request with a fast and strong backend. The routing policy in `cascades/policies.py` may bypass fast inference before it starts or inspect the fast response afterward. A routing decision applies to the complete request.

Supported policies:

- `confidence_threshold`: accept or escalate using request-level minimum, mean or maximum answer confidence and a configurable threshold.
- `max_questions_pre_router`: send requests with too many questions directly to strong inference, without using confidence.
- `always_fast` and `always_strong`: explicit fixed-routing controls.
- `plugin`: a trusted Python factory implementing `before` and `after` hooks.

The benchmark client calls the same inference service used by applications for fast-only, strong-only and cascade arms on identical requests. It measures both client end-to-end HTTP latency and engine-internal latency (including backend HTTP); neither is a pure GPU compute timer. Gold labels and benchmark data preparation do not affect the router.

## Other research directions (not implemented here)

- **Expected-gain routing:** estimate when escalation will improve Gold quality, rather than trusting raw answer confidence.
- **Latent routing:** use hidden representations from the fast model to predict escalation benefit without adding another complete forward pass.
- **Teacher-to-student distillation:** train a small model to approximate a pinned strong model, then test both teacher agreement and independent Gold correctness.
- **Fidelity-risk head:** distinguish predicting teacher agreement from predicting actual correctness.
- **Multiple fast specialists and fusion:** conditionally choose a specialist or fuse representations while accounting for the additional compute cost.

- **Model architecture variants:** study specialized decision heads, adapters, model-internal changes and additional components as parts of a complete cascade, measuring their incremental inference and resource costs.

These are conceptual alternatives, not available routing policies or validated performance claims.

## Data and evaluation boundary

Original datasets are acquired and checksum-verified in `datasets/`; benchmark-owned adapters, Gold interpretation and selection reside under `benchmarks/`. Only model-side processing, routing and fallback execution contribute to timed request inference. Every measured configuration must preserve dataset and model identity for a fair comparison.

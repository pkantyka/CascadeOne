# Candidate architectures

**Implemented Community runtime**: the configured policy may pre-route to strong or inspect a fast decision to accept/escalate. Confidence threshold is one supported policy, not a benchmark requirement. Built-ins include always_fast, always_strong, max_questions_pre_router and confidence_threshold; trusted plug-in policies implement before/after hooks. The baseline always invokes strong directly. One routing decision applies to the whole request.

**Pre-router**: a simple question-count bypass is implemented as a confidence-free Community example. Model-based trained pre-routing remains experimental, not part of the reference package.

**Post-hoc gain router (implemented experimental code in Development only)**: use confidence, margins, entropy and request properties to estimate whether strong inference will improve Gold score. This is distinct from estimating confidence of a fast answer.

**Latent router (prototype in Development)**: use the small model hidden states as extra features; a deployable version must reuse its single forward pass, otherwise the extra inference cost can dominate.

**Distillation (planned test)**: train a small decision model against the pinned strong model's distributions. Measure both fidelity to the teacher and correctness against independent Gold. Distillation does not establish calibrated correctness or guarantee useful routing.

**Dual head / fidelity risk (design only)**: separate a decision head (teacher distribution) from a request-level discrepancy predictor. Distinguish predicting teacher agreement from predicting improvement to original-source Gold.

**Hybrid multi-expert and fusion (design/prototype)**: optional pre-router among different fast specialists, fallback to strong; optional latent fusion of both models, evaluated only under strict controls. Not part of Community v0.1 runtime.

All hypotheses remain falsifiable. Their implementation and verification status must be stated separately.

## Source data vs benchmark setup

Source datasets remain immutable and model-neutral. A versioned setup identifies the cascade, adapter translating source records into model requests, Gold mapping, and optional filtering/sampling. The generic benchmark runner measures execution without consulting confidence itself. Offline dataset conversion is tracked in provenance, not charged to timed request latency.

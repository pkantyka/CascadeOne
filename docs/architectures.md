# Candidate architectures

**Implemented reference path**: request validation → fast eligibility → fast decision → confidence-based accept/fallback → strong decision if escalated. One routing decision applies to the complete request. The baseline invokes strong directly.

**Pre-router (concept)**: decide, before running the small model, whether a request should bypass the fast path. This can save execution time at the cost of routing errors.

**Post-hoc gain router (implemented experimental code in Development only)**: use confidence, margins, entropy and request properties to estimate whether strong inference will improve Gold score. This is distinct from estimating confidence of a fast answer.

**Latent router (prototype in Development)**: use the small model hidden states as extra features; a deployable version must reuse its single forward pass, otherwise the extra inference cost can dominate.

**Distillation (planned test)**: train a small decision model against the pinned strong model's distributions. Measure both fidelity to the teacher and correctness against independent Gold. Distillation does not establish calibrated correctness or guarantee useful routing.

**Dual head / fidelity risk (design only)**: separate a decision head (teacher distribution) from a request-level discrepancy predictor. Distinguish predicting teacher agreement from predicting improvement to original-source Gold.

**Hybrid multi-expert and fusion (design/prototype)**: optional pre-router among different fast specialists, fallback to strong; optional latent fusion of both models, evaluated only under strict controls. Not part of Community v0.1 runtime.

All hypotheses remain falsifiable. Their implementation and verification status must be stated separately.

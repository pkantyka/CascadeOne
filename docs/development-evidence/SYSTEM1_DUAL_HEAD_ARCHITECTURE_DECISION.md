# Decision D-2026-10-15: ModernBERT-base, shared encoder, two heads

**Status: ACCEPTED design decision for the next System One candidate.**
The historical Laya/Clef baselines, frozen routing NO-GO results, and the
existing Clef->Laya pilot handoff are unchanged by this decision.

## Selected architecture

- **Backbone:** pretrained ModernBERT-base, approximately 149M parameters
  *before* adding the task heads; pin the exact model ID and immutable
  revision before any actual model download/training. This is a new
  architecture choice, NOT a claim that its training implementation exists.
- **One encoder forward per request**, shared features reused by both
  heads. No second ModernBERT inference, and no Clef invocation to compute
  routing risk on the fast path.
- **Decision Head:** scores a variable number of options per question and
  returns the full normalized Clef-style distribution for choice, noul,
  score, and multi-question requests. Train encoder + Decision Head
  against frozen Clef distributions (teacher-only soft targets, no Gold).
- **Fidelity Head:** a *separate, small prediction head* consuming shared
  encoder features and optionally Decision Head outputs/metadata
  (stop-gradient for the decision outputs in the initial experiment).
  Predicts *request-level* Student-versus-Clef disagreement risk, not
  P(chosen option). It may later estimate distributional discrepancy
  alongside top-1 disagreement; the risk target and routing tolerances
  must be specified in the experimental protocol before evaluation.
- **Routing:** accept requests based on a calibrated, sufficiently low
  *predicted fidelity risk*. Low Student option confidence can match an
  equally uncertain Clef and is NOT by itself a reason to escalate.
  High option confidence is NOT by itself a reason to accept.

## Initial training sequence

1. Pin ModernBERT weights and preprocessing; implement variable-option
   scoring and lossless eligibility guards. Check full request semantics.
2. Train shared encoder + Decision Head on pinned Clef teacher
   probability distributions from approved **train-only** inputs,
   minimizing soft CE / KL at the frozen teacher temperature.
3. Freeze encoder + Decision Head. Obtain *out-of-training-group*
   Student-versus-Clef predictions (disjoint risk-training subset or
   cross-fitted checkpoints) to prevent in-sample risk optimism.
4. Train Fidelity Head alone on a separately defined *request-level*
   disagreement target (or separately preregistered distributional loss).
   Do not alter Student answers when fitting this head.
5. Choose checkpoint on validation; set the router acceptance threshold
   exclusively on calibration; evaluate frozen architecture/threshold
   on protected holdout. Record selective-risk/coverage, confidence
   intervals, agreement, full-distribution fidelity and real DGX E2E
   mean/p95 including all preprocessing and escalations.
6. Compare against an output-only/post-hoc risk-predictor control, and
   the frozen Laya/Clef baseline, on a common eligibility population.

## Why two heads rather than one combined output projection?

Both designs predict two quantities. Two distinct heads provide separate
losses, pooling, train/freeze boundaries, versioning and replacement of
the fidelity estimator without perturbing Clef-style output probabilities.
The fidelity head still costs parameters and implementation effort, and
may not improve routing if shared features have little predictive signal.
The benefit must be measured, not assumed.

## Boundaries

This decision selects the **research architecture**, not a trained or
production-qualified replacement. The earlier
`docs/DGX_DISTILLATION_HANDOFF.md` and its archived
`cascadeone-distill-06217b7.tar.gz` still implement the **Laya**
distillation *baseline/pilot*; neither implements ModernBERT or the
Fidelity Head. A separate revised DGX handoff is required before
running the selected two-head candidate.

Related: `docs/SYSTEM1_DISTILLATION_DECISION.md`,
`docs/SYSTEM1_FIDELITY_ROUTING_DECISION.md`.

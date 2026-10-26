# Decision D-2026-10-14: pure System Two -> System One distillation

**Accepted direction:** The objective is to approximate the frozen Clef-Flash
conditional choice/noul/score **probability distributions**, not to correct
Clef or maximize accuracy versus arbitrary Gold annotations. The teacher's
mistakes are part of the behavior to reproduce in this experiment.

- Teacher: `Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6`.
- Student starting weights: `convaiinnovations/laya-typed-decisions@1a793eb568e6718f15941d08f85432581df534e3`.
- Objective: full-distribution soft cross entropy (equivalent to KL up to
  teacher entropy), T=1. No ground-truth term and no Gold pre-finetune.
- Gold labels may only be used as **diagnostic** evaluation, and are not
  assumed correct or preferred to the teacher for fitting.
- Main comparison: question top-1 agreement, all-questions request-exact
  agreement, probability KL/TV and **request-level selective risk/coverage**.
  Confidence is calibrated to **agreement with Clef**, not Gold correctness.
- System gate: >=20% mean end-to-end improvement, p95 <=110% of Clef-only,
  with a preregistered agreement-risk tolerance and uncertainty interval.
  These are distinct from the legacy Gold-based V1 quality gate. Both
  evaluations must be reported; no claim of preserved real-world correctness
  follows automatically from Clef agreement.
- Input-only learning is impossible to guarantee OOD generalization on
  unknown future requests. Initially use the approved procedural train split
  for the pilot. Broader domain coverage requires separately audited inputs.
- No student weights / threshold selection using calibration/protected cases.
  Validate checkpoints on validation, fit router threshold on calibration,
  protected suites only after selection is frozen.
- No shadow audit or continual retraining in this handoff.
- Gold-only handoff from `experiment/system1-gold-finetune@f2529ed` is
  **superseded** and must NOT be run for this experiment.

The source branch is `experiment/system1-distillation`, preserving the
historical Gold branch without rewriting it. No weights or results are
produced by preparing this handoff.

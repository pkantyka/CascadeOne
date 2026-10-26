# Decision D-2026-10-15: fidelity-risk routing, not answer confidence

**ACCEPTED.** System One approximates the *full* Clef probability
distributions; the router should skip Clef where **predicted discrepancy
from Clef** is acceptably small, not wherever Student is confident in
one answer.

## Binding architectural principle

- Low Student max-option probability may perfectly match an uncertain
  Clef (e.g. both predict 0.51/0.49). Low answer confidence alone is
  **NOT** a reason for escalation.
- High Student confidence may disagree with Clef (e.g. Student 0.99/0.01,
  Clef 0.01/0.99). High answer confidence alone is **NOT** permission
  to accept.
- No correlation between Student confidence/entropy/margin and
  teacher-fidelity risk may be presumed. Test it empirically.
- At inference Clef has not run, so Student cannot measure true distance
  to Clef; the router may only **estimate** it and calibrate that
  estimate on independent teacher/student predictions.
- Primary acceptance criteria to define in advance: either request-exact
  top-1 disagreement or a distributional distance (e.g. TV), or both.
  These are different targets and must not be conflated. The policy
  requires a specified risk budget and uncertainty bound.
- Gold is not a supervisory target of pure Clef distillation and cannot
  substitute for fidelity evaluation. Keep Gold as diagnostic only.
- Request-level risk must account for multiple questions, not just
  single-option confidence.
- Frozen R0-R4 negative results remain intact and do not change.

## Candidate implementations, not yet selected

A. Single decision head plus simple post-hoc calibrated risk predictor
using probabilities, margin, entropy, request metadata. Cheapest control;
fidelity signal may be insufficient.

B. One shared encoder forward, with two distinct heads: (1) per-option
probability distribution approximating Clef; (2) request-level estimated
disagreement risk or distributional error. Different objectives, pooling
and stop-gradient behavior are possible; the second head adds params and
training work but need not add another encoder pass.

Calling one combined output layer with K answer logits and one risk
score "one head" is largely an implementation distinction: it remains
**two prediction tasks and two supervision targets**. Forcing answer
confidence itself to encode teacher agreement conflicts with faithfully
matching Clef distributions. Do not do it.

## Training / evaluation discipline

First distill the decision predictor on train-only Clef distributions;
then freeze that predictor while initially training the risk model.
Obtain realistic risk targets on held-out-in-train groups / cross-fit
student predictions, **not on the same examples the Student memorized**.
Select models on validation; fit policy thresholds on calibration; use
protected test data only after locking all choices. Compare accepted
request disagreement, distributional fidelity, selective-risk/coverage
curves, uncertainty intervals, DGX E2E mean and p95, and eligible rate.
No guarantee of unknown-domain calibration is assumed.

**Decision made:** route by predicted fidelity risk, not response
confidence. **Decision open:** whether the risk estimator is post-hoc
or an explicitly separate neural head. Do not change the shipped
reference backend or the frozen distillation-pilot handoff implicitly.

See docs/SYSTEM1_DISTILLATION_DECISION.md and
docs/DGX_DISTILLATION_HANDOFF.md.

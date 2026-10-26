# CascadeOne concept

CascadeOne evaluates **conditional execution of decision models**: a lightweight fast model can answer a request, while a stronger fallback model handles selected requests. The measured object is the entire inference path, including preprocessing, routing, and fallback overhead, not simply the fraction routed to the small model.

The reference policy is request-level confidence gating: run the fast model once; if it is ineligible or the minimum question confidence is below a fixed, calibration-derived threshold, re-run the complete request with the strong model. Unlike online adaptive training, the threshold does not change at runtime.

Model vendor and model family are implementation details. Laya/Clef and Basal Pico/Ultra are equally legitimate baseline configurations.

## Engineering questions

- At which escalation rates does a small model reduce mean/p95 latency?
- How much accuracy is lost relative to strong-only on the same inputs?
- Does the cascade increase RAM/VRAM residency, CPU/GPU utilization or energy?
- Does a student distilled from the actual strong model yield a better tradeoff than an independently trained small model?
- Can a learned predictor of fidelity risk or expected Gold improvement outperform output-confidence gating at a matched escalation budget?

No one benchmark establishes universal superiority. Different hardware, tokenization, model revisions, workload distributions, and model scheduling change the answer.

See architectures.md, hypotheses.md and docs/development-evidence for detailed prior work.

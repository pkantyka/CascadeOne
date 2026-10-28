# CascadeOne concept

CascadeOne evaluates **conditional execution of decision models**: a lightweight fast model can answer a request, while the strong model (the higher-capability fallback) handles selected requests. The measured object is the entire inference path, including preprocessing, routing, and fallback overhead, not simply the fraction routed to the small model.

One reference policy is request-level confidence gating: run the fast model once and escalate if confidence fails a configured threshold. This is not mandatory: alternative policies may pre-route requests without running fast inference, use other signals or accept without confidence. Policies and their parameters are part of the cascade definition; the benchmark runner is policy-neutral.

Model vendor and model family are implementation details. Laya/Clef and Basal Pico/Ultra are equally legitimate baseline configurations.

## Architecture experiments

CascadeOne also explores changes to model architecture and model composition in cascades, including specialized decision heads, adapters, distillation, additional fast experts, pre-routers, learned escalation predictors and latent fusion. The aim is to compare alternative complete-system configurations, not just routing thresholds. All added inference, memory and communication overhead must be measured. Proposed variants are not considered implemented until validated here.

## Engineering questions

- At which escalation rates does a small model reduce mean/p95 latency?
- How much accuracy is lost relative to strong-only on the same inputs?
- Does the cascade increase RAM/VRAM residency, CPU/GPU utilization or energy?
- Does a student distilled from the actual strong model yield a better tradeoff than an independently trained small model?
- Can a learned predictor of fidelity risk or expected Gold improvement outperform output-confidence gating at a matched escalation budget?

No one benchmark establishes universal superiority. Different hardware, tokenization, model revisions, workload distributions, and model scheduling change the answer.

For implementation details and research hypotheses, see architectures.md and hypotheses.md.

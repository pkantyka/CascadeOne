# Architecture

CascadeOne separates policy, backend execution and optional Fusion.

## Core path

1. DecisionRequest validates a text-only SystemOne-style request.
2. The fast backend executes once and returns typed answers plus per-question answer confidence.
3. ConfidenceThresholdPolicy aggregates the request by minimum question confidence.
4. If accepted, the fast result is returned and the strong backend is not executed.
5. If escalated, the strong backend processes the complete request.

run_batch is deliberately sequential in V1 so accepted rows can never enter hidden strong-backend
batch computation.

## Fusion path

Fusion is only evaluated on escalated requests. In the current V1 plan, Fusion means GatedResidualFusion. Cross-attention remains experimental code and is not scheduled for pilot or confirmatory execution.

1. The fast backend returns its normal answer and retained latent state from the same forward pass.
2. The strong backend executes encode without its final decision head.
3. FusionAdapter produces a residual correction to the original strong hidden state.
4. The frozen strong finalizer produces typed outputs.
5. Gradients during Stage A pass through the frozen head math into the adapter only.

GatedResidualFusion initializes the residual delta to zero. Therefore the initial fused hidden state is exactly the original strong hidden state.

## Trust boundaries

Upstream models are frozen, kept in eval mode and excluded from the optimizer. Training safety checks
fail on trainable upstream parameters, upstream gradients, training-mode modules or optimizer
tampering.

Adapter checkpoints bind the weights to fast/strong identity, revisions, insertion point, Fusion type
and Fusion configuration.

## V1 limitations

- text only;
- one routing decision per complete request;
- sequential request execution;
- no claim that confidence is calibrated probability of request correctness;
- no implicit truncation in the production reference backends.

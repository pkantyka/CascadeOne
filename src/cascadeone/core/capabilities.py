from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    supports_probabilities: bool = True
    supports_answer_confidence: bool = True
    supports_token_latents: bool = False
    supports_attention_mask: bool = False
    supports_typed_decisions: bool = True
    supports_latent_injection: bool = False
    supports_separate_finalizer: bool = False


def require_fusion(fast, strong):
    for backend in (fast, strong):
        if not backend.capabilities.supports_token_latents or not backend.capabilities.supports_attention_mask:
            raise ValueError("Fusion requires token latents and masks on both backends")
    if not strong.capabilities.supports_latent_injection or not strong.capabilities.supports_separate_finalizer:
        raise ValueError("Fusion requires a separate, differentiable strong finalizer")

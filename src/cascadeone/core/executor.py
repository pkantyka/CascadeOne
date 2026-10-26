import threading
import time

import torch

from cascadeone.core.capabilities import require_fusion
from cascadeone.core.types import DecisionRequest
from cascadeone.routing.confidence_threshold import RouteDecision
from cascadeone.training.safety import assert_frozen


class CascadeExecutor:
    """Backend-independent, request-level cascade. Batches preserve per-request isolation."""
    def __init__(self, fast_backend, strong_backend, routing_policy, fusion_adapter=None):
        self.fast, self.strong = fast_backend, strong_backend
        self.policy, self.fusion = routing_policy, fusion_adapter
        self.lock = threading.RLock()
        if not self.fast.capabilities.supports_answer_confidence:
            raise ValueError("threshold routing requires fast answer confidence")
        if self.fusion is not None:
            require_fusion(self.fast, self.strong)
            self.fusion.eval()
        assert_frozen(self.fast.model, self.strong.model)

    @property
    def mode(self):
        return 'fusion' if self.fusion is not None else 'core'

    def prepare_fast_route(self, request):
        route = self.policy.pre_fast_decide(request)
        prepared = None
        if route is None:
            prepare = getattr(self.fast, "prepare_for_routing", None)
            if prepare is not None:
                preparation = prepare(request)
                if not preparation.eligible:
                    route = RouteDecision(
                        False,
                        self.policy.threshold(request),
                        None,
                        preparation.reason or "fast_backend_ineligible",
                    )
                else:
                    prepared = preparation.prepared
        return route, prepared

    def infer_fast(self, request, *, retain_latent=False, prepared=None):
        if prepared is None:
            return self.fast.infer(request, retain_latent=retain_latent)
        return self.fast.infer(request, retain_latent=retain_latent, prepared=prepared)

    def run(self, request):
        if not isinstance(request, DecisionRequest):
            request = DecisionRequest.model_validate(request)
        with self.lock, torch.no_grad():
            assert_frozen(self.fast.model, self.strong.model)
            started = time.perf_counter()
            route, prepared = self.prepare_fast_route(request)
            fast = None
            if route is not None:
                result, path = self.strong.infer(request), 'strong'
                fast_executions = 0
            else:
                fast = self.infer_fast(
                    request,
                    retain_latent=self.fusion is not None,
                    prepared=prepared,
                )
                route = self.policy.decide(request, fast)
                fast_executions = 1
                if route.accepted:
                    result, path = fast, 'fast'
                elif self.fusion is None:
                    result, path = self.strong.infer(request), 'strong'
                else:
                    if fast.latent is None:
                        raise RuntimeError("backend declared latents but did not return them")
                    strong = self.strong.encode(request)
                    fused = self.fusion(fast.latent, strong)
                    result, path = self.strong.finalize(request, fused), 'strong_fusion'
            if fast is not None:
                fast.latent = None
            metadata = {'framework': 'CascadeOne', 'mode': self.mode, 'route': path,
                        'escalated': not route.accepted, 'fast_backend': self.fast.identity.type,
                        'strong_backend': self.strong.identity.type,
                        'fast_answer_confidence': route.confidence,
                        'escalation_threshold': route.threshold, 'reason': route.reason,
                        'elapsed_ms': (time.perf_counter() - started) * 1000,
                        'fast_executions': fast_executions, 'strong_executions': int(not route.accepted),
                        'fusion_executions': int(path == 'strong_fusion')}
            return {'model': request.model, 'answers': result.answers,
                    'usage': result.metadata.get('usage', {}), 'routing': metadata}

    def run_batch(self, requests):
        # Deliberately sequential V1: no accepted row ever enters strong computation.
        return [self.run(request) for request in requests]

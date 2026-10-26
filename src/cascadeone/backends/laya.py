"""Pinned upstream Laya. A hook observes final per-question token states without replay."""
import numpy as np
import torch

from cascadeone.core.capabilities import Capabilities
from cascadeone.core.types import BackendIdentity, BackendOutput, FastRoutePreparation, Features, diagnostics
from cascadeone.training.safety import freeze

MODEL_ID = 'convaiinnovations/laya-typed-decisions'
REVISION = '1a793eb568e6718f15941d08f85432581df534e3'


class LayaFastBackend:
    capabilities = Capabilities(supports_token_latents=True, supports_attention_mask=True)

    def __init__(self, model_id=MODEL_ID, revision=REVISION, device='cuda'):
        if model_id != MODEL_ID or revision != REVISION:
            raise ValueError('reference backend requires the audited Laya checkpoint/revision')
        if device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('production reference profile runs on CUDA only; use tiny on CPU')
        import laya
        self.agent = laya.load(model_id, revision=revision, device=device, fast=False, compile=False)
        self.model = freeze(self.agent.model)
        self.identity = BackendIdentity('laya', model_id, revision, self.model.encoder.config.hidden_size,
                                        'laya-fa9a2a7-post-head-concat-v1')
        self.calls = self.encoder_calls = 0
        self.model.encoder.register_forward_pre_hook(self._count_encoder)

    def _count_encoder(self, module, args):
        self.encoder_calls += 1

    def prepare_for_routing(self, request):
        from laya.common import collate_items, collapsed_options

        questions = request.backend_record()['questions']
        ids = list(questions)
        for key in request.questions:
            self.agent._check_question(key, questions[key])
        internal = {key: self.agent._to_internal(q) for key, q in questions.items()}
        try:
            items = self.agent._encode_state(request.state, ids, internal)
        except ValueError:
            return FastRoutePreparation(False, "fast_context_or_schema_exceeds_budget")
        if any(item['state_stats']['state_tokens_dropped'] for item in items):
            return FastRoutePreparation(False, "fast_context_would_truncate")
        if collapsed_options(ids, items):
            return FastRoutePreparation(False, "fast_option_spans_would_collapse")
        batch = collate_items([items], self.agent.tok.pad_token_id)
        return FastRoutePreparation(
            True,
            prepared={
                "ids": ids,
                "internal": internal,
                "items": items,
                "batch": batch,
            },
        )

    def infer(self, request, *, retain_latent=False, prepared=None):
        from laya.common import QTYPES, temp_bucket

        self.calls += 1
        if prepared is None:
            preparation = self.prepare_for_routing(request)
            if not preparation.eligible:
                raise ValueError(preparation.reason)
            prepared = preparation.prepared
        ids = prepared["ids"]
        internal = prepared["internal"]
        items = prepared["items"]
        batch = prepared["batch"]
        captured = []
        hook = None
        if retain_latent:
            if self.model.head is None or not self.model.head.layers:
                raise RuntimeError('audited post-head latent is unavailable')
            hook = self.model.head.layers[-1].register_forward_hook(
                lambda module, args, output: captured.append(output.detach()))
        before = self.encoder_calls
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.agent.dtype,
                                               enabled=self.agent.amp_enabled):
                # Deliberately bypass upstream retry/OOM CPU fallback: exactly one model forward.
                logits, act = self.model(*(batch[key].to(self.agent.device) for key in
                                          ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')))
        finally:
            if hook is not None:
                hook.remove()
        if self.encoder_calls - before != 1:
            raise RuntimeError('Laya encoder must execute once')
        raw = logits.float().cpu().numpy()
        answers = self.agent._decode_answers(raw, act.float().softmax(-1).cpu().numpy(), items, ids, internal, 0)
        probabilities, logit_map = {}, {}
        for j, key in enumerate(ids):
            count = len(items[j]['markers'])
            kind = QTYPES[internal[key]['t']]
            temperature = self.agent.temperature_by_options.get(temp_bucket(kind, count), self.agent.temperature[kind])
            z = raw[j, :count] / temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            probabilities[key] = dict(zip(request.questions[key].option_ids(), map(float, p)))
            logit_map[key] = logits[j, :count].detach()
        latent = None
        if retain_latent:
            if len(captured) != 1:
                raise RuntimeError('expected exactly one final Laya hidden-state capture')
            h = captured[0]
            latent = Features(h.reshape(1, -1, h.shape[-1]), batch['attention_mask'].to(h.device).bool().reshape(1, -1))
        return BackendOutput(answers, probabilities, {k: a['answer_confidence'] for k, a in answers.items()},
                             logit_map, latent, {'usage': {'input_tokens': int(batch['attention_mask'].sum()),
                                                          'output_tokens': 0},
                                                'diagnostics': diagnostics(probabilities)})

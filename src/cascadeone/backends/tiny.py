import hashlib
import json

import torch
from torch import nn

from cascadeone.core.capabilities import Capabilities
from cascadeone.core.types import BackendIdentity, BackendOutput, Features, diagnostics
from cascadeone.training.safety import freeze


def make_output(request, logits, option_ids=None):
    answers, probabilities, confidence = {}, {}, {}
    for key, question in request.questions.items():
        ids = option_ids[key] if option_ids else question.option_ids()
        values = logits[key].float().softmax(-1).detach().cpu().tolist()
        probs = dict(zip(ids, values))
        probabilities[key] = probs
        confidence[key] = max(values)
        answer = {'type': question.type, 'probabilities': probs, 'answer_confidence': max(values)}
        if question.type == 'choice':
            answer['choice'] = max(probs, key=probs.get)
        elif question.type == 'noul':
            answer['noul'] = probs['true']
        else:
            answer['score'] = sum(int(i) * p for i, p in probs.items())
            answer['legend'] = dict(zip(question.option_ids(), question.criteria))
        answers[key] = answer
    return BackendOutput(answers, probabilities, confidence, logits,
                         metadata={'diagnostics': diagnostics(probabilities)})


class TinyModel(nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.encoder = nn.Embedding(256, width)
            self.finalizer = nn.Linear(width, 64)

    def encode(self, request):
        text = json.dumps(request.backend_record(), sort_keys=True, ensure_ascii=False).encode()
        tokens = torch.tensor(list(text[:96]) or [0], device=self.encoder.weight.device).unsqueeze(0)
        hidden = self.encoder(tokens)
        return Features(hidden, torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device))

    def finalize(self, request, features):
        pooled = (features.hidden * features.mask.unsqueeze(-1)).sum(1) / features.mask.sum(1, keepdim=True)
        logits = self.finalizer(pooled)[0]
        return make_output(request, {key: logits[:len(q.option_ids())] for key, q in request.questions.items()})


class TinyFastBackend:
    capabilities = Capabilities(supports_token_latents=True, supports_attention_mask=True)

    def __init__(self, width=12, seed=11, confidence=None):
        self.model = freeze(TinyModel(width, seed))
        self.identity = BackendIdentity('tiny_fast', 'tiny-fast', hashlib.sha256(str(seed).encode()).hexdigest(), width)
        self.calls = 0
        self.fixed_confidence = confidence  # test fixture only, never read from request

    def infer(self, request, *, retain_latent=False):
        self.calls += 1
        with torch.no_grad():
            features = self.model.encode(request)
            out = self.model.finalize(request, features)
        if self.fixed_confidence is not None:
            out.answer_confidence = {key: self.fixed_confidence for key in request.questions}
        out.latent = features if retain_latent else None
        return out


class TinyStrongBackend:
    capabilities = Capabilities(supports_token_latents=True, supports_attention_mask=True,
                                supports_latent_injection=True, supports_separate_finalizer=True)
    insertion_point = 'final_hidden_pre_head:v1'

    def __init__(self, width=16, seed=29):
        self.model = freeze(TinyModel(width, seed))
        self.identity = BackendIdentity('tiny_strong', 'tiny-strong', hashlib.sha256(str(seed).encode()).hexdigest(), width)
        self.calls = self.finalizer_calls = 0

    def encode(self, request):
        self.calls += 1
        with torch.no_grad():
            return self.model.encode(request)

    def finalize(self, request, features):
        self.finalizer_calls += 1
        return self.model.finalize(request, features)

    def infer(self, request):
        with torch.no_grad():
            return self.finalize(request, self.encode(request))

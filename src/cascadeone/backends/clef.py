"""Split pinned Clef.forward at its existing final-hidden-state -> JointSchemaHead boundary."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

from cascadeone.core.capabilities import Capabilities
from cascadeone.core.types import BackendIdentity, Features
from cascadeone.backends.tiny import make_output
from cascadeone.training.safety import freeze

MODEL_ID = 'Cloudflare/clef-flash'
REVISION = '17f0b0ad64efb65d273590632833508766b2aae6'
SOURCE_HASH = '0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3'


def import_upstream(path):
    source = Path(path) / 'joint_schema_model.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_HASH:
        raise RuntimeError('Clef source differs from audited revision')
    name = 'cascadeone_pinned_clef'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class ClefStrongBackend:
    capabilities = Capabilities(supports_token_latents=True, supports_attention_mask=True,
                                supports_latent_injection=True, supports_separate_finalizer=True)
    insertion_point = 'ClefModel.final_hidden_pre_JointSchemaHead:v1'

    def __init__(self, model_id=MODEL_ID, revision=REVISION, device='cuda', max_length=16384):
        if model_id != MODEL_ID or revision != REVISION:
            raise ValueError('reference backend requires the audited Clef checkpoint/revision')
        if device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('real Clef requires CUDA; use tiny on CPU')
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, revision=revision)
        self.upstream = import_upstream(path)
        self.model, self.processor = self.upstream.load_release_model(path, device=device, dtype=torch.bfloat16)
        freeze(self.model)
        self.base = self.model.language_model
        self.text_model = self.base.model
        if hasattr(self.text_model, 'language_model'):
            self.text_model = self.text_model.language_model
        self.identity = BackendIdentity('clef', model_id, revision, 4096, 'clef-17f0b0a-text-v1')
        self.max_length = max_length
        self.calls = self.backbone_calls = self.finalizer_calls = 0
        self.text_model.register_forward_pre_hook(self._count_backbone)

    def _count_backbone(self, module, args):
        self.backbone_calls += 1

    def encode(self, request):
        self.calls += 1
        # Encode without truncation, then reject overflow explicitly.
        record = self.upstream.encode_record(self.processor.tokenizer, request.backend_record(),
                                             max_length=2**31 - 1, processor=self.processor)
        if len(record.input_ids) > self.max_length:
            raise ValueError('request exceeds Clef context; silent truncation is disabled')
        device = next(self.model.parameters()).device
        batch = self.upstream.collate_records([record], self.processor.tokenizer.pad_token_id, device)
        with torch.no_grad():
            outputs = self.text_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                      use_cache=False, return_dict=True)
        return Features(outputs.last_hidden_state.detach(), batch['attention_mask'].bool(), batch)

    def finalize(self, request, features):
        self.finalizer_calls += 1
        batch = features.context
        # No no_grad/inference_mode here: Stage A backpropagates through frozen head math.
        rows = self.model.head(features.hidden, batch['input_ids'], batch['attention_mask'],
                               batch['records'], self.base.get_output_embeddings().weight)[0]
        encoded = batch['records'][0]
        ids = {q.question_id: q.option_ids for q in encoded.questions}
        output = make_output(request, dict(zip(ids, rows)), ids)
        output.answers = {key: self.upstream.systemone_answer(request.backend_record()['questions'][key], probs)
                          for key, probs in output.probabilities.items()}
        output.metadata['usage'] = {'input_tokens': len(encoded.input_ids), 'output_tokens': 0}
        return output

    def infer(self, request):
        with torch.no_grad():
            return self.finalize(request, self.encode(request))

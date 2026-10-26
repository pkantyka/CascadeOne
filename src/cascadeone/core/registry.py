import copy
from pathlib import Path

import torch
import yaml

from cascadeone.backends.tiny import TinyFastBackend, TinyStrongBackend
from cascadeone.core.executor import CascadeExecutor
from cascadeone.fusion.adapters import FUSIONS
from cascadeone.routing.confidence_threshold import ConfidenceThresholdPolicy


BACKENDS = {'tiny_fast': TinyFastBackend, 'tiny_strong': TinyStrongBackend}


def register_backend(name, factory):
    if name in BACKENDS:
        raise ValueError(f'backend already registered: {name}')
    BACKENDS[name] = factory


def backend(config):
    cfg = dict(config)
    kind = cfg.pop('type')
    if kind == 'laya':
        from cascadeone.backends.laya import LayaFastBackend
        return LayaFastBackend(**cfg)
    if kind == 'clef':
        from cascadeone.backends.clef import ClefStrongBackend
        return ClefStrongBackend(**cfg)
    if kind == 'opendecider_nano':
        from cascadeone.backends.opendecider import OpenDeciderNanoFastBackend
        return OpenDeciderNanoFastBackend(**cfg)
    if kind == 'decima_base':
        from cascadeone.backends.decima import DecimaBaseFastBackend
        return DecimaBaseFastBackend(**cfg)
    if kind not in BACKENDS:
        raise ValueError(f'unknown backend: {kind}')
    return BACKENDS[kind](**cfg)


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def build(config, *, training=False):
    cfg = copy.deepcopy(config)
    if cfg.get('framework') != 'cascadeone' or cfg.get('mode') not in ('core', 'fusion'):
        raise ValueError('framework=cascadeone and mode=core|fusion required')
    fcfg = cfg.get('fusion', {})
    if (cfg['mode'] == 'core') != (fcfg.get('type', 'none') == 'none'):
        raise ValueError('mode and fusion type disagree')
    fast, strong = backend(cfg['fast_backend']), backend(cfg['strong_backend'])
    fusion = None
    if cfg['mode'] == 'fusion':
        fcfg = dict(fcfg)
        kind, checkpoint = fcfg.pop('type'), fcfg.pop('checkpoint', None)
        if kind not in FUSIONS:
            raise ValueError(f'unknown fusion: {kind}')
        if not training and not checkpoint:
            raise ValueError('fusion inference requires a trained checkpoint')
        torch.manual_seed(cfg.get('training', {}).get('seed', 42))
        fusion = FUSIONS[kind](fast.identity.hidden_size, strong.identity.hidden_size, **fcfg)
        fusion.to(device=next(strong.model.parameters()).device)
        if checkpoint:
            from cascadeone.training.checkpointing import load_adapter
            load_adapter(checkpoint, fusion, fast, strong)
    return CascadeExecutor(fast, strong, ConfidenceThresholdPolicy(**cfg.get('routing', {})), fusion)

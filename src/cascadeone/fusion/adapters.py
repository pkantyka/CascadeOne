from abc import ABC, abstractmethod

import torch
from torch import nn

from cascadeone.core.types import Features


class FusionAdapter(nn.Module, ABC):
    interface_version = '1'
    calls = 0

    @abstractmethod
    def forward(self, fast: Features, strong: Features, routing_features=None) -> Features: ...

    def prepare(self, fast, strong):
        self.calls += 1
        fast.validate()
        strong.validate()
        if fast.hidden.shape[0] != strong.hidden.shape[0]:
            raise ValueError('latent batch sizes differ')
        parameter = next(self.parameters())
        return fast.hidden.to(device=parameter.device, dtype=parameter.dtype), strong.hidden.to(
            device=parameter.device, dtype=parameter.dtype)


class GatedResidualFusion(FusionAdapter):
    fusion_type = 'gated_residual'

    def __init__(self, fast_width, strong_width, width=64, dropout=0.0, **kwargs):
        super().__init__()
        self.config = dict(fast_width=fast_width, strong_width=strong_width, width=width, dropout=dropout)
        self.fast_proj = nn.Linear(fast_width, width)
        self.strong_proj = nn.Linear(strong_width, width)
        self.gate = nn.Linear(width * 2, 1)
        self.delta = nn.Linear(width, strong_width, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.delta.weight)  # exact identity; delta still receives gradients

    def forward(self, fast, strong, routing_features=None):
        f, s = self.prepare(fast, strong)
        mask = fast.mask.to(f.device).unsqueeze(-1)
        pooled = f.masked_fill(~mask, 0).sum(1) / mask.sum(1)
        projected = torch.tanh(self.fast_proj(pooled)).unsqueeze(1).expand(-1, s.shape[1], -1)
        gate = torch.sigmoid(self.gate(torch.cat((projected, self.strong_proj(s)), dim=-1)))
        delta = gate * self.delta(self.dropout(projected))
        delta = delta.masked_fill(~strong.mask.to(s.device).unsqueeze(-1), 0)
        return Features(strong.hidden + delta.to(strong.hidden), strong.mask, strong.context)


class CrossAttentionFusion(FusionAdapter):
    fusion_type = 'cross_attention'

    def __init__(self, fast_width, strong_width, width=64, heads=4, layers=1, dropout=0.0):
        super().__init__()
        if width % heads or layers < 1:
            raise ValueError('fusion width must divide into heads; layers must be positive')
        self.config = dict(fast_width=fast_width, strong_width=strong_width, width=width,
                           heads=heads, layers=layers, dropout=dropout)
        self.q = nn.Linear(strong_width, width)
        self.kv = nn.Linear(fast_width, width)
        self.attention = nn.ModuleList([nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
                                       for _ in range(layers)])
        self.gate = nn.Linear(width, 1)
        self.delta = nn.Linear(width, strong_width, bias=False)
        nn.init.zeros_(self.delta.weight)

    def forward(self, fast, strong, routing_features=None):
        f, s = self.prepare(fast, strong)
        q, kv = self.q(s), self.kv(f)
        memory_mask = fast.mask.to(f.device)
        kv = kv.masked_fill(~memory_mask.unsqueeze(-1), 0)
        for layer in self.attention:
            update, _ = layer(q, kv, kv, key_padding_mask=~memory_mask, need_weights=False)
            q = q + update
        delta = self.delta(q) * torch.sigmoid(self.gate(q))
        delta = delta.masked_fill(~strong.mask.to(s.device).unsqueeze(-1), 0)
        return Features(strong.hidden + delta.to(strong.hidden), strong.mask, strong.context)


FUSIONS = {'gated_residual': GatedResidualFusion, 'cross_attention': CrossAttentionFusion}

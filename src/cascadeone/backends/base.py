from typing import Protocol

from torch import nn

from cascadeone.core.capabilities import Capabilities
from cascadeone.core.types import BackendIdentity, BackendOutput, DecisionRequest, Features


class FastBackend(Protocol):
    model: nn.Module
    identity: BackendIdentity
    capabilities: Capabilities

    def infer(self, request: DecisionRequest, *, retain_latent: bool = False) -> BackendOutput: ...


class StrongBackend(Protocol):
    model: nn.Module
    identity: BackendIdentity
    capabilities: Capabilities

    def infer(self, request: DecisionRequest) -> BackendOutput: ...


class FusionStrongBackend(StrongBackend, Protocol):
    insertion_point: str

    def encode(self, request: DecisionRequest) -> Features: ...
    def finalize(self, request: DecisionRequest, features: Features) -> BackendOutput: ...

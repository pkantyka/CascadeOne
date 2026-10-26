from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def threshold_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("threshold must be a number, not a boolean or string")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    return float(value)


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    escalation_threshold: float | None = None

    @field_validator("escalation_threshold", mode="before")
    @classmethod
    def validate_threshold(cls, value):
        return None if value is None else threshold_value(value)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "noul", "score"]
    instructions: str = ""
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.type == "choice" and (not isinstance(self.criteria, dict) or not self.criteria):
            raise ValueError("choice requires a nonempty mapping of options")
        if self.type == "score" and (not isinstance(self.criteria, list) or len(self.criteria) < 2):
            raise ValueError("score requires at least two ordered levels")
        if self.type == "noul" and self.criteria is not None:
            if not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}:
                raise ValueError("noul criteria must use true/false keys")
        if any(not key for key in self.option_ids()):
            raise ValueError("empty option identifier")
        return self

    def option_ids(self) -> tuple[str, ...]:
        if self.type == "noul":
            return ("false", "true")
        if self.type == "score":
            return tuple(str(i) for i in range(len(self.criteria or [])))
        return tuple(self.criteria or {})


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "cascadeone"
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1, max_length=32)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    min_confidence: float | None = None

    @field_validator("min_confidence", mode="before")
    @classmethod
    def validate_threshold(cls, value):
        return None if value is None else threshold_value(value)

    @field_validator("questions")
    @classmethod
    def question_ids(cls, value):
        if any(not key for key in value):
            raise ValueError("empty question identifier")
        return value

    def backend_record(self):
        return {"model": self.model, "state": self.state,
                "questions": {key: {**q.model_dump(exclude_none=True),
                                     "instructions": q.instructions or key}
                              for key, q in self.questions.items()}}


@dataclass
class Features:
    hidden: torch.Tensor  # [batch, tokens, width]; backend owns sequence packing
    mask: torch.Tensor  # bool, True = valid token
    context: Any = None  # opaque finalizer context, never sent to clients

    def validate(self):
        if self.hidden.ndim != 3 or self.mask.shape != self.hidden.shape[:2]:
            raise ValueError("invalid latent/mask shape")
        if self.mask.dtype != torch.bool or not self.mask.any(dim=1).all():
            raise ValueError("each sequence requires at least one valid token")
        if self.mask.device != self.hidden.device:
            raise ValueError("latent and mask devices differ")


@dataclass
class FastRoutePreparation:
    eligible: bool
    reason: str | None = None
    prepared: Any = None


@dataclass
class BackendOutput:
    answers: dict[str, dict[str, Any]]
    probabilities: dict[str, dict[str, float]]
    answer_confidence: dict[str, float | None]
    logits: dict[str, torch.Tensor] = field(default_factory=dict)
    latent: Features | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendIdentity:
    type: str
    model_id: str
    revision: str
    hidden_size: int
    preprocessing_version: str = "1"


def diagnostics(probabilities):
    result = {}
    for key, values in probabilities.items():
        ps = sorted(values.values(), reverse=True)
        result[key] = {"top1": ps[0], "top2": ps[1] if len(ps) > 1 else None,
                       "margin": ps[0] - ps[1] if len(ps) > 1 else 1.0,
                       "entropy": -sum(p * math.log(p) for p in ps if p > 0),
                       "option_count": len(ps)}
    return result

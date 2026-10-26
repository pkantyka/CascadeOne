"""Pluggable request-level routing policies, independent of benchmark measurement.

A policy decides *whether* to escalate. It does not perform inference.
Plug-ins are trusted Python code installed by the cascade author.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Any


@dataclass(frozen=True)
class Decision:
    escalate: bool
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


class Policy:
    def before(self, request):
        """Return Decision to bypass fast model, otherwise None."""
        return None

    def after(self, request, fast_response):
        """Return Decision after fast inference, when it was eligible."""
        raise NotImplementedError


class ConfidenceThreshold(Policy):
    def __init__(self, threshold, aggregation="minimum"):
        if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite number in [0,1]")
        if aggregation not in ("minimum", "mean", "maximum"):
            raise ValueError("aggregation must be minimum, mean or maximum")
        self.threshold=float(threshold)
        self.aggregation=aggregation

    def after(self, request, fast_response):
        confidences=[]
        for qid in request["questions"]:
            value=fast_response["answers"][qid].get("confidence")
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                return Decision(True, "missing_or_invalid_confidence")
            confidences.append(float(value))
        value={"minimum":min,"maximum":max,
               "mean":lambda values:sum(values)/len(values)}[self.aggregation](confidences)
        return Decision(value < self.threshold, "confidence_threshold",
                        {"confidence":value,"threshold":self.threshold,
                         "aggregation":self.aggregation})


class AlwaysFast(Policy):
    def after(self, request, fast_response):
        return Decision(False, "always_fast")


class AlwaysStrong(Policy):
    def before(self, request):
        return Decision(True, "always_strong")


class MaxQuestionPreRouter(Policy):
    """Example confidence-free pre-router; bypass fast before its execution."""
    def __init__(self, max_questions):
        if type(max_questions) is not int or max_questions < 1:
            raise ValueError("max_questions must be a positive integer")
        self.max_questions=max_questions

    def before(self, request):
        if len(request["questions"]) > self.max_questions:
            return Decision(True, "question_count_pre_route",
                            {"question_count":len(request["questions"])})
        return None

    def after(self, request, fast_response):
        return Decision(False, "pre_router_accepted")


BUILTINS = {
    "confidence_threshold":ConfidenceThreshold,
    "always_fast":AlwaysFast,
    "always_strong":AlwaysStrong,
    "max_questions_pre_router":MaxQuestionPreRouter,
}


def build_policy(config):
    if not isinstance(config,dict) or not isinstance(config.get("type"),str):
        raise ValueError("routing.type is required")
    name=config["type"]
    params=config.get("parameters",{})
    if not isinstance(params,dict): raise ValueError("routing.parameters must be an object")
    if name=="plugin":
        target=config.get("factory")
        if not isinstance(target,str) or target.count(":")!=1:
            raise ValueError("plugin requires module:factory")
        module,attribute=target.split(":")
        if not module or not attribute: raise ValueError("invalid plugin factory")
        factory=getattr(importlib.import_module(module),attribute)
    else:
        if name not in BUILTINS:
            raise ValueError("unknown routing policy: "+name)
        factory=BUILTINS[name]
    policy=factory(**params)
    if not callable(getattr(policy,"before",None)) or not callable(getattr(policy,"after",None)):
        raise TypeError("policy must provide before(request) and after(request,fast_response)")
    return policy


def validate_decision(decision):
    if not isinstance(decision,Decision) or type(decision.escalate) is not bool:
        raise TypeError("policy must return a Decision with boolean escalate")
    if not isinstance(decision.reason,str) or not decision.reason or not isinstance(decision.signals,dict):
        raise TypeError("invalid route decision metadata")
    return decision

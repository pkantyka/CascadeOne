"""Cascade execution independent of benchmark runner and Gold labels.

The invoke callback is an HTTP or custom backend transport:
    invoke(url, request, timeout, fast=True|False) -> (response, elapsed_ms)
Only the policy determines routing; all decisions and fallback overhead are measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from policies import Decision, build_policy, validate_decision


@dataclass
class Outcome:
    response: dict
    route: dict
    fast_ms: float
    strong_ms: float

    @property
    def escalated(self):
        return self.route["escalated"]


class Cascade:
    def __init__(self, config):
        if not isinstance(config,dict) or "routing" not in config:
            raise ValueError("cascade routing configuration is required")
        if "threshold" in config:
            raise ValueError("top-level threshold is obsolete; use routing.parameters")
        self.config=config
        self.policy=build_policy(config["routing"])
        self.fast=config["fast"]["url"]
        self.strong=config["strong"]["url"]

    def execute(self, request, timeout, invoke:Callable):
        fast_ms=strong_ms=0.0
        pre=self.policy.before(request)
        if pre is not None:
            decision=validate_decision(pre)
            if not decision.escalate:
                raise ValueError("pre-routing must escalate or defer to fast inference")
            chosen=None
            fast_executed=False
        else:
            fast_response,fast_ms=invoke(self.fast,request,timeout,fast=True)
            fast_executed=True
            if fast_response.get("fast_eligible") is False:
                decision=Decision(True,"fast_backend_ineligible",
                                  {"eligibility_reason":fast_response.get("reason")})
                chosen=None
            else:
                decision=validate_decision(self.policy.after(request,fast_response))
                chosen=None if decision.escalate else fast_response
        if decision.escalate:
            chosen,strong_ms=invoke(self.strong,request,timeout,fast=False)
        route={"policy":self.config["routing"]["type"],
               "escalated":decision.escalate,"reason":decision.reason,
               "signals":decision.signals,
               "fast_executed":fast_executed,
               "strong_executed":decision.escalate}
        return Outcome(chosen,route,fast_ms,strong_ms)

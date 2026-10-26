import math
from dataclasses import dataclass

from cascadeone.core.types import threshold_value


@dataclass(frozen=True)
class RouteDecision:
    accepted: bool
    threshold: float
    confidence: float | None
    reason: str


class ConfidenceThresholdPolicy:
    def __init__(self, default_threshold=0.9, max_fast_choice_options=None):
        self.default_threshold = threshold_value(default_threshold)
        if max_fast_choice_options is not None:
            if isinstance(max_fast_choice_options, bool) or not isinstance(max_fast_choice_options, int):
                raise ValueError("max_fast_choice_options must be an integer or null")
            if max_fast_choice_options < 1:
                raise ValueError("max_fast_choice_options must be positive")
        self.max_fast_choice_options = max_fast_choice_options

    def threshold(self, request):
        for value in (request.routing.escalation_threshold, request.min_confidence, self.default_threshold):
            if value is not None:
                return threshold_value(value)

    def pre_fast_decide(self, request):
        if self.max_fast_choice_options is None:
            return None
        if any(
            question.type == "choice" and len(question.option_ids()) > self.max_fast_choice_options
            for question in request.questions.values()
        ):
            return RouteDecision(
                False,
                self.threshold(request),
                None,
                "fast_choice_option_count_exceeds_calibrated_limit",
            )
        return None

    def decide(self, request, output):
        threshold = self.threshold(request)
        values = [output.answer_confidence.get(key) for key in request.questions]
        if any(isinstance(c, bool) or not isinstance(c, (int, float)) or
               not math.isfinite(c) or not 0 <= c <= 1 for c in values):
            return RouteDecision(False, threshold, None, "invalid_confidence")
        confidence = min(values)
        return RouteDecision(confidence >= threshold, threshold, confidence, "minimum_question_confidence")

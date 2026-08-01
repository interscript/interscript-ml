"""PER (Phoneme Error Rate) evaluator for secryst tasks.

PER = (S + D + I) / N where counts are over IPA phoneme tokens. Same
math as DER, different domain name. Reuses the framework's
``token_error_rate`` so the edit-distance implementation is shared.
"""

from __future__ import annotations

from typing import Sequence

from framework.evaluator import (
    BaseEvaluator,
    accuracy,
    most_common_error_pairs,
    token_error_rate,
)
from framework.registry import register_evaluator


@register_evaluator("per")
class PEREvaluator(BaseEvaluator):
    """Phoneme Error Rate for transliteration tasks."""

    name = "per"

    def compute_metric(self, predictions: Sequence[str], gold: Sequence[str]) -> float:
        total = 0.0
        for pred, ref in zip(predictions, gold):
            total += token_error_rate(list(pred), list(ref))
        return total / len(predictions)

    def _extra_metrics(self, predictions, gold) -> dict[str, float]:
        return {
            "accuracy": accuracy(predictions, gold),
            **{
                f"err_{k}": v
                for k, v in most_common_error_pairs(predictions, gold).items()
            },
        }

"""DER (Diacritic Error Rate) evaluator for rababa tasks.

DER = (S + D + I) / N, where S/D/I are substitutions/deletions/insertions
of diacritized characters, and N is the count of diacritized characters
in the gold.

Computed via Levenshtein at the character level — the framework's
shared ``edit_distance`` utility. The evaluator is task-specific only
in what it counts (diacritized chars, not raw chars).
"""

from __future__ import annotations

from typing import Sequence

from framework.evaluator import (
    BaseEvaluator,
    accuracy,
    char_error_rate,
    most_common_error_pairs,
)
from framework.registry import register_evaluator


@register_evaluator("der")
class DEREvaluator(BaseEvaluator):
    """Diacritic Error Rate over a held-out corpus."""

    name = "der"

    def compute_metric(self, predictions: Sequence[str], gold: Sequence[str]) -> float:
        total = 0.0
        for pred, ref in zip(predictions, gold):
            total += char_error_rate(pred, ref)
        return total / len(predictions)

    def _extra_metrics(self, predictions, gold) -> dict[str, float]:
        return {
            "accuracy": accuracy(predictions, gold),
            **{
                f"err_{k}": v
                for k, v in most_common_error_pairs(predictions, gold).items()
            },
        }

"""Evaluator abstractions.

Each task defines its own metric (DER for diacritization, PER for
phoneme transliteration, WER for word-level). All inherit
``BaseEvaluator``. The framework calls ``evaluate(predictions, gold)``
— no task knowledge leaks into framework code.

OCP: adding a new metric = new subclass + ``@register_evaluator``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class MetricSet:
    """Frozen snapshot of one evaluation run."""

    metric: str
    value: float
    extra: dict[str, float] = field(default_factory=dict)

    def passes(self, target: float) -> bool:
        """True if ``value`` is at or below ``target`` (lower-is-better metrics)."""
        return self.value <= target

    def format(self) -> str:
        extras = " ".join(f"{k}={v:.4f}" for k, v in self.extra.items())
        return f"{self.metric}={self.value:.4f} {extras}".strip()


class BaseEvaluator(ABC):
    """Abstract evaluator. Subclasses implement ``compute_metric``."""

    name: str = "base"

    @abstractmethod
    def compute_metric(
        self,
        predictions: Sequence[str],
        gold: Sequence[str],
    ) -> float:
        """Return the scalar metric value (lower is better)."""

    def evaluate(
        self,
        predictions: Sequence[str],
        gold: Sequence[str],
    ) -> MetricSet:
        if len(predictions) != len(gold):
            raise ValueError(
                f"predictions/gold length mismatch: {len(predictions)} vs {len(gold)}"
            )
        if not predictions:
            raise ValueError("Cannot evaluate on empty predictions")
        value = self.compute_metric(predictions, gold)
        extras = self._extra_metrics(predictions, gold)
        return MetricSet(metric=self.name, value=value, extra=extras)

    def _extra_metrics(
        self,
        predictions: Sequence[str],
        gold: Sequence[str],
    ) -> dict[str, float]:
        """Hook for subclasses to add complementary metrics (e.g. accuracy)."""
        return {}


def edit_distance(a: Sequence[object], b: Sequence[object]) -> int:
    """Levenshtein distance. Shared by DER/PER — DRY."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i]
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_error_rate(pred: str, gold: str) -> float:
    """CER = edit_distance / len(gold)."""
    if not gold:
        return 0.0 if not pred else 1.0
    return edit_distance(list(pred), list(gold)) / len(gold)


def token_error_rate(
    pred: Sequence[str],
    gold: Sequence[str],
) -> float:
    """TER for token sequences (words, phonemes, characters)."""
    if not gold:
        return 0.0 if not pred else 1.0
    return edit_distance(list(pred), list(gold)) / len(gold)


def accuracy(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Exact-match accuracy across the corpus."""
    if not pred:
        return 0.0
    correct = sum(1 for p, g in zip(pred, gold) if p == g)
    return correct / len(pred)


def most_common_error_pairs(
    pred: Sequence[str],
    gold: Sequence[str],
    top_k: int = 5,
) -> dict[str, float]:
    """Frequency of (gold, pred) character-level substitutions."""
    counts: Counter[tuple[str, str]] = Counter()
    for p, g in zip(pred, gold):
        for pc, gc in zip(p, g):
            if pc != gc:
                counts[(gc, pc)] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {f"{g}->{p}": c / total for (g, p), c in counts.most_common(top_k)}

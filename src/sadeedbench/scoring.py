"""Score predictions against SadeedDiac-25 gold under the campaign's
windowed DER-CE convention.

The convention (docs/paper-a.adoc section 3): every paragraph counts
(zero-skip), haraqat are projected before scoring, and the aggregate is
the Misraj evaluator's Total DER with case endings, reported in
percent. Predictions are whatever a model emitted for the full test
set — same order as the benchmark rows."""

from __future__ import annotations

from typing import Iterable


def score_predictions(preds: list[str], gts: list[str]) -> dict:
    """Aggregate DER-CE over aligned (pred, gt) paragraph pairs."""
    from sadeedbench.vendored_sadeed_evaluator import (
        ArabicDiacritizationEvaluator,
    )

    if len(preds) != len(gts):
        raise ValueError(f"preds/gts length mismatch: {len(preds)} vs {len(gts)}")
    _, _, total_der, _, _ = ArabicDiacritizationEvaluator.caculate_errors_on_sentences(
        preds, gts, gt_missing_diacritic_is_error=False
    )
    return {"der_ce": round(float(total_der), 4), "n": len(preds)}


def per_item_der(preds: list[str], gts: list[str]) -> list[float | None]:
    """Per-paragraph DER for paired bootstrap; None where the paragraph
    carries no scorable positions (ZeroDivisionError in the evaluator's
    single-item mode — the aggregate call skips those paragraphs)."""
    from sadeedbench.vendored_sadeed_evaluator import (
        ArabicDiacritizationEvaluator,
    )

    ders: list[float | None] = []
    for pred, gt in zip(preds, gts, strict=True):
        try:
            _, _, d, _, _ = (
                ArabicDiacritizationEvaluator.caculate_errors_on_sentences(
                    [pred], [gt], gt_missing_diacritic_is_error=False
                )
            )
            ders.append(float(d))
        except ZeroDivisionError:
            ders.append(None)
    return ders

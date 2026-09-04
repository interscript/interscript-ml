"""The protocol as a tool: score any predictions file on SadeedDiac-25
under the campaign's windowed DER-CE convention, with paired
bootstrap against a reference predictions file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytest.importorskip("pyarabic")

from sadeedbench.scoring import score_predictions  # noqa: E402

GT = ["قَوْلُهُ فَحُكْمُهَا"] * 4


def test_perfect_predictions_score_zero() -> None:
    assert score_predictions(GT, GT)["der_ce"] == 0.0


def test_stripped_predictions_score_high() -> None:
    # bare text (all haraqat removed) is the collapse constant
    out = score_predictions(["قوله فحكمها"] * 4, GT)
    assert out["der_ce"] > 50.0
    assert out["n"] == 4


def test_partial_predictions_score_between() -> None:
    perfect = score_predictions(GT, GT)["der_ce"]
    stripped = score_predictions(["قوله فحكمها"] * 4, GT)["der_ce"]
    half = score_predictions([GT[0], "قوله فحكمها", GT[2], "قوله فحكمها"], GT)
    assert perfect < half["der_ce"] < stripped

"""Unit tests for the Sadeed windowed harness — the published protocol,
tested directly instead of through 6-hour GPU runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.sadeed import (  # noqa: E402
    DIACRITICS_RE,
    project_haraqat,
    split_windows,
    strip_diacritics,
)


def test_split_windows_short_text_is_single_window() -> None:
    assert split_windows("قصيرة", 1400) == ["قصيرة"]


def test_split_windows_respects_byte_budget() -> None:
    words = ["كلمة"] * 50  # 2 bytes header + 4 bytes per word + space
    text = " ".join(words)
    budget = 60
    windows = split_windows(text, budget)
    assert len(windows) > 1
    for w in windows:
        assert len(w.encode("utf-8")) <= budget


def test_split_windows_never_drops_words() -> None:
    words = ["a" * 30] * 7
    text = " ".join(words)
    windows = split_windows(text, 100)
    rejoined = " ".join(windows).split()
    assert rejoined == text.split()


def test_project_haraqat_identity_on_equal_letters() -> None:
    text = strip_diacritics("قَالَ")
    assert project_haraqat("قَالَ", text) == "قَالَ"


def test_project_haraqat_survives_prediction_insertions() -> None:
    # prediction drops a letter (ب) — projection must not shift haraqat
    # onto the wrong letters; dropped region emits bare text letters
    text = "كتب"
    pred = "كَتَ"  # missing final ب
    out = project_haraqat(pred, text)
    assert strip_diacritics(out) == text
    assert out.startswith("كَتَ")


def test_project_haraqat_extra_prediction_letters() -> None:
    text = "قل"
    pred = "قَوْلٌ"  # و inserted, haraqat on others
    out = project_haraqat(pred, text)
    assert strip_diacritics(out) == text


def test_diacritics_regex_covers_haraqat_and_shadda() -> None:
    for ch in "ًٌٍَُِّْٰ":
        assert DIACRITICS_RE.match(ch), ch

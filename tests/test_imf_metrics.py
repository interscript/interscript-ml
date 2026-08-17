"""Tests for the WO10 metrics generator (inline markdown; no network —
the network provenance check runs in CI as `imf metrics`)."""

from __future__ import annotations

import pytest

from imf.metrics import MetricsError, TableSpec, _slugify, extract, parse_tables

SAMPLE = """# Results

## G2P (Urdu text → IPA)

### Best result

| Metric | Value | Test set |
|---|---|---|
| PER (word-level) | 72.0%* | 12,699 held-out |
| **CER (char-level)** | **14.77%** | 12,699 held-out |
| Exact match | 33.6% | 12,699 held-out |

## Khmer transliteration (2026-08-14)

| System | EM | CER | n |
|---|---|---|---|
| ByT5-small, early stop @ep15 | **59.66%** | **27.42%** | 895 |

## Key findings

prose without tables
"""


def test_slugify_matches_section_headings() -> None:
    assert _slugify("G2P (Urdu text → IPA)") == "g2p-urdu-text-ipa"
    assert _slugify("Khmer transliteration (2026-08-14)") == "khmer-transliteration-2026-08-14"


def test_parse_tables_scopes_to_anchor_section() -> None:
    tables = parse_tables(SAMPLE, "g2p-urdu-text-ipa")
    assert len(tables) == 1
    labels = [row["label"] for row in tables[0]["rows"]]
    assert "PER (word-level)" in labels


def test_extract_row_mode_takes_first_value_cell() -> None:
    metrics = extract(
        SAMPLE,
        "g2p-urdu-text-ipa",
        (
            TableSpec(row="CER (char-level)", as_name="cer"),
            TableSpec(row="Exact match", as_name="em"),
        ),
    )
    assert metrics == [{"name": "cer", "value": 14.77}, {"name": "em", "value": 33.6}]


def test_extract_column_mode() -> None:
    metrics = extract(
        SAMPLE,
        "khmer-transliteration-2026-08-14",
        (TableSpec(row="ByT5-small, early stop", column="CER", as_name="cer"),),
    )
    assert metrics == [{"name": "cer", "value": 27.42}]


def test_missing_anchor_raises() -> None:
    with pytest.raises(MetricsError, match="anchor"):
        parse_tables(SAMPLE, "no-such-section")


def test_missing_row_raises() -> None:
    with pytest.raises(MetricsError, match="row"):
        extract(SAMPLE, "g2p-urdu-text-ipa", (TableSpec(row="Nonexistent row"),))

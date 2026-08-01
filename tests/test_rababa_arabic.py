"""Tests for the rababa_arabic task package."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_arabic_data_prep_uses_fallback_when_missing(tmp_path) -> None:
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    cfg = DataConfig(
        module="rababa_arabic_data",
        source="missing",
        max_val_samples=2,
    )
    data = RababaArabicData(cfg, tmp_path)
    prepared = data.prepare_data()
    assert len(prepared.train) > 0
    assert len(prepared.val) > 0
    assert prepared.vocab_size > 10
    assert prepared.max_seq_len > 1


def test_arabic_data_prep_is_idempotent(tmp_path) -> None:
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    cfg = DataConfig(module="rababa_arabic_data", source="missing", max_val_samples=2)
    data = RababaArabicData(cfg, tmp_path)
    first = data.prepare_data()
    second = data.prepare_data()
    assert first is second


def test_arabic_encode_source_round_trip() -> None:
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData, clean_arabic, strip_diacritics

    cfg = DataConfig(module="rababa_arabic_data", source="x")
    data = RababaArabicData(cfg, __import__("pathlib").Path("/tmp"))
    ids = data.encode_source("كَتَبَ")
    # Source encoder ignores harakat (INPUT_VOCAB doesn't contain them)
    assert all(isinstance(i, int) for i in ids)
    # Strip-diacritics is the right helper for the bare form
    assert strip_diacritics("كَتَبَ") == "كتب"
    # Cleaner drops non-Arabic chars
    assert "X" not in clean_arabic("كتبXعلم")


def test_arabic_data_module_registered() -> None:
    from framework.registry import resolve_data_module

    assert resolve_data_module("rababa_arabic_data").__name__ == "RababaArabicData"


def test_arabic_evaluator_registered() -> None:
    from framework.registry import resolve_evaluator

    evaluator_cls = resolve_evaluator("der")
    assert evaluator_cls.__name__ == "DEREvaluator"


def test_arabic_model_registered() -> None:
    from framework.registry import resolve_model_module

    cls = resolve_model_module("rababa_student")
    assert cls.__name__ == "RababaStudent"


def test_der_evaluator_computes_correctly() -> None:
    from tasks.rababa_arabic.metrics import DEREvaluator

    evaluator = DEREvaluator()
    metric = evaluator.evaluate(["كَتَبَ"], ["كَتَبَ"])
    assert metric.value == 0.0
    metric2 = evaluator.evaluate(["كَتَبَ"], ["كَتَبْ"])
    assert metric2.value > 0.0

"""Tests for rababa_hebrew task package."""

from __future__ import annotations


def test_hebrew_data_prep_fallback(tmp_path) -> None:
    from framework.config import DataConfig
    from tasks.rababa_hebrew.data import RababaHebrewData

    cfg = DataConfig(module="rababa_hebrew_data", source="missing", max_val_samples=2)
    data = RababaHebrewData(cfg, tmp_path)
    prepared = data.prepare_data()
    assert len(prepared.train) > 0
    assert prepared.vocab_size > 10


def test_hebrew_strip_nikud() -> None:
    from tasks.rababa_hebrew.data import strip_nikud

    assert strip_nikud("בְּרֵאשִׁית") == "בראשית"


def test_hebrew_modules_registered() -> None:
    from framework.registry import resolve_data_module, resolve_evaluator, resolve_model_module

    assert resolve_data_module("rababa_hebrew_data").__name__ == "RababaHebrewData"
    assert resolve_evaluator("der_hebrew").__name__ == "DEREvaluatorHebrew"
    assert resolve_model_module("rababa_hebrew_student").__name__ == "RababaHebrewStudent"

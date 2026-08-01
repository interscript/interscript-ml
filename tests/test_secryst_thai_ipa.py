"""Tests for secryst_thai_ipa task package."""

from __future__ import annotations


def test_thai_data_prep_fallback(tmp_path) -> None:
    from framework.config import DataConfig
    from tasks.secryst_thai_ipa.data import SecrystThaiIpaData

    cfg = DataConfig(module="secryst_thai_ipa_data", source="missing", max_val_samples=2)
    data = SecrystThaiIpaData(cfg, tmp_path)
    prepared = data.prepare_data()
    assert len(prepared.train) > 0
    assert prepared.vocab_size > 10


def test_per_evaluator() -> None:
    from tasks.secryst_thai_ipa.metrics import PEREvaluator

    evaluator = PEREvaluator()
    metric = evaluator.evaluate(["saː.wàt.diː"], ["saː.wàt.diː"])
    assert metric.value == 0.0
    metric2 = evaluator.evaluate(["rak"], ["rák"])
    assert metric2.value >= 0.0


def test_thai_modules_registered() -> None:
    from framework.registry import resolve_data_module, resolve_evaluator, resolve_model_module

    assert resolve_data_module("secryst_thai_ipa_data").__name__ == "SecrystThaiIpaData"
    assert resolve_evaluator("per").__name__ == "PEREvaluator"
    assert resolve_model_module("secryst_student").__name__ == "SecrystThaiIpaStudent"

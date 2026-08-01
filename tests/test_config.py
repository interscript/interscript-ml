"""Tests for ``framework.config``.

Confirms YAML loading + dataclass construction + immutability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from framework.config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    TrainConfig,
    load_task_config,
)


def test_load_task_config_arabic(tmp_path: Path) -> None:
    cfg = load_task_config("rababa_arabic")
    assert cfg.name == "rababa_arabic"
    assert cfg.kind == "rababa"
    assert cfg.data.module == "rababa_arabic_data"
    assert cfg.model.module == "rababa_student"
    assert cfg.eval.module == "der"


def test_load_task_config_hebrew() -> None:
    cfg = load_task_config("rababa_hebrew")
    assert cfg.kind == "rababa"
    assert cfg.data.module == "rababa_hebrew_data"
    assert cfg.eval.module == "der_hebrew"


def test_load_task_config_secryst() -> None:
    cfg = load_task_config("secryst_thai_ipa")
    assert cfg.kind == "secryst"
    assert cfg.eval.module == "per"


def test_task_config_is_frozen() -> None:
    cfg = load_task_config("rababa_arabic")
    with pytest.raises(AttributeError):
        cfg.name = "renamed"  # type: ignore[misc]


def test_missing_task_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_task_config("does_not_exist", tasks_root=tmp_path)


def test_data_config_defaults() -> None:
    c = DataConfig.from_dict({"module": "x", "source": "y"})
    assert c.max_val_samples == 1000
    assert c.cleaner == "basic"


def test_train_config_defaults() -> None:
    c = TrainConfig.from_dict({})
    assert c.epochs == 3
    assert c.batch_size == 16
    assert c.distill_temperature == 4.0


def test_model_config_lora() -> None:
    # teacher_name is optional — direct supervised is the default path
    c = ModelConfig.from_dict(
        {"module": "m", "student_arch": "char_transformer"}
    )
    assert c.teacher_name is None
    assert c.lora_r == 16
    assert c.student_dim == 384


def test_eval_config() -> None:
    c = EvalConfig.from_dict({"module": "der", "metric": "der"})
    assert c.target_value == 0.05

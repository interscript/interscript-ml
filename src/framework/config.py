"""Task configuration loaded from YAML.

A ``TaskConfig`` is the single source of truth for one ML task. It names
the data module, model module, evaluator, and all hyperparameters. The
CLI loads this, then resolves the registered classes by name — no
``if task == "rababa"`` chains anywhere in framework code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Data pipeline knobs. MECE: data layer owns these."""

    module: str
    source: str
    max_train_samples: int | None = None
    max_val_samples: int | None = 1000
    cleaner: str = "basic"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DataConfig:
        return cls(
            module=raw["module"],
            source=raw["source"],
            max_train_samples=raw.get("max_train_samples"),
            max_val_samples=raw.get("max_val_samples", 1000),
            cleaner=raw.get("cleaner", "basic"),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture knobs. MECE: model layer owns these."""

    module: str
    teacher_name: str
    student_arch: str
    student_layers: int = 4
    student_dim: int = 256
    student_heads: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        return cls(
            module=raw["module"],
            teacher_name=raw["teacher_name"],
            student_arch=raw["student_arch"],
            student_layers=raw.get("student_layers", 4),
            student_dim=raw.get("student_dim", 256),
            student_heads=raw.get("student_heads", 4),
            lora_r=raw.get("lora_r", 16),
            lora_alpha=raw.get("lora_alpha", 32),
            device=raw.get("device", "auto"),
        )


@dataclass(frozen=True)
class TrainConfig:
    """Trainer knobs. MECE: trainer layer owns these."""

    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    distill_temperature: float = 4.0
    distill_alpha: float = 0.5
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 1000
    out_dir: str = "models"
    max_steps_per_epoch: int | None = None  # cap for CPU dev mode

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        return cls(
            epochs=raw.get("epochs", 3),
            batch_size=raw.get("batch_size", 16),
            learning_rate=raw.get("learning_rate", 2e-4),
            weight_decay=raw.get("weight_decay", 0.01),
            warmup_steps=raw.get("warmup_steps", 100),
            distill_temperature=raw.get("distill_temperature", 4.0),
            distill_alpha=raw.get("distill_alpha", 0.5),
            grad_clip=raw.get("grad_clip", 1.0),
            log_every=raw.get("log_every", 50),
            save_every=raw.get("save_every", 1000),
            out_dir=raw.get("out_dir", "models"),
            max_steps_per_epoch=raw.get("max_steps_per_epoch"),
        )


@dataclass(frozen=True)
class EvalConfig:
    """Evaluator knobs. MECE: evaluator layer owns these."""

    module: str
    metric: str
    target_value: float = 0.05
    batch_size: int = 32

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalConfig:
        return cls(
            module=raw["module"],
            metric=raw["metric"],
            target_value=raw.get("target_value", 0.05),
            batch_size=raw.get("batch_size", 32),
        )


@dataclass(frozen=True)
class ExportConfig:
    """ONNX export knobs."""

    opset: int = 17
    dynamic_axes: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"input_ids": {0: "batch", 1: "seq"}}
    )
    quantize: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExportConfig:
        return cls(
            opset=raw.get("opset", 17),
            dynamic_axes=raw.get("dynamic_axes", {"input_ids": {0: "batch", 1: "seq"}}),
            quantize=raw.get("quantize", False),
        )


@dataclass(frozen=True)
class TaskConfig:
    """Top-level config for one ML task.

    Loaded from ``src/tasks/<name>/config.yaml``. Frozen so it cannot be
    mutated mid-run — reproducibility requires immutability.
    """

    name: str
    description: str
    kind: str  # "rababa" | "secryst" — used by interscript-ts registry
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    export: ExportConfig

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> TaskConfig:
        return cls(
            name=name,
            description=raw.get("description", ""),
            kind=raw["kind"],
            data=DataConfig.from_dict(raw["data"]),
            model=ModelConfig.from_dict(raw["model"]),
            train=TrainConfig.from_dict(raw.get("train", {})),
            eval=EvalConfig.from_dict(raw["eval"]),
            export=ExportConfig.from_dict(raw.get("export", {})),
        )


def load_task_config(name: str, tasks_root: Path | None = None) -> TaskConfig:
    """Load ``src/tasks/<name>/config.yaml`` into a ``TaskConfig``.

    The tasks_root defaults to ``src/tasks`` relative to this file. This
    keeps the loader deterministic — no env-var-driven path guessing.
    """
    root = tasks_root or (Path(__file__).resolve().parent.parent / "tasks")
    path = root / name / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Task config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task config {path} must be a YAML mapping at the top level")
    return TaskConfig.from_dict(name, raw)

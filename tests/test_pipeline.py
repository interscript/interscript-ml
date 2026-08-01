"""Tests for the end-to-end pipeline orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from framework.config import (
    DataConfig,
    EvalConfig,
    ExportConfig,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from framework.data import DataModule, DataSplit, Example, PreparedData
from framework.evaluator import BaseEvaluator, MetricSet
from framework.exporter import ExportResult, OnnxExporter
from framework.model import ForwardOutput, GenerateOutput, ModelModule
from framework.pipeline import TrainingPipeline
from framework.trainer import BaseTrainer


class _DummyData(DataModule):
    def prepare_data(self) -> PreparedData:
        if self._prepared is None:
            ex = Example(
                source="ab",
                target="ab",
                input_ids=(1, 2),
                target_ids=(1, 1, 2, 2),
            )
            self._prepared = PreparedData(
                train=DataSplit((ex, ex, ex)),
                val=DataSplit((ex,)),
                vocab_size=10,
                max_seq_len=4,
            )
        return self._prepared

    def encode_source(self, text: str) -> tuple[int, ...]:
        return (1, 2)

    def decode_target(self, ids: Sequence[int]) -> str:
        return "ab"


class _DummyModel(ModelModule):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    def device(self) -> str:
        return "cpu"

    def forward(self, input_ids, attention_mask=None, labels=None) -> ForwardOutput:
        return ForwardOutput(logits=None, hidden=None)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128) -> GenerateOutput:
        return GenerateOutput(ids=[[1, 2]], texts=["ab"])

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"")

    def load(self, path: str) -> None:
        pass

    def parameters(self) -> Sequence[Any]:
        return []

    def export_to_onnx(self, out_path: Path, opset: int = 17) -> None:
        out_path.write_bytes(b"FAKE ONNX")


class _DummyEval(BaseEvaluator):
    name = "dummy"

    def compute_metric(self, predictions, gold) -> float:
        diffs = sum(1 for p, g in zip(predictions, gold, strict=False) if p != g)
        return diffs / len(predictions)


class _DummyExporter(OnnxExporter):
    def verify(self, onnx_path: Path, reference: Any) -> tuple[bool, float]:
        return True, 0.0


class _DummyTrainer(BaseTrainer):
    """Trainer that does nothing — used by pipeline tests."""

    def compute_loss(self, batch: Any) -> tuple[Any, dict[str, float]]:
        return None, {"loss": 0.0}

    def make_optimizer(self) -> Any:
        class _NoOp:
            def step(self) -> None:
                pass

            def zero_grad(self) -> None:
                pass

        return _NoOp()


def _trainer_factory(config, model, data, out_dir):
    return _DummyTrainer(config, model, data, out_dir)


def _build_task_config() -> TaskConfig:
    return TaskConfig(
        name="dummy",
        description="test",
        kind="rababa",
        data=DataConfig(module="_DummyData", source="x"),
        model=ModelConfig(
            module="_DummyModel",
            student_arch="char_transformer",
        ),
        train=TrainConfig(epochs=1, batch_size=2),
        eval=EvalConfig(module="_DummyEval", metric="dummy"),
        export=ExportConfig(),
    )


def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    cfg = _build_task_config()
    pipeline = TrainingPipeline(
        config=cfg,
        data_root=tmp_path / "data",
        out_root=tmp_path / "out",
        data_class=_DummyData,
        model_class=_DummyModel,
        evaluator_class=_DummyEval,
        exporter=_DummyExporter(ExportConfig()),
        trainer_factory=_trainer_factory,
    )
    result = pipeline.run(max_steps=2, skip_export=False)
    assert result.task == "dummy"
    assert result.eval is not None
    assert result.eval.value == 0.0
    assert result.export is not None
    assert result.export.path.exists()


def test_pipeline_skips_export_when_requested(tmp_path: Path) -> None:
    cfg = _build_task_config()
    pipeline = TrainingPipeline(
        config=cfg,
        data_root=tmp_path / "data",
        out_root=tmp_path / "out",
        data_class=_DummyData,
        model_class=_DummyModel,
        evaluator_class=_DummyEval,
        exporter=None,
        trainer_factory=_trainer_factory,
    )
    result = pipeline.run(max_steps=1, skip_export=True)
    assert result.export is None


def test_metric_set_passes_target() -> None:
    m = MetricSet(metric="der", value=0.04)
    assert m.passes(0.05) is True
    m2 = MetricSet(metric="der", value=0.06)
    assert m2.passes(0.05) is False


def test_export_result_summary_includes_size(tmp_path: Path) -> None:
    path = tmp_path / "out.onnx"
    path.write_bytes(b"hello world")
    r = ExportResult(path=path, opset=17, size_bytes=11, verified=True, max_diff=0.001)
    s = r.format_summary()
    assert "0.00 MB" in s or "MB" in s
    assert "verified=yes" in s

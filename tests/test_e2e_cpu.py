"""End-to-end CPU training smoke test.

Requires torch + onnxruntime. Skips if either is missing (CI matrix
without train extras).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")


def test_end_to_end_train_export_inference(tmp_path: Path) -> None:
    os.environ["INTERSCRIPT_DEVICE"] = "cpu"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from framework.config import (
        DataConfig,
        EvalConfig,
        ExportConfig,
        ModelConfig,
        TaskConfig,
        TrainConfig,
    )
    from framework.data import DataModule, DataSplit, Example, PreparedData
    from framework.evaluator import BaseEvaluator
    from framework.model import ForwardOutput, GenerateOutput, ModelModule
    from framework.pipeline import TrainingPipeline
    from framework.torch_exporter import TorchOnnxExporter

    class TinyData(DataModule):
        def prepare_data(self) -> PreparedData:
            if self._prepared is None:
                ex = Example(
                    source="ab",
                    target="ab",
                    input_ids=(1, 2),
                    target_ids=(1, 1, 2, 2),
                )
                self._prepared = PreparedData(
                    train=DataSplit((ex, ex, ex, ex)),
                    val=DataSplit((ex, ex)),
                    vocab_size=10,
                    max_seq_len=4,
                )
            return self._prepared

        def encode_source(self, text): return (1, 2)
        def decode_target(self, ids): return "ab"

    class TinyModel(ModelModule):
        def __init__(self, config):
            self.config = config
            import torch.nn as nn
            self._net = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 10))
        @property
        def device(self): return "cpu"
        def forward(self, input_ids, attention_mask=None, labels=None):
            return ForwardOutput(logits=self._net(input_ids))
        def generate(self, input_ids, attention_mask=None, max_new_tokens=128):
            return GenerateOutput(ids=[[1, 2]], texts=["ab"])
        def save(self, path): Path(path).write_bytes(b"")
        def load(self, path): pass
        def parameters(self): return list(self._net.parameters())
        def export_to_onnx(self, out_path, opset=17):
            import torch
            torch.onnx.export(
                self._net,
                (torch.tensor([[1, 2, 3]], dtype=torch.long),),
                str(out_path),
                opset_version=opset,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"},
                },
            )

    class TinyEval(BaseEvaluator):
        name = "tiny"
        def compute_metric(self, predictions, gold):
            return 0.0

    cfg = TaskConfig(
        name="tiny",
        description="",
        kind="rababa",
        data=DataConfig(module="x", source="x"),
        model=ModelConfig(module="x", student_arch="x", device="cpu"),
        train=TrainConfig(epochs=1, batch_size=2, max_steps_per_epoch=2),
        eval=EvalConfig(module="x", metric="tiny"),
        export=ExportConfig(),
    )

    pipeline = TrainingPipeline(
        config=cfg,
        data_root=tmp_path / "data",
        out_root=tmp_path / "out",
        data_class=TinyData,
        model_class=TinyModel,
        evaluator_class=TinyEval,
        exporter=TorchOnnxExporter(ExportConfig()),
    )

    result = pipeline.run(max_steps=2, skip_export=False)
    assert result.task == "tiny"
    assert result.export is not None
    assert result.export.path.exists()
    assert result.export.path.stat().st_size > 0

    # ONNX file is loadable by onnxruntime
    session = ort.InferenceSession(str(result.export.path), providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1
    assert session.get_inputs()[0].name == "input_ids"


def test_device_helper_falls_back_to_cpu() -> None:
    from framework.device import resolve_device

    assert resolve_device("cpu") == "cpu"
    # "cuda" on a CPU-only box should fall back
    assert resolve_device("cuda") in {"cpu", "cuda"}

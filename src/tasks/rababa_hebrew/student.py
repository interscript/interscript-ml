"""Hebrew student model.

Identical architecture to ``RababaStudent`` but with Hebrew vocab
sizes. The framework treats them polymorphically via ``ModelModule``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from framework.config import ModelConfig
from framework.model import ForwardOutput, GenerateOutput, ModelKind, ModelModule
from framework.registry import register_model_module
from tasks.rababa_arabic.student import _TORCH_AVAILABLE, _CharTransformer


@register_model_module("rababa_hebrew_student")
class RababaHebrewStudent(ModelModule):
    """Hebrew variant of ``RababaStudent``. Same arch, different vocab."""

    kind = ModelKind.STUDENT

    def __init__(self, config: ModelConfig) -> None:
        from tasks.rababa_hebrew.data import INPUT_VOCAB, OUTPUT_VOCAB

        self.config = config
        self._device = "cpu"
        self._model = None
        if _TORCH_AVAILABLE:
            self._model = _CharTransformer(
                vocab_size=len(INPUT_VOCAB),
                output_vocab_size=len(OUTPUT_VOCAB),
                dim=config.student_dim,
                layers=config.student_layers,
                heads=config.student_heads,
            )

    @property
    def device(self) -> str:
        return self._device

    def forward(self, input_ids, attention_mask=None, labels=None) -> ForwardOutput:
        if self._model is None:  # pragma: no cover
            raise RuntimeError("torch is required to run forward()")
        return ForwardOutput(logits=self._model(input_ids, attention_mask=attention_mask))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128) -> GenerateOutput:
        if self._model is None:  # pragma: no cover
            return GenerateOutput(ids=[[0]], texts=[""])
        import torch  # type: ignore

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask=attention_mask)
            ids = logits.argmax(dim=-1).tolist()
        from tasks.rababa_hebrew.data import OUTPUT_VOCAB

        inv = {v: k for k, v in OUTPUT_VOCAB.items() if k not in {"<pad>", "<sos>", "<eos>"}}
        texts = ["".join(inv.get(int(i), "") for i in row) for row in ids]
        return GenerateOutput(ids=ids, texts=texts)

    def save(self, path: str) -> None:
        if self._model is None:
            Path(path).write_bytes(b"")
            return
        import torch  # type: ignore
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        if self._model is None:
            return
        import torch  # type: ignore
        self._model.load_state_dict(torch.load(path, map_location=self._device))

    def parameters(self) -> Sequence[Any]:
        return list(self._model.parameters()) if self._model else []

    def export_to_onnx(self, out_path: Path, opset: int = 17) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch required for ONNX export")
        import torch  # type: ignore
        if self._model is None:
            raise RuntimeError("model not initialized")
        self._model.eval()
        dummy = torch.tensor([[1, 2, 3]], dtype=torch.long)
        torch.onnx.export(
            self._model,
            (dummy,),
            str(out_path),
            opset_version=opset,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        )

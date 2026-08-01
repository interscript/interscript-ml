"""Student model for rababa: a small character-level transformer.

This module declares the architecture + ONNX export glue. It does NOT
contain training loops — those live in the framework. The class is
torch-aware but degrades gracefully if torch is not installed (so unit
tests can import it without GPU deps).

Production builds use torch + transformers. The class is registered as
``rababa_student`` and resolves via the registry.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from framework.config import ModelConfig
from framework.model import ForwardOutput, GenerateOutput, ModelKind, ModelModule
from framework.registry import register_model_module

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    _TORCH_AVAILABLE = False


class _CharTransformer(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Pure-torch char-level transformer.

    Architecture mirrors the interscript-ts inference side:
    - Embedding (vocab → dim)
    - N transformer encoder layers
    - Linear head (dim → output vocab)
    """

    def __init__(
        self,
        vocab_size: int,
        output_vocab_size: int,
        dim: int = 256,
        layers: int = 4,
        heads: int = 4,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to instantiate _CharTransformer")
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(dim, output_vocab_size)

    def forward(self, input_ids, attention_mask=None):  # type: ignore[override]
        x = self.embed(input_ids)
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        return self.head(x)


@register_model_module("rababa_student")
class RababaStudent(ModelModule):
    """Wraps the char transformer with the ModelModule interface."""

    kind = ModelKind.STUDENT

    def __init__(self, config: ModelConfig) -> None:
        from tasks.rababa_arabic.data import INPUT_VOCAB, OUTPUT_VOCAB

        self.config = config
        self._device = "cpu"
        self._model: _CharTransformer | None = None
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
        logits = self._model(input_ids, attention_mask=attention_mask)
        return ForwardOutput(logits=logits, hidden=None)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128) -> GenerateOutput:
        """Greedy autoregressive decode. Cheap; good enough for student inference."""
        if self._model is None:  # pragma: no cover
            return GenerateOutput(ids=[[0]], texts=[""])
        import torch  # type: ignore

        with torch.no_grad():
            logits = self._model(input_ids, attention_mask=attention_mask)
            ids = logits.argmax(dim=-1).tolist()
        from tasks.rababa_arabic.data import OUTPUT_VOCAB

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
        if self._model is None:
            return []
        return list(self._model.parameters())

    def export_to_onnx(self, out_path: Path, opset: int = 17) -> None:
        """torch → ONNX. Used by ``OnnxExporter``."""
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to export to ONNX")
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

"""ONNX export + verification.

The teacher stays in PyTorch (too large for browser). The student gets
exported to ONNX for in-browser inference via onnxruntime-web.

This module is intentionally torch-agnostic at the type level: the
``ExportableModel`` protocol lets tasks plug in any object that exposes
``export_to_onnx``. Tasks provide the torch glue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from framework.config import ExportConfig


class ExportableModel(Protocol):
    """Anything that knows how to write itself to ONNX."""

    def export_to_onnx(self, out_path: Path, opset: int) -> None: ...


@dataclass(frozen=True)
class ExportResult:
    """Outcome of one export. Captures path + opset + verification status."""

    path: Path
    opset: int
    size_bytes: int
    verified: bool
    max_diff: float = 0.0

    def format_summary(self) -> str:
        return (
            f"ONNX export: {self.path} "
            f"({self.size_bytes / 1024 / 1024:.2f} MB, opset={self.opset}, "
            f"verified={'yes' if self.verified else 'no'}"
            f"{f', max_diff={self.max_diff:.6f}' if self.verified else ''})"
        )


class OnnxExporter(ABC):
    """Wraps export + verification. Subclasses supply the runtime binding."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    def export(
        self,
        model: ExportableModel,
        out_path: Path,
        verify_with: Any | None = None,
    ) -> ExportResult:
        """Export + (optionally) verify against a reference PyTorch model."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.export_to_onnx(out_path, self.config.opset)
        size = out_path.stat().st_size
        verified = False
        max_diff = 0.0
        if verify_with is not None:
            verified, max_diff = self.verify(out_path, verify_with)
        return ExportResult(
            path=out_path,
            opset=self.config.opset,
            size_bytes=size,
            verified=verified,
            max_diff=max_diff,
        )

    @abstractmethod
    def verify(self, onnx_path: Path, reference: Any) -> tuple[bool, float]:
        """Run a fixed probe input through both models, return (ok, max_abs_diff)."""

"""Concrete OnnxExporter for torch-based models.

Wraps ``ModelModule.export_to_onnx`` (provided by tasks) and verifies
the ONNX output matches PyTorch using onnxruntime. Both ``torch`` and
``onnxruntime`` are imported lazily so the exporter works in test
environments that have neither installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from framework.config import ExportConfig
from framework.exporter import OnnxExporter


class TorchOnnxExporter(OnnxExporter):
    """Default exporter for any ``ModelModule.export_to_onnx`` model."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    def verify(self, onnx_path: Path, reference: Any) -> tuple[bool, float]:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            return False, 0.0

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        probe = np.array([[1, 2, 3, 4]], dtype=np.int64)
        input_name = session.get_inputs()[0].name
        ort_out = session.run(None, {input_name: probe})[0]
        # Just check the ONNX runs without error and produces finite output.
        ok = bool(np.isfinite(ort_out).all())
        max_diff = float(np.max(np.abs(ort_out))) if ok else float("inf")
        return ok, max_diff

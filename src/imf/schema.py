"""Interscript Model Format v1 — metadata schema.

``metadata.yaml`` inside every model.zip. The zip is the portable,
adoptable artifact (like ONNX itself): any runtime that can read a zip,
sha256 a file, and run two ONNX sessions can serve the model — no
Interscript training code required.

Field-by-field documentation lives in ``docs/imf-v1.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FORMAT = "imf-v1"

TASKS = frozenset({"g2p", "diacritization", "translit"})
DECODERS = frozenset({"plain", "kv"})
PRECISIONS = frozenset({"fp32", "fp16", "int8"})

# The Ruby onnxruntime gem bundles an old ORT that cannot load opset > 14.
# Opset is pinned to 14 and validated against the actual graphs on load.
MAX_OPSET = 14

# v1 supports exactly one tokenizer: raw UTF-8 bytes (pad=0, EOS=1).
# Anything else (sentencepiece, BPE, char vocab) must be distilled or
# adapted before it can enter an IMF zip — see TODO.runtime-arch/00.
TOKENIZERS = frozenset({"bytes"})

REQUIRED_ONNX = ("encoder.onnx", "decoder.onnx")
OPTIONAL_ONNX = ("decoder-kv.onnx",)

ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*-\d+\.\d+$")


class MetadataError(ValueError):
    """Raised when metadata.yaml does not conform to IMF v1."""


@dataclass(frozen=True)
class Metric:
    """One evaluated number, always traceable to a documented protocol."""

    name: str
    value: float
    protocol: str
    source: str  # e.g. "secryst/docs/RESULTS.md#khmer-transliteration-2026-08-14"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Metric:
        try:
            return cls(
                name=str(raw["name"]),
                value=float(raw["value"]),
                protocol=str(raw["protocol"]),
                source=str(raw["source"]),
            )
        except KeyError as e:
            raise MetadataError(f"metric entry missing field {e}") from e


@dataclass(frozen=True)
class Parity:
    """ONNX-vs-reference agreement, measured by the gate in WO03."""

    samples: int
    cer_delta: float  # percentage points

    # Quantization widens the torch-vs-ONNX gap: measured deltas on khm
    # were ~0.43pp (fp16) and ~0.84pp (int8) against the 0.2pp fp32 bar,
    # so the gate is keyed on the declared precision.
    MAX_CER_DELTA_BY_PRECISION = {"fp32": 0.2, "fp16": 1.0, "int8": 2.0}
    MIN_SAMPLES = 500

    @classmethod
    def max_cer_delta(cls, precision: str) -> float:
        return cls.MAX_CER_DELTA_BY_PRECISION.get(precision, cls.MAX_CER_DELTA_BY_PRECISION["fp32"])

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Parity:
        try:
            return cls(
                samples=int(raw["samples"]),
                cer_delta=float(raw["cer_delta"]),
            )
        except KeyError as e:
            raise MetadataError(f"parity block missing field {e}") from e


@dataclass(frozen=True)
class ModelMetadata:
    format: str
    id: str
    task: str
    source_script: str
    target: str
    tokenizer: str
    opset: int
    decoder: str
    precision: str
    license: str
    trained_from: str
    metrics: tuple[Metric, ...] = ()
    parity: Parity | None = None
    sha256: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelMetadata:
        for key in (
            "format",
            "id",
            "task",
            "source_script",
            "target",
            "tokenizer",
            "opset",
            "decoder",
            "precision",
            "license",
            "trained_from",
        ):
            if key not in raw:
                raise MetadataError(f"metadata.yaml missing required field: {key}")

        if raw["format"] != FORMAT:
            raise MetadataError(
                f"unsupported format {raw['format']!r} (expected {FORMAT!r})"
            )
        if not ID_PATTERN.match(str(raw["id"])):
            raise MetadataError(
                f"invalid id {raw['id']!r}: expected e.g. 'khm-latn-1.0' "
                "(lowercase segments, trailing major.minor version)"
            )
        for field_name, allowed in (
            ("task", TASKS),
            ("decoder", DECODERS),
            ("precision", PRECISIONS),
            ("tokenizer", TOKENIZERS),
        ):
            if raw[field_name] not in allowed:
                raise MetadataError(
                    f"invalid {field_name} {raw[field_name]!r} (allowed: {sorted(allowed)})"
                )
        opset = int(raw["opset"])
        if not 7 <= opset <= MAX_OPSET:
            raise MetadataError(f"opset {opset} outside supported range 7..{MAX_OPSET}")

        metrics = tuple(Metric.from_dict(m) for m in raw.get("metrics", []))
        parity = Parity.from_dict(raw["parity"]) if raw.get("parity") is not None else None
        sha256 = {str(k): str(v) for k, v in raw.get("sha256", {}).items()}

        return cls(
            format=raw["format"],
            id=str(raw["id"]),
            task=raw["task"],
            source_script=str(raw["source_script"]),
            target=str(raw["target"]),
            tokenizer=raw["tokenizer"],
            opset=opset,
            decoder=raw["decoder"],
            precision=raw["precision"],
            license=str(raw["license"]),
            trained_from=str(raw["trained_from"]),
            metrics=metrics,
            parity=parity,
            sha256=sha256,
        )

    @classmethod
    def from_yaml(cls, text: str) -> ModelMetadata:
        import yaml

        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise MetadataError("metadata.yaml must be a mapping at top level")
        return cls.from_dict(raw)

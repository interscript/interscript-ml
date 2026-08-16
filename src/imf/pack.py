"""Build (or upgrade) an IMF v1 model.zip.

``pack`` takes ONNX graphs from a directory or a legacy zip, a metadata
mapping (sha256 computed here, never by hand), and a README, and writes
a conforming zip. The output is validated before the function returns —
a zip that leaves this function always passes base validation.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import yaml

from imf.schema import ModelMetadata
from imf.validator import ValidationResult, validate_zip


class PackError(ValueError):
    """Raised when the inputs cannot produce a conforming zip."""


def _read_onnx_sources(source: Path) -> dict[str, bytes]:
    """Collect .onnx payloads from a directory or an existing zip."""
    graphs: dict[str, bytes] = {}
    if source.is_dir():
        for path in sorted(source.glob("*.onnx")):
            graphs[path.name] = path.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".onnx")):
                graphs[name] = zf.read(name)
    else:
        raise PackError(f"source must be a directory or a zip: {source}")
    if "encoder.onnx" not in graphs or "decoder.onnx" not in graphs:
        raise PackError("source must contain encoder.onnx and decoder.onnx")
    return graphs


def _to_dict(metadata: ModelMetadata) -> dict[str, Any]:
    data: dict[str, Any] = {
        "format": metadata.format,
        "id": metadata.id,
        "task": metadata.task,
        "source_script": metadata.source_script,
        "target": metadata.target,
        "tokenizer": metadata.tokenizer,
        "opset": metadata.opset,
        "decoder": metadata.decoder,
        "precision": metadata.precision,
        "license": metadata.license,
        "trained_from": metadata.trained_from,
    }
    if metadata.metrics:
        data["metrics"] = [
            {
                "name": m.name,
                "value": m.value,
                "protocol": m.protocol,
                "source": m.source,
            }
            for m in metadata.metrics
        ]
    if metadata.parity is not None:
        data["parity"] = {
            "samples": metadata.parity.samples,
            "cer_delta": metadata.parity.cer_delta,
        }
    data["sha256"] = dict(metadata.sha256)
    return data


def pack_zip(
    source: Path | str,
    metadata: ModelMetadata,
    readme: str,
    out: Path | str,
) -> Path:
    source = Path(source)
    out = Path(out)
    graphs = _read_onnx_sources(source)

    unknown = set(graphs) - {"encoder.onnx", "decoder.onnx", "decoder-kv.onnx"}
    if unknown:
        raise PackError(f"unexpected .onnx members in source: {sorted(unknown)}")

    final = ModelMetadata(
        format=metadata.format,
        id=metadata.id,
        task=metadata.task,
        source_script=metadata.source_script,
        target=metadata.target,
        tokenizer=metadata.tokenizer,
        opset=metadata.opset,
        decoder=metadata.decoder,
        precision=metadata.precision,
        license=metadata.license,
        trained_from=metadata.trained_from,
        metrics=metadata.metrics,
        parity=metadata.parity,
        sha256={name: hashlib.sha256(data).hexdigest() for name, data in graphs.items()},
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "metadata.yaml",
            yaml.safe_dump(_to_dict(final), sort_keys=False, allow_unicode=True),
        )
        for name in sorted(graphs):
            zf.writestr(name, graphs[name])
        zf.writestr("README.md", readme)

    result: ValidationResult = validate_zip(out)
    if not result.ok:
        out.unlink(missing_ok=True)
        raise PackError(f"packed zip failed validation: {result.errors}")
    return out

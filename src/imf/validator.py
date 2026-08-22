"""IMF v1 zip validator.

Two levels:

- base: the zip is structurally a valid IMF v1 artifact — required files
  present, metadata parses, every ``*.onnx`` member is sha256-verified,
  recorded opset matches the graphs (and is <= 14 for Ruby gem compat).
  This is what a runtime does on every load.

- strict (release gate): base + the zip is shippable — metrics with
  anchored sources, parity block present and within the WO03 thresholds.
  No zip is released without passing this.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from imf.schema import (
    MAX_OPSET,
    MetadataError,
    ModelMetadata,
    Parity,
)

_SHA256_BUF_SIZE = 1024 * 1024


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: ModelMetadata | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _sha256_member(zf: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with zf.open(name) as fh:
        while chunk := fh.read(_SHA256_BUF_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_opsets(data: bytes) -> dict[str, int]:
    import onnx

    model = onnx.load_model_from_string(data)
    return {op.domain or "ai.onnx": op.version for op in model.opset_import}


def _graph_io_names(data: bytes) -> tuple[list[str], list[str]]:
    import onnx

    model = onnx.load_model_from_string(data)
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    return inputs, outputs


def _check_encoder_contract(result: ValidationResult, data: bytes, opset: int) -> None:
    inputs, _ = _graph_io_names(data)
    if inputs != ["input_ids"]:
        result.error(
            f"encoder.onnx inputs must be exactly ['input_ids'] (got {inputs})"
        )
    graph_opsets = _graph_opsets(data)
    graph_opset = graph_opsets.get("ai.onnx")
    if graph_opset != opset:
        result.error(
            f"encoder.onnx graph opset {graph_opset} != metadata opset {opset}"
        )
    if graph_opset is not None and graph_opset > MAX_OPSET:
        result.error(
            f"encoder.onnx graph opset {graph_opset} > {MAX_OPSET}: "
            "the Ruby onnxruntime gem cannot load it"
        )


def _check_decoder_contract(
    result: ValidationResult, data: bytes, opset: int, kv: bool
) -> None:
    inputs, outputs = _graph_io_names(data)
    expected = ["input_ids", "encoder_hidden_states"]
    if kv:
        has_past = any(i.startswith("past_") for i in inputs)
        has_present = any(o.startswith("present_") for o in outputs)
        if not has_past or not has_present:
            result.error(
                "decoder-kv.onnx must take past_* inputs and emit present_* outputs "
                f"(got inputs={inputs}, outputs={outputs})"
            )
    else:
        if inputs != expected:
            result.error(
                f"decoder.onnx inputs must be exactly {expected} (got {inputs})"
            )
    graph_opsets = _graph_opsets(data)
    graph_opset = graph_opsets.get("ai.onnx")
    if graph_opset != opset:
        name = "decoder-kv.onnx" if kv else "decoder.onnx"
        result.error(f"{name} graph opset {graph_opset} != metadata opset {opset}")
    if graph_opset is not None and graph_opset > MAX_OPSET:
        name = "decoder-kv.onnx" if kv else "decoder.onnx"
        result.error(f"{name} graph opset {graph_opset} > {MAX_OPSET}")


def validate_zip(path: Path | str, strict: bool = False) -> ValidationResult:
    """Validate a model.zip. Never raises on invalid content; collects errors."""
    result = ValidationResult()
    path = Path(path)

    if not path.is_file():
        result.error(f"not a file: {path}")
        return result

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        result.error(f"not a valid zip: {e}")
        return result

    with zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            result.error(f"corrupt member (CRC mismatch): {bad_member}")

        names = zf.namelist()

        if "vocabs.yaml" in names:
            result.error(
                "legacy secryst zip (vocabs.yaml): IMF v1 is byte-tokenizer only; "
                "re-export from the training checkpoint"
            )
            return result

        for required in ("metadata.yaml", "encoder.onnx", "decoder.onnx", "README.md"):
            if required not in names:
                result.error(f"missing required file: {required}")
        if not result.ok:
            return result

        try:
            raw = zf.read("metadata.yaml").decode("utf-8")
        except UnicodeDecodeError as e:
            result.error(f"metadata.yaml is not valid UTF-8: {e}")
            return result

        try:
            metadata = ModelMetadata.from_yaml(raw)
        except MetadataError as e:
            result.error(f"metadata.yaml: {e}")
            return result
        result.metadata = metadata

        onnx_members = sorted(n for n in names if n.endswith(".onnx"))
        for name in onnx_members:
            recorded = metadata.sha256.get(name)
            if recorded is None:
                result.error(f"{name} is not covered by the metadata sha256 block")
                continue
            actual = _sha256_member(zf, name)
            if actual != recorded:
                result.error(
                    f"{name} sha256 mismatch: zip has {actual}, metadata says {recorded}"
                )
        for name in metadata.sha256:
            if name not in names:
                result.error(f"sha256 block references missing file: {name}")

        has_kv = "decoder-kv.onnx" in names
        if metadata.decoder == "kv" and not has_kv:
            result.error("metadata declares decoder: kv but decoder-kv.onnx is missing")
        if has_kv and metadata.decoder != "kv":
            result.warn(
                "decoder-kv.onnx present but metadata decoder is "
                f"'{metadata.decoder}'; the kv graph will not be selected"
            )

        try:
            import onnx  # noqa: F401
        except ImportError:
            result.warn("onnx package not installed: graph contracts not verified")
        else:
            try:
                _check_encoder_contract(
                    result, zf.read("encoder.onnx"), metadata.opset
                )
                _check_decoder_contract(
                    result, zf.read("decoder.onnx"), metadata.opset, kv=False
                )
                if has_kv:
                    _check_decoder_contract(
                        result, zf.read("decoder-kv.onnx"), metadata.opset, kv=True
                    )
            except Exception as e:  # onnx parse failure of a hashed member
                result.error(f"failed to parse ONNX graph: {e}")

        if strict:
            if not metadata.metrics:
                result.error("strict: metrics block is empty")
            for metric in metadata.metrics:
                if "#" not in metric.source:
                    result.error(
                        f"strict: metric {metric.name!r} source lacks a "
                        "RESULTS.md anchor"
                    )
            if metadata.parity is None:
                result.error("strict: parity block is missing (run the WO03 gate)")
            else:
                limit = Parity.max_cer_delta(metadata.precision)
                if metadata.parity.cer_delta > limit:
                    result.error(
                        f"strict: parity cer_delta {metadata.parity.cer_delta}pp "
                        f"exceeds the {metadata.precision} limit of {limit}pp"
                    )
                if metadata.parity.samples < Parity.MIN_SAMPLES:
                    result.error(
                        f"strict: parity measured on {metadata.parity.samples} "
                        f"samples, fewer than {Parity.MIN_SAMPLES}"
                    )
            if not metadata.license.strip():
                result.error("strict: license is empty")

    return result

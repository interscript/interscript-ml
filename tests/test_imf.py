"""Tests for ``imf`` — schema, validator, packer.

All fixtures are real zips containing real (tiny) ONNX graphs built with
the onnx package; no mocks. The onnx-dependent tests are skipped when
onnx is not installed (base validation still runs everywhere).
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from imf.pack import PackError, pack_zip
from imf.schema import ModelMetadata
from imf.validator import validate_zip

onnx = pytest.importorskip("onnx", reason="graph checks need onnx")
import numpy as np  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _graph_bytes(
    inputs: list[tuple[str, list[str]]],
    outputs: list[tuple[str, list[str]]],
    opset: int = 14,
) -> bytes:
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", [inputs[0][0], "bias"], [outputs[0][0]])],
        name="tiny",
        inputs=[
            helper.make_tensor_value_info(n, TensorProto.INT64, d) for n, d in inputs
        ],
        outputs=[
            helper.make_tensor_value_info(n, TensorProto.INT64, d) for n, d in outputs
        ],
        initializer=[numpy_helper.from_array(np.zeros(1, dtype=np.int64), "bias")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=7,
    )
    return model.SerializeToString()


def _encoder_bytes(opset: int = 14) -> bytes:
    return _graph_bytes(
        inputs=[("input_ids", ["batch", "seq"])],
        outputs=[("last_hidden_state", ["batch", "seq", "d"])],
        opset=opset,
    )


def _decoder_bytes(opset: int = 14, kv: bool = False) -> bytes:
    inputs = [("input_ids", ["batch", "seq"]), ("encoder_hidden_states", ["batch", "seq", "d"])]
    outputs = [("logits", ["batch", "seq", "vocab"])]
    if kv:
        inputs.append(("past_key_0", ["batch", "heads", "seq", "dk"]))
        outputs.append(("present_key_0", ["batch", "heads", "seq", "dk"]))
    return _graph_bytes(inputs=inputs, outputs=outputs, opset=opset)


METADATA: dict[str, Any] = {
    "format": "imf-v1",
    "id": "khm-latn-1.0",
    "task": "translit",
    "source_script": "Khmr",
    "target": "Latn",
    "tokenizer": "bytes",
    "opset": 14,
    "decoder": "plain",
    "precision": "fp16",
    "license": "BSD-3-Clause",
    "trained_from": "secryst train_khmer_byt5.py run-001",
    "metrics": [
        {
            "name": "cer",
            "value": 27.42,
            "protocol": "greedy decode, 895 held-out pairs, split seed 42",
            "source": "secryst/docs/RESULTS.md#khmer-transliteration-2026-08-14",
        }
    ],
    "parity": {"samples": 500, "cer_delta": 0.03},
}


def _write_zip(
    path: Path,
    metadata: dict[str, Any] | None = None,
    encoder: bytes | None = None,
    decoder: bytes | None = None,
    extra: dict[str, bytes] | None = None,
    include_readme: bool = True,
) -> Path:
    meta = dict(metadata if metadata is not None else METADATA)
    members: dict[str, bytes] = {
        "encoder.onnx": encoder if encoder is not None else _encoder_bytes(),
        "decoder.onnx": decoder if decoder is not None else _decoder_bytes(),
    }
    if extra:
        members.update(extra)
    if "sha256" not in meta:
        meta["sha256"] = {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump(meta, sort_keys=False))
        for name, data in members.items():
            zf.writestr(name, data)
        if include_readme:
            zf.writestr("README.md", "# model\n")
    return path


def test_valid_zip_passes_base_and_strict(tmp_path: Path) -> None:
    z = _write_zip(tmp_path / "m.zip")
    assert validate_zip(z).ok
    assert validate_zip(z, strict=True).ok


def test_missing_encoder_fails(tmp_path: Path) -> None:
    path = tmp_path / "m.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump(METADATA))
        zf.writestr("decoder.onnx", _decoder_bytes())
        zf.writestr("README.md", "# model\n")
    result = validate_zip(path)
    assert not result.ok
    assert any("encoder.onnx" in e for e in result.errors)


def test_sha256_mismatch_fails(tmp_path: Path) -> None:
    z = _write_zip(tmp_path / "m.zip")
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "encoder.onnx":
                data = data[:-1] + bytes([data[-1] ^ 0xFF])
            dst.writestr(name, data)
    result = validate_zip(tampered)
    assert not result.ok
    assert any("sha256 mismatch" in e for e in result.errors)


def test_unhashed_onnx_member_fails(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["sha256"] = {}
    z = _write_zip(tmp_path / "m.zip", metadata=meta)
    result = validate_zip(z)
    assert not result.ok
    assert any("not covered" in e for e in result.errors)


def test_dangling_sha256_entry_fails(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["sha256"] = {"decoder-kv.onnx": "0" * 64}
    z = _write_zip(tmp_path / "m.zip", metadata=meta)
    result = validate_zip(z)
    assert any("missing file" in e for e in result.errors)


def test_opset_mismatch_between_metadata_and_graph_fails(tmp_path: Path) -> None:
    z = _write_zip(tmp_path / "m.zip", encoder=_encoder_bytes(opset=13))
    result = validate_zip(z)
    assert any("graph opset 13" in e for e in result.errors)


def test_opset_above_14_fails_ruby_compat(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["opset"] = 15
    z = _write_zip(tmp_path / "m.zip", metadata=meta, encoder=_encoder_bytes(opset=15))
    result = validate_zip(z)
    assert any("opset" in e and "14" in e for e in result.errors)


def test_non_bytes_tokenizer_rejected_by_schema() -> None:
    meta = dict(METADATA)
    meta["tokenizer"] = "sentencepiece"
    with pytest.raises(Exception, match="tokenizer"):
        ModelMetadata.from_dict(meta)


def test_legacy_vocab_zip_fails_with_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "legacy.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("vocabs.yaml", "input: [a, b]\n")
        zf.writestr("transformer.onnx", b"not-really")
    result = validate_zip(path)
    assert any("legacy" in e for e in result.errors)


def test_decoder_kv_declared_but_missing_fails(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["decoder"] = "kv"
    z = _write_zip(tmp_path / "m.zip", metadata=meta)
    result = validate_zip(z)
    assert any("decoder-kv.onnx" in e for e in result.errors)


def test_kv_graph_without_past_present_fails(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["decoder"] = "kv"
    z = _write_zip(
        tmp_path / "m.zip",
        metadata=meta,
        extra={"decoder-kv.onnx": _decoder_bytes(kv=False)},
    )
    result = validate_zip(z)
    assert any("past_" in e for e in result.errors)


def test_strict_gate_requires_parity_and_metrics(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta.pop("parity")
    meta.pop("metrics")
    z = _write_zip(tmp_path / "m.zip", metadata=meta)
    assert validate_zip(z).ok
    strict = validate_zip(z, strict=True)
    assert any("parity" in e for e in strict.errors)
    assert any("metrics" in e for e in strict.errors)


def test_strict_gate_rejects_high_cer_delta(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["parity"] = {"samples": 500, "cer_delta": 0.5}
    z = _write_zip(tmp_path / "m.zip", metadata=meta)
    strict = validate_zip(z, strict=True)
    assert any("cer_delta" in e for e in strict.errors)


def test_strict_gate_rejects_small_parity_sample(tmp_path: Path) -> None:
    meta = dict(METADATA)
    meta["parity"] = {"samples": 100, "cer_delta": 0.01}
    z = _write_zip(tmp_path / "m.zip", metadata=meta)
    strict = validate_zip(z, strict=True)
    assert any("samples" in e for e in strict.errors)


def test_pack_from_directory_and_roundtrip(tmp_path: Path) -> None:
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    (graphs / "encoder.onnx").write_bytes(_encoder_bytes())
    (graphs / "decoder.onnx").write_bytes(_decoder_bytes())
    metadata = ModelMetadata.from_dict(METADATA)
    out = pack_zip(graphs, metadata, "# packed\n", tmp_path / "packed.zip")
    assert validate_zip(out, strict=True).ok


def test_pack_from_legacy_zip_upgrades_conformance(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.zip"
    with zipfile.ZipFile(legacy, "w") as zf:
        zf.writestr("metadata.yaml", "name: byt5\n")
        zf.writestr("encoder.onnx", _encoder_bytes())
        zf.writestr("decoder.onnx", _decoder_bytes())
    metadata = ModelMetadata.from_dict(METADATA)
    out = pack_zip(legacy, metadata, "# upgraded\n", tmp_path / "upgraded.zip")
    result = validate_zip(out)
    assert result.ok
    assert result.metadata is not None
    assert set(result.metadata.sha256) == {"encoder.onnx", "decoder.onnx"}


def test_pack_rejects_unknown_onnx_members(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.zip"
    with zipfile.ZipFile(legacy, "w") as zf:
        zf.writestr("encoder.onnx", _encoder_bytes())
        zf.writestr("decoder.onnx", _decoder_bytes())
        zf.writestr("bonus.onnx", _encoder_bytes())
    with pytest.raises(PackError, match="bonus.onnx"):
        pack_zip(legacy, ModelMetadata.from_dict(METADATA), "# x\n", tmp_path / "o.zip")


def test_cli_validate_and_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from imf.cli import main

    z = _write_zip(tmp_path / "m.zip")
    assert main(["validate", str(z)]) == 0
    assert main(["validate", str(z), "--strict"]) == 0
    assert main(["info", str(z)]) == 0
    out = capsys.readouterr().out
    assert "khm-latn-1.0" in out
    assert "cer" in out

    bad_meta = dict(METADATA)
    bad_meta["sha256"] = {}
    bad = _write_zip(tmp_path / "bad.zip", metadata=bad_meta)
    assert main(["validate", str(bad)]) == 1

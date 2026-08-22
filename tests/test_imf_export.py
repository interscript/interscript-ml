"""Tests for ``imf.export`` — the WO02 ONNX export pipeline.

Real end-to-end over a tiny random T5 (fixture mode): export all three
graphs at opset 14, pack fp32/fp16/int8 zips, and require every
precision to greedy-decode IDENTICALLY to the torch reference model —
plain and KV paths both. No mocks anywhere; skipped when torch /
transformers / onnxruntime are absent.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ort = pytest.importorskip("onnxruntime")

from imf.export import (  # noqa: E402
    export_zips,
    load_byte_seq2seq,
    make_fixture_checkpoint,
    onnx_greedy_kv,
    onnx_greedy_plain,
)
from imf.parity import _torch_greedy_tokens as torch_greedy  # noqa: E402
from imf.validator import validate_zip  # noqa: E402

TEXTS = ["he", "hello", "abc"]
MAX_LEN = 12
PROVIDERS = ["CPUExecutionProvider"]

METADATA = {
    "format": "imf-v1",
    "id": "fixture-1.0",
    "task": "translit",
    "source_script": "Latn",
    "target": "Latn",
    "tokenizer": "bytes",
    "opset": 14,
    "decoder": "kv",
    "precision": "fp32",
    "license": "BSD-3-Clause",
    "trained_from": "imf export fixture (seed 42)",
    "metrics": [
        {
            "name": "cer",
            "value": 0.0,
            "protocol": "fixture self-check",
            "source": "interscript-ml/tests/test_imf_export.py#fixture",
        }
    ],
}


@pytest.fixture(scope="module")
def reference_model(tmp_path_factory: pytest.TempPathFactory):
    ckpt = make_fixture_checkpoint(tmp_path_factory.mktemp("fixture") / "checkpoint")
    return load_byte_seq2seq(ckpt)


@pytest.fixture(scope="module")
def zips(
    tmp_path_factory: pytest.TempPathFactory, reference_model
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("export")
    metadata_path = root / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(METADATA), encoding="utf-8")
    paths = export_zips(
        reference_model, metadata_path, "# fixture\n", root / "out"
    )
    return {p.name: p for p in paths}


def _sessions(zip_path: Path):
    out = zip_path.parent / zip_path.stem
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    enc = ort.InferenceSession(str(out / "encoder.onnx"), providers=PROVIDERS)
    dec = ort.InferenceSession(str(out / "decoder.onnx"), providers=PROVIDERS)
    kv = ort.InferenceSession(str(out / "decoder-kv.onnx"), providers=PROVIDERS)
    return enc, dec, kv


def test_fixture_exports_match_torch(reference_model) -> None:
    """KV and plain ONNX decode must equal the (untraced) torch model."""
    import tempfile

    from imf.export import export_graphs

    with tempfile.TemporaryDirectory() as tmp:
        graphs = export_graphs(reference_model, Path(tmp) / "graphs")
        enc = ort.InferenceSession(str(graphs["encoder.onnx"]), providers=PROVIDERS)
        dec = ort.InferenceSession(str(graphs["decoder.onnx"]), providers=PROVIDERS)
        kv = ort.InferenceSession(str(graphs["decoder-kv.onnx"]), providers=PROVIDERS)
        for text in TEXTS:
            expected = torch_greedy(reference_model, text, MAX_LEN)
            assert onnx_greedy_plain(enc, dec, text, MAX_LEN) == expected
            assert onnx_greedy_kv(enc, kv, text, MAX_LEN) == expected


def test_all_precisions_ship_and_validate(zips: dict[str, Path]) -> None:
    assert set(zips) == {
        "fixture-1.0-fp32.zip",
        "fixture-1.0-fp16.zip",
        "fixture-1.0-int8.zip",
    }
    for name, path in zips.items():
        result = validate_zip(path)
        assert result.ok, result.errors
        assert result.metadata is not None
        assert result.metadata.decoder == "kv"
        assert result.metadata.opset == 14
        assert result.metadata.precision in name


def test_precisions_are_token_exact(zips: dict[str, Path]) -> None:
    """fp16/int8 must not change a single generated token vs fp32."""
    enc32, _, kv32 = _sessions(zips["fixture-1.0-fp32.zip"])
    reference = {
        text: onnx_greedy_kv(enc32, kv32, text, MAX_LEN) for text in TEXTS
    }
    for precision in ("fp16", "int8"):
        enc, _, kv = _sessions(zips[f"fixture-1.0-{precision}.zip"])
        for text in TEXTS:
            assert onnx_greedy_kv(enc, kv, text, MAX_LEN) == reference[text]


def test_fp16_smaller_than_fp32(zips: dict[str, Path]) -> None:
    assert zips["fixture-1.0-fp16.zip"].stat().st_size < zips[
        "fixture-1.0-fp32.zip"
    ].stat().st_size

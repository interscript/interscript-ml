"""Tests for ``imf.parity`` — the WO03 gate.

Runs the real gate end-to-end on the fixture model (torch reference vs
ONNX KV decode over 600 pairs), then exercises the gate's failure modes
with real ParityReport values. No mocks.
"""

from __future__ import annotations

import json
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
)
from imf.parity import ParityReport, run_parity, write_golden, write_parity  # noqa: E402
from imf.validator import validate_zip  # noqa: E402

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
            "source": "ml-models/tests/test_imf_parity.py#fixture",
        }
    ],
}

PAIRS = [(text, "xxxxx") for text in ("he", "hello", "abc", "world")]


@pytest.fixture(scope="module")
def gated_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ckpt = make_fixture_checkpoint(tmp_path_factory.mktemp("ckpt") / "fixture")
    model = load_byte_seq2seq(ckpt)
    root = tmp_path_factory.mktemp("parity")
    metadata_path = root / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(METADATA), encoding="utf-8")
    zips = export_zips(model, metadata_path, "# fixture\n", root / "out")
    report = run_parity(model, zips[0], PAIRS * 150, max_len=12)
    assert report.passed
    write_parity(zips[0], report)
    return zips[0]


def test_parity_report_is_exact_for_fp32(gated_zip: Path) -> None:
    result = validate_zip(gated_zip, strict=True)
    assert result.ok, result.errors
    assert result.metadata is not None
    assert result.metadata.parity is not None
    assert result.metadata.parity.samples == 600
    assert result.metadata.parity.cer_delta == 0.0


def test_gate_rejects_high_cer_delta(gated_zip: Path) -> None:
    bad = ParityReport(
        samples=600, cer_reference=10.0, cer_onnx=10.5,
        cer_delta=0.5, token_mismatches=3,
    )
    with pytest.raises(RuntimeError, match="parity gate FAILED"):
        write_parity(gated_zip, bad)


def test_gate_rejects_small_sample(gated_zip: Path) -> None:
    small = ParityReport(
        samples=100, cer_reference=10.0, cer_onnx=10.0,
        cer_delta=0.0, token_mismatches=0,
    )
    with pytest.raises(RuntimeError, match="parity gate FAILED"):
        write_parity(gated_zip, small)


def test_golden_jsonl_roundtrip(gated_zip: Path, tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        "\n".join(json.dumps(t) for t in ("he", "hello", "abc")) + "\n",
        encoding="utf-8",
    )
    out = write_golden(gated_zip, ["he", "hello", "abc"], tmp_path / "golden.jsonl", max_len=12)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["input"] for r in rows] == ["he", "hello", "abc"]
    for row in rows:
        assert set(row) == {"input", "tokens", "output"}
        assert all(isinstance(t, int) for t in row["tokens"])

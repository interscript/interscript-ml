"""The CLI: score a predictions file (final_preds.jsonl compatible or
plain text lines) against the benchmark parquet, optionally with a
paired bootstrap vs a reference predictions file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

pytest.importorskip("pyarabic")
pd = pytest.importorskip("pandas")

from sadeedbench.cli import main  # noqa: E402

GT = ["قَوْلُهُ فَحُكْمُهَا", "مُكْتَبَّةٌ جَمِيلَةٌ"]


def _parquet(tmp_path: Path, name: str = "bench.parquet") -> Path:
    p = tmp_path / name
    pd.DataFrame({"input": ["قوله فحكمها", "مكتبة جميلة"], "output": GT}).to_parquet(p)
    return p


def _preds(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_score_final_preds_format(tmp_path: Path, capsys) -> None:
    data = _parquet(tmp_path)
    preds = _preds(tmp_path, "p.jsonl", [{"idx": i, "student": gt} for i, gt in enumerate(GT)])
    rc = main(["score", "--preds", str(preds), "--data", str(data), "--key", "student"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["der_ce"] == 0.0
    assert out["n"] == 2


def test_score_plain_text_lines(tmp_path: Path, capsys) -> None:
    data = _parquet(tmp_path)
    preds = tmp_path / "plain.txt"
    preds.write_text("\n".join(GT) + "\n", encoding="utf-8")
    rc = main(["score", "--preds", str(preds), "--data", str(data)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["der_ce"] == 0.0


def test_bootstrap_vs_reference(tmp_path: Path, capsys) -> None:
    data = _parquet(tmp_path)
    cand = _preds(tmp_path, "c.jsonl", [{"idx": i, "student": s} for i, s in enumerate(GT)])
    marks = "ًَُّْ"
    stripped = ["".join(c for c in g if c not in marks) for g in GT]
    ref = _preds(tmp_path, "r.jsonl", [{"idx": i, "student": s} for i, s in enumerate(stripped)])
    rc = main(["score", "--preds", str(cand), "--data", str(data), "--key", "student",
               "--vs", str(ref), "--vs-key", "student"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["vs"]["delta"] < 0  # candidate (perfect) better than stripped reference


def test_data_accepts_hf_dataset_id(monkeypatch, tmp_path, capsys) -> None:
    # --data may name the HF dataset instead of a local parquet; the
    # loader resolves it through huggingface_hub with a local cache
    import sadeedbench.cli as cli

    cached = _parquet(tmp_path, name="train.parquet")
    calls = {}

    def fake_snapshot(repo_id, repo_type):
        calls["repo_id"] = repo_id
        return str(cached.parent)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot, raising=False)
    rc = cli.main(["score", "--preds", str(_preds(tmp_path, "p.jsonl", [
        {"idx": i, "student": g} for i, g in enumerate(GT)])), "--data", "Misraj/SadeedDiac-25"])
    assert rc == 0
    assert calls["repo_id"] == "Misraj/SadeedDiac-25"
    out = json.loads(capsys.readouterr().out)
    assert out["der_ce"] == 0.0

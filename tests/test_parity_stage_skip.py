"""Stage-skip resume for parity_model: preemption restarts must not
redo precision stages whose margin reports are already durable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

modal = pytest.importorskip("modal")

from gpu.modal_export import pending_precisions  # noqa: E402


def _write_zip(path: Path, parity: bool) -> None:
    import zipfile

    import yaml

    meta = {"id": "m", "precision": "int4"}
    if parity:
        meta["parity"] = {"cer_delta": 0.05}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump(meta))


def test_completed_stage_is_skipped(tmp_path: Path) -> None:
    _write_zip(tmp_path / "ara-diac-small-2.1-fp32.zip", parity=True)
    (tmp_path / "ara-diac-small-2.1-margins-fp32.json").write_text("{}")
    got = pending_precisions(tmp_path, "ara-diac-small-2.1", ["fp32", "fp16", "int8"])
    assert got == ["fp16", "int8"]


def test_fresh_run_keeps_all_stages(tmp_path: Path) -> None:
    got = pending_precisions(tmp_path, "ara-diac-small-2.1", ["fp32", "fp16", "int8"])
    assert got == ["fp32", "fp16", "int8"]


def test_zip_without_margin_report_still_runs(tmp_path: Path) -> None:
    # a partial stage (zip present, margin report missing) must rerun —
    # the margin report is the last artifact written, so its absence
    # means the stage never completed
    (tmp_path / "ara-diac-small-2.1-int8.zip").write_text("partial")
    got = pending_precisions(tmp_path, "ara-diac-small-2.1", ["int8"])
    assert got == ["int8"]


def test_list_input_from_entrypoint(tmp_path: Path) -> None:
    # the parity/margins entrypoints pass precisions.split(",") — a
    # list — into the remote functions; the functions must accept both
    # forms (direct ::parity_model CLI invocation passes a string)
    _write_zip(tmp_path / "ara-diac-small-2.1-fp32.zip", parity=True)
    (tmp_path / "ara-diac-small-2.1-margins-fp32.json").write_text("{}")
    got = pending_precisions(tmp_path, "ara-diac-small-2.1", ["fp32", "int8"])
    assert got == ["int8"]


def test_string_input_strips_whitespace(tmp_path: Path) -> None:
    got = pending_precisions(tmp_path, "m", "fp32, int8")
    assert got == ["fp32", "int8"]


def test_margins_without_zip_parity_reruns(tmp_path: Path) -> None:
    # a standalone margin_model run writes the margins json but never
    # the zip's parity block; such a stage must rerun, not skip
    (tmp_path / "m-margins-int4.json").write_text("{}")
    _write_zip(tmp_path / "m-int4.zip", parity=False)
    assert pending_precisions(tmp_path, "m", ["int4"]) == ["int4"]


def test_margins_and_zip_parity_skip(tmp_path: Path) -> None:
    (tmp_path / "m-margins-int4.json").write_text("{}")
    _write_zip(tmp_path / "m-int4.zip", parity=True)
    assert pending_precisions(tmp_path, "m", ["int4"]) == []


def test_corrupt_zip_is_pending(tmp_path: Path) -> None:
    (tmp_path / "m-margins-int4.json").write_text("{}")
    (tmp_path / "m-int4.zip").write_text("not a zip")
    assert pending_precisions(tmp_path, "m", ["int4"]) == ["int4"]

"""Stage-skip resume for parity_model: preemption restarts must not
redo precision stages whose margin reports are already durable."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

modal = pytest.importorskip("modal")

from gpu.modal_export import pending_precisions  # noqa: E402


def test_completed_stage_is_skipped(tmp_path: Path) -> None:
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

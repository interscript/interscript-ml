"""RunState marker protocol tests — pure CPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu.runstate import RunState  # noqa: E402


def test_fresh_run_state_reports_nothing_done(tmp_path: Path) -> None:
    state = RunState(tmp_path / "run-x")
    assert not state.training_done()
    assert not state.eval_done()
    assert state.latest_step() == -1
    assert state.read_eval() is None


def test_log_writes_into_missing_run_dir(tmp_path: Path) -> None:
    # the arm-4 bug: first log must create the run dir, not crash
    state = RunState(tmp_path / "brand-new-arm")
    state.log("watch step=-1")
    lines = (tmp_path / "brand-new-arm" / "chain_log.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert "watch step=-1" in lines[0]


def test_step_and_marker_semantics(tmp_path: Path) -> None:
    state = RunState(tmp_path / "run-y")
    for n in (500, 2000, 1000):
        (tmp_path / "run-y" / f"step-{n}").mkdir(parents=True)
    assert state.latest_step() == 2000
    (tmp_path / "run-y" / "best").mkdir()
    (tmp_path / "run-y" / "best" / "config.json").write_text("{}")
    assert state.training_done()
    assert not state.eval_done()


def test_read_eval_roundtrip(tmp_path: Path) -> None:
    state = RunState(tmp_path / "run-z")
    (tmp_path / "run-z").mkdir()
    (tmp_path / "run-z" / "final_eval.json").write_text('{"gate_pass": true}')
    assert state.eval_done()
    assert state.read_eval()["gate_pass"] is True

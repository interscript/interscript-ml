"""Paired bootstrap lives in its pure home (no modal import) so the
CLI and the training harness share one implementation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sadeedbench.bootstrap import bootstrap_delta  # noqa: E402


def test_delta_and_ci_exclude_zero() -> None:
    # (candidate, reference): positive delta = candidate worse
    cand = [2.0] * 50
    ref = [1.0] * 50
    out = bootstrap_delta(cand, ref)
    assert abs(out["delta"] - 1.0) < 1e-9
    assert out["ci95"][0] > 0
    assert out["p_leq0"] < 0.001


def test_none_items_dropped_pairwise() -> None:
    cand = [2.0, None, 2.0]
    ref = [1.0, None, 1.0]
    out = bootstrap_delta(cand, ref)
    assert out["n"] == 2
    assert abs(out["delta"] - 1.0) < 1e-9


def test_deterministic_seed() -> None:
    cand = [2.0, 3.0, 4.0, 5.0]
    ref = [1.0, 2.0, 3.0, 4.0]
    assert bootstrap_delta(cand, ref) == bootstrap_delta(cand, ref)

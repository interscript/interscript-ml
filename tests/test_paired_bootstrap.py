"""Paired bootstrap for per-item DER deltas (microkimi protocol,
generalized to the windowed Arabic harness)."""

import pytest

pytest.importorskip("modal")

from src.gpu.modal_distill import paired_bootstrap


def test_delta_outside_zero_ci():
    # student uniformly +1 DER on every item -> CI excludes 0
    out = paired_bootstrap([1.0] * 100)
    assert abs(out["delta"] - 1.0) < 1e-9
    assert out["ci95"][0] > 0
    assert out["p_leq0"] < 0.001


def test_zero_delta_ci_straddles_zero():
    out = paired_bootstrap([0.0] * 100)
    assert out["ci95"][0] <= 0.0 <= out["ci95"][1]
    assert out["p_leq0"] > 0.05


def test_seed_reproducible():
    a = paired_bootstrap([0.5, -0.2, 1.3, 0.1, -0.4] * 20)
    b = paired_bootstrap([0.5, -0.2, 1.3, 0.1, -0.4] * 20)
    assert a == b

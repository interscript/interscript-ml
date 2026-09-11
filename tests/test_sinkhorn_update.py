"""Sinkhorn-balanced momentum update specs (TODO.impl/06, DeepSeek
V4.1-Flash Algorithm 1): Nesterov momentum, alternating row/column L2
normalization over K steps, near-zero row masking, sqrt(n) RMS
conversion, gamma-corrected learning rate, no weight decay. Replaces
Adam for embedding tables and prediction heads."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

torch = pytest.importorskip("torch")

from gpu.sinkhorn_update import SinkhornUpdate, sinkhorn_balance  # noqa: E402


def _randn(rows, cols, seed):
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))


def test_balance_yields_unit_row_norms() -> None:
    w = _randn(6, 4, seed=1)
    u = sinkhorn_balance(w, k=11)
    row_norms = u.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-4)


def test_balance_masks_near_zero_rows() -> None:
    """Rows at or below tau * mean-row-norm contribute nothing — their
    update stays zero after balancing."""
    w = _randn(6, 4, seed=2)
    w[0] = 1e-9  # dead row
    u = sinkhorn_balance(w, k=11, tau=1e-3)
    assert u[0].abs().max().item() == 0.0
    live = u[1:]
    assert torch.allclose(live.norm(dim=1), torch.ones(live.size(0)), atol=1e-4)


def test_balance_survives_small_magnitudes() -> None:
    """The mHC log-domain lesson: direct division NaNs when magnitudes
    shrink across alternating normalizations; eps guards must hold at
    1e-20-class scales."""
    w = _randn(8, 5, seed=3) * 1e-8
    u = sinkhorn_balance(w, k=11, eps=1e-20)
    assert torch.isfinite(u).all()


def test_balance_matches_hand_computed_case() -> None:
    """k=1 is a single ROW step: each row is L2-normalized; the sqrt(n)
    RMS conversion is the optimizer's job, not the balance's."""
    w = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    u = sinkhorn_balance(w, k=1)
    expect = torch.tensor([[0.6, 0.8], [0.6, 0.8]])
    torch.testing.assert_close(u, expect)


def test_optimizer_step_matches_manual_algorithm() -> None:
    """One SinkhornUpdate step on an embedding-shaped weight equals the
    algorithm written out by hand: Nesterov momentum, balance, sqrt(n),
    gamma-corrected lr, no weight decay."""
    rows, cols = 5, 3
    weight = _randn(rows, cols, seed=4)
    grad = _randn(rows, cols, seed=5)
    beta, lr, gamma, n = 0.9, 0.02, 0.18, rows

    before = weight.clone()
    opt = SinkhornUpdate([weight], lr=lr, beta=beta, gamma=gamma)
    weight.grad = grad.clone()
    opt.step()

    momentum = grad.clone() * (1 - beta)  # first step: buf = (1-beta)*g
    lookahead = beta * momentum + (1 - beta) * grad
    u = sinkhorn_balance(lookahead, k=11)
    n = before.size(1)  # hidden feature count
    delta = math.sqrt(n) * u
    expect = before - (gamma * lr) * delta
    torch.testing.assert_close(weight, expect)


def test_no_weight_decay_applied() -> None:
    weight = _randn(4, 3, seed=6)
    before = weight.clone()
    opt = SinkhornUpdate([weight], weight_decay=0.1)  # ignored by contract
    weight.grad = _randn(4, 3, seed=7)
    opt.step()
    # balanced rows have unit L2 norm; movement is exactly
    # lr * gamma * sqrt(n_hidden), untouched by weight decay
    movement = (weight - before).norm(dim=1)
    expect = opt.defaults["lr"] * opt.defaults["gamma"] * math.sqrt(3)
    assert torch.allclose(movement, torch.full_like(movement, expect), atol=1e-5)

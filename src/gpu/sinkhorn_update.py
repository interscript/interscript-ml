"""Sinkhorn-balanced momentum update for embedding tables and
prediction heads (DeepSeek-V4.1-Flash Algorithm 1, TODO.impl/06).

Replaces Adam for large row-structured matrices: Nesterov momentum,
then alternating row/column L2 normalization (odd number of steps,
ending on rows), near-zero row masking, sqrt(n) to convert unit row
L2 into unit row RMS, and a gamma-corrected learning rate to match
Adam's update magnitude. Momentum-only state; no weight decay.

Numerical note (the mHC lesson): alternating direct division produced
NaNs for ~10% of inits when magnitudes shrink; the eps here is 1e-20
and every division is guarded.
"""

from __future__ import annotations

import math

import torch


def sinkhorn_balance(
    update: torch.Tensor,
    k: int = 11,
    tau: float = 1e-3,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Alternating row/column L2 normalization ending on rows (k odd),
    with near-zero rows masked. Returns the balanced update; the
    caller applies the sqrt(n) RMS conversion."""
    if k % 2 == 0:
        k += 1  # the algorithm requires an odd count (ends row-wise)
    rows = update.norm(dim=1)
    mean_row = rows.mean()
    work = update.clone()
    work[(rows <= tau * mean_row)] = 0.0
    for step in range(1, k + 1):
        if step % 2 == 1:  # odd: rows
            norms = work.norm(dim=1, keepdim=True).add_(eps)
            work = work / norms
        else:  # even: columns
            norms = work.norm(dim=0, keepdim=True).add_(eps)
            work = work / norms
    work[(rows <= tau * mean_row)] = 0.0
    return work


class SinkhornUpdate(torch.optim.Optimizer):
    """Momentum + Sinkhorn balancing in place of Adam's second moment,
    for embedding tables and prediction heads (row = token/n-gram,
    column = hidden feature)."""

    def __init__(self, params, lr: float = 2.6e-4, beta: float = 0.95,
                 gamma: float = 0.18, k: int = 11, tau: float = 1e-3,
                 weight_decay: float = 0.0) -> None:  # noqa: ARG002 (ignored by contract)
        # rows of the update matrix carry the row structure (token /
        # n-gram identity); n = hidden feature count (columns), whose
        # sqrt converts unit row L2 norm into unit row RMS
        settings = {"lr": lr, "momentum": beta, "gamma": gamma,
                    "k": k, "tau": tau, "weight_decay": 0.0}
        super().__init__(list(params), settings)

    @torch.no_grad()
    def step(self, closure=None) -> None:  # noqa: ARG002
        for group in self.param_groups:
            beta, lr, gamma = group["momentum"], group["lr"], group["gamma"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - beta)
                g = p.grad.lerp(buf, beta)  # Nesterov lookahead
                u = sinkhorn_balance(g, k=group["k"], tau=group["tau"])
                p.add_(u.to(p.dtype), alpha=-lr * gamma * math.sqrt(p.size(1)))

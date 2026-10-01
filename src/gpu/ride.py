"""RIDE displacement-arm math (TODO.sota-2026/05).

The probe (rababa/probe_ride_direction.py, 2026-10-01) measured the
r7-over-r6 SFT residual direction as domain-general in encoder layers
0-8 (cos 0.59-0.95 across classical/news text) and idiosyncratic
deeper. These helpers implement the training arm: regress the
student's encoder hidden states toward ridge-projected, extrapolated
teacher targets h_t' = h_teacher + lam*(h_teacher - h_base).

The ridge map is fit at arm start (teacher hiddens -> the student's
initial hiddens on the same inputs); by linearity the displacement
survives the projection: W(h_t + lam*(h_t - h_b)) = W h_t +
lam*(W h_t - W h_b).
"""

from __future__ import annotations

import torch


def displaced_targets(h_teacher: torch.Tensor, h_base: torch.Tensor, lam: float) -> torch.Tensor:
    return h_teacher + lam * (h_teacher - h_base)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over kept positions and all dims.

    pred/target: (B, T, D); attention_mask: (B, T), 1 = kept.
    Returns 0.0 when nothing is kept.
    """
    mask = attention_mask.to(pred.dtype).unsqueeze(-1)
    diff2 = (pred - target).pow(2) * mask
    n = mask.sum() * pred.shape[-1]
    if n == 0:
        return pred.new_tensor(0.0)
    return diff2.sum() / n


def fit_ridge(H_teacher: torch.Tensor, H_student: torch.Tensor, alpha: float) -> torch.Tensor:
    """Closed-form ridge W minimizing ||H_teacher @ W.T - H_student||^2
    + alpha*||W||^2. Returns W of shape (d_student, d_teacher)."""
    Ht = H_teacher.double()
    Hs = H_student.double()
    d_t = Ht.shape[-1]
    A = Ht.T @ Ht + alpha * torch.eye(d_t, dtype=Ht.dtype, device=Ht.device)
    W = torch.linalg.solve(A, Ht.T @ Hs).T
    return W.to(H_teacher.dtype)

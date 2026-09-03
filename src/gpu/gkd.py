"""On-policy GKD helpers (EXPERIMENTS.md "GKD — on-policy distillation
rung"). Pure torch, no modal import — unit-testable outside CI's GPU
image. The training-side wiring lives in gpu.modal_distill."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gkd_weight(step: int, total_steps: int, ratio: float,
               anneal_from: float = 2 / 3) -> float:
    """Full ratio until `anneal_from` of training, then linear to zero
    at total_steps (the registered anneal over the final third)."""
    if total_steps <= 0 or step >= total_steps:
        return 0.0
    frac = step / total_steps
    if frac < anneal_from:
        return ratio
    return ratio * max(0.0, 1.0 - (frac - anneal_from) / (1.0 - anneal_from))


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs of `targets` under `logits` (teacher-forced
    shift): logits [B, L, V], targets [B, L] -> [B, L-1]."""
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return lp.gather(-1, targets[:, 1:].unsqueeze(-1)).squeeze(-1)


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
               targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """KL(student || teacher) on the student's own sampled tokens:
    mean over (masked) positions of logp_s - logp_t. Gradient flows
    through the student term only — call under no_grad for teacher."""
    s_lp = token_logprobs(student_logits, targets)
    with torch.no_grad():
        t_lp = token_logprobs(teacher_logits, targets)
    diff = s_lp - t_lp
    if mask is not None:
        m = mask[:, 1:].to(diff.dtype)
        return (diff * m).sum() / m.sum().clamp(min=1.0)
    return diff.mean()

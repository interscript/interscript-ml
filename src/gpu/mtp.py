"""Multi-token-prediction auxiliary head (E5, TODO 07-hy4).

Per-position multi-step prediction as a TRAINING auxiliary: each
decoder position's hidden state additionally predicts the target
tokens at t+1..t+k, densifying supervision on the decode path. The
head is discarded at inference — the shipped student stays vanilla
and size-identical; only the training run carries it (saved as
mtp_head.pt beside student.pt for resume, never exported).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    def __init__(self, d_model: int, vocab: int, steps: int = 3):
        super().__init__()
        self.steps = steps
        self.heads = nn.ModuleList(
            nn.Linear(d_model, vocab) for _ in range(steps)
        )

    def aux_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """hidden [B, T, d] is the decoder's final hidden state whose
        position t already predicts labels[t] through the main head;
        step-k heads predict labels[t+k] from the same position."""
        total = 0.0
        n = 0
        for k, head in enumerate(self.heads, start=1):
            tgt = labels[:, k:]
            m = tgt != -100
            if m.any():
                total = total + F.cross_entropy(
                    head(hidden[:, :-k])[m].float(), tgt[m]
                )
                n += 1
        return total / max(n, 1)


def build_mtp(student, steps: int = 3) -> MTPHead:
    return MTPHead(
        student.config.d_model, student.config.vocab_size, steps=steps
    ).to(student.device)


def mtp_named(head: MTPHead):
    for name, p in head.named_parameters():
        yield f"mtp.{name}", p

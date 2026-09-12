"""Engram: byte-n-gram conditional memory for byte-level students
(TODO.impl/10, gated open 2026-09-12; mechanism corrected from
DeepSeek-V4.1-Flash §2.4.2 — hash-addressed lookup tables, not example
retrieval).

Haraqat are strongly lexical: a lookup keyed on byte n-grams captures
idiomatic vocalization that a 300M byte model must otherwise spend
capacity memorizing. The module sums ONE embedding per position into
the encoder stream at ONE layer — the proportionate dose for a 300M
backbone (DeepSeek places two modules in a 552B model).

Addressing: each byte position addresses the table with k independent
hashes of the n-grams ENDING at it (orders {2,3,4}); the k looked-up
vectors are averaged, projected to d_model, and added. Deterministic
hashing (FNV-1a variants per order/head) — no learned addressing, so
the table is pure memorization decoupled from compute, prefetchable,
and export-stable (Gather ops only).

Training pairs with the Sinkhorn-balanced update (gpu.sinkhorn_update):
row-structured embedding tables are exactly its intended parameter
class. Storage in the artifact is int8 with per-table scales; the
module holds fp32/fp16 at runtime.
"""

from __future__ import annotations

import torch
from torch import nn

FNV_OFFSET = 0x811C9DC5
FNV_PRIMES = (0x01000193, 0x01000193**2 % (1 << 32), 0x85EBCA6B)


def _fnv1a(data: bytes, seed: int) -> int:
    h = (FNV_OFFSET ^ seed) & 0xFFFFFFFF
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def ngram_addresses(seq: list[int], orders=(2, 3, 4)) -> list[list[int]]:
    """Per-position table addresses: order-n hashes of the byte
    n-grams ENDING at each position. Positions without a full n-gram
    (sequence starts) address 0 — the null row."""
    addresses: list[list[int]] = []
    for end in range(len(seq)):
        row = [0] * len(orders)
        for i, order in enumerate(orders):
            if end + 1 >= order:
                gram = bytes((t - 3) % 256 for t in seq[end + 1 - order : end + 1])
                row[i] = _fnv1a(gram, seed=i + 1) or 1
        addresses.append(row)
    return addresses


class Engram(nn.Module):
    """Byte-n-gram lookup memory summed into the encoder stream.

    table_shape: (entries, dim). Projection maps dim -> d_model; the
    added vector is zero-initialized (gated by a zero scalar) so a
    freshly attached module leaves the backbone's function identical
    at step 0 — the same safety property as the PKM gate.
    """

    def __init__(self, d_model: int, entries: int = 1 << 21, dim: int = 32,
                 orders=(2, 3, 4)):
        super().__init__()
        self.orders = tuple(orders)
        self.table = nn.Embedding(entries, dim)
        self.proj = nn.Linear(dim, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[B, T] token ids -> [B, T, d_model] additive memory.
        Python-side address computation — the training path."""
        b, t = input_ids.shape
        flat = input_ids.flatten().tolist()
        rows = []
        for bi in range(b):
            seq = flat[bi * t : (bi + 1) * t]
            rows.extend(ngram_addresses(seq, self.orders))
        addresses = torch.tensor(rows, dtype=torch.long,
                                 device=input_ids.device).reshape(b, t, -1)
        return self.from_addresses(addresses)

    def from_addresses(self, addresses: torch.Tensor) -> torch.Tensor:
        """[B, T, k] table addresses -> [B, T, d_model]. The pure-graph
        path: Gather + mean + Linear only, exportable and runtime-
        portable. The hash itself is computed by the caller (training:
        ngram_addresses in python; runtimes: the same 20-line function
        per runtime — the IMF contract carries addresses as an input."""
        # remainder, not bitwise-and: ONNX exports int64 mod (and
        # prime table sizes — the report's distinct-primes choice —
        # become available)
        looked = self.table(torch.remainder(addresses, self.table.num_embeddings))
        return self.proj(looked.mean(dim=2))

    def export_int8_state(self) -> dict[str, torch.Tensor]:
        """The artifact form: int8 table with one scale, fp16
        projection (the projection is d_model*dim — tiny)."""
        w = self.table.weight.detach()
        scale = w.abs().max().clamp(min=1e-8) / 127.0
        return {
            "table_int8": (w / scale).round().to(torch.int8),
            "table_scale": scale.reshape(1),
            "proj_fp16": self.proj.weight.detach().half(),
        }

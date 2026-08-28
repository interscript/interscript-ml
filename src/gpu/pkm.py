"""Product-key memory layers for byte-level students.

The Qwen3.8-Flash-Next / LongCat-Flash-Lite capacity axis (arXiv
2601.21204): parameters and compute are separable — a lookup memory adds
knowledge capacity at near-zero FLOPs. Precedent at character level:
Lample et al., "Large Memory Layers with Product Keys", NeurIPS 2019.

Design notes:
- Injected as a parallel residual branch on decoder FFNs:
  ``y = FFN(x) + g * mem(LN(x))`` with the gate ``g`` zero-initialized, so
  a pretrained backbone's function is preserved exactly at step 0
  (ReZero-style bootstrap; memory parameters receive gradient once the
  gate moves).
- The value table is a random-access tensor: it belongs to the
  embedding-like quantization class (see TODO.qwen-next/01) and stays on
  AdamW in the Muon split (TODO.qwen-next/03).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProductKeyMemory(nn.Module):
    """Sparse memory read: two half-codebooks of ``n_keys`` keys span
    ``n_keys**2`` slots; per position, top-k candidates from each half
    combine into a k*k grid from which the final ``topk`` slots are
    gathered and softmax-weighted."""

    def __init__(self, d_model: int, n_keys: int = 128, topk: int = 32) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.n_keys = n_keys
        self.topk = topk
        d_k = d_model // 2
        self.ln = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.k1 = nn.Parameter(torch.randn(n_keys, d_k) / d_k**0.5)
        self.k2 = nn.Parameter(torch.randn(n_keys, d_k) / d_k**0.5)
        self.values = nn.Parameter(torch.randn(n_keys * n_keys, d_model) / d_model**0.5)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_k**-0.5

    def forward(self, x):
        h = self.ln(x)
        q1, q2 = self.wq(h).chunk(2, dim=-1)
        s1 = torch.einsum("btd,cd->btc", q1, self.k1) * self.scale
        s2 = torch.einsum("btd,cd->btc", q2, self.k2) * self.scale
        t1, i1 = s1.topk(self.topk, dim=-1)
        t2, i2 = s2.topk(self.topk, dim=-1)
        cand = t1[:, :, :, None] + t2[:, :, None, :]
        w, idx = cand.flatten(-2).topk(self.topk, dim=-1)
        a, b = idx // self.topk, idx % self.topk
        slot = torch.gather(i1, -1, a) * self.n_keys + torch.gather(i2, -1, b)
        v = self.values[slot]  # (B, T, topk, d_model)
        mem = torch.einsum("btk,btkd->btd", torch.softmax(w, dim=-1), v)
        return self.wo(mem)


class _FFNWithMemory(nn.Module):
    """Wraps a T5LayerFF: keeps its output contract, adds the gated
    memory branch computed from the same (pre-FFN) hidden states."""

    def __init__(self, ffn: nn.Module, memory: ProductKeyMemory) -> None:
        super().__init__()
        self.ffn = ffn
        self.memory = memory
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states, **kwargs):
        out = self.ffn(hidden_states, **kwargs)
        mem = self.gate * self.memory(hidden_states)
        if isinstance(out, tuple):
            return (out[0] + mem,) + out[1:]
        return out + mem


def inject_pkm(model, layer_indices=(-2, -4, -6), n_keys: int = 128, topk: int = 32):
    """Wrap decoder-block FFNs with memory branches in place. Negative
    indices count from the output side — the late decoder blocks, where
    lexical (table-friendly) decisions crystallize."""
    blocks = model.decoder.block
    n = len(blocks)
    for i in layer_indices:
        idx = i if i >= 0 else n + i
        if not 0 <= idx < n:
            raise IndexError(f"layer index {i} out of range for {n} decoder blocks")
        blocks[idx].layer[1] = _FFNWithMemory(blocks[idx].layer[1], ProductKeyMemory(
            model.config.d_model, n_keys=n_keys, topk=topk))
    base = sum(p.numel() for p in model.parameters())
    mem = sum(p.numel() for m in model.modules() if isinstance(m, ProductKeyMemory)
              for p in m.parameters())
    print(f"[pkm] injected {len(tuple(layer_indices))} memory layers: "
          f"+{mem / 1e6:.1f}M params on a {base / 1e6:.0f}M model", flush=True)
    return model


def load_student_with_pkm(path, pkm_cfg: dict):
    """Load a PKM student saved with ``save_pretrained``: the vanilla
    class ignores the injected parameters, so re-inject then pull them
    from the checkpoint file. Raises if any PKM parameter is missing."""
    from transformers import AutoModelForSeq2SeqLM

    student = AutoModelForSeq2SeqLM.from_pretrained(path)
    inject_pkm(student, **pkm_cfg)
    sd = None
    import glob

    for pattern in ("model.safetensors", "model*.safetensors", "pytorch_model.bin"):
        hits = glob.glob(str(path / pattern))
        if hits:
            if hits[0].endswith(".bin"):
                sd = torch.load(hits[0], map_location="cpu", weights_only=True)
            else:
                from safetensors.torch import load_file

                sd = load_file(hits[0])
            break
    if sd is None:
        raise RuntimeError(f"no weight file found in {path}")
    missing, unexpected = student.load_state_dict(sd, strict=False)
    # tied embeddings (shared/lm_head) are intentionally absent from the
    # checkpoint file — only the injected memory keys are load-bearing here
    missing_pkm = [k for k in missing if "memory." in k or k.endswith(".gate")]
    if missing_pkm:
        raise RuntimeError(f"PKM parameters missing from checkpoint: {missing_pkm[:5]}")
    return student

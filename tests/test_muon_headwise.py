"""Head-wise Muon specs (TODO.impl/03): per-head preconditioning for
Q/K weights — the update each attention head would get from vanilla
Muon applied to its slice alone, reassembled into the full matrix.
Validated externally by DeepSeek-V4.1-Flash, GLM-5, Kimi-K3."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

torch = pytest.importorskip("torch")

from gpu.muon import Muon, qk_named, split_parameters  # noqa: E402


def _grad_like(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_headwise_equals_per_slice_vanilla() -> None:
    """One step of head-wise Muon on [H*d, D] must equal vanilla Muon
    applied to each [d, D] slice with the same grad — that is the
    entire semantic content of 'different preconditioners per head'."""
    heads, d, dim = 3, 8, 16
    weight = torch.randn(heads * d, dim, generator=torch.Generator().manual_seed(7))
    grad = _grad_like((heads * d, dim), seed=11)

    slices = list(weight.chunk(heads))
    slice_grads = list(grad.chunk(heads))
    # clone BEFORE vanilla mutates `weight` through its chunk views
    hw_weight = weight.clone()
    hw = Muon([hw_weight], lr=0.01, momentum=0.95)
    hw.add_headwise_group(list(hw.param_groups[0]["params"]), heads=heads)
    # the base group must be replaced, not duplicated: the param moves
    # into the headwise group
    assert hw.param_groups[0].get("headwise") is True
    hw_weight.grad = grad.clone()
    hw.step()

    vanilla = Muon(slices, lr=0.01, momentum=0.95)
    for p, g in zip(slices, slice_grads, strict=True):
        p.grad = g.clone()
    vanilla.step()

    for h in range(heads):
        torch.testing.assert_close(hw_weight.chunk(heads)[h], weight.chunk(heads)[h])


def test_heads_one_matches_vanilla_whole() -> None:
    weight = torch.randn(12, 16, generator=torch.Generator().manual_seed(3))
    grad = _grad_like((12, 16), seed=5)

    v = Muon([weight.clone()], lr=0.01, momentum=0.95)
    v.param_groups[0]["params"][0].grad = grad.clone()
    v.step()

    hw_w = weight.clone()
    hw = Muon([hw_w], lr=0.01, momentum=0.95)
    hw.add_headwise_group(list(hw.param_groups[0]["params"]), heads=1)
    hw_w.grad = grad.clone()
    hw.step()

    torch.testing.assert_close(hw_w, v.param_groups[0]["params"][0])


def test_headwise_differs_from_vanilla_for_heterogeneous_heads() -> None:
    """If every head were preconditioned identically the split would be
    a no-op; give the slices different singular structure and the two
    updates must diverge."""
    heads, d, dim = 2, 8, 16
    base = _grad_like((heads * d, dim), seed=13)
    base[:d] *= 0.01  # near-isotropic head vs anisotropic head
    weight = torch.zeros(heads * d, dim)

    v = Muon([weight.clone()], lr=0.01, momentum=0.95)
    v.param_groups[0]["params"][0].grad = base.clone()
    v.step()

    hw_w = weight.clone()
    hw = Muon([hw_w], lr=0.01, momentum=0.95)
    hw.add_headwise_group(list(hw.param_groups[0]["params"]), heads=heads)
    hw_w.grad = base.clone()
    hw.step()

    assert not torch.allclose(hw_w, v.param_groups[0]["params"][0])


def test_qk_named_selects_only_qk_projections() -> None:
    named = [
        ("encoder.block.0.layer.0.SelfAttention.q.weight", torch.zeros(2)),
        ("encoder.block.0.layer.0.SelfAttention.k.weight", torch.zeros(2)),
        ("encoder.block.0.layer.0.SelfAttention.v.weight", torch.zeros(2)),
        ("encoder.block.0.layer.0.SelfAttention.o.weight", torch.zeros(2)),
        ("decoder.block.0.layer.0.EncDecAttention.q.weight", torch.zeros(2)),
        ("decoder.block.0.layer.0.EncDecAttention.k.weight", torch.zeros(2)),
        ("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", torch.zeros(2)),
    ]
    selected = {n for n, _ in qk_named(named)}
    assert selected == {
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.block.0.layer.0.SelfAttention.k.weight",
        "decoder.block.0.layer.0.EncDecAttention.q.weight",
        "decoder.block.0.layer.0.EncDecAttention.k.weight",
    }


def test_split_parameters_unchanged_when_headwise_unused() -> None:
    """Off-state must be bit-identical routing: the feature adds a
    group kind, it does not touch the default split."""
    named = [
        ("shared.weight", torch.zeros(3, 4, requires_grad=True)),
        ("encoder.block.0.layer.1.DenseReluDense.wi_0.weight",
         torch.zeros(4, 3, requires_grad=True)),
        ("encoder.block.0.layer.0.layer_norm.weight", torch.zeros(3, requires_grad=True)),
    ]
    muon, adamw = split_parameters(named)
    assert [p.shape for p in muon] == [torch.Size([4, 3])]
    assert [p.shape for p in adamw] == [torch.Size([3, 4]), torch.Size([3])]

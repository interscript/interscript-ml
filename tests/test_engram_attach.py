"""Engram attachment specs (TODO.impl/10's run): zero-init identity,
hook plumbing, param split for optimizer routing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gpu.engram import attach_engram, engram_param_split  # noqa: E402


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration

    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=259, d_model=32, d_ff=64, d_kv=16,
            num_layers=4, num_decoder_layers=4, num_heads=2,
            feed_forward_proj="relu", decoder_start_token_id=0,
        )
    )


def test_attached_model_is_identity_at_step0() -> None:
    """The PKM rule: zero-init projection means the attached model's
    forward equals the backbone's, bit-for-bit."""
    torch.manual_seed(0)
    base = _tiny_t5().eval()
    with torch.no_grad():
        for p in base.parameters():
            p.copy_(torch.randn_like(p).mul(0.1))
    attached = _tiny_t5().eval()
    attached.load_state_dict(base.state_dict())
    attach_engram(attached, layer=2, entries=1024, dim=8)

    ids = torch.tensor([[5, 6, 7, 8, 1]])
    with torch.no_grad():
        out_base = base(input_ids=ids, decoder_input_ids=torch.tensor([[0]]))
        out_att = attached(input_ids=ids, decoder_input_ids=torch.tensor([[0]]))
    torch.testing.assert_close(out_att.logits, out_base.logits)


def test_memory_flows_once_projection_trains() -> None:
    attached = _tiny_t5()
    attach_engram(attached, layer=1, entries=4096, dim=8)
    with torch.no_grad():
        attached._engram.proj.weight.normal_(std=0.1)
    ids = torch.tensor([[5, 6, 7, 8, 1]])
    out = attached(input_ids=ids, decoder_input_ids=torch.tensor([[0]])).logits
    ids2 = torch.tensor([[5, 6, 9, 8, 1]])
    out2 = attached(input_ids=ids2, decoder_input_ids=torch.tensor([[0]])).logits
    assert not torch.allclose(out, out2), "memory must reach the logits"


def test_param_split_routes_table_and_projection() -> None:
    model = _tiny_t5()
    attach_engram(model, layer=0, entries=512, dim=8)
    tables, others = engram_param_split(model)
    assert len(tables) == 1 and len(others) == 1
    assert tables[0].shape == (512, 8)
    assert tuple(others[0].shape) == (model.config.d_model, 8)
    assert not any(p.requires_grad is False for p in tables + others)

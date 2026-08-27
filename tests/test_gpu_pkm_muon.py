"""Smoke tests for ``gpu.pkm`` and ``gpu.muon`` (CPU, tiny models).

The identity property matters most: a zero-initialized memory gate must
leave the pretrained backbone's function bit-identical at step 0 — that
is what makes injection safe on a pretrained student.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gpu.muon import Muon, split_parameters  # noqa: E402
from gpu.pkm import inject_pkm, load_student_with_pkm  # noqa: E402


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(
        vocab_size=259, d_model=32, d_ff=64, d_kv=16,
        num_layers=4, num_decoder_layers=4, num_heads=2,
        feed_forward_proj="relu", decoder_start_token_id=0,
    )
    return T5ForConditionalGeneration(config)


def _gates(model):
    return [b.layer[1].gate for b in model.decoder.block if hasattr(b.layer[1], "gate")]


def test_pkm_gate_zero_preserves_function() -> None:
    model = _tiny_t5().eval()
    ids = torch.randint(3, 259, (2, 7))
    labels = torch.randint(3, 259, (2, 5))
    with torch.no_grad():
        before = model(input_ids=ids, labels=labels).logits.clone()
    inject_pkm(model, layer_indices=[-1, -3], n_keys=8, topk=4)
    with torch.no_grad():
        after = model(input_ids=ids, labels=labels).logits
    assert torch.equal(before, after)
    assert len(_gates(model)) == 2


def test_pkm_gradients_flow_once_gate_moves() -> None:
    model = _tiny_t5().train()
    inject_pkm(model, layer_indices=[-1], n_keys=8, topk=4)
    with torch.no_grad():
        _gates(model)[0].fill_(1.0)
    ids = torch.randint(3, 259, (2, 7))
    labels = torch.randint(3, 259, (2, 5))
    model(input_ids=ids, labels=labels).loss.backward()
    memory = model.decoder.block[-1].layer[1].memory
    for name in ("wq", "wo", "values", "k1", "k2"):
        p = getattr(memory, name)
        p = p.weight if not isinstance(p, torch.Tensor) else p
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


def test_pkm_generate_smoke() -> None:
    model = _tiny_t5().eval()
    inject_pkm(model, layer_indices=[-2], n_keys=8, topk=4)
    with torch.no_grad():
        _gates(model)[0].fill_(0.5)
    ids = torch.randint(3, 259, (1, 6))
    out = model.generate(input_ids=ids, max_length=10, num_beams=1)
    assert out.shape[0] == 1


def test_pkm_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = {"layer_indices": [-1], "n_keys": 8, "topk": 4}
    model = _tiny_t5().eval()
    inject_pkm(model, **cfg)
    with torch.no_grad():
        _gates(model)[0].fill_(0.5)
        _gates(model)[0].add_(0.13)  # non-trivial gate value must survive
    ids = torch.randint(3, 259, (2, 7))
    labels = torch.randint(3, 259, (2, 5))
    with torch.no_grad():
        before = model(input_ids=ids, labels=labels).logits
    out_dir = tmp_path / "best"
    model.save_pretrained(str(out_dir))
    loaded = load_student_with_pkm(out_dir, cfg).eval()
    with torch.no_grad():
        after = loaded(input_ids=ids, labels=labels).logits
    assert torch.allclose(before, after, atol=1e-5)


def test_muon_split_routes_embedding_like_params() -> None:
    model = _tiny_t5()
    inject_pkm(model, layer_indices=[-1], n_keys=8, topk=4)
    muon_params, adamw_params = split_parameters(model.named_parameters())
    assert muon_params and adamw_params
    assert all(p.ndim == 2 for p in muon_params)
    id_adamw = {id(p) for p in adamw_params}
    names_adamw = [n for n, p in model.named_parameters() if id(p) in id_adamw]
    names_muon = [n for n, p in model.named_parameters() if id(p) not in id_adamw]
    assert any("embed_tokens" in n or n == "shared.weight" for n in names_adamw)
    assert any("memory.values" in n for n in names_adamw)
    assert any("decoder.block" in n for n in names_muon)


def test_muon_step_updates_params_and_saves_state() -> None:
    model = _tiny_t5()
    inject_pkm(model, layer_indices=[-1], n_keys=8, topk=4)
    muon_params, adamw_params = split_parameters(model.named_parameters())
    opt = Muon(muon_params, lr=0.01)
    opt.add_adamw_group(adamw_params, lr=1e-3)
    p0 = muon_params[0].detach().clone()
    q0 = adamw_params[0].detach().clone()
    loss = model(
        input_ids=torch.randint(3, 259, (2, 7)),
        labels=torch.randint(3, 259, (2, 5)),
    ).loss
    loss.backward()
    opt.step()
    assert not torch.equal(p0, muon_params[0])
    assert not torch.equal(q0, adamw_params[0])
    state = opt.state_dict()
    has_momentum = any(
        "momentum_buffer" in v for s in state["state"].values() for v in [s] if isinstance(s, dict)
    )
    assert has_momentum

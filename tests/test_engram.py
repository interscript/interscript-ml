"""Engram specs (TODO.impl/10): byte-n-gram addressing, zero-init
safety, int8 export roundtrip, and the ONNX gather-survival probe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

torch = pytest.importorskip("torch")

from gpu.engram import Engram, ngram_addresses  # noqa: E402


def test_addresses_are_deterministic_and_order_separated() -> None:
    seq = [117, 114, 110, 1]  # "rok" + EOS in the byte table
    a1 = ngram_addresses(seq)
    a2 = ngram_addresses(seq)
    assert a1 == a2
    # different orders address independently (seeded hashes)
    assert a1[3][0] != a1[3][1] != a1[3][2]
    # positions without a full n-gram address the null row
    assert a1[0] == [0, 0, 0]
    assert a1[1][1:] == [0, 0]


def test_addresses_track_the_byte_not_the_token() -> None:
    # the hash input is (id-3)%256: ids 259+3k wrap to the same byte
    a = ngram_addresses([103 + 256, 104 + 256])[1][0]
    b = ngram_addresses([103, 104])[1][0]
    assert a == b


def test_zero_init_leaves_the_backbone_untouched() -> None:
    eng = Engram(d_model=32, entries=1024, dim=8)
    ids = torch.tensor([[117, 114, 110, 1]])
    out = eng(ids)
    assert out.shape == (1, 4, 32)
    assert torch.all(out == 0), "fresh module must add nothing (the PKM rule)"


def test_lookup_varies_with_context() -> None:
    eng = Engram(d_model=8, entries=4096, dim=4)
    with torch.no_grad():
        eng.proj.weight.normal_()
    ctx = torch.tensor([[5, 6, 7, 8, 9]])
    shifted = torch.tensor([[5, 6, 7, 9, 8]])  # last two bytes swapped
    assert not torch.allclose(eng(ctx)[0, 4], eng(shifted)[0, 4])


def test_int8_export_roundtrip() -> None:
    eng = Engram(d_model=16, entries=512, dim=8)
    with torch.no_grad():
        eng.table.weight.normal_(std=0.05)
    state = eng.export_int8_state()
    deq = state["table_int8"].float() * state["table_scale"]
    err = (deq - eng.table.weight.detach()).abs().max().item()
    assert err < 0.05 / 127.0 * 4  # within a few int8 quanta of the scale
    assert state["table_int8"].dtype == torch.int8


def test_gather_survives_onnx_and_matches_torch() -> None:
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    eng = Engram(d_model=16, entries=512, dim=8)
    with torch.no_grad():
        eng.proj.weight.normal_(std=0.1)
    eng.requires_grad_(False)  # parameters as constants, not live tensors

    # the pure-graph path: addresses as a graph INPUT (runtimes compute
    # the same hash — ngram_addresses is the portable contract)
    addresses = torch.tensor(ngram_addresses([5, 6, 7, 8, 9]),
                             dtype=torch.long).unsqueeze(0)

    class Wrapper(torch.nn.Module):
        def forward(self, addrs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return eng.from_addresses(addrs)

    wrapper = Wrapper().eval()
    path = Path(__file__).parent / "fixtures" / "engram-probe.onnx"
    path.parent.mkdir(exist_ok=True)
    torch.onnx.export(wrapper, addresses, str(path), opset_version=14,
                      input_names=["addresses"], output_names=["memory"],
                      dynamo=False)
    sess = ort.InferenceSession(str(path))
    got = sess.run(None, {"addresses": addresses.numpy().astype(np.int64)})[0]
    want = wrapper(addresses).detach().numpy()
    np.testing.assert_allclose(got, want, atol=1e-4)
    path.unlink()  # probe artifact, not a fixture

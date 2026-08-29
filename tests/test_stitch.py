"""SVD width-stitch: project pretrained wide T5 weights into a narrow
config, preserving the top singular subspace (the microkimi
closed-form bridge)."""

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from src.gpu.modal_distill import svd_stitch_state  # noqa: E402


def _wide_narrow():
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    wide = T5ForConditionalGeneration(
        T5Config(vocab_size=384, d_model=64, d_ff=128, d_kv=16, num_layers=2,
                 num_decoder_layers=2, feed_forward_proj="relu",
                 decoder_start_token_id=0, eos_token_id=1)
    )
    narrow = T5ForConditionalGeneration(
        T5Config(vocab_size=384, d_model=32, d_ff=64, d_kv=16, num_layers=2,
                 num_decoder_layers=2, feed_forward_proj="relu",
                 decoder_start_token_id=0, eos_token_id=1)
    )
    return wide, narrow


def test_stitch_fills_every_parameter_with_target_shapes():
    wide, narrow = _wide_narrow()
    wide_sd = wide.state_dict()
    filled = svd_stitch_state(wide_sd, narrow.state_dict())
    for name, target in narrow.state_dict().items():
        assert name in filled, name
        assert filled[name].shape == target.shape, name
        assert torch.isfinite(filled[name]).all(), name


def test_stitch_preserves_top_subspace_action():
    wide, narrow = _wide_narrow()
    wide_sd = wide.state_dict()
    filled = svd_stitch_state(wide_sd, narrow.state_dict())
    w = wide_sd["encoder.block.0.layer.0.SelfAttention.q.weight"]  # (128, 64)
    w2 = filled["encoder.block.0.layer.0.SelfAttention.q.weight"]  # (128, 32)
    _, _, vh = torch.linalg.svd(w, full_matrices=False)
    V = vh[:32, :].T  # top-32 right singular vectors, orthonormal columns
    torch.manual_seed(1)
    z = torch.randn(32)
    # for inputs in the kept right subspace (x = V z), the stitched
    # layer reproduces the wide layer's action exactly
    assert torch.allclose(w2 @ z, w @ (V @ z), atol=1e-4)

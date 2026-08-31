import torch

from gpu.mtp import MTPHead, mtp_named


def _labels():
    t = torch.full((2, 6), -100)
    t[0, :5] = torch.tensor([10, 11, 12, 13, 14])
    t[1, :3] = torch.tensor([4, 5, 6])
    return t


def test_mtp_head_shapes_and_shift():
    torch.manual_seed(0)
    head = MTPHead(d_model=8, vocab=16, steps=3)
    hidden = torch.randn(2, 6, 8)
    loss = head.aux_loss(hidden, _labels())
    assert loss.ndim == 0 and float(loss) > 0


def test_mtp_shift_targets_not_inputs():
    torch.manual_seed(0)
    head = MTPHead(d_model=8, vocab=16, steps=1)
    head.heads[0].weight.data.zero_()
    head.heads[0].bias.data.zero_()

    labels = _labels()
    hidden = torch.randn(2, 6, 8)
    # uniform-logit head: CE is log(vocab) wherever any target is valid
    loss = head.aux_loss(hidden, labels)
    import math

    assert math.isclose(float(loss), math.log(16), rel_tol=1e-4)

    # a step-1 head must see labels[:, 1:] as targets: with all -100
    # beyond position 0 there is nothing to predict
    only_first = torch.full((1, 4), -100)
    only_first[0, 0] = 7.0
    assert float(head.aux_loss(torch.randn(1, 4, 8), only_first)) == 0.0


def test_mtp_learnable_and_named():
    head = MTPHead(d_model=8, vocab=16, steps=2)
    named = dict(mtp_named(head))
    assert set(named) == {
        "mtp.heads.0.weight", "mtp.heads.0.bias",
        "mtp.heads.1.weight", "mtp.heads.1.bias",
    }
    out = head.aux_loss(torch.randn(2, 6, 8), _labels())
    out.backward()
    assert head.heads[0].weight.grad is not None

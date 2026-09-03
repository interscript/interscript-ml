import pytest

torch = pytest.importorskip("torch")

from gpu.gkd import gkd_weight, reverse_kl, token_logprobs  # noqa: E402


class TestGkdWeight:
    def test_full_ratio_before_anneal(self):
        assert gkd_weight(0, 300, 0.3) == 0.3
        assert gkd_weight(199, 300, 0.3) == 0.3

    def test_linear_anneal_over_final_third(self):
        assert gkd_weight(200, 300, 0.3) == pytest.approx(0.3)  # anneal starts at full
        assert gkd_weight(250, 300, 0.3) == pytest.approx(0.15)
        assert gkd_weight(299, 300, 0.3) == pytest.approx(0.003)

    def test_zero_at_end_and_degenerate(self):
        assert gkd_weight(300, 300, 0.3) == 0.0
        assert gkd_weight(0, 0, 0.3) == 0.0


class TestTokenLogprobs:
    def test_shift_and_gather(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 7)
        targets = torch.randint(0, 7, (2, 5))
        lp = token_logprobs(logits, targets)
        assert lp.shape == (2, 4)
        ref = torch.log_softmax(logits[0, :-1].float(), -1)
        expect = ref.gather(-1, targets[0, 1:].unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(lp[0], expect)


class TestReverseKl:
    def test_identical_distributions_zero(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 6, 7)
        targets = torch.randint(0, 7, (2, 6))
        assert reverse_kl(logits, logits.clone(), targets).abs() < 1e-6

    def test_mask_restricts_positions(self):
        torch.manual_seed(0)
        s = torch.randn(1, 6, 7)
        t = torch.randn(1, 6, 7)
        targets = torch.randint(0, 7, (1, 6))
        mask = torch.zeros(1, 6, dtype=torch.bool)
        mask[:, 4:] = True  # only the continuation region (positions >= 4)
        masked = reverse_kl(s, t, targets, mask)
        full = reverse_kl(s, t, targets)
        assert not torch.allclose(masked, full)

    def test_gradient_flows_through_student_only(self):
        s = torch.randn(1, 4, 5, requires_grad=True)
        t = torch.randn(1, 4, 5, requires_grad=True)
        targets = torch.randint(0, 5, (1, 4))
        loss = reverse_kl(s, t, targets)
        loss.backward()
        assert s.grad is not None
        assert t.grad is None

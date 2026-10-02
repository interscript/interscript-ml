"""Tests for the RIDE displacement-arm math (TODO.sota-2026/05)."""

import sys
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpu.ride import displaced_targets, fit_ridge, masked_mse


class TestDisplacedTargets(unittest.TestCase):
    def test_lambda_zero_is_teacher(self):
        h_t = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
        h_b = torch.tensor([[3.0, 6.0], [0.0, -3.0]])
        self.assertTrue(torch.equal(displaced_targets(h_t, h_b, lam=0.0), h_t))

    def test_lambda_one_extrapolates(self):
        h_t = torch.tensor([[2.0, 4.0]])
        h_b = torch.tensor([[0.0, 0.0]])
        got = displaced_targets(h_t, h_b, lam=1.0)
        self.assertTrue(torch.equal(got, torch.tensor([[4.0, 8.0]])))

    def test_fractional_lambda(self):
        h_t = torch.tensor([[3.0]])
        h_b = torch.tensor([[1.0]])
        self.assertTrue(torch.equal(displaced_targets(h_t, h_b, lam=0.5), torch.tensor([[4.0]])))


class TestMaskedMse(unittest.TestCase):
    def test_exact_match_is_zero(self):
        h = torch.tensor([[1.0, 2.0]])
        self.assertEqual(masked_mse(h, h.clone(), torch.tensor([1])), 0.0)

    def test_masked_out_positions_ignored(self):
        pred = torch.tensor([[[0.0], [9.0]]])
        tgt = torch.tensor([[[1.0], [0.0]]])
        am = torch.tensor([[1, 0]])  # second position padded/masked
        self.assertAlmostEqual(
            masked_mse(pred, tgt, am).item(), 1.0)

    def test_averages_over_kept_positions_and_dims(self):
        pred = torch.tensor([[0.0, 2.0], [4.0, 4.0]])
        tgt = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        # diffs: 2,0,2,2 -> mse 12/4 = 3
        self.assertAlmostEqual(
            masked_mse(pred, tgt, torch.tensor([1, 1])).item(), 3.0)

    def test_zero_kept_is_zero(self):
        self.assertEqual(
            masked_mse(torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([0])).item(), 0.0)

    def test_batch_shape_with_padding(self):
        # (B=2, T=2, D=1); row 1 fully padded
        pred = torch.tensor([[[0.0], [2.0]], [[9.0], [9.0]]])
        tgt = torch.tensor([[[2.0], [2.0]], [[0.0], [0.0]]])
        am = torch.tensor([[1, 1], [0, 0]])
        # kept row 0: diffs 4, 0 -> sum 4 / (2 kept x D=1) = 2
        self.assertAlmostEqual(masked_mse(pred, tgt, am).item(), 2.0)


class TestFitRidge(unittest.TestCase):
    def test_identity_when_spaces_align(self):
        H = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        W = fit_ridge(H, H, alpha=1e-8)
        got = H @ W.T
        self.assertTrue(torch.allclose(got, H, atol=1e-4))

    def test_scales_output(self):
        H = torch.tensor([[2.0], [4.0]])
        S = torch.tensor([[4.0], [8.0]])  # student = 2x teacher
        W = fit_ridge(H, S, alpha=1e-8)
        self.assertAlmostEqual(W[0][0].item(), 2.0, places=4)

    def test_alpha_regularizes(self):
        H = torch.tensor([[1.0], [0.0]])
        W0 = fit_ridge(H, H, alpha=0.0)
        W1 = fit_ridge(H, H, alpha=10.0)
        self.assertLess(abs(W1[0][0].item()), abs(W0[0][0].item()) + 1e-9)

    def test_maps_width(self):
        # teacher d=1 -> student d=2
        H = torch.tensor([[1.0], [2.0], [3.0]])
        S = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        W = fit_ridge(H, S, alpha=1e-8)  # (2, 1)
        self.assertEqual(W.shape, (2, 1))
        got = H @ W.T
        self.assertTrue(torch.allclose(got, S, atol=1e-4))


if __name__ == "__main__":
    unittest.main()

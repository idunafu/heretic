import unittest

import torch
import torch.nn.functional as F

from heretic.config import RowNormalization
from heretic.directional import additive_factors


class DirectionalTests(unittest.TestCase):
    def setUp(self):
        rng = torch.Generator().manual_seed(12)
        self.W = torch.randn(8, 11, generator=rng)
        self.V = F.normalize(torch.randn(3, 8, generator=rng), dim=1)
        self.strengths = torch.tensor([0.6, 0.9, 0.4])

    def test_none_and_pre_match_dense_nonorthogonal_updates(self):
        for mode in (RowNormalization.NONE, RowNormalization.PRE):
            with self.subTest(mode=mode):
                W = (
                    self.W
                    if mode == RowNormalization.NONE
                    else F.normalize(self.W, dim=1)
                )
                expected = sum(
                    -s * torch.outer(v, v @ W) for v, s in zip(self.V, self.strengths)
                )
                if mode == RowNormalization.PRE:
                    expected *= self.W.norm(dim=1, keepdim=True)
                B, A = additive_factors(self.W, self.V, self.strengths, mode, 4, 9)
                self.assertEqual(A.shape, (4, 11))
                self.assertEqual(B.shape, (8, 4))
                torch.testing.assert_close(B @ A, expected)
                self.assertEqual(torch.linalg.matrix_rank(B @ A).item(), 3)

    def test_full_matches_dense_when_rank_is_sufficient_and_preserves_rng(self):
        row_norms = self.W.norm(dim=1, keepdim=True)
        W = F.normalize(self.W, dim=1)
        expected = W - (self.V.T * self.strengths) @ (self.V @ W)
        expected = F.normalize(expected, dim=1) * row_norms - self.W
        rng = torch.get_rng_state().clone()
        B, A = additive_factors(
            self.W, self.V, self.strengths, RowNormalization.FULL, 12, 9
        )
        torch.testing.assert_close(B @ A, expected, atol=2e-6, rtol=2e-5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(A.shape, (12, 11))

    def test_zero_rows_and_zero_strength_are_finite(self):
        self.W[0] = 0
        for mode in RowNormalization:
            B, A = additive_factors(self.W, self.V, torch.zeros(3), mode, 8, 9)
            torch.testing.assert_close(
                B @ A, torch.zeros_like(self.W), atol=2e-6, rtol=0
            )

    def test_rejects_rank_truncation(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            additive_factors(
                self.W, self.V, self.strengths, RowNormalization.NONE, 1, 0
            )


if __name__ == "__main__":
    unittest.main()

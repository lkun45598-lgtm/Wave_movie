from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
sys.path.insert(0, str(TRAINING_ROOT))

from utils.metrics import masked_psnr, masked_ssim  # noqa: E402


class MaskedMetricsTest(unittest.TestCase):
    def test_empty_mask_returns_finite_neutral_metrics(self) -> None:
        pred = torch.zeros(1, 2, 2, 1)
        target = torch.ones(1, 2, 2, 1)
        mask = torch.zeros(1, 2, 2, 1, dtype=torch.bool)

        psnr = masked_psnr(pred, target, shape=[2, 2], mask=mask)
        ssim = masked_ssim(pred, target, shape=[2, 2], mask=mask)

        self.assertTrue(torch.isfinite(psnr))
        self.assertTrue(torch.isfinite(ssim))
        self.assertEqual(float(psnr), 0.0)
        self.assertEqual(float(ssim), 0.0)


if __name__ == "__main__":
    unittest.main()

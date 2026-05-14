from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_cnn_detail_refiner_bohai_vz import (  # noqa: E402
    CNNDetailRefiner,
    Sample,
    apply_observed_constraint,
    case_from_base_name,
    compute_metrics,
    peak_preserving_loss,
    topk_amplitude_loss,
)


class CNNDetailRefinerPilotTest(unittest.TestCase):
    def test_apply_observed_constraint_preserves_sparse_points(self) -> None:
        pred = torch.zeros(1, 4, 4, 1)
        sparse = torch.full((1, 4, 4, 1), 7.0)
        mask = torch.zeros(1, 4, 4, 1)
        mask[:, ::2, ::2, :] = 1.0

        constrained = apply_observed_constraint(pred, sparse, mask)

        torch.testing.assert_close(constrained[:, ::2, ::2, :], sparse[:, ::2, ::2, :])
        torch.testing.assert_close(constrained[:, 1::2, 1::2, :], pred[:, 1::2, 1::2, :])

    def test_cnn_detail_refiner_shape_and_zero_initial_delta(self) -> None:
        model = CNNDetailRefiner(in_channels=5, features=8, num_blocks=2)
        x = torch.randn(2, 5, 16, 16)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 1, 16, 16))
        torch.testing.assert_close(y, torch.zeros_like(y))

    def test_case_from_base_name(self) -> None:
        self.assertEqual(case_from_base_name("S1_TTTZ_000050"), "S1_TTTZ")
        self.assertEqual(case_from_base_name("S1_WRRZ_000013"), "S1_WRRZ")

    def test_sample_is_hashable_metadata(self) -> None:
        sample = Sample(base_name="S1_WTVZ_000090", case_name="S1_WTVZ")
        self.assertEqual(sample.base_name, "S1_WTVZ_000090")
        self.assertEqual(sample.case_name, "S1_WTVZ")

    def test_compute_metrics_reports_missing_and_active_missing(self) -> None:
        target = torch.tensor([[0.0, 0.1], [0.2, 0.0]]).numpy()
        pred = torch.zeros(2, 2).numpy()
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()

        metrics = compute_metrics(pred, target, observed_mask=mask, active_threshold=0.05)

        self.assertAlmostEqual(metrics["rmse"], ((0.1**2 + 0.2**2) / 4) ** 0.5)
        self.assertAlmostEqual(metrics["missing_rmse"], ((0.1**2 + 0.2**2) / 2) ** 0.5)
        self.assertAlmostEqual(metrics["active_missing_rmse"], metrics["missing_rmse"])

    def test_peak_preserving_loss_penalizes_underestimated_target_peaks(self) -> None:
        target = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]]])
        pred_under = target * 0.5
        pred_exact = target.clone()
        missing_mask = torch.ones_like(target, dtype=torch.bool)

        under_loss = peak_preserving_loss(
            pred_under,
            target,
            missing_mask,
            peak_quantile=0.75,
        )
        exact_loss = peak_preserving_loss(
            pred_exact,
            target,
            missing_mask,
            peak_quantile=0.75,
        )

        self.assertGreater(float(under_loss), 0.0)
        self.assertAlmostEqual(float(exact_loss), 0.0, places=6)

    def test_topk_amplitude_loss_matches_target_peak_energy(self) -> None:
        target = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]]])
        pred_under = target * 0.5
        pred_exact = target.clone()
        missing_mask = torch.ones_like(target, dtype=torch.bool)

        under_loss = topk_amplitude_loss(
            pred_under,
            target,
            missing_mask,
            topk_fraction=0.5,
        )
        exact_loss = topk_amplitude_loss(
            pred_exact,
            target,
            missing_mask,
            topk_fraction=0.5,
        )

        self.assertGreater(float(under_loss), 0.0)
        self.assertAlmostEqual(float(exact_loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

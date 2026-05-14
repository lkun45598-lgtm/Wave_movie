from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
sys.path.insert(0, str(TRAINING_ROOT))

from trainers.resshift import (  # noqa: E402
    ResshiftTrainer,
    apply_sparse_known_constraint_nchw,
)


class ResShiftDataConsistencyTest(unittest.TestCase):
    def test_nchw_projection_preserves_observed_points_and_supports_soft_strength(self) -> None:
        pred = torch.full((1, 1, 2, 3), -9.0)
        x = torch.zeros(1, 2, 3, 3)
        x[..., 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x[..., 2] = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        hard = apply_sparse_known_constraint_nchw(
            pred,
            x,
            observed_value_channels=[0],
            mask_channel=2,
            strength=1.0,
        )
        expected_hard = torch.tensor(
            [[[[1.0, -9.0, 3.0], [-9.0, 5.0, -9.0]]]]
        )
        torch.testing.assert_close(hard, expected_hard)

        soft = apply_sparse_known_constraint_nchw(
            pred,
            x,
            observed_value_channels=[0],
            mask_channel=2,
            strength=0.25,
        )
        expected_soft = pred.clone()
        expected_soft[0, 0, 0, 0] = -9.0 * 0.75 + 1.0 * 0.25
        expected_soft[0, 0, 0, 2] = -9.0 * 0.75 + 3.0 * 0.25
        expected_soft[0, 0, 1, 1] = -9.0 * 0.75 + 5.0 * 0.25
        torch.testing.assert_close(soft, expected_soft)

    def test_trainer_builds_denoised_fn_for_res_shift_sampling_projection(self) -> None:
        trainer = ResshiftTrainer.__new__(ResshiftTrainer)
        trainer.resshift_data_consistency = True
        trainer.resshift_data_consistency_strength = 1.0
        trainer.sparse_known_value_channels = [0]
        trainer.sparse_known_mask_channel = 2

        x = torch.zeros(1, 2, 3, 3)
        x[..., 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x[..., 2] = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        pred_xstart = torch.full((1, 1, 2, 3), -9.0)

        denoised_fn = ResshiftTrainer._build_data_consistency_denoised_fn(
            trainer,
            x,
        )

        self.assertIsNotNone(denoised_fn)
        constrained = denoised_fn(pred_xstart)
        expected = torch.tensor([[[[1.0, -9.0, 3.0], [-9.0, 5.0, -9.0]]]])
        torch.testing.assert_close(constrained, expected)


if __name__ == "__main__":
    unittest.main()

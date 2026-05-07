from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
sys.path.insert(0, str(TRAINING_ROOT))

from trainers.base import BaseTrainer, build_hr_bicubic_baseline  # noqa: E402
from utils.loss import MaskedCompositeSRLoss  # noqa: E402
from utils.normalizer import GaussianNormalizer  # noqa: E402


class ResidualHighFrequencyTrainingTest(unittest.TestCase):
    def test_masked_composite_loss_combines_l1_and_peak_weight(self) -> None:
        loss_fn = MaskedCompositeSRLoss(
            {
                "l1": 1.0,
                "peak_l1": 2.0,
                "peak_quantile": 0.5,
                "peak_boost": 3.0,
            }
        )
        pred = torch.zeros(1, 2, 2, 1)
        target = torch.tensor([[[[0.0], [1.0]], [[2.0], [4.0]]]])
        loss = loss_fn(pred, target)

        base_l1 = target.abs().mean()
        weights = torch.tensor([[[[1.0], [1.0]], [[3.0], [3.0]]]])
        peak_l1 = (target.abs() * weights).mean()
        expected = base_l1 + 2.0 * peak_l1
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_masked_composite_loss_has_gradient_and_fft_terms(self) -> None:
        loss_fn = MaskedCompositeSRLoss({"gradient_l1": 1.0, "fft_hf_l1": 1.0})
        pred = torch.zeros(1, 8, 8, 1)
        target = torch.zeros(1, 8, 8, 1)
        target[:, ::2, ::2, :] = 1.0
        target[:, 1::2, 1::2, :] = 1.0

        self.assertGreater(float(loss_fn(pred, target)), 0.0)
        self.assertAlmostEqual(float(loss_fn(target, target)), 0.0, places=6)

    def test_build_hr_bicubic_baseline_decodes_lr_and_encodes_hr_space(self) -> None:
        lr_phys = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]]]])
        hr_phys = F.interpolate(
            lr_phys.permute(0, 3, 1, 2),
            size=(4, 4),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        norm_lr = GaussianNormalizer(lr_phys.reshape(1, -1, 1))
        norm_hr = GaussianNormalizer(hr_phys.reshape(1, -1, 1))
        lr_norm = norm_lr.encode(lr_phys.reshape(1, -1, 1)).reshape(lr_phys.shape)
        expected = norm_hr.encode(hr_phys.reshape(1, -1, 1)).reshape(hr_phys.shape)

        baseline = build_hr_bicubic_baseline(
            lr_norm,
            target_shape=hr_phys.shape,
            normalizer={"lr": norm_lr, "hr": norm_hr},
        )
        torch.testing.assert_close(baseline, expected)

    def test_load_ckpt_restores_model_weights(self) -> None:
        class DummyTrainer:
            pass

        trainer = DummyTrainer()
        trainer.model = torch.nn.Linear(2, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
        trainer.scaler = torch.amp.GradScaler(enabled=False)
        trainer.use_amp = False
        trainer.device = torch.device("cpu")
        trainer.dist = False
        trainer.dist_mode = None
        trainer.main_log = lambda message: None

        with torch.no_grad():
            trainer.model.weight.zero_()
            trainer.model.bias.zero_()

        saved_model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            saved_model.weight.fill_(2.0)
            saved_model.bias.fill_(3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model_epoch_4.pth"
            torch.save(
                {
                    "epoch": 4,
                    "model_state_dict": saved_model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "scaler_state_dict": None,
                },
                ckpt_path,
            )

            BaseTrainer.load_ckpt(trainer, ckpt_path)

        torch.testing.assert_close(trainer.model.weight, saved_model.weight)
        torch.testing.assert_close(trainer.model.bias, saved_model.bias)
        self.assertEqual(trainer.start_epoch, 5)


if __name__ == "__main__":
    unittest.main()

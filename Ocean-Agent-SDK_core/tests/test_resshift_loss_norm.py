from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
sys.path.insert(0, str(TRAINING_ROOT))

from models.resshift.models.gaussian_diffusion import (  # noqa: E402
    GaussianDiffusion,
    LossType,
    ModelMeanType,
)


class ConstantModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x, timesteps, **kwargs):
        return torch.full_like(x, self.value)


class ResShiftLossNormTest(unittest.TestCase):
    def test_mse_loss_norm_does_not_explode_for_near_zero_targets(self) -> None:
        diffusion = GaussianDiffusion(
            sqrt_etas=np.array([0.1, 0.2], dtype=np.float64),
            kappa=2.0,
            model_mean_type=ModelMeanType.START_X,
            loss_type=LossType.MSE,
            diffusion_loss_norm="mse",
            sf=1,
            scale_factor=1.0,
            normalize_input=False,
            latent_flag=False,
        )
        x_start = torch.full((2, 1, 4, 4), 1e-6)
        degraded = torch.zeros_like(x_start)
        timesteps = torch.tensor([0, 1])

        losses, _z_t, _pred_zstart = diffusion.training_losses(
            ConstantModel(1.0),
            x_start,
            degraded,
            timesteps,
            model_kwargs={},
            noise=torch.zeros_like(x_start),
        )

        self.assertGreater(float(losses["mse"]), 0.9)
        self.assertLess(float(losses["mse"]), 1.1)


if __name__ == "__main__":
    unittest.main()

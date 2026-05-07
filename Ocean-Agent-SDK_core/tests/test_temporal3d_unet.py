from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
sys.path.insert(0, str(TRAINING_ROOT))

from models import _model_dict  # noqa: E402
from trainers import _trainer_dict  # noqa: E402


class Temporal3DUNetTest(unittest.TestCase):
    def test_temporal3d_unet_is_registered_and_returns_center_frame_shape(self) -> None:
        model = _model_dict["Temporal3DUNet"](
            {
                "in_channels": 15,
                "out_channels": 1,
                "temporal_window": 5,
                "channels_per_frame": 3,
                "init_features": 4,
            }
        )

        x = torch.randn(2, 32, 32, 15)

        y = model(x)

        self.assertEqual(tuple(y.shape), (2, 32, 32, 1))

    def test_temporal3d_unet_rejects_mismatched_temporal_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_channels"):
            _model_dict["Temporal3DUNet"](
                {
                    "in_channels": 14,
                    "out_channels": 1,
                    "temporal_window": 5,
                    "channels_per_frame": 3,
                }
            )

    def test_temporal3d_unet_uses_default_trainer(self) -> None:
        self.assertIn("Temporal3DUNet", _trainer_dict)


if __name__ == "__main__":
    unittest.main()

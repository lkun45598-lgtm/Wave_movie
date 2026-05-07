from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
sys.path.insert(0, str(TRAINING_ROOT))

from datasets.OceanNPY import OceanNPYDataset  # noqa: E402
from trainers.base import (  # noqa: E402
    apply_sparse_known_constraint,
    build_hr_bicubic_baseline,
    build_sparse_loss_mask,
)


def _write_fake_split(root: Path, split: str) -> None:
    dyn_vars = ("Vx", "Vy")
    cases = {"A": 0, "B": 100}

    for space, shape, space_offset in (("lr", (2, 2), 0), ("hr", (4, 4), 1000)):
        for var_index, var_name in enumerate(dyn_vars):
            var_dir = root / split / space / var_name
            var_dir.mkdir(parents=True, exist_ok=True)
            for case_name, case_offset in cases.items():
                for frame in range(1, 4):
                    value = space_offset + case_offset + frame + var_index * 10
                    arr = np.full(shape, value, dtype=np.float32)
                    np.save(var_dir / f"{case_name}_{frame:06d}.npy", arr)


def _write_sparse_split(root: Path, split: str) -> None:
    sample_name = "CaseA_000001"
    hr_dir = root / split / "hr" / "Vz"
    lr_interp_dir = root / split / "lr" / "Vz_interp"
    lr_mask_dir = root / split / "lr" / "mask_observed"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_interp_dir.mkdir(parents=True, exist_ok=True)
    lr_mask_dir.mkdir(parents=True, exist_ok=True)

    hr = np.arange(16, dtype=np.float32).reshape(4, 4)
    interp = hr + 0.5
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[::2, ::2] = 1.0

    np.save(hr_dir / f"{sample_name}.npy", hr)
    np.save(lr_interp_dir / f"{sample_name}.npy", interp)
    np.save(lr_mask_dir / f"{sample_name}.npy", mask)


class TemporalOceanNPYTest(unittest.TestCase):
    def test_temporal_window_stacks_lr_frames_without_crossing_case_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "valid", "test"):
                _write_fake_split(root, split)

            dataset = OceanNPYDataset(
                {
                    "dataset_root": str(root),
                    "dyn_vars": ["Vx", "Vy"],
                    "normalize": False,
                    "sample_factor": 2,
                    "patch_size": None,
                    "temporal_window": 3,
                    "temporal_boundary": "clamp",
                }
            )

            x_first, y_first = dataset.train_dataset[0]
            self.assertEqual(tuple(x_first.shape), (2, 2, 6))
            self.assertEqual(tuple(y_first.shape), (4, 4, 2))
            torch.testing.assert_close(
                x_first[0, 0],
                torch.tensor([1.0, 11.0, 1.0, 11.0, 2.0, 12.0]),
            )
            torch.testing.assert_close(y_first[0, 0], torch.tensor([1001.0, 1011.0]))

            x_b_start, y_b_start = dataset.train_dataset[3]
            torch.testing.assert_close(
                x_b_start[0, 0],
                torch.tensor([101.0, 111.0, 101.0, 111.0, 102.0, 112.0]),
            )
            torch.testing.assert_close(y_b_start[0, 0], torch.tensor([1101.0, 1111.0]))

    def test_lr_and_hr_dynamic_variables_can_differ_for_sparse_mask_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "valid", "test"):
                _write_sparse_split(root, split)

            dataset = OceanNPYDataset(
                {
                    "dataset_root": str(root),
                    "dyn_vars": ["Vz"],
                    "hr_dyn_vars": ["Vz"],
                    "lr_dyn_vars": ["Vz_interp", "mask_observed"],
                    "normalize": False,
                    "sample_factor": 1,
                    "patch_size": None,
                }
            )

            x, y = dataset.train_dataset[0]
            self.assertEqual(tuple(x.shape), (4, 4, 2))
            self.assertEqual(tuple(y.shape), (4, 4, 1))
            torch.testing.assert_close(x[..., 0], y[..., 0] + 0.5)
            torch.testing.assert_close(
                x[..., 1],
                torch.tensor(
                    [
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            )
            self.assertEqual(dataset.dyn_vars, ["Vz"])
            self.assertEqual(dataset.lr_dyn_vars, ["Vz_interp", "mask_observed"])

    def test_static_mask_is_loaded_and_combined_with_nan_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "valid", "test"):
                _write_fake_split(root, split)

            hr_mask = np.ones((4, 4), dtype=np.float32)
            hr_mask[1, 2] = 0.0
            lr_mask = np.ones((2, 2), dtype=np.float32)
            lr_mask[0, 1] = 0.0
            (root / "static_variables" / "hr").mkdir(parents=True)
            (root / "static_variables" / "lr").mkdir(parents=True)
            np.save(root / "static_variables" / "hr" / "30_mask_rho.npy", hr_mask)
            np.save(root / "static_variables" / "lr" / "30_mask_rho.npy", lr_mask)

            dataset = OceanNPYDataset(
                {
                    "dataset_root": str(root),
                    "dyn_vars": ["Vx", "Vy"],
                    "normalize": False,
                    "sample_factor": 2,
                    "patch_size": None,
                }
            )

            self.assertEqual(tuple(dataset.mask_hr.shape), (1, 4, 4, 1))
            self.assertEqual(tuple(dataset.mask_lr.shape), (1, 2, 2, 1))
            self.assertFalse(bool(dataset.mask_hr[0, 1, 2, 0]))
            self.assertFalse(bool(dataset.mask_lr[0, 0, 1, 0]))
            self.assertEqual(int(dataset.mask_hr.sum().item()), 15)
            self.assertEqual(int(dataset.mask_lr.sum().item()), 3)

    def test_temporal_residual_baseline_uses_only_center_lr_channels(self) -> None:
        x = torch.zeros(1, 2, 2, 6)
        x[..., 0] = 100.0
        x[..., 1] = 200.0
        x[..., 2] = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        x[..., 3] = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        x[..., 4] = 300.0
        x[..., 5] = 400.0

        expected = F.interpolate(
            x[..., 2:4].permute(0, 3, 1, 2),
            size=(4, 4),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        baseline = build_hr_bicubic_baseline(
            x,
            target_shape=expected.shape,
            normalizer=None,
            source_channels=[2, 3],
        )
        torch.testing.assert_close(baseline, expected)

    def test_sparse_known_constraint_preserves_observed_values(self) -> None:
        pred = torch.full((1, 2, 3, 1), -9.0)
        x = torch.zeros(1, 2, 3, 3)
        x[..., 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x[..., 2] = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        constrained = apply_sparse_known_constraint(
            pred,
            x,
            observed_value_channels=[0],
            mask_channel=2,
        )

        expected = torch.tensor(
            [[[[1.0], [-9.0], [3.0]], [[-9.0], [5.0], [-9.0]]]]
        )
        torch.testing.assert_close(constrained, expected)

    def test_sparse_known_constraint_keeps_observed_precision_under_amp(self) -> None:
        pred = torch.full((1, 1, 1, 1), -9.0, dtype=torch.float16)
        x = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
        x[..., 0] = 4.9947075843811035
        x[..., 2] = 1.0

        constrained = apply_sparse_known_constraint(
            pred,
            x,
            observed_value_channels=[0],
            mask_channel=2,
        )

        self.assertEqual(constrained.dtype, torch.float32)
        torch.testing.assert_close(constrained[..., 0], x[..., 0], rtol=0.0, atol=0.0)

    def test_active_missing_loss_mask_uses_unobserved_active_target_pixels(self) -> None:
        base_mask = torch.ones(1, 2, 3, 1, dtype=torch.bool)
        base_mask[:, 1, 2, :] = False

        x = torch.zeros(1, 2, 3, 3)
        x[..., 2] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        y = torch.zeros(1, 2, 3, 1)
        y[..., 0] = torch.tensor([[0.2, 0.3, 0.001], [0.4, 0.5, 0.6]])

        mask = build_sparse_loss_mask(
            base_mask,
            x,
            y,
            mode="active_missing",
            observed_mask_channel=2,
            active_threshold=0.01,
        )

        expected = torch.tensor(
            [[[[False], [True], [False]], [[True], [False], [False]]]]
        )
        torch.testing.assert_close(mask, expected)

    def test_static_loss_mask_keeps_existing_behavior(self) -> None:
        base_mask = torch.tensor([[[[True], [False]], [[True], [True]]]])
        x = torch.zeros(1, 2, 2, 3)
        y = torch.ones(1, 2, 2, 1)

        mask = build_sparse_loss_mask(
            base_mask,
            x,
            y,
            mode="static",
            observed_mask_channel=2,
            active_threshold=0.01,
        )

        torch.testing.assert_close(mask, base_mask)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_PATH = ROOT / "scripts" / "preprocess_bohai_wave_xyz.py"

spec = importlib.util.spec_from_file_location("preprocess_bohai_wave_xyz", PREPROCESS_PATH)
preprocess = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = preprocess
spec.loader.exec_module(preprocess)


class BohaiPreprocessDownsampleTest(unittest.TestCase):
    def test_anti_alias_point_prefilters_before_sampling(self) -> None:
        field = np.zeros((8, 8), dtype=np.float32)
        field[2, 2] = 1.0

        point = preprocess.downsample_2d(field, 4, "point", 2)
        anti_alias = preprocess.downsample_2d(
            field,
            4,
            "anti_alias_point",
            2,
            anti_alias_sigma=1.0,
            anti_alias_truncate=3.0,
        )

        self.assertEqual(point.shape, (2, 2))
        self.assertEqual(anti_alias.shape, (2, 2))
        self.assertAlmostEqual(float(point[0, 0]), 1.0)
        self.assertGreater(float(anti_alias[0, 0]), 0.0)
        self.assertLess(float(anti_alias[0, 0]), 1.0)

    def test_sparse_mask_keeps_observed_points_and_interpolates_to_full_grid(self) -> None:
        yy, xx = np.meshgrid(
            np.arange(5, dtype=np.float32),
            np.arange(5, dtype=np.float32),
            indexing="ij",
        )
        field = yy * 10.0 + xx

        mask = preprocess.sparse_observation_mask_2d(field.shape, scale=2, offset=0)
        sparse = preprocess.sparse_zero_fill_2d(field, scale=2, offset=0)
        interp = preprocess.sparse_linear_interpolate_2d(field, scale=2, offset=0)

        self.assertEqual(mask.shape, field.shape)
        self.assertEqual(sparse.shape, field.shape)
        self.assertEqual(interp.shape, field.shape)
        self.assertEqual(int(mask.sum()), 9)
        self.assertEqual(float(sparse[0, 0]), float(field[0, 0]))
        self.assertEqual(float(sparse[1, 1]), 0.0)
        np.testing.assert_allclose(interp, field, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_bohai_total_sparsemask.py"

spec = importlib.util.spec_from_file_location("build_bohai_total_sparsemask", SCRIPT_PATH)
build_total = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = build_total
spec.loader.exec_module(build_total)


class BuildBohaiTotalSparseMaskTest(unittest.TestCase):
    def test_total_magnitude_combines_three_components_pointwise(self) -> None:
        vx = np.array([[3.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        vy = np.array([[4.0, 0.0], [2.0, 2.0]], dtype=np.float32)
        vz = np.array([[0.0, 5.0], [2.0, 1.0]], dtype=np.float32)

        total = build_total.total_magnitude(vx, vy, vz)

        expected = np.sqrt(vx * vx + vy * vy + vz * vz, dtype=np.float32)
        np.testing.assert_allclose(total, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(total.dtype, np.float32)

    def test_build_sparse_products_keep_only_observed_points_and_interpolate_full_grid(self) -> None:
        yy, xx = np.meshgrid(
            np.arange(5, dtype=np.float32),
            np.arange(5, dtype=np.float32),
            indexing="ij",
        )
        total = yy * 10.0 + xx

        sparse, interp, mask = build_total.build_sparse_products(
            total,
            scale=2,
            offset=0,
        )

        self.assertEqual(tuple(sparse.shape), (5, 5))
        self.assertEqual(tuple(interp.shape), (5, 5))
        self.assertEqual(tuple(mask.shape), (5, 5))
        self.assertEqual(int(mask.sum()), 9)
        self.assertEqual(float(sparse[0, 0]), float(total[0, 0]))
        self.assertEqual(float(sparse[1, 1]), 0.0)
        np.testing.assert_allclose(interp, total, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

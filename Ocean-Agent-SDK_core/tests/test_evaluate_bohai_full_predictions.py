from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_bohai_full_predictions import (  # noqa: E402
    COMPONENTS,
    compute_metrics,
    load_frame_triplet,
    parse_case_frame,
    prediction_base_name,
)


class EvaluateBohaiFullPredictionsTest(unittest.TestCase):
    def test_prediction_base_name_strips_sr_suffix(self) -> None:
        self.assertEqual(
            prediction_base_name(Path("S1_TTTZ_000001_sr.npy")),
            "S1_TTTZ_000001",
        )
        with self.assertRaises(ValueError):
            prediction_base_name(Path("S1_TTTZ_000001.npy"))

    def test_parse_case_frame_splits_last_numeric_token(self) -> None:
        self.assertEqual(parse_case_frame("S1_TTTZ_000042"), ("S1_TTTZ", 42))
        self.assertEqual(parse_case_frame("S1_UTU_000100"), ("S1_UTU", 100))

    def test_compute_metrics_matches_known_values(self) -> None:
        pred = np.array([1.0, 3.0], dtype=np.float32)
        target = np.array([1.0, 1.0], dtype=np.float32)
        result = compute_metrics(pred, target)

        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2.0))
        self.assertAlmostEqual(result["rfne"], math.sqrt(2.0))
        self.assertAlmostEqual(result["max_abs_error"], 2.0)
        self.assertAlmostEqual(result["acc"], 4.0 / (math.sqrt(10.0) * math.sqrt(2.0)))

    def test_load_frame_triplet_stacks_components_in_vx_vy_vz_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = "S1_TTTZ_000001"
            for split in ("hr", "lr"):
                for value, component in enumerate(COMPONENTS, start=1):
                    path = root / "test" / split / component
                    path.mkdir(parents=True, exist_ok=True)
                    np.save(path / f"{base}.npy", np.full((2, 3), value, dtype=np.float32))

            hr, lr = load_frame_triplet(root, base)

        self.assertEqual(hr.shape, (2, 3, 3))
        self.assertEqual(lr.shape, (2, 3, 3))
        np.testing.assert_array_equal(hr[..., 0], np.full((2, 3), 1, dtype=np.float32))
        np.testing.assert_array_equal(hr[..., 1], np.full((2, 3), 2, dtype=np.float32))
        np.testing.assert_array_equal(hr[..., 2], np.full((2, 3), 3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_visualize_bohai_vz.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_visualize_bohai_vz", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvaluateVisualizeSingleComponentTest(unittest.TestCase):
    def test_expected_bases_and_hr_loader_accept_component_name(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hr_dir = root / "test" / "hr" / "Total"
            hr_dir.mkdir(parents=True)
            np.save(hr_dir / "S1_TTTZ_000001.npy", np.ones((2, 3), dtype=np.float32))

            bases = module.resolve_expected_bases(
                root,
                max_frames=None,
                component="Total",
            )
            hr = module.load_hr(root, "S1_TTTZ_000001", component="Total")

        self.assertEqual(bases, ["S1_TTTZ_000001"])
        np.testing.assert_array_equal(hr, np.ones((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_bohai_sr_eval.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_bohai_sr_eval", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotBohaiSREvalTest(unittest.TestCase):
    def test_temporal_lr_uses_center_frame_channels(self) -> None:
        module = load_plot_module()
        lr = np.arange(2 * 4 * 5 * 15, dtype=np.float32).reshape(2, 4, 5, 15)

        selected = module.select_lr_display_channels(lr, target_channels=3)

        np.testing.assert_array_equal(selected, lr[..., 6:9])


if __name__ == "__main__":
    unittest.main()

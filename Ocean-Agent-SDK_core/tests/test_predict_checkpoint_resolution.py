from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "scripts" / "ocean-SR-training-masked"
MAIN_PATH = TRAINING_ROOT / "main.py"


def _load_main_module():
    sys.path.insert(0, str(TRAINING_ROOT))
    spec = importlib.util.spec_from_file_location("ocean_sr_training_main", MAIN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_uses_best_model_by_default_even_when_train_ckpt_path_is_set() -> None:
    main_module = _load_main_module()
    args = {
        "train": {
            "ckpt_path": "/tmp/old_resume_checkpoint.pth",
        }
    }

    ckpt_path = main_module.resolve_predict_checkpoint_path(args, "/tmp/current_run")

    assert ckpt_path == "/tmp/current_run/best_model.pth"


def test_predict_can_use_explicit_predict_ckpt_path() -> None:
    main_module = _load_main_module()
    args = {
        "train": {
            "ckpt_path": "/tmp/old_resume_checkpoint.pth",
            "predict_ckpt_path": "/tmp/explicit_predict_checkpoint.pth",
        }
    }

    ckpt_path = main_module.resolve_predict_checkpoint_path(args, "/tmp/current_run")

    assert ckpt_path == "/tmp/explicit_predict_checkpoint.pth"

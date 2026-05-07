from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "scripts" / "ocean-SR-training-masked"
SOURCE_CONFIG_DIR = ROOT / "configs" / "bohai_xyz_4x"
DERIVED_CONFIG_DIR = SOURCE_CONFIG_DIR / "gpu5_light"
OUTPUT_ROOT = Path("/data1/user/lz/wave_movie/testouts")
SUMMARY_PATH = OUTPUT_ROOT / "light_gpu5_queue_summary.json"
PYTHON = Path("/home/lz/miniconda3/envs/pytorch/bin/python")


LIGHT_MODELS = [
    ("FNO2d", "bohai_xyz_4x_fno2d.yaml"),
    ("UNet2d", "bohai_xyz_4x_unet2d.yaml"),
    ("M2NO2d", "bohai_xyz_4x_m2no2d.yaml"),
    ("Galerkin_Transformer", "bohai_xyz_4x_galerkin_transformer.yaml"),
    ("MWT2d", "bohai_xyz_4x_mwt2d.yaml"),
    ("SRNO", "bohai_xyz_4x_srno.yaml"),
    ("Swin_Transformer", "bohai_xyz_4x_swin_transformer.yaml"),
    ("EDSR", "bohai_xyz_4x_edsr.yaml"),
    ("HiNOTE", "bohai_xyz_4x_hinote.yaml"),
    ("SwinIR", "bohai_xyz_4x_swinir.yaml"),
]

def multiply_positive_int(value: int | None, factor: int) -> int:
    if value is None:
        return factor
    return max(int(value) * factor, 1)


def build_config(model_name: str, source_name: str) -> tuple[Path, Path]:
    source_path = SOURCE_CONFIG_DIR / source_name
    config = yaml.safe_load(source_path.read_text())

    data = config.setdefault("data", {})
    data["normalizer_type"] = "PGN"
    data["train_batchsize"] = multiply_positive_int(data.get("train_batchsize"), 3)
    data["eval_batchsize"] = multiply_positive_int(data.get("eval_batchsize"), 3)

    train = config.setdefault("train", {})
    train["cuda"] = True
    train["device"] = 0
    train["distribute"] = False
    train["distribute_mode"] = "DDP"
    train["device_ids"] = [0]
    train["eval_freq"] = 50
    train["saving_ckpt"] = True
    train["ckpt_freq"] = 50

    output_dir = OUTPUT_ROOT / model_name
    log = config.setdefault("log", {})
    log["log"] = True
    log["log_dir"] = str(output_dir)
    log["wandb"] = False

    DERIVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DERIVED_CONFIG_DIR / f"bohai_xyz_4x_{model_name.lower()}_gpu5.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return output_path, output_dir


def run_model(model_name: str, config_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
    launch_log = output_dir / "launch.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "5"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = str(output_dir / "mplconfig")

    cmd = [
        str(PYTHON),
        "-u",
        "main.py",
        "--mode",
        "train",
        "--config",
        str(config_path),
    ]

    start = time.time()
    print(f"[{time.strftime('%F %T')}] START {model_name} config={config_path}", flush=True)
    with launch_log.open("w") as log_file:
        process = subprocess.run(
            cmd,
            cwd=TRAIN_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    duration = time.time() - start
    status = {
        "model": model_name,
        "config": str(config_path),
        "log_dir": str(output_dir),
        "returncode": process.returncode,
        "duration_seconds": duration,
        "finished_at": time.strftime("%FT%T"),
    }
    (output_dir / "queue_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(
        f"[{time.strftime('%F %T')}] END {model_name} "
        f"returncode={process.returncode} duration={duration:.1f}s",
        flush=True,
    )
    return process.returncode


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = []
    for model_name, source_name in LIGHT_MODELS:
        config_path, output_dir = build_config(model_name, source_name)
        configs.append((model_name, config_path, output_dir))
        print(f"CONFIG {model_name}: {config_path} -> {output_dir}", flush=True)

    results = {}
    for model_name, config_path, output_dir in configs:
        results[model_name] = run_model(model_name, config_path, output_dir)

    SUMMARY_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    failed = {name: code for name, code in results.items() if code != 0}
    if failed:
        print(f"FAILED: {failed}", flush=True)
        return 1
    print("ALL_LIGHT_MODELS_FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

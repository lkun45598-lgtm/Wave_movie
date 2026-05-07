#!/usr/bin/env python3
"""
Generate Bohai Sea XYZ 4x OceanNPY training configs for every SR model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml


MODEL_NAMES = [
    "FNO2d",
    "UNet2d",
    "M2NO2d",
    "Galerkin_Transformer",
    "MWT2d",
    "SRNO",
    "Swin_Transformer",
    "EDSR",
    "HiNOTE",
    "SwinIR",
    "DDPM",
    "SR3",
    "MG-DDPM",
    "Resshift",
    "ReMiG",
]


MODEL_SLUGS = {
    "FNO2d": "fno2d",
    "UNet2d": "unet2d",
    "M2NO2d": "m2no2d",
    "Galerkin_Transformer": "galerkin_transformer",
    "MWT2d": "mwt2d",
    "SRNO": "srno",
    "Swin_Transformer": "swin_transformer",
    "EDSR": "edsr",
    "HiNOTE": "hinote",
    "SwinIR": "swinir",
    "DDPM": "ddpm",
    "SR3": "sr3",
    "MG-DDPM": "mg_ddpm",
    "Resshift": "resshift",
    "ReMiG": "remig",
}


PATCH_64_MODELS = {
    "M2NO2d",
    "UNet2d",
    "Galerkin_Transformer",
    "SRNO",
    "Swin_Transformer",
    "EDSR",
    "SwinIR",
    "DDPM",
    "SR3",
    "MG-DDPM",
    "Resshift",
    "ReMiG",
}


DIFFUSION_MODELS = {"DDPM", "SR3", "MG-DDPM", "Resshift", "ReMiG"}


PER_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "FNO2d": {"batch_size": 4, "eval_batch_size": 4, "patch_size": None},
    "M2NO2d": {"batch_size": 2, "eval_batch_size": 2, "patch_size": 64},
    "MWT2d": {"batch_size": 2, "eval_batch_size": 2, "patch_size": None},
    "HiNOTE": {"batch_size": 2, "eval_batch_size": 2, "patch_size": None},
    "UNet2d": {"batch_size": 8, "eval_batch_size": 4, "patch_size": 64},
    "Galerkin_Transformer": {"batch_size": 4, "eval_batch_size": 4, "patch_size": 64},
    "SRNO": {"batch_size": 4, "eval_batch_size": 4, "patch_size": 64},
    "Swin_Transformer": {"batch_size": 4, "eval_batch_size": 4, "patch_size": 64},
    "EDSR": {"batch_size": 8, "eval_batch_size": 4, "patch_size": 64},
    "SwinIR": {"batch_size": 4, "eval_batch_size": 4, "patch_size": 64},
    "DDPM": {"batch_size": 1, "eval_batch_size": 1, "patch_size": 64, "lr": 1e-4},
    "SR3": {"batch_size": 1, "eval_batch_size": 1, "patch_size": 64, "lr": 1e-4},
    "MG-DDPM": {"batch_size": 1, "eval_batch_size": 1, "patch_size": 64, "lr": 1e-4},
    "Resshift": {"batch_size": 1, "eval_batch_size": 1, "patch_size": 64, "lr": 5e-5},
    "ReMiG": {"batch_size": 1, "eval_batch_size": 1, "patch_size": 64, "lr": 1e-4},
}


def load_generate_config(repo_root: Path):
    module_path = (
        repo_root
        / "scripts"
        / "ocean-SR-training-masked"
        / "generate_config.py"
    )
    spec = importlib.util.spec_from_file_location("bohai_generate_config", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generate_config.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_config


def build_params(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_name": model_name,
        "dataset_root": str(args.dataset_root),
        "dyn_vars": ["Vx", "Vy", "Vz"],
        "scale": 4,
        "log_dir": str(args.log_root / f"{MODEL_SLUGS[model_name]}_xyz_4x"),
        "epochs": args.epochs,
        "batch_size": 4,
        "eval_batch_size": 4,
        "normalize": True,
        "normalizer_type": "PGN",
        "gradient_checkpointing": True,
        "saving_ckpt": False,
        "device": args.device,
        "device_ids": [args.device],
        "distribute": False,
        "wandb": False,
        "seed": 42,
    }
    params.update(PER_MODEL_OVERRIDES.get(model_name, {}))
    return params


def patch_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    config = deepcopy(config)
    model = config.get("model", {})
    data = config.get("data", {})

    if model_name in PATCH_64_MODELS:
        data["patch_size"] = 64

    if model_name == "SwinIR":
        model["img_size"] = 16

    if model_name in DIFFUSION_MODELS:
        data["eval_batchsize"] = 1
        config["train"]["eval_freq"] = max(int(config["train"].get("eval_freq", 5)), 5)

    if model_name == "MG-DDPM":
        model["channels"] = 3
        model["in_channels"] = 6
        model["out_channels"] = 3
        model["resolutions"] = [64, 64]

    if model_name == "Resshift":
        config.setdefault("model", {})["name"] = "Resshift"

    config["model"] = model
    config["data"] = data
    return config


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(data, file_obj, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all Bohai XYZ 4x model configs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/Bohai_Sea/process_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data1/user/lz/wave_movie/Ocean-Agent-SDK_core/configs/bohai_xyz_4x"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/data/Bohai_Sea/process_data/training_outputs"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    generate_config = load_generate_config(repo_root)

    generated = []
    for model_name in MODEL_NAMES:
        params = build_params(args, model_name)
        config = generate_config(params)
        config = patch_config(config, model_name)
        config_path = args.output_dir / f"bohai_xyz_4x_{MODEL_SLUGS[model_name]}.yaml"
        write_yaml(config_path, config)
        generated.append(
            {
                "model": model_name,
                "config": str(config_path),
                "patch_size": config["data"].get("patch_size"),
                "batch_size": config["data"].get("train_batchsize"),
                "eval_batch_size": config["data"].get("eval_batchsize"),
                "log_dir": config["log"].get("log_dir"),
            }
        )

    manifest = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "dynamic_variables": ["Vx", "Vy", "Vz"],
        "scale": 4,
        "normalizer_type": "PGN",
        "epochs": args.epochs,
        "models": generated,
        "notes": [
            "All configs target OceanNPY data under /data/Bohai_Sea/process_data.",
            "Patch-sensitive models use 64x64 HR patches and 16x16 LR patches.",
            "FNO2d, MWT2d, and HiNOTE keep full-image training by default.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

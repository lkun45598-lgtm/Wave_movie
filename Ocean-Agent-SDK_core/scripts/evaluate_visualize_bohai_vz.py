from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


EPS = 1e-12
COMPONENT = "Vz"
DEFAULT_DATASET_ROOT = Path("/data/Bohai_Sea/process_data")
DEFAULT_OUTPUT_ROOT = Path("/data1/user/lz/wave_movie/testouts/vz_single_eval")
DEFAULT_MODELS = (
    (
        "EDSR_PGN_Vz",
        Path(
            "/data1/user/lz/wave_movie/testouts/EDSR_PGN_Vz/"
            "full_predict_run/predictions"
        ),
    ),
    (
        "FNO2d_PGN_Vz",
        Path(
            "/data1/user/lz/wave_movie/testouts/FNO2d_PGN_Vz/"
            "full_predict_run/predictions"
        ),
    ),
    (
        "Resshift_PGN_Vz",
        Path(
            "/data1/user/lz/wave_movie/testouts/Resshift_PGN_Vz/"
            "full_predict_run/predictions"
        ),
    ),
)
METRIC_FIELDS = (
    "rmse",
    "mae",
    "rfne",
    "acc",
    "bias",
    "p99_abs_error",
    "max_abs_error",
    "psnr",
    "ssim",
    "peak_abs_pred",
    "peak_abs_target",
    "peak_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and visualize single-component Bohai Vz 4x super-resolution "
            "predictions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Processed dataset root containing test/hr/<component> and test/lr variables.",
    )
    parser.add_argument(
        "--component",
        default=COMPONENT,
        help="HR target component name, for example Vz or Total.",
    )
    parser.add_argument(
        "--lr-component",
        default=COMPONENT,
        help=(
            "LR/input component to compare against HR. Use Vz_interp for "
            "sparse-mask full-grid interpolation inputs."
        ),
    )
    parser.add_argument(
        "--baseline-label",
        default="LR bicubic",
        help="Display label for the baseline input in figures and CSV summaries.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for combined summaries and comparison figures.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PRED_DIR",
        help=(
            "Model prediction directory. Can be repeated. If omitted, known "
            "EDSR/FNO/ResShift Vz output directories are used."
        ),
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Evaluate prediction directories even if they contain fewer files than test/hr/Vz.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional cap for quick debugging. Omit for full test evaluation.",
    )
    parser.add_argument(
        "--case",
        default="S1_TTTZ",
        help="Case name used for every-N-frame diagnostic PNGs.",
    )
    parser.add_argument("--frame-start", type=int, default=10)
    parser.add_argument("--frame-end", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-spectral", action="store_true")
    return parser.parse_args()


def safe_model_key(model_name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", model_name).strip("_")
    if not key:
        raise ValueError("model name cannot be empty")
    return key


def short_model_label(model_name: str) -> str:
    normalized = model_name.lower()
    if normalized.startswith("edsr"):
        return "EDSR SR"
    if normalized.startswith("fno2d"):
        return "FNO2d SR"
    if normalized.startswith("resshift") or normalized.startswith("res_shift"):
        return "ResShift SR"
    if normalized == "sr":
        return "SR"

    cleaned = re.sub(r"[_-]+", " ", model_name).strip()
    cleaned = re.sub(r"\b(PGN|Vz|POINT)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return f"{cleaned} SR" if cleaned else "SR"


def model_display_labels(summaries: list[dict[str, object]]) -> dict[str, str]:
    names = [str(summary["model_name"]) for summary in summaries]
    labels = ["SR" if len(names) == 1 else short_model_label(name) for name in names]
    counts = Counter(labels)
    return {
        name: label if counts[label] == 1 else f"{label} ({name})"
        for name, label in zip(names, labels)
    }


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--model must be NAME=PRED_DIR, got: {value}")
    name, pred_dir = value.split("=", 1)
    name = name.strip()
    pred_dir = pred_dir.strip()
    if not name or not pred_dir:
        raise ValueError(f"--model must be NAME=PRED_DIR, got: {value}")
    return name, Path(pred_dir)


def resolve_models(model_args: list[str]) -> list[tuple[str, Path]]:
    if not model_args:
        return list(DEFAULT_MODELS)
    return [parse_model_arg(value) for value in model_args]


def base_from_prediction(path: Path) -> str:
    if path.suffix != ".npy" or not path.stem.endswith("_sr"):
        raise ValueError(f"Prediction file must match *_sr.npy: {path}")
    return path.stem[: -len("_sr")]


def parse_case_frame(base_name: str) -> tuple[str, int]:
    case, sep, frame_text = base_name.rpartition("_")
    if not sep or not frame_text.isdigit():
        raise ValueError(f"Cannot parse case/frame from {base_name}")
    return case, int(frame_text)


def sort_key_from_base(base_name: str) -> tuple[str, int, str]:
    case, frame = parse_case_frame(base_name)
    return case, frame, base_name


def sort_key_from_prediction(path: Path) -> tuple[str, int, str]:
    return sort_key_from_base(base_from_prediction(path))


def component_key(component: str) -> str:
    return safe_model_key(component).lower()


def resolve_expected_bases(
    dataset_root: Path,
    max_frames: int | None,
    component: str = COMPONENT,
) -> list[str]:
    hr_dir = dataset_root / "test" / "hr" / component
    if not hr_dir.exists():
        raise FileNotFoundError(f"Missing HR directory: {hr_dir}")
    bases = sorted((path.stem for path in hr_dir.glob("*.npy")), key=sort_key_from_base)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be >= 1")
        bases = bases[:max_frames]
    if not bases:
        raise FileNotFoundError(f"No HR frames found in {hr_dir}")
    return bases


def resolve_prediction_map(prediction_dir: Path) -> dict[str, Path]:
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Missing prediction directory: {prediction_dir}")
    paths = sorted(prediction_dir.glob("*_sr.npy"), key=sort_key_from_prediction)
    return {base_from_prediction(path): path for path in paths}


def load_field(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
        arr = arr[0, ..., 0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected a 2D single-channel field, got {arr.shape}: {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {path}")
    return arr.astype(np.float32, copy=False)


def load_hr(
    dataset_root: Path,
    base_name: str,
    component: str = COMPONENT,
) -> np.ndarray:
    return load_field(dataset_root / "test" / "hr" / component / f"{base_name}.npy")


def load_lr(dataset_root: Path, base_name: str, component: str = COMPONENT) -> np.ndarray:
    return load_field(dataset_root / "test" / "lr" / component / f"{base_name}.npy")


def upsample_lr(lr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(tensor, size=target_hw, mode="bicubic", align_corners=False)
    return up.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def baseline_from_lr(lr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if lr.shape == target_hw:
        return lr.astype(np.float32, copy=False)
    return upsample_lr(lr, target_hw)


def psnr_from_rmse(rmse: float, data_range: float) -> float:
    if rmse <= EPS or data_range <= EPS:
        return float("nan")
    return float(20.0 * math.log10(data_range / rmse))


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred64 = np.asarray(pred, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    if pred64.shape != target64.shape:
        raise ValueError(f"Metric shape mismatch: {pred.shape} vs {target.shape}")

    diff = pred64 - target64
    abs_diff = np.abs(diff)
    flat_pred = pred64.reshape(-1)
    flat_target = target64.reshape(-1)
    flat_diff = diff.reshape(-1)
    sum_sq_diff = float(np.dot(flat_diff, flat_diff))
    sum_sq_pred = float(np.dot(flat_pred, flat_pred))
    sum_sq_target = float(np.dot(flat_target, flat_target))
    dot = float(np.dot(flat_pred, flat_target))
    count = int(flat_diff.size)
    rmse = float(np.sqrt(sum_sq_diff / max(count, 1)))
    data_range = float(np.max(target64) - np.min(target64))
    if data_range <= EPS:
        ssim_value = float("nan")
    else:
        ssim_value = float(structural_similarity(target64, pred64, data_range=data_range))

    peak_abs_pred = float(np.max(np.abs(pred64))) if count else 0.0
    peak_abs_target = float(np.max(np.abs(target64))) if count else 0.0
    return {
        "rmse": rmse,
        "mae": float(np.mean(abs_diff)) if count else 0.0,
        "rfne": float(np.sqrt(sum_sq_diff) / (np.sqrt(sum_sq_target) + EPS)),
        "acc": float(dot / (np.sqrt(sum_sq_pred) * np.sqrt(sum_sq_target) + EPS)),
        "bias": float(np.mean(diff)) if count else 0.0,
        "p99_abs_error": float(np.percentile(abs_diff, 99.0)) if count else 0.0,
        "max_abs_error": float(np.max(abs_diff)) if count else 0.0,
        "psnr": psnr_from_rmse(rmse, data_range),
        "ssim": ssim_value,
        "peak_abs_pred": peak_abs_pred,
        "peak_abs_target": peak_abs_target,
        "peak_ratio": float(peak_abs_pred / (peak_abs_target + EPS)),
        "count": count,
    }


@dataclass
class MetricAccumulator:
    sum_sq_diff: float = 0.0
    sum_abs_diff: float = 0.0
    sum_diff: float = 0.0
    sum_sq_target: float = 0.0
    sum_sq_pred: float = 0.0
    dot: float = 0.0
    max_abs_error: float = 0.0
    count: int = 0
    target_min: float = float("inf")
    target_max: float = float("-inf")
    frame_p99: list[float] = field(default_factory=list)
    frame_psnr: list[float] = field(default_factory=list)
    frame_ssim: list[float] = field(default_factory=list)
    frame_peak_ratio: list[float] = field(default_factory=list)

    def update(self, pred: np.ndarray, target: np.ndarray, frame_metrics: dict[str, float | int]) -> None:
        pred64 = np.asarray(pred, dtype=np.float64).reshape(-1)
        target64 = np.asarray(target, dtype=np.float64).reshape(-1)
        diff = pred64 - target64
        abs_diff = np.abs(diff)
        self.sum_sq_diff += float(np.dot(diff, diff))
        self.sum_abs_diff += float(np.sum(abs_diff))
        self.sum_diff += float(np.sum(diff))
        self.sum_sq_target += float(np.dot(target64, target64))
        self.sum_sq_pred += float(np.dot(pred64, pred64))
        self.dot += float(np.dot(pred64, target64))
        self.max_abs_error = max(self.max_abs_error, float(np.max(abs_diff)))
        self.count += int(diff.size)
        self.target_min = min(self.target_min, float(np.min(target64)))
        self.target_max = max(self.target_max, float(np.max(target64)))
        self.frame_p99.append(float(frame_metrics["p99_abs_error"]))
        if math.isfinite(float(frame_metrics["psnr"])):
            self.frame_psnr.append(float(frame_metrics["psnr"]))
        if math.isfinite(float(frame_metrics["ssim"])):
            self.frame_ssim.append(float(frame_metrics["ssim"]))
        self.frame_peak_ratio.append(float(frame_metrics["peak_ratio"]))

    def to_metrics(self) -> dict[str, float | int]:
        rmse = float(np.sqrt(self.sum_sq_diff / max(self.count, 1)))
        data_range = self.target_max - self.target_min
        return {
            "rmse": rmse,
            "mae": float(self.sum_abs_diff / max(self.count, 1)),
            "rfne": float(np.sqrt(self.sum_sq_diff) / (np.sqrt(self.sum_sq_target) + EPS)),
            "acc": float(
                self.dot / (np.sqrt(self.sum_sq_pred) * np.sqrt(self.sum_sq_target) + EPS)
            ),
            "bias": float(self.sum_diff / max(self.count, 1)),
            "p99_abs_error_mean": float(np.mean(self.frame_p99)) if self.frame_p99 else 0.0,
            "p99_abs_error_max": float(np.max(self.frame_p99)) if self.frame_p99 else 0.0,
            "max_abs_error": float(self.max_abs_error),
            "psnr_global": psnr_from_rmse(rmse, data_range),
            "psnr_mean": float(np.mean(self.frame_psnr)) if self.frame_psnr else float("nan"),
            "ssim_mean": float(np.mean(self.frame_ssim)) if self.frame_ssim else float("nan"),
            "peak_ratio_mean": float(np.mean(self.frame_peak_ratio))
            if self.frame_peak_ratio
            else float("nan"),
            "count": int(self.count),
        }


def relative_metrics(
    model_metrics: dict[str, float | int],
    bicubic_metrics: dict[str, float | int],
) -> dict[str, float]:
    return {
        "rmse_reduction_percent": float(
            100.0
            * (float(bicubic_metrics["rmse"]) - float(model_metrics["rmse"]))
            / (float(bicubic_metrics["rmse"]) + EPS)
        ),
        "rfne_reduction_percent": float(
            100.0
            * (float(bicubic_metrics["rfne"]) - float(model_metrics["rfne"]))
            / (float(bicubic_metrics["rfne"]) + EPS)
        ),
        "mae_reduction_percent": float(
            100.0
            * (float(bicubic_metrics["mae"]) - float(model_metrics["mae"]))
            / (float(bicubic_metrics["mae"]) + EPS)
        ),
        "acc_delta": float(float(model_metrics["acc"]) - float(bicubic_metrics["acc"])),
    }


def evaluate_model(
    model_name: str,
    prediction_map: dict[str, Path],
    bases: list[str],
    dataset_root: Path,
    component: str,
    lr_component: str,
    baseline_label: str,
) -> dict[str, object]:
    model_acc = MetricAccumulator()
    bicubic_acc = MetricAccumulator()
    by_case: dict[str, dict[str, object]] = {}
    per_frame: list[dict[str, object]] = []

    key = safe_model_key(model_name)
    for index, base_name in enumerate(bases, start=1):
        pred_path = prediction_map[base_name]
        case, frame = parse_case_frame(base_name)
        hr = load_hr(dataset_root, base_name, component)
        lr = load_lr(dataset_root, base_name, lr_component)
        pred = load_field(pred_path)
        bicubic = baseline_from_lr(lr, hr.shape)
        if pred.shape != hr.shape:
            raise ValueError(f"{model_name} shape mismatch for {base_name}: {pred.shape} vs {hr.shape}")

        model_frame = compute_metrics(pred, hr)
        bicubic_frame = compute_metrics(bicubic, hr)
        per_frame.append(
            {
                "filename": base_name,
                "case": case,
                "frame": frame,
                key: model_frame,
                "bicubic": bicubic_frame,
            }
        )

        model_acc.update(pred, hr, model_frame)
        bicubic_acc.update(bicubic, hr, bicubic_frame)

        case_row = by_case.setdefault(
            case,
            {
                "frames": 0,
                "_model": MetricAccumulator(),
                "_bicubic": MetricAccumulator(),
            },
        )
        case_row["frames"] = int(case_row["frames"]) + 1
        case_row["_model"].update(pred, hr, model_frame)  # type: ignore[union-attr]
        case_row["_bicubic"].update(bicubic, hr, bicubic_frame)  # type: ignore[union-attr]

        if index == 1 or index % 100 == 0 or index == len(bases):
            print(f"[{model_name}] evaluated {index}/{len(bases)}: {base_name}")

    model_metrics = model_acc.to_metrics()
    bicubic_metrics = bicubic_acc.to_metrics()
    by_case_out: dict[str, dict[str, object]] = {}
    for case in sorted(by_case):
        row = by_case[case]
        by_case_out[case] = {
            "frames": int(row["frames"]),
            key: row["_model"].to_metrics(),  # type: ignore[union-attr]
            "bicubic": row["_bicubic"].to_metrics(),  # type: ignore[union-attr]
        }

    return {
        "model_name": model_name,
        "model_key": key,
        "component": component,
        "lr_component": lr_component,
        "baseline_label": baseline_label,
        "num_predictions": len(bases),
        "global": {
            key: model_metrics,
            "bicubic": bicubic_metrics,
            "relative": relative_metrics(model_metrics, bicubic_metrics),
        },
        "by_case": by_case_out,
        "per_frame": per_frame,
    }


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def write_per_frame_csv(summary: dict[str, object], path: Path) -> None:
    key = str(summary["model_key"])
    baseline_label = str(summary.get("baseline_label", "baseline"))
    rows = summary["per_frame"]  # type: ignore[assignment]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "case", "frame", "source", *METRIC_FIELDS])
        for row in rows:  # type: ignore[union-attr]
            for source, source_label in (("bicubic", baseline_label), (key, key)):
                metrics = row[source]  # type: ignore[index]
                writer.writerow(
                    [
                        row["filename"],  # type: ignore[index]
                        row["case"],  # type: ignore[index]
                        row["frame"],  # type: ignore[index]
                        source_label,
                        *[metrics[field] for field in METRIC_FIELDS],  # type: ignore[index]
                    ]
                )


def load_coordinate_extent(dataset_root: Path) -> tuple[list[float], str, str]:
    static_hr = dataset_root / "static_variables" / "hr"
    lon_path = static_hr / "00_lon_rho.npy"
    lat_path = static_hr / "10_lat_rho.npy"
    if lon_path.exists() and lat_path.exists():
        lon = np.asarray(np.load(lon_path), dtype=np.float64)
        lat = np.asarray(np.load(lat_path), dtype=np.float64)
        x = (lon - np.nanmin(lon)) / 1000.0
        y = (lat - np.nanmin(lat)) / 1000.0
        extent = [
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            float(np.nanmin(y)),
            float(np.nanmax(y)),
        ]
        return extent, "X offset (km)", "Y offset (km)"
    return [0.0, 147.0, 0.0, 199.0], "X index", "Y index"


def finite_concat_abs(arrays: list[np.ndarray]) -> np.ndarray:
    values = []
    for arr in arrays:
        flat = np.ravel(np.abs(arr))
        values.append(flat[np.isfinite(flat)])
    if not values:
        return np.asarray([1.0], dtype=np.float64)
    merged = np.concatenate(values)
    if merged.size == 0:
        return np.asarray([1.0], dtype=np.float64)
    return merged


def robust_sym_limit(arrays: list[np.ndarray], percentile: float = 99.5) -> float:
    values = finite_concat_abs(arrays)
    return max(float(np.percentile(values, percentile)), 1e-6)


def robust_pos_limit(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
    values = finite_concat_abs(arrays)
    return max(float(np.percentile(values, percentile)), 1e-6)


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, show_ylabel: bool) -> None:
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=10)
    ax.set_aspect("equal", adjustable="box")


def plot_diagnostic_frame(
    model_name: str,
    pred_path: Path,
    base_name: str,
    dataset_root: Path,
    component: str,
    lr_component: str,
    baseline_label: str,
    out_dir: Path,
    extent: list[float],
    xlabel: str,
    ylabel: str,
    dpi: int,
) -> Path:
    hr = load_hr(dataset_root, base_name, component)
    lr = load_lr(dataset_root, base_name, lr_component)
    pred = load_field(pred_path)
    bicubic = baseline_from_lr(lr, hr.shape)
    model_metrics = compute_metrics(pred, hr)
    bicubic_metrics = compute_metrics(bicubic, hr)

    field_lim = robust_sym_limit([bicubic, pred, hr], percentile=99.5)
    bicubic_err = np.abs(bicubic - hr)
    model_err = np.abs(pred - hr)
    err_lim = robust_pos_limit([bicubic_err, model_err], percentile=99.0)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(25, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.18, wspace=0.18)

    field_items = [
        (baseline_label, bicubic),
        ("SR", pred),
        ("HR target", hr),
    ]
    field_im = None
    for ax, (title, data) in zip(axes[:3], field_items):
        field_im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="seismic",
            vmin=-field_lim,
            vmax=field_lim,
        )
        ax.set_title(f"{title}\nmin={data.min():.3g}, max={data.max():.3g}")
        style_axes(ax, xlabel, ylabel, show_ylabel=(ax is axes[0]))

    error_items = [
        (f"|{baseline_label}-HR|", bicubic_err, bicubic_metrics),
        ("|SR-HR|", model_err, model_metrics),
    ]
    err_im = None
    for ax, (title, data, metrics) in zip(axes[3:], error_items):
        err_im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=err_lim,
        )
        ax.set_title(
            f"{title}\n"
            f"p99={metrics['p99_abs_error']:.3g}, max={metrics['max_abs_error']:.3g}"
        )
        style_axes(ax, xlabel, ylabel, show_ylabel=False)

    if field_im is not None:
        cbar = fig.colorbar(
            field_im,
            ax=list(axes[:3]),
            location="bottom",
            fraction=0.055,
            pad=0.09,
        )
        cbar.set_label(f"{component} field")
    if err_im is not None:
        cbar = fig.colorbar(
            err_im,
            ax=list(axes[3:]),
            location="bottom",
            fraction=0.08,
            pad=0.09,
        )
        cbar.set_label("Absolute error")

    fig.suptitle(
        (
            f"Bohai {component} sparse reconstruction - {base_name}\n"
            f"SR RMSE={model_metrics['rmse']:.4g}, ACC={model_metrics['acc']:.4f}, "
            f"MaxErr={model_metrics['max_abs_error']:.4g}, "
            f"P99Err={model_metrics['p99_abs_error']:.4g}, "
            f"PeakRatio={model_metrics['peak_ratio']:.3f}; "
            f"{baseline_label} RMSE={bicubic_metrics['rmse']:.4g}"
        ),
        fontsize=18,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def selected_visual_bases(case: str, frame_start: int, frame_end: int, frame_step: int) -> list[str]:
    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if frame_start > frame_end:
        raise ValueError("--frame-start cannot be greater than --frame-end")
    return [f"{case}_{frame:06d}" for frame in range(frame_start, frame_end + 1, frame_step)]


def plot_comparison_metrics(
    summaries: list[dict[str, object]],
    output_root: Path,
    component: str,
    dpi: int,
) -> Path:
    if not summaries:
        raise ValueError("No summaries to plot")
    per_frame0 = summaries[0]["per_frame"]  # type: ignore[assignment]
    baseline_label = str(summaries[0].get("baseline_label", "baseline"))
    x = np.arange(1, len(per_frame0) + 1)
    metric_names = ("rmse", "rfne", "acc", "max_abs_error")
    fig, axes = plt.subplots(4, 1, figsize=(20, 15), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16, hspace=0.2)

    bicubic_values = {
        metric: np.asarray(
            [float(row["bicubic"][metric]) for row in per_frame0],  # type: ignore[index]
            dtype=np.float64,
        )
        for metric in metric_names
    }
    display_labels = model_display_labels(summaries)

    for ax, metric in zip(axes, metric_names):
        ax.plot(
            x,
            bicubic_values[metric],
            color="#d9822b",
            lw=1.4,
            label=baseline_label,
            alpha=0.9,
        )
        for summary in summaries:
            key = str(summary["model_key"])
            model_name = str(summary["model_name"])
            values = np.asarray(
                [float(row[key][metric]) for row in summary["per_frame"]],  # type: ignore[index]
                dtype=np.float64,
            )
            ax.plot(x, values, lw=1.35, label=display_labels[model_name], alpha=0.92)
        ax.grid(alpha=0.25)
        ax.set_ylabel(metric.upper())
        if metric in {"rmse", "rfne", "max_abs_error"}:
            ax.set_ylabel(f"{metric.upper()} lower")
        else:
            ax.set_ylabel("ACC higher")

    axes[-1].set_xlabel("Test frame index, sorted by case and time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=min(len(labels), 4),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(f"Bohai single-component {component} full-test metrics", fontsize=20)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"comparison_per_frame_metrics_{component_key(component)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_case_metric_bars(
    summaries: list[dict[str, object]],
    output_root: Path,
    component: str,
    dpi: int,
) -> Path:
    cases = sorted(summaries[0]["by_case"])  # type: ignore[arg-type]
    labels = [case.replace("S1_", "") for case in cases]
    baseline_label = str(summaries[0].get("baseline_label", "baseline"))
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.22, wspace=0.22)
    metrics = ("rmse", "rfne", "acc")
    width = 0.8 / (len(summaries) + 1)
    bicubic_offsets = -0.4 + width / 2
    model_offsets = [bicubic_offsets + width * (idx + 1) for idx in range(len(summaries))]
    display_labels = model_display_labels(summaries)
    for ax, metric in zip(axes, metrics):
        bicubic = [
            float(summaries[0]["by_case"][case]["bicubic"][metric])  # type: ignore[index]
            for case in cases
        ]
        ax.bar(x + bicubic_offsets, bicubic, width, label=baseline_label, color="#d9822b")
        for offset, summary in zip(model_offsets, summaries):
            key = str(summary["model_key"])
            values = [
                float(summary["by_case"][case][key][metric])  # type: ignore[index]
                for case in cases
            ]
            model_name = str(summary["model_name"])
            ax.bar(x + offset, values, width, label=display_labels[model_name])
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Higher is better" if metric == "acc" else "Lower is better")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(legend_labels), 4),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(f"Bohai {component} per-case metrics", fontsize=20)
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"comparison_case_metrics_{component_key(component)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def radial_frequency_bins(shape: tuple[int, int], n_bins: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = np.linspace(0.0, float(radius.max()) + EPS, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_index = np.digitize(radius.ravel(), bins) - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    return centers, bin_index, counts


def accumulate_error_spectrum(
    error: np.ndarray,
    bin_index: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    power = np.abs(np.fft.fft2(error)) ** 2 / float(error.size)
    binned = np.bincount(bin_index, weights=power.ravel(), minlength=len(counts))
    return binned / counts


def plot_spectral_error_energy(
    summaries: list[dict[str, object]],
    prediction_maps: dict[str, dict[str, Path]],
    bases: list[str],
    dataset_root: Path,
    component: str,
    lr_component: str,
    output_root: Path,
    dpi: int,
) -> tuple[Path, Path]:
    sample_hr = load_hr(dataset_root, bases[0], component)
    centers, bin_index, counts = radial_frequency_bins(sample_hr.shape)
    display_labels = model_display_labels(summaries)
    baseline_label = str(summaries[0].get("baseline_label", "baseline"))
    spectra: dict[str, np.ndarray] = {
        baseline_label: np.zeros_like(centers, dtype=np.float64),
    }
    for summary in summaries:
        label = display_labels[str(summary["model_name"])]
        spectra[label] = np.zeros_like(centers, dtype=np.float64)

    for index, base_name in enumerate(bases, start=1):
        hr = load_hr(dataset_root, base_name, component)
        lr = load_lr(dataset_root, base_name, lr_component)
        bicubic = baseline_from_lr(lr, hr.shape)
        spectra[baseline_label] += accumulate_error_spectrum(bicubic - hr, bin_index, counts)
        for summary in summaries:
            name = str(summary["model_name"])
            label = display_labels[name]
            pred = load_field(prediction_maps[name][base_name])
            spectra[label] += accumulate_error_spectrum(pred - hr, bin_index, counts)
        if index == 1 or index % 100 == 0 or index == len(bases):
            print(f"[spectral] accumulated {index}/{len(bases)}: {base_name}")

    for name in spectra:
        spectra[name] /= float(len(bases))

    output_root.mkdir(parents=True, exist_ok=True)
    key = component_key(component)
    csv_path = output_root / f"average_error_energy_{key}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["radial_frequency", *spectra.keys()])
        for idx, center in enumerate(centers):
            writer.writerow([center, *[spectra[name][idx] for name in spectra]])

    fig, ax = plt.subplots(figsize=(12.5, 7.6), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.22)
    colors = {
        baseline_label: "#d9822b",
        "EDSR SR": "#1764ab",
        "FNO2d SR": "#1b8a5a",
        "ResShift SR": "#7b3294",
    }
    for name, values in spectra.items():
        ax.plot(
            centers,
            values,
            lw=2.0,
            label=name,
            color=colors.get(name),
            alpha=0.95,
        )
    ax.set_yscale("log")
    ax.grid(alpha=0.28, which="both")
    ax.set_xlabel("Radial spatial frequency (cycles / HR grid cell)")
    ax.set_ylabel("Mean FFT error energy")
    ax.set_title(f"Average spectral error energy on Bohai {component} test set")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(len(spectra), 4),
        frameon=False,
    )
    png_path = output_root / f"average_error_energy_{key}_legend_bottom.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, csv_path


def write_model_summary_csv(
    summaries: list[dict[str, object]],
    output_root: Path,
    component: str,
) -> Path:
    path = output_root / f"model_summary_{component_key(component)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "source",
        "num_predictions",
        "rmse",
        "mae",
        "rfne",
        "acc",
        "max_abs_error",
        "p99_abs_error_mean",
        "p99_abs_error_max",
        "psnr_global",
        "psnr_mean",
        "ssim_mean",
        "peak_ratio_mean",
        "rmse_reduction_percent",
        "rfne_reduction_percent",
        "acc_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            key = str(summary["model_key"])
            baseline_label = str(summary.get("baseline_label", "baseline"))
            global_metrics = summary["global"]  # type: ignore[assignment]
            relative = global_metrics["relative"]  # type: ignore[index]
            for source in ("bicubic", key):
                metrics = global_metrics[source]  # type: ignore[index]
                row = {
                    "model": summary["model_name"],
                    "source": baseline_label if source == "bicubic" else source,
                    "num_predictions": summary["num_predictions"],
                    **{field: metrics.get(field, "") for field in fields if field in metrics},
                    "rmse_reduction_percent": "",
                    "rfne_reduction_percent": "",
                    "acc_delta": "",
                }
                if source == key:
                    row.update(
                        {
                            "rmse_reduction_percent": relative["rmse_reduction_percent"],
                            "rfne_reduction_percent": relative["rfne_reduction_percent"],
                            "acc_delta": relative["acc_delta"],
                        }
                    )
                writer.writerow(row)
    return path


def resolve_model_root(pred_dir: Path) -> Path:
    if pred_dir.name != "predictions":
        return pred_dir.parent
    if pred_dir.parent.name == "full_predict_run":
        return pred_dir.parent.parent
    return pred_dir.parent


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    component = str(args.component)
    key = component_key(component)
    output_root.mkdir(parents=True, exist_ok=True)

    expected_bases = resolve_expected_bases(dataset_root, args.max_frames, component)
    visual_bases = selected_visual_bases(
        args.case,
        args.frame_start,
        args.frame_end,
        args.frame_step,
    )
    extent, xlabel, ylabel = load_coordinate_extent(dataset_root)

    summaries: list[dict[str, object]] = []
    prediction_maps_for_completed: dict[str, dict[str, Path]] = {}
    skipped: list[dict[str, object]] = []
    generated_figures: list[str] = []

    for model_name, pred_dir in resolve_models(args.model):
        pred_dir = pred_dir.resolve()
        try:
            prediction_map = resolve_prediction_map(pred_dir)
        except FileNotFoundError as exc:
            skipped.append({"model": model_name, "reason": str(exc)})
            print(f"[skip] {model_name}: {exc}")
            continue

        missing = [base for base in expected_bases if base not in prediction_map]
        if missing and not args.include_incomplete:
            skipped.append(
                {
                    "model": model_name,
                    "prediction_dir": str(pred_dir),
                    "available": len(prediction_map),
                    "expected": len(expected_bases),
                    "reason": "incomplete prediction set",
                    "first_missing": missing[0],
                }
            )
            print(
                f"[skip] {model_name}: incomplete predictions "
                f"{len(prediction_map)}/{len(expected_bases)}, first missing={missing[0]}"
            )
            continue

        eval_bases = [base for base in expected_bases if base in prediction_map]
        if not eval_bases:
            skipped.append({"model": model_name, "reason": "no matching prediction files"})
            print(f"[skip] {model_name}: no matching prediction files")
            continue

        summary = evaluate_model(
            model_name,
            prediction_map,
            eval_bases,
            dataset_root,
            component,
            args.lr_component,
            args.baseline_label,
        )
        summary["prediction_dir"] = str(pred_dir)
        model_root = resolve_model_root(pred_dir)
        metrics_path = model_root / f"{key}_single_metrics.json"
        csv_path = model_root / f"{key}_single_per_frame_metrics.csv"
        write_json(summary, metrics_path)
        write_per_frame_csv(summary, csv_path)
        print(f"[write] {metrics_path}")
        print(f"[write] {csv_path}")

        if not args.no_figures:
            fig_dir = model_root / f"figs_eval_every10_{key}_maxerr"
            for base_name in visual_bases:
                if base_name not in prediction_map:
                    print(f"[warn] {model_name}: missing visual frame {base_name}")
                    continue
                out_path = plot_diagnostic_frame(
                    model_name=model_name,
                    pred_path=prediction_map[base_name],
                    base_name=base_name,
                    dataset_root=dataset_root,
                    component=component,
                    lr_component=args.lr_component,
                    baseline_label=args.baseline_label,
                    out_dir=fig_dir,
                    extent=extent,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    dpi=args.dpi,
                )
                generated_figures.append(str(out_path))
            print(f"[write] diagnostic figures: {fig_dir}")

        summaries.append(summary)
        prediction_maps_for_completed[model_name] = prediction_map

    combined = {
        "dataset_root": str(dataset_root),
        "component": component,
        "num_expected_test_frames": len(expected_bases),
        "evaluated_models": [summary["model_name"] for summary in summaries],
        "skipped_models": skipped,
        "summaries": summaries,
    }
    combined_path = output_root / f"combined_metrics_{key}.json"
    write_json(combined, combined_path)
    print(f"[write] {combined_path}")

    if summaries:
        summary_csv = write_model_summary_csv(summaries, output_root, component)
        print(f"[write] {summary_csv}")

    if not args.no_figures and summaries:
        metric_fig = plot_comparison_metrics(summaries, output_root, component, args.dpi)
        case_fig = plot_case_metric_bars(summaries, output_root, component, args.dpi)
        generated_figures.extend([str(metric_fig), str(case_fig)])
        print(f"[write] {metric_fig}")
        print(f"[write] {case_fig}")

    if not args.no_figures and not args.no_spectral and summaries:
        spectral_fig, spectral_csv = plot_spectral_error_energy(
            summaries=summaries,
            prediction_maps=prediction_maps_for_completed,
            bases=expected_bases,
            dataset_root=dataset_root,
            component=component,
            lr_component=args.lr_component,
            output_root=output_root,
            dpi=args.dpi,
        )
        generated_figures.append(str(spectral_fig))
        print(f"[write] {spectral_fig}")
        print(f"[write] {spectral_csv}")

    run_summary = {
        "combined_metrics": str(combined_path),
        "generated_figures_count": len(generated_figures),
        "generated_figures_sample": generated_figures[:20],
        "skipped_models": skipped,
    }
    run_summary_path = output_root / f"run_summary_{key}.json"
    write_json(run_summary, run_summary_path)
    print(f"[write] {run_summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12
DEFAULT_DATASET_ROOT = Path("/data/Bohai_Sea/process_data_sparsemask_2x")
DEFAULT_OUTPUT_DIR = Path("/data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics")
DEFAULT_MODELS = (
    (
        "DirectActiveMissing",
        Path(
            "/data1/user/lz/wave_movie/testouts/"
            "Temporal3DUNet_SparseMask2xHardActiveMissing_Vz_Test/predictions"
        ),
    ),
    (
        "E14_ActiveHF",
        Path(
            "/data1/user/lz/wave_movie/testouts/"
            "Temporal3DUNet_SparseMask2xActiveHF_E14_Vz/predictions"
        ),
    ),
    (
        "ResShift200ep",
        Path(
            "/data1/user/lz/wave_movie/testouts/"
            "Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions"
        ),
    ),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    prediction_dir: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why Bohai Vz sparse reconstruction fails: peak position, "
            "peak amplitude, active-missing errors, and spectral energy retention."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--component", default="Vz")
    parser.add_argument("--lr-component", default="Vz_interp")
    parser.add_argument("--mask-component", default="mask_observed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PRED_DIR",
        help="Prediction directory to diagnose. Can be repeated.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--active-threshold", type=float, default=0.005)
    parser.add_argument("--energetic-peak-threshold", type=float, default=0.05)
    parser.add_argument(
        "--low-cutoff",
        type=float,
        default=0.125,
        help="Radial frequency cutoff for low-band power in cycles / HR pixel.",
    )
    parser.add_argument(
        "--high-cutoff",
        type=float,
        default=0.25,
        help="Radial frequency cutoff for high-band power in cycles / HR pixel.",
    )
    parser.add_argument("--hard-frame-count", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_model_arg(value: str) -> ModelSpec:
    if "=" not in value:
        raise ValueError(f"--model must be NAME=PRED_DIR, got: {value}")
    name, pred_dir = value.split("=", 1)
    name = name.strip()
    pred_dir = pred_dir.strip()
    if not name or not pred_dir:
        raise ValueError(f"--model must be NAME=PRED_DIR, got: {value}")
    return ModelSpec(name=name, prediction_dir=Path(pred_dir))


def resolve_models(model_args: list[str]) -> list[ModelSpec]:
    models = [ModelSpec("Interp", None)]
    if model_args:
        models.extend(parse_model_arg(value) for value in model_args)
    else:
        models.extend(ModelSpec(name, pred_dir) for name, pred_dir in DEFAULT_MODELS)
    return models


def parse_case_frame(stem: str) -> tuple[str, int]:
    case, sep, frame_text = stem.rpartition("_")
    if not sep or not frame_text.isdigit():
        raise ValueError(f"Cannot parse case/frame from {stem}")
    return case, int(frame_text)


def frame_sort_key(path: Path) -> tuple[str, int, str]:
    case, frame = parse_case_frame(path.stem)
    return case, frame, path.stem


def load_2d(path: Path) -> np.ndarray:
    arr = np.squeeze(np.load(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got {arr.shape}: {path}")
    return arr.astype(np.float64, copy=False)


def resolve_hr_frames(dataset_root: Path, component: str, max_frames: int | None) -> list[Path]:
    hr_dir = dataset_root / "test" / "hr" / component
    if not hr_dir.exists():
        raise FileNotFoundError(f"Missing HR directory: {hr_dir}")
    frames = sorted(hr_dir.glob("*.npy"), key=frame_sort_key)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be >= 1")
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"No frames found in {hr_dir}")
    return frames


def rmse(diff: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = diff if mask is None else diff[mask]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def mae(diff: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.abs(diff if mask is None else diff[mask])
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def peak_info(field: np.ndarray) -> tuple[float, tuple[int, int], float]:
    abs_field = np.abs(field)
    flat_index = int(np.argmax(abs_field))
    row, col = np.unravel_index(flat_index, field.shape)
    return float(abs_field[row, col]), (int(row), int(col)), float(field[row, col])


def radial_frequency_grid(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def spectral_band_powers(
    field: np.ndarray,
    radius: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
) -> dict[str, float]:
    centered = field.astype(np.float64, copy=False) - float(np.mean(field))
    power = np.abs(np.fft.fft2(centered)) ** 2 / max(field.size, 1)
    low_mask = radius < low_cutoff
    mid_mask = (radius >= low_cutoff) & (radius < high_cutoff)
    high_mask = radius >= high_cutoff
    return {
        "low_power": float(np.sum(power[low_mask])),
        "mid_power": float(np.sum(power[mid_mask])),
        "high_power": float(np.sum(power[high_mask])),
        "total_power": float(np.sum(power)),
    }


def normalized_dot(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    av = a.reshape(-1) if mask is None else a[mask].reshape(-1)
    bv = b.reshape(-1) if mask is None else b[mask].reshape(-1)
    if av.size == 0:
        return float("nan")
    numerator = float(np.dot(av, bv))
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return numerator / (denominator + EPS)


def frame_metrics(
    *,
    case: str,
    frame: int,
    model_name: str,
    pred: np.ndarray,
    target: np.ndarray,
    observed_mask: np.ndarray,
    radius: np.ndarray,
    active_threshold: float,
    low_cutoff: float,
    high_cutoff: float,
) -> dict[str, object]:
    diff = pred - target
    abs_diff = np.abs(diff)
    missing_mask = ~observed_mask
    active_missing_mask = missing_mask & (np.abs(target) > active_threshold)
    inactive_missing_mask = missing_mask & (np.abs(target) <= active_threshold)

    target_peak_abs, target_peak_xy, target_peak_signed = peak_info(target)
    pred_peak_abs, pred_peak_xy, pred_peak_signed = peak_info(pred)
    target_peak_value_pred = float(pred[target_peak_xy])
    target_peak_abs_pred = float(abs(target_peak_value_pred))
    peak_distance_px = math.dist(target_peak_xy, pred_peak_xy)

    target_power = spectral_band_powers(target, radius, low_cutoff, high_cutoff)
    pred_power = spectral_band_powers(pred, radius, low_cutoff, high_cutoff)

    return {
        "case": case,
        "frame": frame,
        "model": model_name,
        "rmse": rmse(diff),
        "mae": mae(diff),
        "missing_rmse": rmse(diff, missing_mask),
        "active_missing_rmse": rmse(diff, active_missing_mask),
        "inactive_missing_rmse": rmse(diff, inactive_missing_mask),
        "p99_abs_error": float(np.percentile(abs_diff, 99.0)),
        "max_abs_error": float(np.max(abs_diff)),
        "target_peak_abs": target_peak_abs,
        "pred_peak_abs": pred_peak_abs,
        "peak_ratio": float(pred_peak_abs / (target_peak_abs + EPS)),
        "target_peak_abs_pred": target_peak_abs_pred,
        "target_peak_ratio": float(target_peak_abs_pred / (target_peak_abs + EPS)),
        "target_peak_signed": target_peak_signed,
        "pred_peak_signed": pred_peak_signed,
        "target_peak_pred_signed": target_peak_value_pred,
        "target_peak_row": target_peak_xy[0],
        "target_peak_col": target_peak_xy[1],
        "pred_peak_row": pred_peak_xy[0],
        "pred_peak_col": pred_peak_xy[1],
        "peak_distance_px": peak_distance_px,
        "acc": normalized_dot(pred, target),
        "active_missing_acc": normalized_dot(pred, target, active_missing_mask),
        "low_power_ratio": float(pred_power["low_power"] / (target_power["low_power"] + EPS)),
        "mid_power_ratio": float(pred_power["mid_power"] / (target_power["mid_power"] + EPS)),
        "high_power_ratio": float(pred_power["high_power"] / (target_power["high_power"] + EPS)),
        "total_power_ratio": float(pred_power["total_power"] / (target_power["total_power"] + EPS)),
        "target_low_power": target_power["low_power"],
        "target_mid_power": target_power["mid_power"],
        "target_high_power": target_power["high_power"],
        "target_total_power": target_power["total_power"],
    }


def aggregate_rows(rows: list[dict[str, object]], energetic_peak_threshold: float) -> dict[str, dict[str, float]]:
    numeric_fields = [
        "rmse",
        "mae",
        "missing_rmse",
        "active_missing_rmse",
        "inactive_missing_rmse",
        "p99_abs_error",
        "max_abs_error",
        "peak_ratio",
        "target_peak_ratio",
        "peak_distance_px",
        "acc",
        "active_missing_acc",
        "low_power_ratio",
        "mid_power_ratio",
        "high_power_ratio",
        "total_power_ratio",
    ]
    summary: dict[str, dict[str, float]] = {}
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        energetic_rows = [
            row for row in model_rows if float(row["target_peak_abs"]) >= energetic_peak_threshold
        ]
        for prefix, subset in (("all", model_rows), ("energetic", energetic_rows)):
            key = f"{model}:{prefix}"
            summary[key] = {"count": float(len(subset))}
            for field in numeric_fields:
                values = np.asarray([float(row[field]) for row in subset], dtype=np.float64)
                if values.size == 0:
                    summary[key][f"{field}_mean"] = float("nan")
                    summary[key][f"{field}_median"] = float("nan")
                    continue
                summary[key][f"{field}_mean"] = float(np.nanmean(values))
                summary[key][f"{field}_median"] = float(np.nanmedian(values))
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, dict[str, float]]) -> None:
    rows = []
    fields = sorted({field for metrics in summary.values() for field in metrics})
    for key, metrics in sorted(summary.items()):
        model, subset = key.split(":", 1)
        row: dict[str, object] = {"model": model, "subset": subset}
        row.update({field: metrics.get(field, float("nan")) for field in fields})
        rows.append(row)
    write_csv(path, rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "model"


def plot_summary_bars(
    rows: list[dict[str, object]],
    output_dir: Path,
    energetic_peak_threshold: float,
    dpi: int,
) -> None:
    energetic = [
        row for row in rows if float(row["target_peak_abs"]) >= energetic_peak_threshold
    ]
    models = sorted({str(row["model"]) for row in rows})
    metrics = [
        ("active_missing_rmse", "Active-missing RMSE"),
        ("target_peak_ratio", "Amplitude at HR peak / HR peak"),
        ("peak_distance_px", "Peak distance (px)"),
        ("high_power_ratio", "High-frequency power ratio"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(21, 5.2), constrained_layout=True)
    for ax, (field, title) in zip(axes, metrics):
        values = []
        for model in models:
            subset = [row for row in energetic if str(row["model"]) == model]
            values.append(float(np.nanmean([float(row[field]) for row in subset])))
        ax.bar(models, values, color=["#d9822b", "#1764ab", "#52a365", "#8c5fbf"][: len(models)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        if field.endswith("ratio"):
            ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
    fig.suptitle(
        f"E15 failure diagnostics on energetic frames (HR peak >= {energetic_peak_threshold:g})",
        fontsize=15,
    )
    fig.savefig(output_dir / "fig1_failure_mode_summary.png", dpi=dpi)
    plt.close(fig)


def plot_peak_scatter(
    rows: list[dict[str, object]],
    output_dir: Path,
    energetic_peak_threshold: float,
    dpi: int,
) -> None:
    energetic = [
        row for row in rows if float(row["target_peak_abs"]) >= energetic_peak_threshold
    ]
    models = sorted({str(row["model"]) for row in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 5.2), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        subset = [row for row in energetic if str(row["model"]) == model]
        x = [float(row["target_peak_abs"]) for row in subset]
        y = [float(row["target_peak_ratio"]) for row in subset]
        c = [float(row["peak_distance_px"]) for row in subset]
        sc = ax.scatter(x, y, c=c, s=14, cmap="magma", alpha=0.75)
        ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_title(model)
        ax.set_xlabel("HR peak amplitude")
        ax.set_ylabel("Pred amplitude at HR peak / HR peak")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="Peak distance (px)")
    fig.suptitle("Peak amplitude recovery and peak-location error", fontsize=15)
    fig.savefig(output_dir / "fig2_peak_recovery_scatter.png", dpi=dpi)
    plt.close(fig)


def plot_spectral_boxplot(
    rows: list[dict[str, object]],
    output_dir: Path,
    energetic_peak_threshold: float,
    dpi: int,
) -> None:
    energetic = [
        row for row in rows if float(row["target_peak_abs"]) >= energetic_peak_threshold
    ]
    models = sorted({str(row["model"]) for row in rows})
    fields = [
        ("low_power_ratio", "Low"),
        ("mid_power_ratio", "Mid"),
        ("high_power_ratio", "High"),
    ]
    fig, axes = plt.subplots(1, len(fields), figsize=(17, 5), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        data = [
            [float(row[field]) for row in energetic if str(row["model"]) == model]
            for model in models
        ]
        ax.boxplot(data, tick_labels=models, showfliers=False)
        ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.set_title(f"{title}-band power retention")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Spectral power retention relative to HR target", fontsize=15)
    fig.savefig(output_dir / "fig3_spectral_retention_boxplot.png", dpi=dpi)
    plt.close(fig)


def hard_frame_rows(rows: list[dict[str, object]], metric: str, count: int) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row[metric]), reverse=True)[:count]


def diagnose(args: argparse.Namespace) -> dict[str, object]:
    frames = resolve_hr_frames(args.dataset_root, args.component, args.max_frames)
    lr_dir = args.dataset_root / "test" / "lr" / args.lr_component
    mask_dir = args.dataset_root / "test" / "lr" / args.mask_component
    models = resolve_models(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_target = load_2d(frames[0])
    radius = radial_frequency_grid(first_target.shape)
    rows: list[dict[str, object]] = []

    for hr_path in frames:
        case, frame = parse_case_frame(hr_path.stem)
        target = load_2d(hr_path)
        lr = load_2d(lr_dir / hr_path.name)
        observed_mask = load_2d(mask_dir / hr_path.name) > 0.5
        for model in models:
            if model.prediction_dir is None:
                pred = lr
            else:
                pred_path = model.prediction_dir / f"{hr_path.stem}_sr.npy"
                if not pred_path.exists():
                    raise FileNotFoundError(pred_path)
                pred = load_2d(pred_path)
            if pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {model.name} {hr_path.stem}: {pred.shape} vs {target.shape}"
                )
            rows.append(
                frame_metrics(
                    case=case,
                    frame=frame,
                    model_name=model.name,
                    pred=pred,
                    target=target,
                    observed_mask=observed_mask,
                    radius=radius,
                    active_threshold=args.active_threshold,
                    low_cutoff=args.low_cutoff,
                    high_cutoff=args.high_cutoff,
                )
            )

    summary = aggregate_rows(rows, args.energetic_peak_threshold)
    write_csv(args.output_dir / "per_frame_failure_diagnostics.csv", rows)
    write_summary_csv(args.output_dir / "summary_failure_diagnostics.csv", summary)
    (args.output_dir / "summary_failure_diagnostics.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "component": args.component,
                "frame_count": len(frames),
                "models": [
                    {"name": model.name, "prediction_dir": str(model.prediction_dir)}
                    for model in models
                ],
                "active_threshold": args.active_threshold,
                "energetic_peak_threshold": args.energetic_peak_threshold,
                "low_cutoff": args.low_cutoff,
                "high_cutoff": args.high_cutoff,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for metric in ("active_missing_rmse", "p99_abs_error", "peak_distance_px", "max_abs_error"):
        write_csv(
            args.output_dir / f"hard_frames_by_{metric}.csv",
            hard_frame_rows(rows, metric, args.hard_frame_count),
        )

    plot_summary_bars(rows, args.output_dir, args.energetic_peak_threshold, args.dpi)
    plot_peak_scatter(rows, args.output_dir, args.energetic_peak_threshold, args.dpi)
    plot_spectral_boxplot(rows, args.output_dir, args.energetic_peak_threshold, args.dpi)
    return {
        "output_dir": str(args.output_dir),
        "frame_count": len(frames),
        "model_count": len(models),
        "summary": summary,
    }


def main() -> None:
    result = diagnose(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

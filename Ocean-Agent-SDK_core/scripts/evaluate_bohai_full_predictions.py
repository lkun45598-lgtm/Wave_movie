from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


COMPONENTS = ("Vx", "Vy", "Vz")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full Bohai XYZ 4x SR prediction directories."
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing *_sr.npy prediction files.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/Bohai_Sea/process_data"),
        help="Processed Bohai dataset root with test/hr and test/lr folders.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model label used in metrics and figures, e.g. FNO2d.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Figure output directory. Defaults to <prediction-dir>/../figs_eval_full.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Metrics JSON path. Defaults to <prediction-dir>/../full_metrics.json.",
    )
    parser.add_argument("--dpi", type=int, default=170, help="Output figure DPI.")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional cap for debugging. Omit for full evaluation.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only write metrics JSON, skip PNG generation.",
    )
    return parser.parse_args()


def model_key(model_name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", model_name.strip()).strip("_").lower()
    if not key:
        raise ValueError("model_name cannot be empty")
    return key


def prediction_base_name(path: Path) -> str:
    if path.suffix != ".npy" or not path.stem.endswith("_sr"):
        raise ValueError(f"Prediction file must match *_sr.npy: {path.name}")
    return path.stem[: -len("_sr")]


def parse_case_frame(base_name: str) -> tuple[str, int]:
    case, sep, frame_text = base_name.rpartition("_")
    if not sep or not frame_text.isdigit():
        raise ValueError(f"Cannot parse case/frame from filename base: {base_name}")
    return case, int(frame_text)


def prediction_sort_key(path: Path) -> tuple[str, int, str]:
    base = prediction_base_name(path)
    case, frame = parse_case_frame(base)
    return case, frame, path.name


def resolve_prediction_files(prediction_dir: Path, max_frames: int | None = None) -> list[Path]:
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Missing prediction directory: {prediction_dir}")
    files = sorted(prediction_dir.glob("*_sr.npy"), key=prediction_sort_key)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be >= 1 when provided")
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"No *_sr.npy files found in {prediction_dir}")
    return files


def load_component(root: Path, split: str, component: str, base_name: str) -> np.ndarray:
    path = root / "test" / split / component / f"{base_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split}/{component} frame: {path}")
    return np.asarray(np.load(path), dtype=np.float32)


def load_frame_triplet(dataset_root: Path, base_name: str) -> tuple[np.ndarray, np.ndarray]:
    hr = np.stack(
        [load_component(dataset_root, "hr", component, base_name) for component in COMPONENTS],
        axis=-1,
    )
    lr = np.stack(
        [load_component(dataset_root, "lr", component, base_name) for component in COMPONENTS],
        axis=-1,
    )
    return hr.astype(np.float32, copy=False), lr.astype(np.float32, copy=False)


def load_prediction(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Prediction must be [H,W,C], got {arr.shape}: {path}")
    if arr.shape[-1] != len(COMPONENTS):
        raise ValueError(f"Prediction must have {len(COMPONENTS)} channels, got {arr.shape}: {path}")
    return arr


def upsample_lr(lr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if lr.ndim != 3 or lr.shape[-1] != len(COMPONENTS):
        raise ValueError(f"LR must be [H,W,{len(COMPONENTS)}], got {lr.shape}")
    tensor = torch.from_numpy(lr).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(tensor, size=target_hw, mode="bicubic", align_corners=False)
    return up.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred64 = np.asarray(pred, dtype=np.float64).reshape(-1)
    target64 = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred64.shape != target64.shape:
        raise ValueError(f"Metric shape mismatch: {pred.shape} vs {target.shape}")
    diff = pred64 - target64
    sum_sq_diff = float(np.dot(diff, diff))
    sum_abs_diff = float(np.sum(np.abs(diff)))
    sum_sq_target = float(np.dot(target64, target64))
    sum_sq_pred = float(np.dot(pred64, pred64))
    dot = float(np.dot(pred64, target64))
    count = int(diff.size)
    return {
        "rmse": float(np.sqrt(sum_sq_diff / max(count, 1))),
        "mae": float(sum_abs_diff / max(count, 1)),
        "rfne": float(np.sqrt(sum_sq_diff) / (np.sqrt(sum_sq_target) + EPS)),
        "acc": float(dot / (np.sqrt(sum_sq_pred) * np.sqrt(sum_sq_target) + EPS)),
        "max_abs_error": float(np.max(np.abs(diff))) if count else 0.0,
        "count": count,
    }


@dataclass
class MetricAccumulator:
    sum_sq_diff: float = 0.0
    sum_abs_diff: float = 0.0
    sum_sq_target: float = 0.0
    sum_sq_pred: float = 0.0
    dot: float = 0.0
    max_abs_error: float = 0.0
    count: int = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred64 = np.asarray(pred, dtype=np.float64).reshape(-1)
        target64 = np.asarray(target, dtype=np.float64).reshape(-1)
        if pred64.shape != target64.shape:
            raise ValueError(f"Accumulator shape mismatch: {pred.shape} vs {target.shape}")
        diff = pred64 - target64
        self.sum_sq_diff += float(np.dot(diff, diff))
        self.sum_abs_diff += float(np.sum(np.abs(diff)))
        self.sum_sq_target += float(np.dot(target64, target64))
        self.sum_sq_pred += float(np.dot(pred64, pred64))
        self.dot += float(np.dot(pred64, target64))
        if diff.size:
            self.max_abs_error = max(self.max_abs_error, float(np.max(np.abs(diff))))
        self.count += int(diff.size)

    def to_metrics(self) -> dict[str, float | int]:
        return {
            "rmse": float(np.sqrt(self.sum_sq_diff / max(self.count, 1))),
            "mae": float(self.sum_abs_diff / max(self.count, 1)),
            "rfne": float(np.sqrt(self.sum_sq_diff) / (np.sqrt(self.sum_sq_target) + EPS)),
            "acc": float(
                self.dot / (np.sqrt(self.sum_sq_pred) * np.sqrt(self.sum_sq_target) + EPS)
            ),
            "max_abs_error": float(self.max_abs_error),
            "count": int(self.count),
        }


def relative_metrics(model_metrics: dict[str, float | int], bicubic_metrics: dict[str, float | int]) -> dict[str, float]:
    bicubic_rfne = float(bicubic_metrics["rfne"])
    bicubic_rmse = float(bicubic_metrics["rmse"])
    return {
        "rfne_reduction_percent": float(
            100.0 * (bicubic_rfne - float(model_metrics["rfne"])) / (bicubic_rfne + EPS)
        ),
        "rmse_reduction_percent": float(
            100.0 * (bicubic_rmse - float(model_metrics["rmse"])) / (bicubic_rmse + EPS)
        ),
        "acc_delta": float(float(model_metrics["acc"]) - float(bicubic_metrics["acc"])),
    }


def empty_component_accumulators() -> dict[str, MetricAccumulator]:
    return {component: MetricAccumulator() for component in COMPONENTS}


def evaluate_predictions(
    prediction_files: list[Path],
    dataset_root: Path,
    key: str,
) -> dict[str, object]:
    global_model = MetricAccumulator()
    global_bicubic = MetricAccumulator()
    by_var_model = empty_component_accumulators()
    by_var_bicubic = empty_component_accumulators()
    by_case: dict[str, dict[str, object]] = {}
    per_frame: list[dict[str, object]] = []

    for index, pred_path in enumerate(prediction_files, start=1):
        base = prediction_base_name(pred_path)
        case, frame = parse_case_frame(base)
        pred = load_prediction(pred_path)
        hr, lr = load_frame_triplet(dataset_root, base)
        if pred.shape != hr.shape:
            raise ValueError(f"Prediction/HR shape mismatch for {base}: {pred.shape} vs {hr.shape}")
        bicubic = upsample_lr(lr, target_hw=hr.shape[:2])
        if bicubic.shape != hr.shape:
            raise ValueError(f"Bicubic/HR shape mismatch for {base}: {bicubic.shape} vs {hr.shape}")

        frame_model = compute_metrics(pred, hr)
        frame_bicubic = compute_metrics(bicubic, hr)
        per_frame.append(
            {
                "filename": base,
                "case": case,
                "frame": frame,
                key: frame_model,
                "bicubic": frame_bicubic,
            }
        )

        global_model.update(pred, hr)
        global_bicubic.update(bicubic, hr)

        case_row = by_case.setdefault(
            case,
            {
                "frames": 0,
                "_model": MetricAccumulator(),
                "_bicubic": MetricAccumulator(),
            },
        )
        case_row["frames"] = int(case_row["frames"]) + 1
        case_row["_model"].update(pred, hr)  # type: ignore[union-attr]
        case_row["_bicubic"].update(bicubic, hr)  # type: ignore[union-attr]

        for component_index, component in enumerate(COMPONENTS):
            by_var_model[component].update(pred[..., component_index], hr[..., component_index])
            by_var_bicubic[component].update(bicubic[..., component_index], hr[..., component_index])

        if index == 1 or index % 100 == 0 or index == len(prediction_files):
            print(f"Evaluated {index}/{len(prediction_files)}: {base}")

    global_model_metrics = global_model.to_metrics()
    global_bicubic_metrics = global_bicubic.to_metrics()
    by_var = {
        component: {
            key: by_var_model[component].to_metrics(),
            "bicubic": by_var_bicubic[component].to_metrics(),
        }
        for component in COMPONENTS
    }

    by_case_out: dict[str, dict[str, object]] = {}
    for case in sorted(by_case):
        row = by_case[case]
        by_case_out[case] = {
            "frames": int(row["frames"]),
            key: row["_model"].to_metrics(),  # type: ignore[union-attr]
            "bicubic": row["_bicubic"].to_metrics(),  # type: ignore[union-attr]
        }

    return {
        "num_predictions": len(prediction_files),
        "global": {
            key: global_model_metrics,
            "bicubic": global_bicubic_metrics,
            "relative": relative_metrics(global_model_metrics, global_bicubic_metrics),
        },
        "by_var": by_var,
        "by_case": by_case_out,
        "per_frame": per_frame,
    }


def metric_series(per_frame: list[dict[str, object]], key: str, source: str, metric: str) -> np.ndarray:
    return np.asarray(
        [float(row[source][metric]) for row in per_frame],  # type: ignore[index]
        dtype=np.float64,
    )


def case_boundaries(per_frame: list[dict[str, object]]) -> tuple[list[int], list[int], list[str]]:
    starts: list[int] = []
    centers: list[int] = []
    labels: list[str] = []
    current_case: str | None = None
    start = 0
    for idx, row in enumerate(per_frame):
        case = str(row["case"])
        if current_case is None:
            current_case = case
            start = idx
        elif case != current_case:
            starts.append(idx)
            centers.append((start + idx - 1) // 2)
            labels.append(current_case.replace("S1_", ""))
            current_case = case
            start = idx
    if current_case is not None:
        centers.append((start + len(per_frame) - 1) // 2)
        labels.append(current_case.replace("S1_", ""))
    return starts, centers, labels


def plot_per_frame_metrics(
    per_frame: list[dict[str, object]],
    key: str,
    model_name: str,
    out_dir: Path,
    dpi: int,
) -> None:
    x = np.arange(1, len(per_frame) + 1)
    metrics = ("rmse", "rfne", "acc")
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
    boundaries, centers, labels = case_boundaries(per_frame)

    for ax, metric in zip(axes, metrics):
        model_values = metric_series(per_frame, key, key, metric)
        bicubic_values = metric_series(per_frame, key, "bicubic", metric)
        ax.plot(x, bicubic_values, color="#d9822b", lw=1.2, label="Bicubic", alpha=0.9)
        ax.plot(x, model_values, color="#1764ab", lw=1.35, label=model_name, alpha=0.95)
        for boundary in boundaries:
            ax.axvline(boundary + 0.5, color="0.75", lw=0.8, alpha=0.65)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.24)
        if metric == "acc":
            ax.set_ylabel("ACC")
        else:
            ax.set_ylabel(f"{metric.upper()} lower is better")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Test frame index, sorted by case and time")
    axes[-1].set_xticks([center + 1 for center in centers])
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle(f"Full Test Per-Frame Metrics - {model_name}", fontsize=20)
    save_figure(fig, out_dir / "fig1_full_per_frame_metrics.png", dpi)


def plot_case_metrics(
    by_case: dict[str, dict[str, object]],
    key: str,
    model_name: str,
    out_dir: Path,
    dpi: int,
) -> None:
    cases = sorted(by_case)
    labels = [case.replace("S1_", "") for case in cases]
    x = np.arange(len(cases))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, metric in zip(axes, ("rmse", "rfne", "acc")):
        model_values = [float(by_case[case][key][metric]) for case in cases]  # type: ignore[index]
        bicubic_values = [float(by_case[case]["bicubic"][metric]) for case in cases]  # type: ignore[index]
        ax.bar(x - width / 2, bicubic_values, width, color="#d9822b", label="Bicubic")
        ax.bar(x + width / 2, model_values, width, color="#1764ab", label=model_name)
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if metric == "acc":
            ax.set_ylabel("Higher is better")
        else:
            ax.set_ylabel("Lower is better")
        ax.legend(loc="best")
    fig.suptitle(f"Per-Case Full Test Metrics - {model_name}", fontsize=20)
    save_figure(fig, out_dir / "fig2_case_metrics.png", dpi)


def build_heatmap(
    per_frame: list[dict[str, object]],
    source: str,
    metric: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    cases = sorted({str(row["case"]) for row in per_frame})
    frames = sorted({int(row["frame"]) for row in per_frame})
    case_to_idx = {case: idx for idx, case in enumerate(cases)}
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
    heatmap = np.full((len(cases), len(frames)), np.nan, dtype=np.float64)
    for row in per_frame:
        heatmap[case_to_idx[str(row["case"])], frame_to_idx[int(row["frame"])]] = float(
            row[source][metric]  # type: ignore[index]
        )
    return heatmap, cases, frames


def plot_metric_heatmaps(
    per_frame: list[dict[str, object]],
    key: str,
    model_name: str,
    out_dir: Path,
    dpi: int,
) -> None:
    model_rmse, cases, frames = build_heatmap(per_frame, key, "rmse")
    bicubic_rmse, _, _ = build_heatmap(per_frame, "bicubic", "rmse")
    improvement = 100.0 * (bicubic_rmse - model_rmse) / (bicubic_rmse + EPS)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    items = [
        (model_rmse, f"{model_name} RMSE", "viridis", None, None, "RMSE"),
        (bicubic_rmse, "Bicubic RMSE", "viridis", None, None, "RMSE"),
        (improvement, "RMSE reduction vs Bicubic (%)", "RdBu", -20, 20, "%"),
    ]
    ylabels = [case.replace("S1_", "") for case in cases]
    xtick_positions = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
    xtick_labels = [str(frames[pos]) for pos in xtick_positions]

    for ax, (data, title, cmap, vmin, vmax, cbar_label) in zip(axes, items):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Frame index within case")
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(np.arange(len(cases)))
        ax.set_yticklabels(ylabels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cbar_label)
    axes[0].set_ylabel("Case")
    fig.suptitle(f"Full Test RMSE Heatmaps - {model_name}", fontsize=20)
    save_figure(fig, out_dir / "fig3_full_metric_heatmaps.png", dpi)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def generate_figures(metrics: dict[str, object], key: str, model_name: str, out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 20,
        }
    )
    per_frame = metrics["per_frame"]  # type: ignore[assignment]
    by_case = metrics["by_case"]  # type: ignore[assignment]
    plot_per_frame_metrics(per_frame, key, model_name, out_dir, dpi)  # type: ignore[arg-type]
    plot_case_metrics(by_case, key, model_name, out_dir, dpi)  # type: ignore[arg-type]
    plot_metric_heatmaps(per_frame, key, model_name, out_dir, dpi)  # type: ignore[arg-type]


def main() -> None:
    args = parse_args()
    key = model_key(args.model_name)
    prediction_dir = args.prediction_dir.resolve()
    out_dir = args.out_dir or prediction_dir.parent / "figs_eval_full"
    metrics_path = args.metrics_path or prediction_dir.parent / "full_metrics.json"

    files = resolve_prediction_files(prediction_dir, max_frames=args.max_frames)
    metrics = evaluate_predictions(files, args.dataset_root, key)
    metrics["prediction_dir"] = str(prediction_dir)
    ordered_metrics = {
        "prediction_dir": metrics["prediction_dir"],
        "num_predictions": metrics["num_predictions"],
        "global": metrics["global"],
        "by_var": metrics["by_var"],
        "by_case": metrics["by_case"],
        "per_frame": metrics["per_frame"],
    }

    write_metrics(ordered_metrics, metrics_path)
    if not args.no_figures:
        generate_figures(ordered_metrics, key, args.model_name, out_dir, args.dpi)

    print(f"Saved metrics: {metrics_path}")
    if not args.no_figures:
        print(f"Saved figures: {out_dir}")
        for path in sorted(out_dir.glob("fig*.png")):
            print(f"  {path.name}")


if __name__ == "__main__":
    main()

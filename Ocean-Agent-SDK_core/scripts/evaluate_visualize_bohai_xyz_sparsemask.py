from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


COMPONENTS = ("Vx", "Vy", "Vz")
INTERP_COMPONENTS = ("Vx_interp", "Vy_interp", "Vz_interp")
SPARSE_COMPONENTS = ("Vx_sparse", "Vy_sparse", "Vz_sparse")
EPS = 1e-12

DEFAULT_RUN_ROOT = Path(
    "/data1/user/lz/wave_movie/testouts/"
    "Resshift_SparseMask2xObserved_XYZ_MSEAux_BS32_200ep"
)
DEFAULT_DATASET_ROOT = Path("/data/Bohai_Sea/process_data_sparsemask_2x")
DEFAULT_SINGLE_VZ_DIR = Path(
    "/data1/user/lz/wave_movie/testouts/"
    "Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and visualize Bohai sparse-mask 2x XYZ ResShift predictions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Sparse-mask processed dataset root.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "predictions",
        help="XYZ prediction directory containing *_sr.npy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "xyz_eval",
        help="Output directory for metrics and figures.",
    )
    parser.add_argument(
        "--single-vz-prediction-dir",
        type=Path,
        default=DEFAULT_SINGLE_VZ_DIR,
        help=(
            "Optional previous best single-Vz prediction directory. If present, "
            "Vz comparison figures include it."
        ),
    )
    parser.add_argument("--model-name", default="XYZ-joint ResShift")
    parser.add_argument("--single-vz-name", default="Single-Vz ResShift200ep")
    parser.add_argument("--baseline-label", default="Interp")
    parser.add_argument("--active-threshold", type=float, default=0.005)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--case", default="S1_TTTZ")
    parser.add_argument("--frame-start", type=int, default=10)
    parser.add_argument("--frame-end", type=int, default=100)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def safe_key(value: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return key or "model"


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


def resolve_expected_bases(dataset_root: Path, max_frames: int | None) -> list[str]:
    hr_dir = dataset_root / "test" / "hr" / "Vz"
    if not hr_dir.exists():
        raise FileNotFoundError(f"Missing HR directory: {hr_dir}")
    bases = sorted((path.stem for path in hr_dir.glob("*.npy")), key=sort_key_from_base)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be >= 1")
        bases = bases[:max_frames]
    if not bases:
        raise FileNotFoundError(f"No test HR frames found in {hr_dir}")
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
        raise ValueError(f"Expected 2D field, got {arr.shape}: {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {path}")
    return arr.astype(np.float32, copy=False)


def load_xyz_prediction(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != len(COMPONENTS):
        raise ValueError(f"Expected [H,W,3] prediction, got {arr.shape}: {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {path}")
    return arr.astype(np.float32, copy=False)


def load_stack(dataset_root: Path, split: str, variables: tuple[str, ...], base_name: str) -> np.ndarray:
    root = dataset_root / "test" / split
    return np.stack(
        [load_field(root / variable / f"{base_name}.npy") for variable in variables],
        axis=-1,
    ).astype(np.float32, copy=False)


def load_mask(dataset_root: Path, base_name: str) -> np.ndarray:
    return load_field(dataset_root / "test" / "lr" / "mask_observed" / f"{base_name}.npy") > 0.5


def load_coordinate_extent(dataset_root: Path) -> tuple[list[float], str, str]:
    static_hr = dataset_root / "static_variables" / "hr"
    x_path = static_hr / "00_lon_rho.npy"
    y_path = static_hr / "10_lat_rho.npy"
    if x_path.exists() and y_path.exists():
        x = np.asarray(np.load(x_path), dtype=np.float64)
        y = np.asarray(np.load(y_path), dtype=np.float64)
        x = (x - np.nanmin(x)) / 1000.0
        y = (y - np.nanmin(y)) / 1000.0
        return [
            float(np.nanmin(x)),
            float(np.nanmax(x)),
            float(np.nanmin(y)),
            float(np.nanmax(y)),
        ], "X offset (km)", "Y offset (km)"
    return [0.0, 149.0, 0.0, 199.0], "X index", "Y index"


def values_for_mask(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return arr
    if arr.ndim == 2:
        return arr[mask]
    if arr.ndim == 3:
        return arr[mask, :]
    raise ValueError(f"Unsupported array rank: {arr.shape}")


def psnr_from_rmse(rmse: float, data_range: float) -> float:
    if rmse <= EPS or data_range <= EPS:
        return float("nan")
    return float(20.0 * math.log10(data_range / rmse))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    compute_ssim: bool = False,
) -> dict[str, float | int]:
    pred_values = np.asarray(values_for_mask(pred, mask), dtype=np.float64)
    target_values = np.asarray(values_for_mask(target, mask), dtype=np.float64)
    if pred_values.shape != target_values.shape:
        raise ValueError(f"Metric shape mismatch: {pred_values.shape} vs {target_values.shape}")
    pred_flat = pred_values.reshape(-1)
    target_flat = target_values.reshape(-1)
    count = int(target_flat.size)
    if count == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "rfne": float("nan"),
            "acc": float("nan"),
            "bias": float("nan"),
            "p99_abs_error": float("nan"),
            "max_abs_error": float("nan"),
            "psnr": float("nan"),
            "ssim": float("nan"),
            "peak_abs_pred": float("nan"),
            "peak_abs_target": float("nan"),
            "peak_ratio": float("nan"),
            "count": 0,
        }

    diff = pred_flat - target_flat
    abs_diff = np.abs(diff)
    sum_sq_diff = float(np.dot(diff, diff))
    sum_sq_pred = float(np.dot(pred_flat, pred_flat))
    sum_sq_target = float(np.dot(target_flat, target_flat))
    dot = float(np.dot(pred_flat, target_flat))
    rmse = float(np.sqrt(sum_sq_diff / count))
    data_range = float(np.max(target_flat) - np.min(target_flat))
    ssim_value = float("nan")
    if compute_ssim and mask is None and pred.ndim == 2 and data_range > EPS:
        ssim_value = float(structural_similarity(target, pred, data_range=data_range))

    peak_abs_pred = float(np.max(np.abs(pred_flat)))
    peak_abs_target = float(np.max(np.abs(target_flat)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(abs_diff)),
        "rfne": float(np.sqrt(sum_sq_diff) / (np.sqrt(sum_sq_target) + EPS)),
        "acc": float(dot / (np.sqrt(sum_sq_pred) * np.sqrt(sum_sq_target) + EPS)),
        "bias": float(np.mean(diff)),
        "p99_abs_error": float(np.percentile(abs_diff, 99.0)),
        "max_abs_error": float(np.max(abs_diff)),
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
    frame_ssim: list[float] = field(default_factory=list)
    frame_psnr: list[float] = field(default_factory=list)
    frame_peak_ratio: list[float] = field(default_factory=list)

    def update(self, pred: np.ndarray, target: np.ndarray, metrics: dict[str, float | int]) -> None:
        pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
        target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
        if pred_flat.shape != target_flat.shape:
            raise ValueError(f"Accumulator shape mismatch: {pred.shape} vs {target.shape}")
        if pred_flat.size == 0:
            return
        diff = pred_flat - target_flat
        abs_diff = np.abs(diff)
        self.sum_sq_diff += float(np.dot(diff, diff))
        self.sum_abs_diff += float(np.sum(abs_diff))
        self.sum_diff += float(np.sum(diff))
        self.sum_sq_target += float(np.dot(target_flat, target_flat))
        self.sum_sq_pred += float(np.dot(pred_flat, pred_flat))
        self.dot += float(np.dot(pred_flat, target_flat))
        self.max_abs_error = max(self.max_abs_error, float(np.max(abs_diff)))
        self.count += int(diff.size)
        self.target_min = min(self.target_min, float(np.min(target_flat)))
        self.target_max = max(self.target_max, float(np.max(target_flat)))
        self.frame_p99.append(float(metrics["p99_abs_error"]))
        if math.isfinite(float(metrics["ssim"])):
            self.frame_ssim.append(float(metrics["ssim"]))
        if math.isfinite(float(metrics["psnr"])):
            self.frame_psnr.append(float(metrics["psnr"]))
        if math.isfinite(float(metrics["peak_ratio"])):
            self.frame_peak_ratio.append(float(metrics["peak_ratio"]))

    def to_metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "rmse": float("nan"),
                "mae": float("nan"),
                "rfne": float("nan"),
                "acc": float("nan"),
                "bias": float("nan"),
                "p99_abs_error_mean": float("nan"),
                "p99_abs_error_max": float("nan"),
                "max_abs_error": float("nan"),
                "psnr_global": float("nan"),
                "psnr_mean": float("nan"),
                "ssim_mean": float("nan"),
                "peak_ratio_mean": float("nan"),
                "count": 0,
            }
        rmse = float(np.sqrt(self.sum_sq_diff / self.count))
        data_range = self.target_max - self.target_min
        return {
            "rmse": rmse,
            "mae": float(self.sum_abs_diff / self.count),
            "rfne": float(np.sqrt(self.sum_sq_diff) / (np.sqrt(self.sum_sq_target) + EPS)),
            "acc": float(self.dot / (np.sqrt(self.sum_sq_pred) * np.sqrt(self.sum_sq_target) + EPS)),
            "bias": float(self.sum_diff / self.count),
            "p99_abs_error_mean": float(np.mean(self.frame_p99)) if self.frame_p99 else float("nan"),
            "p99_abs_error_max": float(np.max(self.frame_p99)) if self.frame_p99 else float("nan"),
            "max_abs_error": float(self.max_abs_error),
            "psnr_global": psnr_from_rmse(rmse, data_range),
            "psnr_mean": float(np.mean(self.frame_psnr)) if self.frame_psnr else float("nan"),
            "ssim_mean": float(np.mean(self.frame_ssim)) if self.frame_ssim else float("nan"),
            "peak_ratio_mean": float(np.mean(self.frame_peak_ratio))
            if self.frame_peak_ratio
            else float("nan"),
            "count": int(self.count),
        }


def relative_metrics(model: dict[str, float | int], baseline: dict[str, float | int]) -> dict[str, float]:
    return {
        "rmse_reduction_percent": float(
            100.0 * (float(baseline["rmse"]) - float(model["rmse"])) / (float(baseline["rmse"]) + EPS)
        ),
        "rfne_reduction_percent": float(
            100.0 * (float(baseline["rfne"]) - float(model["rfne"])) / (float(baseline["rfne"]) + EPS)
        ),
        "mae_reduction_percent": float(
            100.0 * (float(baseline["mae"]) - float(model["mae"])) / (float(baseline["mae"]) + EPS)
        ),
        "acc_delta": float(float(model["acc"]) - float(baseline["acc"])),
    }


def masked_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values_for_mask(arr, mask)


def update_accumulators(
    accumulators: dict[str, dict[str, MetricAccumulator]],
    source: str,
    scope: str,
    pred: np.ndarray,
    target: np.ndarray,
    frame_metrics: dict[str, float | int],
) -> None:
    accumulators.setdefault(source, {}).setdefault(scope, MetricAccumulator()).update(
        pred,
        target,
        frame_metrics,
    )


def evaluate(
    bases: list[str],
    dataset_root: Path,
    prediction_map: dict[str, Path],
    single_vz_map: dict[str, Path] | None,
    active_threshold: float,
    model_name: str,
    single_vz_name: str,
    baseline_label: str,
) -> dict[str, object]:
    accumulators: dict[str, dict[str, MetricAccumulator]] = {}
    component_accumulators: dict[str, dict[str, dict[str, MetricAccumulator]]] = {}
    per_frame: list[dict[str, object]] = []
    by_case: dict[str, dict[str, dict[str, MetricAccumulator] | int]] = {}

    missing_predictions = [base for base in bases if base not in prediction_map]
    if missing_predictions:
        raise FileNotFoundError(
            f"Missing {len(missing_predictions)} XYZ predictions, first: {missing_predictions[0]}"
        )

    has_single_vz = single_vz_map is not None
    if single_vz_map is not None:
        missing_single = [base for base in bases if base not in single_vz_map]
        if missing_single:
            print(
                f"Warning: single-Vz comparison disabled because {len(missing_single)} "
                f"files are missing, first: {missing_single[0]}"
            )
            has_single_vz = False

    for index, base in enumerate(bases, start=1):
        case, frame = parse_case_frame(base)
        hr = load_stack(dataset_root, "hr", COMPONENTS, base)
        interp = load_stack(dataset_root, "lr", INTERP_COMPONENTS, base)
        pred = load_xyz_prediction(prediction_map[base])
        mask_observed = load_mask(dataset_root, base)
        if pred.shape != hr.shape:
            raise ValueError(f"Prediction/HR shape mismatch for {base}: {pred.shape} vs {hr.shape}")
        if interp.shape != hr.shape:
            raise ValueError(f"Interp/HR shape mismatch for {base}: {interp.shape} vs {hr.shape}")

        mask_missing = ~mask_observed
        active_vector = np.linalg.norm(hr, axis=-1) > active_threshold
        active_missing_vector = mask_missing & active_vector
        masks = {
            "all": None,
            "observed": mask_observed,
            "missing": mask_missing,
            "active_missing": active_missing_vector,
        }

        frame_row: dict[str, object] = {"filename": base, "case": case, "frame": frame}
        case_row = by_case.setdefault(case, {"frames": 0})
        case_row["frames"] = int(case_row["frames"]) + 1

        for source, arr in ((baseline_label, interp), (model_name, pred)):
            frame_row[source] = {}
            for scope, mask in masks.items():
                metrics = compute_metrics(arr, hr, mask=mask)
                frame_row[source][scope] = metrics  # type: ignore[index]
                target_values = hr if mask is None else masked_values(hr, mask)
                pred_values = arr if mask is None else masked_values(arr, mask)
                update_accumulators(accumulators, source, scope, pred_values, target_values, metrics)
                case_accs = case_row.setdefault(source, {})  # type: ignore[assignment]
                if isinstance(case_accs, dict):
                    case_accs.setdefault(scope, MetricAccumulator()).update(
                        pred_values,
                        target_values,
                        metrics,
                    )

        frame_row["components"] = {}
        for component_index, component in enumerate(COMPONENTS):
            component_accumulators.setdefault(component, {})
            component_target = hr[..., component_index]
            component_active_missing = mask_missing & (np.abs(component_target) > active_threshold)
            component_masks = {
                "all": None,
                "observed": mask_observed,
                "missing": mask_missing,
                "active_missing": component_active_missing,
            }
            frame_row["components"][component] = {}  # type: ignore[index]
            for source, arr in (
                (baseline_label, interp[..., component_index]),
                (model_name, pred[..., component_index]),
            ):
                frame_row["components"][component][source] = {}  # type: ignore[index]
                component_accumulators[component].setdefault(source, {})
                for scope, mask in component_masks.items():
                    metrics = compute_metrics(
                        arr,
                        component_target,
                        mask=mask,
                        compute_ssim=(scope == "all"),
                    )
                    frame_row["components"][component][source][scope] = metrics  # type: ignore[index]
                    target_values = component_target if mask is None else masked_values(component_target, mask)
                    pred_values = arr if mask is None else masked_values(arr, mask)
                    component_accumulators[component][source].setdefault(scope, MetricAccumulator()).update(
                        pred_values,
                        target_values,
                        metrics,
                    )

        if has_single_vz and single_vz_map is not None:
            single_pred = load_field(single_vz_map[base])
            vz_target = hr[..., 2]
            single_active_missing = mask_missing & (np.abs(vz_target) > active_threshold)
            frame_row["components"]["Vz"][single_vz_name] = {}  # type: ignore[index]
            component_accumulators["Vz"].setdefault(single_vz_name, {})
            for scope, mask in {
                "all": None,
                "observed": mask_observed,
                "missing": mask_missing,
                "active_missing": single_active_missing,
            }.items():
                metrics = compute_metrics(
                    single_pred,
                    vz_target,
                    mask=mask,
                    compute_ssim=(scope == "all"),
                )
                frame_row["components"]["Vz"][single_vz_name][scope] = metrics  # type: ignore[index]
                target_values = vz_target if mask is None else masked_values(vz_target, mask)
                pred_values = single_pred if mask is None else masked_values(single_pred, mask)
                component_accumulators["Vz"][single_vz_name].setdefault(scope, MetricAccumulator()).update(
                    pred_values,
                    target_values,
                    metrics,
                )

        per_frame.append(frame_row)
        if index == 1 or index % 100 == 0 or index == len(bases):
            print(f"Evaluated {index}/{len(bases)}: {base}")

    global_summary = {
        source: {scope: acc.to_metrics() for scope, acc in scopes.items()}
        for source, scopes in accumulators.items()
    }
    component_summary = {
        component: {
            source: {scope: acc.to_metrics() for scope, acc in scopes.items()}
            for source, scopes in sources.items()
        }
        for component, sources in component_accumulators.items()
    }
    case_summary: dict[str, object] = {}
    for case, row in by_case.items():
        out: dict[str, object] = {"frames": int(row["frames"])}
        for source in (baseline_label, model_name):
            source_accs = row.get(source)
            if isinstance(source_accs, dict):
                out[source] = {
                    scope: acc.to_metrics()
                    for scope, acc in source_accs.items()
                    if isinstance(acc, MetricAccumulator)
                }
        case_summary[case] = out

    comparisons = {
        "xyz_vs_interp": {
            scope: relative_metrics(global_summary[model_name][scope], global_summary[baseline_label][scope])
            for scope in global_summary[model_name]
        },
        "components_vs_interp": {
            component: {
                scope: relative_metrics(
                    component_summary[component][model_name][scope],
                    component_summary[component][baseline_label][scope],
                )
                for scope in component_summary[component][model_name]
            }
            for component in COMPONENTS
        },
    }
    if has_single_vz:
        comparisons["vz_xyz_vs_single"] = {
            scope: relative_metrics(
                component_summary["Vz"][model_name][scope],
                component_summary["Vz"][single_vz_name][scope],
            )
            for scope in component_summary["Vz"][model_name]
            if scope in component_summary["Vz"][single_vz_name]
        }

    return {
        "dataset_root": str(dataset_root),
        "num_frames": len(bases),
        "components": list(COMPONENTS),
        "baseline_label": baseline_label,
        "model_name": model_name,
        "single_vz_name": single_vz_name if has_single_vz else None,
        "active_threshold": active_threshold,
        "global": global_summary,
        "by_component": component_summary,
        "by_case": case_summary,
        "comparisons": comparisons,
        "per_frame": per_frame,
    }


def write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=True)


def metric_fields() -> list[str]:
    return [
        "rmse",
        "mae",
        "rfne",
        "acc",
        "bias",
        "p99_abs_error_mean",
        "p99_abs_error_max",
        "max_abs_error",
        "psnr_global",
        "psnr_mean",
        "ssim_mean",
        "peak_ratio_mean",
        "count",
    ]


def write_summary_csv(summary: dict[str, object], path: Path) -> None:
    rows: list[list[object]] = []
    fields = metric_fields()
    for source, scopes in summary["global"].items():  # type: ignore[union-attr]
        for scope, metrics in scopes.items():
            rows.append(["XYZ", source, scope, *[metrics.get(field, "") for field in fields]])
    for component, sources in summary["by_component"].items():  # type: ignore[union-attr]
        for source, scopes in sources.items():
            for scope, metrics in scopes.items():
                rows.append([component, source, scope, *[metrics.get(field, "") for field in fields]])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "source", "scope", *fields])
        writer.writerows(rows)


def write_vz_compare_csv(summary: dict[str, object], path: Path) -> None:
    fields = metric_fields()
    vz_sources = summary["by_component"]["Vz"]  # type: ignore[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "scope", *fields])
        for source, scopes in vz_sources.items():
            for scope, metrics in scopes.items():
                writer.writerow([source, scope, *[metrics.get(field, "") for field in fields]])


def write_per_frame_csv(summary: dict[str, object], path: Path) -> None:
    baseline_label = str(summary["baseline_label"])
    model_name = str(summary["model_name"])
    single_vz_name = summary.get("single_vz_name")
    fields = [
        "rmse",
        "mae",
        "rfne",
        "acc",
        "p99_abs_error",
        "max_abs_error",
        "psnr",
        "ssim",
        "peak_ratio",
        "count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "case", "frame", "component", "source", "scope", *fields])
        for row in summary["per_frame"]:  # type: ignore[union-attr]
            for source in (baseline_label, model_name):
                for scope, metrics in row[source].items():  # type: ignore[index]
                    writer.writerow(
                        [
                            row["filename"],
                            row["case"],
                            row["frame"],
                            "XYZ",
                            source,
                            scope,
                            *[metrics.get(field, "") for field in fields],
                        ]
                    )
            for component, comp_sources in row["components"].items():  # type: ignore[index]
                source_names = [baseline_label, model_name]
                if component == "Vz" and single_vz_name:
                    source_names.append(str(single_vz_name))
                for source in source_names:
                    if source not in comp_sources:
                        continue
                    for scope, metrics in comp_sources[source].items():
                        writer.writerow(
                            [
                                row["filename"],
                                row["case"],
                                row["frame"],
                                component,
                                source,
                                scope,
                                *[metrics.get(field, "") for field in fields],
                            ]
                        )


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def metric_series(
    summary: dict[str, object],
    component: str,
    source: str,
    scope: str,
    metric: str,
) -> np.ndarray:
    values = []
    for row in summary["per_frame"]:  # type: ignore[union-attr]
        if component == "XYZ":
            values.append(float(row[source][scope][metric]))  # type: ignore[index]
        else:
            values.append(float(row["components"][component][source][scope][metric]))  # type: ignore[index]
    return np.asarray(values, dtype=np.float64)


def case_boundaries(summary: dict[str, object]) -> tuple[list[int], list[int], list[str]]:
    rows = summary["per_frame"]  # type: ignore[assignment]
    starts: list[int] = []
    centers: list[int] = []
    labels: list[str] = []
    current_case: str | None = None
    start = 0
    for idx, row in enumerate(rows):  # type: ignore[union-attr]
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
        centers.append((start + len(rows) - 1) // 2)  # type: ignore[arg-type]
        labels.append(current_case.replace("S1_", ""))
    return starts, centers, labels


def plot_per_frame(
    summary: dict[str, object],
    component: str,
    sources: list[str],
    out_path: Path,
    dpi: int,
) -> Path:
    x = np.arange(1, int(summary["num_frames"]) + 1)
    metrics = ("rmse", "rfne", "acc", "max_abs_error")
    fig, axes = plt.subplots(4, 1, figsize=(21, 15), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.15, hspace=0.22)
    boundaries, centers, labels = case_boundaries(summary)
    colors = {
        str(summary["baseline_label"]): "#d9822b",
        str(summary["model_name"]): "#1764ab",
        str(summary.get("single_vz_name")): "#287d3c",
    }

    for ax, metric in zip(axes, metrics):
        for source in sources:
            values = metric_series(summary, component, source, "all", metric)
            ax.plot(x, values, lw=1.25, label=source, color=colors.get(source), alpha=0.94)
        for boundary in boundaries:
            ax.axvline(boundary + 0.5, color="0.78", lw=0.8, alpha=0.65)
        ax.grid(alpha=0.25)
        ax.set_ylabel("ACC higher" if metric == "acc" else f"{metric.upper()} lower")

    axes[-1].set_xlabel("Test frame index, sorted by case and time")
    axes[-1].set_xticks([center + 1 for center in centers])
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=min(len(labels_), 4),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(f"Bohai {component} full-test metrics", fontsize=20)
    save_figure(fig, out_path, dpi)
    return out_path


def plot_case_bars(
    summary: dict[str, object],
    component: str,
    sources: list[str],
    out_path: Path,
    dpi: int,
) -> Path:
    cases = sorted({str(row["case"]) for row in summary["per_frame"]})  # type: ignore[union-attr]
    labels = [case.replace("S1_", "") for case in cases]
    metrics = ("rmse", "rfne", "acc")
    width = 0.78 / len(sources)
    offsets = np.linspace(-0.39 + width / 2.0, 0.39 - width / 2.0, len(sources))
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.23, wspace=0.22)

    grouped: dict[str, list[dict[str, object]]] = {case: [] for case in cases}
    for row in summary["per_frame"]:  # type: ignore[union-attr]
        grouped[str(row["case"])].append(row)  # type: ignore[arg-type]

    for ax, metric in zip(axes, metrics):
        for offset, source in zip(offsets, sources):
            values = []
            for case in cases:
                vals = []
                for row in grouped[case]:
                    if component == "XYZ":
                        vals.append(float(row[source]["all"][metric]))  # type: ignore[index]
                    else:
                        vals.append(float(row["components"][component][source]["all"][metric]))  # type: ignore[index]
                values.append(float(np.nanmean(vals)))
            ax.bar(x + offset, values, width, label=source)
        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Higher is better" if metric == "acc" else "Lower is better")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=min(len(labels_), 4),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(f"Bohai {component} per-case metrics", fontsize=20)
    save_figure(fig, out_path, dpi)
    return out_path


def finite_concat_abs(arrays: list[np.ndarray]) -> np.ndarray:
    values = []
    for arr in arrays:
        flat = np.ravel(np.abs(arr))
        values.append(flat[np.isfinite(flat)])
    return np.concatenate(values) if values else np.asarray([1.0], dtype=np.float64)


def robust_sym_limit(arrays: list[np.ndarray], percentile: float = 99.5) -> float:
    return max(float(np.percentile(finite_concat_abs(arrays), percentile)), 1e-6)


def robust_pos_limit(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
    return max(float(np.percentile(finite_concat_abs(arrays), percentile)), 1e-6)


def style_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    show_ylabel: bool,
    show_xlabel: bool = True,
) -> None:
    ax.set_xlabel(xlabel if show_xlabel else "")
    if not show_xlabel:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=10)
    ax.set_aspect("equal", adjustable="box")


def style_colorbar(cbar: matplotlib.colorbar.Colorbar, title: str) -> None:
    cbar.set_label("")
    cbar.ax.set_title(title, fontsize=10, pad=8)
    cbar.ax.tick_params(labelsize=9)


def plot_component_frame(
    base: str,
    dataset_root: Path,
    prediction_map: dict[str, Path],
    single_vz_map: dict[str, Path] | None,
    component: str,
    out_dir: Path,
    extent: list[float],
    xlabel: str,
    ylabel: str,
    model_name: str,
    single_vz_name: str,
    baseline_label: str,
    dpi: int,
) -> Path:
    if component not in COMPONENTS:
        raise ValueError(f"Unsupported component: {component}")
    component_index = COMPONENTS.index(component)
    hr = load_field(dataset_root / "test" / "hr" / component / f"{base}.npy")
    sparse = load_field(dataset_root / "test" / "lr" / f"{component}_sparse" / f"{base}.npy")
    interp = load_field(dataset_root / "test" / "lr" / f"{component}_interp" / f"{base}.npy")
    pred = load_xyz_prediction(prediction_map[base])[..., component_index]

    field_arrays = [interp, pred, hr]
    field_lim = robust_sym_limit(field_arrays, percentile=99.5)

    xyz_err = np.abs(pred - hr)
    interp_err = np.abs(interp - hr)
    error_arrays = [interp_err, xyz_err]
    err_lim = robust_pos_limit(error_arrays, percentile=99.0)

    field_items = [
        (f"{component} sparse observation\n25% regular grid, zero-filled", sparse),
        (f"{component} linear interpolation\nsparse samples -> full field", interp),
        (f"{component} ResShift reconstruction\n5-frame XYZ input", pred),
        (f"{component} HR reference\nfull-field target", hr),
    ]
    error_items = [
        (f"{component} interpolation error\n|linear interpolation - HR|", interp_err),
        (f"{component} ResShift error\n|ResShift reconstruction - HR|", xyz_err),
    ]

    fig = plt.figure(figsize=(23.5, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        9,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.045, 0.10, 0.045],
        left=0.035,
        right=0.98,
        top=0.88,
        bottom=0.14,
        wspace=0.055,
    )

    field_im = None
    for col, (title, data) in enumerate(field_items):
        ax = fig.add_subplot(gs[0, col])
        field_im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="seismic",
            vmin=-field_lim,
            vmax=field_lim,
            interpolation="nearest",
        )
        ax.set_title(title, pad=6, fontsize=12)
        style_axes(ax, xlabel, ylabel, show_ylabel=(col == 0), show_xlabel=True)

    err_im = None
    error_cols = (4, 5)
    for col, (title, data) in zip(error_cols, error_items):
        ax = fig.add_subplot(gs[0, col])
        err_im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=err_lim,
            interpolation="nearest",
        )
        ax.set_title(title, pad=6, fontsize=12)
        style_axes(ax, xlabel, ylabel, show_ylabel=False, show_xlabel=True)

    if field_im is not None:
        cax = fig.add_subplot(gs[0, 6])
        cbar = fig.colorbar(field_im, cax=cax)
        style_colorbar(cbar, f"{component}\nvalue")
    if err_im is not None:
        cax = fig.add_subplot(gs[0, 8])
        cbar = fig.colorbar(err_im, cax=cax)
        style_colorbar(cbar, f"{component}\nabs. err.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}.png"
    save_figure(fig, out_path, dpi)
    return out_path


def plot_magnitude_frame(
    base: str,
    dataset_root: Path,
    prediction_map: dict[str, Path],
    out_dir: Path,
    extent: list[float],
    xlabel: str,
    ylabel: str,
    baseline_label: str,
    dpi: int,
) -> Path:
    hr = load_stack(dataset_root, "hr", COMPONENTS, base)
    sparse = load_stack(dataset_root, "lr", SPARSE_COMPONENTS, base)
    interp = load_stack(dataset_root, "lr", INTERP_COMPONENTS, base)
    pred = load_xyz_prediction(prediction_map[base])
    hr_mag = np.linalg.norm(hr, axis=-1)
    sparse_mag = np.linalg.norm(sparse, axis=-1)
    interp_mag = np.linalg.norm(interp, axis=-1)
    pred_mag = np.linalg.norm(pred, axis=-1)
    sr_err = np.abs(pred_mag - hr_mag)
    interp_err = np.abs(interp_mag - hr_mag)
    field_lim = max(
        float(np.percentile(finite_concat_abs([sparse_mag, hr_mag, interp_mag, pred_mag]), 99.5)),
        1e-6,
    )
    err_lim = robust_pos_limit([sr_err, interp_err], percentile=99.0)

    error_items = [
        ("|V| interpolation error\nabs(|V| interp - |V| HR)", interp_err),
        ("|V| ResShift error\nabs(|V| ResShift - |V| HR)", sr_err),
    ]
    field_items = [
        ("|V| sparse observation\n25% regular grid, zero-filled", sparse_mag),
        ("|V| linear interpolation\nfrom interpolated XYZ", interp_mag),
        ("|V| ResShift reconstruction\nfrom predicted XYZ", pred_mag),
        ("|V| HR reference\nfrom full-field HR XYZ", hr_mag),
    ]
    fig = plt.figure(figsize=(23.5, 5.6), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        9,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.045, 0.10, 0.045],
        left=0.035,
        right=0.98,
        top=0.88,
        bottom=0.14,
        wspace=0.055,
    )

    field_im = None
    for col, (title, data) in enumerate(field_items):
        ax = fig.add_subplot(gs[0, col])
        field_im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="Reds",
            vmin=0.0,
            vmax=field_lim,
            interpolation="nearest",
        )
        ax.set_title(title, pad=6, fontsize=12)
        style_axes(ax, xlabel, ylabel, show_ylabel=(col == 0), show_xlabel=True)

    err_im = None
    error_cols = (4, 5)
    for col, (title, data) in zip(error_cols, error_items):
        ax = fig.add_subplot(gs[0, col])
        err_im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=err_lim,
            interpolation="nearest",
        )
        ax.set_title(title, pad=6, fontsize=12)
        style_axes(ax, xlabel, ylabel, show_ylabel=False, show_xlabel=True)

    if field_im is not None:
        cax = fig.add_subplot(gs[0, 6])
        cbar = fig.colorbar(field_im, cax=cax)
        style_colorbar(cbar, "|V|\nvalue")
    if err_im is not None:
        cax = fig.add_subplot(gs[0, 8])
        cbar = fig.colorbar(err_im, cax=cax)
        style_colorbar(cbar, "|V|\nabs. err.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}.png"
    save_figure(fig, out_path, dpi)
    return out_path


def selected_bases(case: str, frame_start: int, frame_end: int, frame_step: int) -> list[str]:
    if frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    return [f"{case}_{frame:06d}" for frame in range(frame_start, frame_end + 1, frame_step)]


def generate_figures(
    summary: dict[str, object],
    dataset_root: Path,
    prediction_map: dict[str, Path],
    single_vz_map: dict[str, Path] | None,
    out_dir: Path,
    case: str,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    dpi: int,
) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 11,
        }
    )
    baseline_label = str(summary["baseline_label"])
    model_name = str(summary["model_name"])
    single_vz_name = str(summary.get("single_vz_name") or "Single-Vz")
    sources_xyz = [baseline_label, model_name]
    component_sources = {
        "Vx": [baseline_label, model_name],
        "Vy": [baseline_label, model_name],
        "Vz": [baseline_label, model_name],
    }
    if summary.get("single_vz_name"):
        component_sources["Vz"].append(single_vz_name)

    generated: list[Path] = []
    generated.append(plot_per_frame(summary, "XYZ", sources_xyz, out_dir / "comparison_per_frame_metrics_xyz.png", dpi))
    generated.append(plot_case_bars(summary, "XYZ", sources_xyz, out_dir / "comparison_case_metrics_xyz.png", dpi))
    for component in COMPONENTS:
        component_key = component.lower()
        generated.append(
            plot_per_frame(
                summary,
                component,
                component_sources[component],
                out_dir / f"comparison_per_frame_metrics_{component_key}.png",
                dpi,
            )
        )
        generated.append(
            plot_case_bars(
                summary,
                component,
                component_sources[component],
                out_dir / f"comparison_case_metrics_{component_key}.png",
                dpi,
            )
        )

    extent, xlabel, ylabel = load_coordinate_extent(dataset_root)
    for base in selected_bases(case, frame_start, frame_end, frame_step):
        if base not in prediction_map:
            print(f"Warning: skip missing visualization frame {base}")
            continue
        for component in COMPONENTS:
            generated.append(
                plot_component_frame(
                    base,
                    dataset_root,
                    prediction_map,
                    single_vz_map,
                    component,
                    out_dir / f"figs_every10_{component.lower()}",
                    extent,
                    xlabel,
                    ylabel,
                    model_name,
                    single_vz_name,
                    baseline_label,
                    dpi,
                )
            )
        generated.append(
            plot_magnitude_frame(
                base,
                dataset_root,
                prediction_map,
                out_dir / "figs_every10_magnitude",
                extent,
                xlabel,
                ylabel,
                baseline_label,
                dpi,
            )
        )
    return generated


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    prediction_dir = args.prediction_dir.resolve()
    out_dir = args.out_dir.resolve()
    bases = resolve_expected_bases(dataset_root, args.max_frames)
    prediction_map = resolve_prediction_map(prediction_dir)

    single_vz_map = None
    if args.single_vz_prediction_dir and args.single_vz_prediction_dir.exists():
        single_vz_map = resolve_prediction_map(args.single_vz_prediction_dir.resolve())

    summary = evaluate(
        bases,
        dataset_root,
        prediction_map,
        single_vz_map,
        args.active_threshold,
        args.model_name,
        args.single_vz_name,
        args.baseline_label,
    )
    summary["prediction_dir"] = str(prediction_dir)
    if single_vz_map is not None:
        summary["single_vz_prediction_dir"] = str(args.single_vz_prediction_dir.resolve())

    write_json(summary, out_dir / "xyz_metrics.json")
    write_summary_csv(summary, out_dir / "model_summary_xyz.csv")
    write_vz_compare_csv(summary, out_dir / "model_summary_vz_compare.csv")
    write_per_frame_csv(summary, out_dir / "per_frame_metrics.csv")

    generated: list[Path] = []
    if not args.no_figures:
        generated = generate_figures(
            summary,
            dataset_root,
            prediction_map,
            single_vz_map,
            out_dir,
            args.case,
            args.frame_start,
            args.frame_end,
            args.frame_step,
            args.dpi,
        )

    print(f"Saved metrics JSON: {out_dir / 'xyz_metrics.json'}")
    print(f"Saved summary CSV: {out_dir / 'model_summary_xyz.csv'}")
    print(f"Saved Vz comparison CSV: {out_dir / 'model_summary_vz_compare.csv'}")
    print(f"Saved per-frame CSV: {out_dir / 'per_frame_metrics.csv'}")
    if generated:
        print(f"Saved {len(generated)} figures under: {out_dir}")
        for path in generated[:12]:
            print(f"  {path}")
        if len(generated) > 12:
            print(f"  ... {len(generated) - 12} more")


if __name__ == "__main__":
    main()

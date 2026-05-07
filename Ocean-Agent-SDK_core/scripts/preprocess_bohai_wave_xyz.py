#!/usr/bin/env python3
"""
Convert Bohai Sea AVS wave snapshots into the OceanNPY super-resolution layout.

Output layout:
    process_data/
    ├── train/hr/Vx/*.npy
    ├── train/hr/Vy/*.npy
    ├── train/hr/Vz/*.npy
    ├── train/lr/Vx/*.npy
    ├── valid/...
    ├── test/...
    ├── static_variables/hr/{00_lon_rho,10_lat_rho,20_z,30_mask_rho}.npy
    ├── static_variables/lr/{00_lon_rho,10_lat_rho,20_z,30_mask_rho}.npy
    ├── var_names.json
    ├── data_stats.json
    └── preprocess_manifest.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np


SOURCE_SUBDIR = "To_ZGT_wave_movie"
SOURCE_COMPONENTS = ("X", "Y", "Z")
TARGET_VARS = {"X": "Vx", "Y": "Vy", "Z": "Vz"}
SPARSE_MASK_METHOD = "sparse_mask"

_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass(frozen=True)
class Geometry:
    node_count: int
    element_count: int
    scalar_skip_rows: int
    grid_shape: tuple[int, int]
    hr_shape: tuple[int, int]
    lr_shape: tuple[int, int]
    crop: dict[str, int]
    flat_indices: np.ndarray
    x_grid_hr: np.ndarray
    y_grid_hr: np.ndarray
    z_grid_hr: np.ndarray
    x_grid_lr: np.ndarray
    y_grid_lr: np.ndarray
    z_grid_lr: np.ndarray


@dataclass(frozen=True)
class Sample:
    split: str
    case_name: str
    frame_number: int
    frame_stem: str
    output_stem: str
    component_paths: dict[str, str]


class StatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        finite64 = finite.astype(np.float64, copy=False)
        self.count += int(finite64.size)
        self.sum += float(finite64.sum())
        self.sumsq += float(np.square(finite64).sum())
        self.min = min(self.min, float(finite64.min()))
        self.max = max(self.max, float(finite64.max()))

    def merge(self, other: dict[str, float | int]) -> None:
        count = int(other.get("count", 0))
        if count == 0:
            return
        self.count += count
        self.sum += float(other["sum"])
        self.sumsq += float(other["sumsq"])
        self.min = min(self.min, float(other["min"]))
        self.max = max(self.max, float(other["max"]))

    def to_json(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.sum / self.count
        variance = max(self.sumsq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.min,
            "max": self.max,
        }


def natural_sort_key(text: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def frame_number_from_path(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot find frame number in {path.name}")
    return int(match.group(1))


def resolve_source_root(path: Path) -> Path:
    if path.name == SOURCE_SUBDIR:
        return path
    candidate = path / SOURCE_SUBDIR
    if candidate.is_dir():
        return candidate
    return path


def parse_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", errors="replace") as file_obj:
        first_line = file_obj.readline().split()
    if len(first_line) < 2:
        raise ValueError(f"Invalid AVS header in {path}")
    return int(first_line[0]), int(first_line[1])


def center_crop_bounds(height: int, width: int, scale: int) -> dict[str, int]:
    crop_h = (height // scale) * scale
    crop_w = (width // scale) * scale
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(
            f"Grid shape {(height, width)} is too small for scale={scale}"
        )
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return {
        "top": top,
        "bottom": top + crop_h,
        "left": left,
        "right": left + crop_w,
        "removed_top": top,
        "removed_bottom": height - (top + crop_h),
        "removed_left": left,
        "removed_right": width - (left + crop_w),
    }


def block_mean_2d(array: np.ndarray, scale: int) -> np.ndarray:
    height, width = array.shape
    if height % scale != 0 or width % scale != 0:
        raise ValueError(f"Array shape {array.shape} is not divisible by scale={scale}")
    reshaped = array.reshape(height // scale, scale, width // scale, scale)
    return reshaped.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def gaussian_kernel_1d(sigma: float, truncate: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"anti-alias sigma must be > 0, got {sigma}")
    if truncate <= 0:
        raise ValueError(f"anti-alias truncate must be > 0, got {truncate}")
    radius = max(1, int(truncate * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_reflect_1d(
    array: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    radius = kernel.size // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=kernel.size,
        axis=axis,
    )
    return np.tensordot(windows, kernel.astype(np.float64), axes=([-1], [0]))


def gaussian_blur_2d(
    array: np.ndarray,
    sigma: float,
    truncate: float,
) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma, truncate)
    blurred = convolve_reflect_1d(array.astype(np.float64, copy=False), kernel, axis=0)
    blurred = convolve_reflect_1d(blurred, kernel, axis=1)
    return blurred.astype(np.float32)


def point_sample_2d(array: np.ndarray, scale: int, offset: int) -> np.ndarray:
    height, width = array.shape
    if height % scale != 0 or width % scale != 0:
        raise ValueError(f"Array shape {array.shape} is not divisible by scale={scale}")
    if offset < 0 or offset >= scale:
        raise ValueError(f"point offset must satisfy 0 <= offset < scale, got {offset}")
    return array[offset::scale, offset::scale].astype(np.float32, copy=True)


def anti_alias_point_sample_2d(
    array: np.ndarray,
    scale: int,
    offset: int,
    sigma: float,
    truncate: float,
) -> np.ndarray:
    return point_sample_2d(
        gaussian_blur_2d(array, sigma=sigma, truncate=truncate),
        scale,
        offset,
    )


def sparse_sample_indices(length: int, scale: int, offset: int) -> np.ndarray:
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if offset < 0 or offset >= scale:
        raise ValueError(f"point offset must satisfy 0 <= offset < scale, got {offset}")
    indices = np.arange(offset, length, scale, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(
            f"No sparse samples for length={length}, scale={scale}, offset={offset}"
        )
    return indices


def sparse_observation_mask_2d(
    shape: tuple[int, int],
    scale: int,
    offset: int,
) -> np.ndarray:
    height, width = shape
    rows = sparse_sample_indices(height, scale, offset)
    cols = sparse_sample_indices(width, scale, offset)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[np.ix_(rows, cols)] = 1.0
    return mask


def sparse_zero_fill_2d(
    array: np.ndarray,
    scale: int,
    offset: int,
) -> np.ndarray:
    mask = sparse_observation_mask_2d(array.shape, scale, offset)
    return (array.astype(np.float32, copy=False) * mask).astype(np.float32, copy=False)


def sparse_linear_interpolate_2d(
    array: np.ndarray,
    scale: int,
    offset: int,
) -> np.ndarray:
    """Interpolate regularly sparse observations back to the full tensor grid."""
    height, width = array.shape
    rows = sparse_sample_indices(height, scale, offset)
    cols = sparse_sample_indices(width, scale, offset)
    sampled = array[np.ix_(rows, cols)].astype(np.float64, copy=False)

    full_cols = np.arange(width, dtype=np.float64)
    row_interp = np.empty((rows.size, width), dtype=np.float64)
    for row_index in range(rows.size):
        row_interp[row_index] = np.interp(full_cols, cols, sampled[row_index])

    full_rows = np.arange(height, dtype=np.float64)
    full_interp = np.empty((height, width), dtype=np.float64)
    for col_index in range(width):
        full_interp[:, col_index] = np.interp(full_rows, rows, row_interp[:, col_index])

    return full_interp.astype(np.float32)


def downsample_2d(
    array: np.ndarray,
    scale: int,
    method: str,
    point_offset: int,
    anti_alias_sigma: float | None = None,
    anti_alias_truncate: float = 3.0,
) -> np.ndarray:
    if method == "block_mean":
        return block_mean_2d(array, scale)
    if method == "point":
        return point_sample_2d(array, scale, point_offset)
    if method == "anti_alias_point":
        sigma = anti_alias_sigma if anti_alias_sigma is not None else scale / 2.0
        return anti_alias_point_sample_2d(
            array,
            scale,
            point_offset,
            sigma=sigma,
            truncate=anti_alias_truncate,
        )
    raise ValueError(f"Unsupported downsample method: {method}")


def hr_var_names() -> list[str]:
    return list(TARGET_VARS.values())


def lr_var_names(downsample_method: str) -> list[str]:
    if downsample_method == SPARSE_MASK_METHOD:
        names: list[str] = []
        for target_var in TARGET_VARS.values():
            names.extend([f"{target_var}_sparse", f"{target_var}_interp"])
        names.append("mask_observed")
        return names
    return list(TARGET_VARS.values())


def read_geometry(
    sample_file: Path,
    scale: int,
    downsample_method: str,
    point_offset: int,
    anti_alias_sigma: float | None,
    anti_alias_truncate: float,
) -> Geometry:
    node_count, element_count = parse_header(sample_file)
    nodes = np.loadtxt(
        sample_file,
        skiprows=1,
        max_rows=node_count,
        usecols=(1, 2, 3),
        dtype=np.float64,
    )
    if nodes.shape != (node_count, 3):
        raise ValueError(f"Unexpected node table shape in {sample_file}: {nodes.shape}")

    x_values = nodes[:, 0]
    y_values = nodes[:, 1]
    z_values = nodes[:, 2]
    unique_x = np.unique(x_values)
    unique_y = np.unique(y_values)
    height = int(unique_y.size)
    width = int(unique_x.size)
    if height * width != node_count:
        raise ValueError(
            f"Nodes do not form a complete tensor grid: "
            f"{height} * {width} != {node_count}"
        )

    x_indices = np.searchsorted(unique_x, x_values)
    y_indices = np.searchsorted(unique_y, y_values)
    flat_indices = (y_indices * width + x_indices).astype(np.int64)

    x_grid = np.empty((height, width), dtype=np.float32)
    y_grid = np.empty((height, width), dtype=np.float32)
    z_grid = np.empty((height, width), dtype=np.float32)
    x_grid.reshape(-1)[flat_indices] = x_values.astype(np.float32)
    y_grid.reshape(-1)[flat_indices] = y_values.astype(np.float32)
    z_grid.reshape(-1)[flat_indices] = z_values.astype(np.float32)

    crop = center_crop_bounds(height, width, scale)
    row_slice = slice(crop["top"], crop["bottom"])
    col_slice = slice(crop["left"], crop["right"])
    x_grid_hr = x_grid[row_slice, col_slice].copy()
    y_grid_hr = y_grid[row_slice, col_slice].copy()
    z_grid_hr = z_grid[row_slice, col_slice].copy()

    if downsample_method == SPARSE_MASK_METHOD:
        x_grid_lr = x_grid_hr.copy()
        y_grid_lr = y_grid_hr.copy()
        z_grid_lr = z_grid_hr.copy()
    else:
        x_grid_lr = downsample_2d(
            x_grid_hr,
            scale,
            downsample_method,
            point_offset,
            anti_alias_sigma,
            anti_alias_truncate,
        )
        y_grid_lr = downsample_2d(
            y_grid_hr,
            scale,
            downsample_method,
            point_offset,
            anti_alias_sigma,
            anti_alias_truncate,
        )
        z_grid_lr = downsample_2d(
            z_grid_hr,
            scale,
            downsample_method,
            point_offset,
            anti_alias_sigma,
            anti_alias_truncate,
        )

    scalar_skip_rows = 1 + node_count + element_count + 2
    return Geometry(
        node_count=node_count,
        element_count=element_count,
        scalar_skip_rows=scalar_skip_rows,
        grid_shape=(height, width),
        hr_shape=tuple(x_grid_hr.shape),
        lr_shape=tuple(x_grid_lr.shape),
        crop=crop,
        flat_indices=flat_indices,
        x_grid_hr=x_grid_hr,
        y_grid_hr=y_grid_hr,
        z_grid_hr=z_grid_hr,
        x_grid_lr=x_grid_lr,
        y_grid_lr=y_grid_lr,
        z_grid_lr=z_grid_lr,
    )


def read_scalar_grid(path: Path | str) -> np.ndarray:
    context = _WORKER_CONTEXT
    values = np.loadtxt(
        path,
        skiprows=context["scalar_skip_rows"],
        max_rows=context["node_count"],
        usecols=1,
        dtype=np.float32,
    )
    if values.shape[0] != context["node_count"]:
        raise ValueError(f"Unexpected scalar length in {path}: {values.shape[0]}")

    height, width = context["grid_shape"]
    grid = np.empty((height, width), dtype=np.float32)
    grid.reshape(-1)[context["flat_indices"]] = values
    crop = context["crop"]
    return grid[crop["top"] : crop["bottom"], crop["left"] : crop["right"]].copy()


def discover_samples(
    source_root: Path,
    train_ratio: float,
    valid_ratio: float,
    max_cases: int | None,
    max_frames: int | None,
) -> tuple[list[Sample], list[str], dict[str, Any]]:
    case_dirs = sorted(
        [path for path in source_root.iterdir() if path.is_dir()],
        key=lambda path: natural_sort_key(path.name),
    )
    complete_cases: list[tuple[Path, list[tuple[int, str, dict[str, str]]]]] = []
    skipped_cases: dict[str, str] = {}

    for case_dir in case_dirs:
        component_frames: dict[str, dict[int, Path]] = {}
        missing_component = False
        for component in SOURCE_COMPONENTS:
            component_dir = case_dir / component
            if not component_dir.is_dir():
                missing_component = True
                break
            frames = {}
            for path in component_dir.glob("AVS_movie_*.inp"):
                frames[frame_number_from_path(path)] = path
            if not frames:
                missing_component = True
                break
            component_frames[component] = frames

        if missing_component:
            skipped_cases[case_dir.name] = "missing X/Y/Z component files"
            continue

        common_frames = sorted(
            set(component_frames["X"])
            & set(component_frames["Y"])
            & set(component_frames["Z"])
        )
        if max_frames is not None:
            common_frames = common_frames[:max_frames]
        if not common_frames:
            skipped_cases[case_dir.name] = "no common frames across X/Y/Z"
            continue

        frame_records = []
        for frame_number in common_frames:
            paths = {
                component: str(component_frames[component][frame_number])
                for component in SOURCE_COMPONENTS
            }
            frame_stem = component_frames["X"][frame_number].stem
            frame_records.append((frame_number, frame_stem, paths))
        complete_cases.append((case_dir, frame_records))

    if max_cases is not None:
        complete_cases = complete_cases[:max_cases]

    if not complete_cases:
        raise FileNotFoundError(f"No complete X/Y/Z cases found under {source_root}")

    n_cases = len(complete_cases)
    train_count = int(n_cases * train_ratio)
    valid_count = int(n_cases * valid_ratio)
    if n_cases >= 3:
        train_count = max(1, min(train_count, n_cases - 2))
        valid_count = max(1, min(valid_count, n_cases - train_count - 1))
    else:
        train_count = max(1, n_cases - 1)
        valid_count = 0
    split_by_case: dict[str, str] = {}
    samples: list[Sample] = []
    split_case_names: dict[str, list[str]] = {"train": [], "valid": [], "test": []}

    for case_index, (case_dir, frame_records) in enumerate(complete_cases):
        if case_index < train_count:
            split = "train"
        elif case_index < train_count + valid_count:
            split = "valid"
        else:
            split = "test"
        split_by_case[case_dir.name] = split
        split_case_names[split].append(case_dir.name)

        safe_case_name = case_dir.name.replace(".", "_")
        for frame_number, frame_stem, paths in frame_records:
            samples.append(
                Sample(
                    split=split,
                    case_name=case_dir.name,
                    frame_number=frame_number,
                    frame_stem=frame_stem,
                    output_stem=f"{safe_case_name}_{frame_number:06d}",
                    component_paths=paths,
                )
            )

    summary = {
        "source_case_count": len(case_dirs),
        "complete_case_count": len(complete_cases),
        "skipped_cases": skipped_cases,
        "split_cases": split_case_names,
        "samples_per_split": {
            split: sum(1 for sample in samples if sample.split == split)
            for split in ("train", "valid", "test")
        },
    }
    return samples, [case_dir.name for case_dir, _ in complete_cases], summary


def prepare_output_dirs(
    output_root: Path,
    overwrite: bool,
    downsample_method: str,
) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    for split in ("train", "valid", "test"):
        for var_name in hr_var_names():
            (output_root / split / "hr" / var_name).mkdir(parents=True, exist_ok=True)
        for var_name in lr_var_names(downsample_method):
            (output_root / split / "lr" / var_name).mkdir(parents=True, exist_ok=True)
    (output_root / "static_variables" / "hr").mkdir(parents=True, exist_ok=True)
    (output_root / "static_variables" / "lr").mkdir(parents=True, exist_ok=True)


def save_static_variables(output_root: Path, geometry: Geometry) -> None:
    static_hr = output_root / "static_variables" / "hr"
    static_lr = output_root / "static_variables" / "lr"
    mask_hr = np.ones(geometry.hr_shape, dtype=np.float32)
    mask_lr = np.ones(geometry.lr_shape, dtype=np.float32)

    np.save(static_hr / "00_lon_rho.npy", geometry.x_grid_hr)
    np.save(static_hr / "10_lat_rho.npy", geometry.y_grid_hr)
    np.save(static_hr / "20_z.npy", geometry.z_grid_hr)
    np.save(static_hr / "30_mask_rho.npy", mask_hr)

    np.save(static_lr / "00_lon_rho.npy", geometry.x_grid_lr)
    np.save(static_lr / "10_lat_rho.npy", geometry.y_grid_lr)
    np.save(static_lr / "20_z.npy", geometry.z_grid_lr)
    np.save(static_lr / "30_mask_rho.npy", mask_lr)


def init_worker(
    node_count: int,
    scalar_skip_rows: int,
    grid_shape: tuple[int, int],
    crop: dict[str, int],
    flat_indices: np.ndarray,
    scale: int,
    downsample_method: str,
    point_offset: int,
    anti_alias_sigma: float | None,
    anti_alias_truncate: float,
    output_root: str,
) -> None:
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "node_count": node_count,
            "scalar_skip_rows": scalar_skip_rows,
            "grid_shape": grid_shape,
            "crop": crop,
            "flat_indices": flat_indices,
            "scale": scale,
            "downsample_method": downsample_method,
            "point_offset": point_offset,
            "anti_alias_sigma": anti_alias_sigma,
            "anti_alias_truncate": anti_alias_truncate,
            "output_root": output_root,
        }
    )


def stats_dict(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": 0.0, "max": 0.0}
    finite64 = finite.astype(np.float64, copy=False)
    return {
        "count": int(finite64.size),
        "sum": float(finite64.sum()),
        "sumsq": float(np.square(finite64).sum()),
        "min": float(finite64.min()),
        "max": float(finite64.max()),
    }


def process_sample(sample: Sample) -> dict[str, Any]:
    context = _WORKER_CONTEXT
    output_root = Path(context["output_root"])
    scale = int(context["scale"])
    downsample_method = str(context["downsample_method"])
    point_offset = int(context["point_offset"])
    anti_alias_sigma = context["anti_alias_sigma"]
    anti_alias_truncate = float(context["anti_alias_truncate"])
    sample_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    sparse_mask: np.ndarray | None = None

    for source_component, target_var in TARGET_VARS.items():
        hr = read_scalar_grid(sample.component_paths[source_component]).astype(
            np.float32, copy=False
        )

        hr_path = output_root / sample.split / "hr" / target_var / f"{sample.output_stem}.npy"
        np.save(hr_path, hr)
        sample_stats[target_var] = {
            "hr": stats_dict(hr),
        }

        if downsample_method == SPARSE_MASK_METHOD:
            sparse = sparse_zero_fill_2d(hr, scale, point_offset)
            interp = sparse_linear_interpolate_2d(hr, scale, point_offset)
            sparse_name = f"{target_var}_sparse"
            interp_name = f"{target_var}_interp"
            sparse_path = (
                output_root
                / sample.split
                / "lr"
                / sparse_name
                / f"{sample.output_stem}.npy"
            )
            interp_path = (
                output_root
                / sample.split
                / "lr"
                / interp_name
                / f"{sample.output_stem}.npy"
            )
            np.save(sparse_path, sparse)
            np.save(interp_path, interp)
            sample_stats[sparse_name] = {"lr": stats_dict(sparse)}
            sample_stats[interp_name] = {"lr": stats_dict(interp)}

            if sparse_mask is None:
                sparse_mask = sparse_observation_mask_2d(hr.shape, scale, point_offset)
                mask_path = (
                    output_root
                    / sample.split
                    / "lr"
                    / "mask_observed"
                    / f"{sample.output_stem}.npy"
                )
                np.save(mask_path, sparse_mask)
                sample_stats["mask_observed"] = {"lr": stats_dict(sparse_mask)}
            continue

        lr = downsample_2d(
            hr,
            scale,
            downsample_method,
            point_offset,
            anti_alias_sigma,
            anti_alias_truncate,
        )
        lr_path = output_root / sample.split / "lr" / target_var / f"{sample.output_stem}.npy"
        np.save(lr_path, lr)
        sample_stats[target_var]["lr"] = stats_dict(lr)

    return {
        "split": sample.split,
        "case_name": sample.case_name,
        "frame_number": sample.frame_number,
        "output_stem": sample.output_stem,
        "stats": sample_stats,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def process_all_samples(
    samples: list[Sample],
    geometry: Geometry,
    output_root: Path,
    scale: int,
    downsample_method: str,
    point_offset: int,
    anti_alias_sigma: float | None,
    anti_alias_truncate: float,
    workers: int,
) -> dict[str, Any]:
    stats: dict[str, dict[str, StatsAccumulator]] = {}
    for var_name in hr_var_names():
        stats.setdefault(var_name, {})["hr"] = StatsAccumulator()
    for var_name in lr_var_names(downsample_method):
        stats.setdefault(var_name, {})["lr"] = StatsAccumulator()
    completed = 0
    progress_interval = max(1, len(samples) // 20)

    def merge_result(result: dict[str, Any]) -> None:
        for var_name, resolution_stats in result["stats"].items():
            for resolution, partial in resolution_stats.items():
                stats[var_name][resolution].merge(partial)

    if workers <= 1:
        init_worker(
            geometry.node_count,
            geometry.scalar_skip_rows,
            geometry.grid_shape,
            geometry.crop,
            geometry.flat_indices,
            scale,
            downsample_method,
            point_offset,
            anti_alias_sigma,
            anti_alias_truncate,
            str(output_root),
        )
        for sample in samples:
            merge_result(process_sample(sample))
            completed += 1
            if completed == 1 or completed % progress_interval == 0 or completed == len(samples):
                print(f"[progress] {completed}/{len(samples)} samples", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(
                geometry.node_count,
                geometry.scalar_skip_rows,
                geometry.grid_shape,
                geometry.crop,
                geometry.flat_indices,
                scale,
                downsample_method,
                point_offset,
                anti_alias_sigma,
                anti_alias_truncate,
                str(output_root),
            ),
        ) as executor:
            futures = [executor.submit(process_sample, sample) for sample in samples]
            for future in as_completed(futures):
                merge_result(future.result())
                completed += 1
                if completed == 1 or completed % progress_interval == 0 or completed == len(samples):
                    print(f"[progress] {completed}/{len(samples)} samples", flush=True)

    return {
        var_name: {
            resolution: accumulator.to_json()
            for resolution, accumulator in resolution_stats.items()
        }
        for var_name, resolution_stats in stats.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Bohai Sea X/Y/Z AVS wave snapshots for OceanNPY training."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/Bohai_Sea/To_ZGT_wave_movie"),
        help="Source directory, either /data/Bohai_Sea or /data/Bohai_Sea/To_ZGT_wave_movie.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/Bohai_Sea/process_data"),
        help="Output dataset root in OceanNPY layout.",
    )
    parser.add_argument("--scale", type=int, default=4, help="LR downsample scale.")
    parser.add_argument(
        "--downsample-method",
        choices=("block_mean", "point", "anti_alias_point", SPARSE_MASK_METHOD),
        default="block_mean",
        help=(
            "LR generation method. block_mean averages each scale x scale block; "
            "point samples one fixed grid point from each block; "
            "anti_alias_point applies Gaussian anti-alias filtering before point sampling; "
            "sparse_mask keeps the HR grid and writes sparse/interpolated observations "
            "plus mask_observed for missing-trace reconstruction."
        ),
    )
    parser.add_argument(
        "--point-offset",
        type=int,
        help=(
            "Offset used by point-style downsampling. Defaults to scale//2, "
            "so scale=4 uses hr[2::4, 2::4] after optional anti-alias filtering."
        ),
    )
    parser.add_argument(
        "--anti-alias-sigma",
        type=float,
        help="Gaussian sigma for --downsample-method=anti_alias_point. Defaults to scale/2.",
    )
    parser.add_argument(
        "--anti-alias-truncate",
        type=float,
        default=3.0,
        help="Gaussian kernel truncate radius in sigmas for anti_alias_point.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-cases", type=int, help="Limit cases for smoke tests.")
    parser.add_argument("--max-frames", type=int, help="Limit frames per case for smoke tests.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect layout and print the planned conversion without writing samples.",
    )
    args = parser.parse_args()

    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.point_offset is None:
        args.point_offset = args.scale // 2
    if args.point_offset < 0 or args.point_offset >= args.scale:
        raise ValueError("--point-offset must satisfy 0 <= point_offset < scale")
    if args.anti_alias_sigma is None:
        args.anti_alias_sigma = args.scale / 2.0
    if args.anti_alias_sigma <= 0:
        raise ValueError("--anti-alias-sigma must be > 0")
    if args.anti_alias_truncate <= 0:
        raise ValueError("--anti-alias-truncate must be > 0")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= args.valid_ratio < 1:
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("--train-ratio + --valid-ratio must be < 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    source_root = resolve_source_root(args.source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    samples, complete_cases, discovery = discover_samples(
        source_root=source_root,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        max_cases=args.max_cases,
        max_frames=args.max_frames,
    )
    first_sample = samples[0]
    geometry = read_geometry(
        Path(first_sample.component_paths["X"]),
        args.scale,
        args.downsample_method,
        args.point_offset,
        args.anti_alias_sigma,
        args.anti_alias_truncate,
    )

    print(f"source_root: {source_root}")
    print(f"output_root: {args.output_root}")
    print(f"complete_cases: {len(complete_cases)}")
    print(f"samples: {len(samples)}")
    print(f"grid_shape: {geometry.grid_shape}")
    print(f"hr_shape: {geometry.hr_shape}")
    print(f"lr_shape: {geometry.lr_shape}")
    print(f"scale: {args.scale}")
    print(f"downsample_method: {args.downsample_method}")
    if args.downsample_method in {"point", "anti_alias_point"}:
        print(f"point_offset: {args.point_offset}")
    if args.downsample_method == "anti_alias_point":
        print(f"anti_alias_sigma: {args.anti_alias_sigma}")
        print(f"anti_alias_truncate: {args.anti_alias_truncate}")
    print(f"crop: {geometry.crop}")
    print(f"splits: {discovery['samples_per_split']}")
    print(f"workers: {args.workers}")

    if args.dry_run:
        return

    prepare_output_dirs(args.output_root, args.overwrite, args.downsample_method)
    save_static_variables(args.output_root, geometry)

    data_stats = process_all_samples(
        samples=samples,
        geometry=geometry,
        output_root=args.output_root,
        scale=args.scale,
        downsample_method=args.downsample_method,
        point_offset=args.point_offset,
        anti_alias_sigma=args.anti_alias_sigma,
        anti_alias_truncate=args.anti_alias_truncate,
        workers=args.workers,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(args.output_root),
        "format": "OceanNPY",
        "task": (
            "sparse full-grid wavefield reconstruction from observed points and mask"
            if args.downsample_method == SPARSE_MASK_METHOD
            else "4x super-resolution from downsampled LR to HR wavefield snapshots"
        ),
        "source_components": list(SOURCE_COMPONENTS),
        "dynamic_variables": list(TARGET_VARS.values()),
        "hr_dynamic_variables": hr_var_names(),
        "lr_dynamic_variables": lr_var_names(args.downsample_method),
        "scale": args.scale,
        "downsample_method": args.downsample_method,
        "point_offset": args.point_offset
        if args.downsample_method in {"point", "anti_alias_point"}
        else None,
        "anti_alias_sigma": args.anti_alias_sigma
        if args.downsample_method == "anti_alias_point"
        else None,
        "anti_alias_truncate": args.anti_alias_truncate
        if args.downsample_method == "anti_alias_point"
        else None,
        "grid_shape_original": list(geometry.grid_shape),
        "hr_shape": list(geometry.hr_shape),
        "lr_shape": list(geometry.lr_shape),
        "crop": geometry.crop,
        "node_count": geometry.node_count,
        "element_count": geometry.element_count,
        "split_policy": "case-level split; every frame from one case stays in one split",
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.valid_ratio,
        "complete_cases": complete_cases,
        **discovery,
    }
    var_names = {
        "dynamic": list(TARGET_VARS.values()),
        "dyn_vars": list(TARGET_VARS.values()),
        "hr_dynamic": hr_var_names(),
        "lr_dynamic": lr_var_names(args.downsample_method),
        "static": ["x", "y", "z", "mask"],
        "spatial_shape": list(geometry.hr_shape),
        "lr_spatial_shape": list(geometry.lr_shape),
        "scale": args.scale,
        "downsample_method": args.downsample_method,
        "point_offset": args.point_offset
        if args.downsample_method in {"point", "anti_alias_point"}
        else None,
        "anti_alias_sigma": args.anti_alias_sigma
        if args.downsample_method == "anti_alias_point"
        else None,
        "anti_alias_truncate": args.anti_alias_truncate
        if args.downsample_method == "anti_alias_point"
        else None,
    }

    write_json(args.output_root / "preprocess_manifest.json", manifest)
    write_json(args.output_root / "var_names.json", var_names)
    write_json(args.output_root / "data_stats.json", data_stats)

    print("done")
    print(f"manifest: {args.output_root / 'preprocess_manifest.json'}")
    print(f"stats: {args.output_root / 'data_stats.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

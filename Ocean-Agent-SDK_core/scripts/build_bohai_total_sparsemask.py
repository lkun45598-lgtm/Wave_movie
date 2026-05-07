from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np


DEFAULT_SOURCE_ROOT = Path("/data/Bohai_Sea/process_data_sparsemask_2x")
DEFAULT_OUTPUT_ROOT = Path("/data/Bohai_Sea/process_data_sparsemask_2x_total")
SPLITS = ("train", "valid", "test")


def sparse_sample_indices(length: int, scale: int, offset: int) -> np.ndarray:
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if offset < 0 or offset >= scale:
        raise ValueError(f"offset must satisfy 0 <= offset < scale, got {offset}")
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


def sparse_linear_interpolate_2d(
    array: np.ndarray,
    scale: int,
    offset: int,
) -> np.ndarray:
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


def total_magnitude(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    if vx.shape != vy.shape or vx.shape != vz.shape:
        raise ValueError(f"Component shapes differ: {vx.shape}, {vy.shape}, {vz.shape}")
    total = np.sqrt(
        vx.astype(np.float64) ** 2
        + vy.astype(np.float64) ** 2
        + vz.astype(np.float64) ** 2
    )
    return total.astype(np.float32)


def build_sparse_products(
    total: np.ndarray,
    scale: int,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = sparse_observation_mask_2d(total.shape, scale=scale, offset=offset)
    sparse = (total.astype(np.float32, copy=False) * mask).astype(
        np.float32,
        copy=False,
    )
    interp = sparse_linear_interpolate_2d(total, scale=scale, offset=offset)
    return sparse, interp, mask


@dataclass
class StatsAccumulator:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, array: np.ndarray) -> None:
        values = np.asarray(array, dtype=np.float64)
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        self.min_value = min(self.min_value, float(values.min()))
        self.max_value = max(self.max_value, float(values.max()))

    def to_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.min_value,
            "max": self.max_value,
        }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    for split in SPLITS:
        (output_root / split / "hr" / "Total").mkdir(parents=True, exist_ok=True)
        (output_root / split / "lr" / "Total_sparse").mkdir(parents=True, exist_ok=True)
        (output_root / split / "lr" / "Total_interp").mkdir(parents=True, exist_ok=True)
        (output_root / split / "lr" / "mask_observed").mkdir(parents=True, exist_ok=True)


def link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def copy_static_variables(source_root: Path, output_root: Path) -> None:
    static_source = source_root / "static_variables"
    static_target = output_root / "static_variables"
    if not static_source.exists():
        return

    for source_file in static_source.rglob("*"):
        if not source_file.is_file():
            continue
        target_file = static_target / source_file.relative_to(static_source)
        link_or_copy_file(source_file, target_file)


def split_stems(source_root: Path, split: str) -> list[str]:
    vx_dir = source_root / split / "hr" / "Vx"
    if not vx_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {vx_dir}")
    return [path.stem for path in sorted(vx_dir.glob("*.npy"))]


def derive_split(
    source_root: Path,
    output_root: Path,
    split: str,
    scale: int,
    offset: int,
    stats: dict[str, StatsAccumulator],
) -> int:
    stems = split_stems(source_root, split)
    mask_cache: dict[tuple[int, int], np.ndarray] = {}

    for index, stem in enumerate(stems, start=1):
        vx = np.load(source_root / split / "hr" / "Vx" / f"{stem}.npy")
        vy = np.load(source_root / split / "hr" / "Vy" / f"{stem}.npy")
        vz = np.load(source_root / split / "hr" / "Vz" / f"{stem}.npy")

        total = total_magnitude(vx, vy, vz)
        sparse, interp, mask = build_sparse_products(total, scale=scale, offset=offset)

        np.save(output_root / split / "hr" / "Total" / f"{stem}.npy", total)
        np.save(output_root / split / "lr" / "Total_sparse" / f"{stem}.npy", sparse)
        np.save(output_root / split / "lr" / "Total_interp" / f"{stem}.npy", interp)

        mask_source = source_root / split / "lr" / "mask_observed" / f"{stem}.npy"
        mask_target = output_root / split / "lr" / "mask_observed" / f"{stem}.npy"
        if mask_source.exists():
            link_or_copy_file(mask_source, mask_target)
        else:
            shape_key = tuple(mask.shape)
            if shape_key not in mask_cache:
                mask_cache[shape_key] = mask
            np.save(mask_target, mask_cache[shape_key])

        stats["Total"].update(total)
        stats["Total_sparse"].update(sparse)
        stats["Total_interp"].update(interp)
        stats["mask_observed"].update(mask)

        if index == 1 or index % 500 == 0 or index == len(stems):
            print(f"[{split}] derived {index}/{len(stems)}: {stem}", flush=True)

    return len(stems)


def write_metadata(
    source_root: Path,
    output_root: Path,
    scale: int,
    offset: int,
    split_counts: dict[str, int],
    stats: dict[str, StatsAccumulator],
) -> None:
    source_manifest = read_json(source_root / "preprocess_manifest.json")
    manifest = dict(source_manifest)
    manifest.update(
        {
            "source_processed_root": str(source_root),
            "output_root": str(output_root),
            "task": "sparse full-grid scalar-magnitude wavefield reconstruction",
            "source_components": ["X", "Y", "Z"],
            "dynamic_variables": ["Total"],
            "hr_dynamic_variables": ["Total"],
            "lr_dynamic_variables": [
                "Total_sparse",
                "Total_interp",
                "mask_observed",
            ],
            "scale": scale,
            "downsample_method": "sparse_mask",
            "point_offset": offset,
            "split_counts": split_counts,
        }
    )

    var_names = {
        "dynamic": ["Total"],
        "dyn_vars": ["Total"],
        "hr_dynamic": ["Total"],
        "lr_dynamic": ["Total_sparse", "Total_interp", "mask_observed"],
        "static": ["x", "y", "z", "mask"],
        "spatial_shape": source_manifest.get("hr_shape", [200, 150]),
        "lr_spatial_shape": source_manifest.get("lr_shape", [200, 150]),
        "scale": scale,
        "downsample_method": "sparse_mask",
        "point_offset": offset,
    }

    data_stats = {
        "Total": {"hr": stats["Total"].to_dict()},
        "Total_sparse": {"lr": stats["Total_sparse"].to_dict()},
        "Total_interp": {"lr": stats["Total_interp"].to_dict()},
        "mask_observed": {"lr": stats["mask_observed"].to_dict()},
    }

    write_json(output_root / "preprocess_manifest.json", manifest)
    write_json(output_root / "var_names.json", var_names)
    write_json(output_root / "data_stats.json", data_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive Total=sqrt(Vx^2+Vy^2+Vz^2) sparse-mask OceanNPY data "
            "from the existing Bohai Vx/Vy/Vz processed dataset."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_manifest = read_json(source_root / "preprocess_manifest.json")
    scale = int(args.scale if args.scale is not None else source_manifest.get("scale", 2))
    offset_value = source_manifest.get("point_offset", 0)
    offset = int(args.offset if args.offset is not None else (offset_value or 0))

    prepare_output_root(output_root, overwrite=args.overwrite)
    copy_static_variables(source_root, output_root)

    stats = {
        "Total": StatsAccumulator(),
        "Total_sparse": StatsAccumulator(),
        "Total_interp": StatsAccumulator(),
        "mask_observed": StatsAccumulator(),
    }
    split_counts = {}
    for split in SPLITS:
        split_counts[split] = derive_split(
            source_root,
            output_root,
            split,
            scale=scale,
            offset=offset,
            stats=stats,
        )

    write_metadata(
        source_root,
        output_root,
        scale=scale,
        offset=offset,
        split_counts=split_counts,
        stats=stats,
    )
    print(f"[write] {output_root}", flush=True)


if __name__ == "__main__":
    main()

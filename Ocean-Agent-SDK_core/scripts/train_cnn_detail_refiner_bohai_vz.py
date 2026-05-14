from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency fallback
    structural_similarity = None


DEFAULT_DATASET_ROOT = Path("/data/Bohai_Sea/process_data_sparsemask_2x")
DEFAULT_BASE_PRED_DIR = Path(
    "/data1/user/lz/wave_movie/testouts/"
    "Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot"
)

TRAIN_CASES = (
    "S1_TTTZ",
    "S1_TWVZ",
    "S1_URZ",
    "S1_UTU",
    "S1_WATZ",
    "S1_WCZ",
    "S1_WHTZ",
    "S1_WIAZ",
)
VALID_CASES = ("S1_WLCZ", "S1_WTAZ")
EVAL_CASES = ("S1_WRRZ", "S1_WTVZ")


@dataclass(frozen=True)
class Sample:
    base_name: str
    case_name: str


def case_from_base_name(base_name: str) -> str:
    parts = base_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse case/frame from {base_name!r}")
    return parts[0]


def load_field(path: Path) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D field or HWC single-channel field: {path}")
    return np.nan_to_num(array, nan=0.0).astype(np.float32, copy=False)


def list_samples(dataset_root: Path, cases: tuple[str, ...]) -> list[Sample]:
    hr_dir = dataset_root / "test" / "hr" / "Vz"
    case_set = set(cases)
    samples: list[Sample] = []
    for path in sorted(hr_dir.glob("*.npy")):
        base_name = path.stem
        case_name = case_from_base_name(base_name)
        if case_name in case_set:
            samples.append(Sample(base_name=base_name, case_name=case_name))
    if not samples:
        raise FileNotFoundError(f"No samples found in {hr_dir} for cases={cases}")
    return samples


def split_samples(dataset_root: Path) -> dict[str, list[Sample]]:
    return {
        "train": list_samples(dataset_root, TRAIN_CASES),
        "valid": list_samples(dataset_root, VALID_CASES),
        "eval": list_samples(dataset_root, EVAL_CASES),
    }


def sample_paths(dataset_root: Path, base_pred_dir: Path, base_name: str) -> dict[str, Path]:
    return {
        "hr": dataset_root / "test" / "hr" / "Vz" / f"{base_name}.npy",
        "interp": dataset_root / "test" / "lr" / "Vz_interp" / f"{base_name}.npy",
        "sparse": dataset_root / "test" / "lr" / "Vz_sparse" / f"{base_name}.npy",
        "mask": dataset_root / "test" / "lr" / "mask_observed" / f"{base_name}.npy",
        "base": base_pred_dir / f"{base_name}_sr.npy",
    }


def load_sample_arrays(
    dataset_root: Path,
    base_pred_dir: Path,
    base_name: str,
) -> dict[str, np.ndarray]:
    paths = sample_paths(dataset_root, base_pred_dir, base_name)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required sample files:\n" + "\n".join(missing))

    hr = load_field(paths["hr"])
    interp = load_field(paths["interp"])
    sparse = load_field(paths["sparse"])
    mask = load_field(paths["mask"])
    base = load_field(paths["base"])
    if not (hr.shape == interp.shape == sparse.shape == mask.shape == base.shape):
        raise ValueError(
            f"Shape mismatch for {base_name}: "
            f"hr={hr.shape}, interp={interp.shape}, sparse={sparse.shape}, "
            f"mask={mask.shape}, base={base.shape}"
        )
    return {
        "hr": hr,
        "interp": interp,
        "sparse": sparse,
        "mask": mask,
        "base": base,
    }


def build_input_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    base = arrays["base"]
    interp = arrays["interp"]
    return np.stack(
        [
            base,
            interp,
            arrays["sparse"],
            arrays["mask"],
            base - interp,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def compute_input_stats(
    samples: list[Sample],
    dataset_root: Path,
    base_pred_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(5, dtype=np.float64)
    channel_sum_sq = np.zeros(5, dtype=np.float64)
    count = 0
    for sample in samples:
        arrays = load_sample_arrays(dataset_root, base_pred_dir, sample.base_name)
        features = build_input_features(arrays)
        flat = features.reshape(features.shape[0], -1)
        channel_sum += flat.sum(axis=1)
        channel_sum_sq += np.square(flat, dtype=np.float64).sum(axis=1)
        count += flat.shape[1]
    mean = channel_sum / max(count, 1)
    variance = channel_sum_sq / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(variance, 1e-12))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class BohaiCNNRefinerDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        dataset_root: Path,
        base_pred_dir: Path,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        mode: str,
        patch_size: int,
        samples_per_epoch: int | None = None,
    ) -> None:
        self.samples = list(samples)
        self.dataset_root = dataset_root
        self.base_pred_dir = base_pred_dir
        self.input_mean = input_mean.reshape(5, 1, 1).astype(np.float32)
        self.input_std = input_std.reshape(5, 1, 1).astype(np.float32)
        self.mode = mode
        self.patch_size = int(patch_size)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        if self.mode == "train" and self.samples_per_epoch is not None:
            return int(self.samples_per_epoch)
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index % len(self.samples)]
        arrays = load_sample_arrays(self.dataset_root, self.base_pred_dir, sample.base_name)
        features = build_input_features(arrays)
        target = arrays["hr"][None, ...].astype(np.float32)
        base = arrays["base"][None, ...].astype(np.float32)
        sparse = arrays["sparse"][None, ...].astype(np.float32)
        observed_mask = arrays["mask"][None, ...].astype(np.float32)

        if self.mode == "train":
            _, height, width = features.shape
            if self.patch_size > height or self.patch_size > width:
                raise ValueError(
                    f"patch_size={self.patch_size} exceeds field shape {height}x{width}"
                )
            top = int(torch.randint(0, height - self.patch_size + 1, (1,)).item())
            left = int(torch.randint(0, width - self.patch_size + 1, (1,)).item())
            sl_y = slice(top, top + self.patch_size)
            sl_x = slice(left, left + self.patch_size)
            features = features[:, sl_y, sl_x]
            target = target[:, sl_y, sl_x]
            base = base[:, sl_y, sl_x]
            sparse = sparse[:, sl_y, sl_x]
            observed_mask = observed_mask[:, sl_y, sl_x]

        features = (features - self.input_mean) / self.input_std
        return {
            "input": torch.from_numpy(features.astype(np.float32, copy=False)),
            "target": torch.from_numpy(target),
            "base": torch.from_numpy(base),
            "sparse": torch.from_numpy(sparse),
            "observed_mask": torch.from_numpy(observed_mask),
            "base_name": sample.base_name,
            "case_name": sample.case_name,
        }


class ResidualBlock(nn.Module):
    def __init__(self, features: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=padding, dilation=dilation),
            nn.SiLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=padding, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CNNDetailRefiner(nn.Module):
    def __init__(self, in_channels: int = 5, features: int = 32, num_blocks: int = 8) -> None:
        super().__init__()
        blocks = []
        for index in range(num_blocks):
            dilation = 2 if index % 3 == 2 else 1
            blocks.append(ResidualBlock(features, dilation=dilation))
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(features, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(self.head(x)))


def apply_observed_constraint(
    pred: torch.Tensor,
    sparse: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(observed_mask > 0.5, sparse, pred)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype, device=values.device).expand_as(values)
    return (values * weights).sum() / weights.sum().clamp(min=1.0)


def gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx = torch.abs((pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1]))
    dy = torch.abs((pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :]))
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    return masked_mean(dx, mask_x) + masked_mean(dy, mask_y)


def laplacian_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3)
    pred_lap = torch.nn.functional.conv2d(pred, kernel, padding=1)
    target_lap = torch.nn.functional.conv2d(target, kernel, padding=1)
    return masked_mean(torch.abs(pred_lap - target_lap), mask)


def target_peak_mask(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    peak_quantile: float,
) -> torch.Tensor:
    if not 0.0 <= peak_quantile <= 1.0:
        raise ValueError("peak_quantile must be in [0, 1]")

    abs_target = torch.abs(target)
    valid_mask = valid_mask.to(device=target.device, dtype=torch.bool).expand_as(target)
    peak_mask = torch.zeros_like(valid_mask)
    for batch_index in range(target.shape[0]):
        values = abs_target[batch_index][valid_mask[batch_index]]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values.float(), peak_quantile)
        peak_mask[batch_index] = valid_mask[batch_index] & (
            abs_target[batch_index] >= threshold
        )
    return peak_mask


def peak_preserving_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    peak_quantile: float = 0.9,
) -> torch.Tensor:
    peak_mask = target_peak_mask(target, valid_mask, peak_quantile)
    under_peak = torch.relu(torch.abs(target) - torch.abs(pred))
    return masked_mean(under_peak, peak_mask)


def topk_amplitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    topk_fraction: float = 0.05,
) -> torch.Tensor:
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")

    valid_mask = valid_mask.to(device=target.device, dtype=torch.bool).expand_as(target)
    abs_pred = torch.abs(pred)
    abs_target = torch.abs(target)
    losses = []
    for batch_index in range(target.shape[0]):
        valid = valid_mask[batch_index]
        target_values = abs_target[batch_index][valid]
        pred_values = abs_pred[batch_index][valid]
        if target_values.numel() == 0:
            continue
        k = max(1, int(math.ceil(target_values.numel() * topk_fraction)))
        target_topk, topk_indices = torch.topk(target_values, k=k, largest=True)
        pred_at_target_peaks = pred_values[topk_indices]
        losses.append(torch.abs(pred_at_target_peaks.mean() - target_topk.mean()))
    if not losses:
        return target.new_tensor(0.0)
    return torch.stack(losses).mean()


def refiner_loss(
    refined: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    observed_mask: torch.Tensor,
    active_threshold: float,
    peak_preserve_weight: float = 0.0,
    topk_amplitude_weight: float = 0.0,
    peak_quantile: float = 0.9,
    topk_fraction: float = 0.05,
) -> torch.Tensor:
    missing = observed_mask <= 0.5
    active = torch.abs(target) > active_threshold
    active_missing = missing & active
    inactive_missing = missing & (~active)
    abs_err = torch.abs(refined - target)
    loss = masked_mean(abs_err, active_missing)
    loss = loss + 0.2 * masked_mean(abs_err, missing)
    loss = loss + 0.05 * masked_mean(abs_err, inactive_missing)
    loss = loss + 0.2 * gradient_l1(refined, target, missing)
    loss = loss + 0.1 * laplacian_l1(refined, target, missing)
    loss = loss + 0.05 * torch.mean(delta * delta)
    if peak_preserve_weight > 0:
        loss = loss + peak_preserve_weight * peak_preserving_loss(
            refined,
            target,
            active_missing,
            peak_quantile=peak_quantile,
        )
    if topk_amplitude_weight > 0:
        loss = loss + topk_amplitude_weight * topk_amplitude_loss(
            refined,
            target,
            active_missing,
            topk_fraction=topk_fraction,
        )
    return loss


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    observed_mask: np.ndarray | None = None,
    active_threshold: float = 0.005,
) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target[..., 0]
    diff = pred - target
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(abs_diff))
    rfne = float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(target.ravel()), 1e-12))
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    acc = float(
        np.sum(pred_centered * target_centered)
        / max(np.sqrt(np.sum(pred_centered**2) * np.sum(target_centered**2)), 1e-12)
    )
    data_range = float(np.nanmax(target) - np.nanmin(target))
    min_side = min(target.shape)
    if structural_similarity is not None and data_range > 0 and min_side >= 7:
        ssim = float(structural_similarity(target, pred, data_range=data_range))
    else:
        ssim = float("nan")
    peak_target = float(np.max(np.abs(target)))
    peak_pred = float(np.max(np.abs(pred)))
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "rfne": rfne,
        "acc": acc,
        "ssim": ssim,
        "p99_abs_error": float(np.percentile(abs_diff, 99)),
        "max_abs_error": float(np.max(abs_diff)),
        "peak_abs_pred": peak_pred,
        "peak_abs_target": peak_target,
        "peak_ratio": float(peak_pred / max(peak_target, 1e-12)),
    }
    if observed_mask is not None:
        observed = np.asarray(observed_mask) > 0.5
        if observed.ndim == 3 and observed.shape[-1] == 1:
            observed = observed[..., 0]
        missing = ~observed
        active_missing = missing & (np.abs(target) > active_threshold)
        if np.any(missing):
            metrics["missing_rmse"] = float(np.sqrt(np.mean(diff[missing] ** 2)))
        else:
            metrics["missing_rmse"] = float("nan")
        if np.any(active_missing):
            metrics["active_missing_rmse"] = float(
                np.sqrt(np.mean(diff[active_missing] ** 2))
            )
        else:
            metrics["active_missing_rmse"] = float("nan")
    return metrics


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[key] = float(values.mean()) if values.size else float("nan")
    return summary


def evaluate_model(
    model: CNNDetailRefiner,
    dataset: BohaiCNNRefinerDataset,
    device: torch.device,
    output_dir: Path,
    active_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    metrics_by_method: dict[str, list[dict[str, float]]] = {
        "Interp": [],
        "ResShift200ep": [],
        "CNNRefined": [],
    }
    with torch.no_grad():
        for index in range(len(dataset.samples)):
            item = dataset[index]
            x = item["input"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            base = item["base"].unsqueeze(0).to(device)
            sparse = item["sparse"].unsqueeze(0).to(device)
            mask = item["observed_mask"].unsqueeze(0).to(device)
            delta = model(x)
            refined = apply_observed_constraint(base + delta, sparse, mask)

            base_name = str(item["base_name"])
            arrays = load_sample_arrays(dataset.dataset_root, dataset.base_pred_dir, base_name)
            target_np = arrays["hr"]
            interp_np = arrays["interp"]
            base_np = arrays["base"]
            mask_np = arrays["mask"]
            refined_np = refined.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(output_dir / f"{base_name}_sr.npy", refined_np[..., None])

            for method, pred_np in (
                ("Interp", interp_np),
                ("ResShift200ep", base_np),
                ("CNNRefined", refined_np),
            ):
                metrics = compute_metrics(
                    pred_np,
                    target_np,
                    observed_mask=mask_np,
                    active_threshold=active_threshold,
                )
                metrics_by_method[method].append(metrics)
                rows.append(
                    {
                        "base_name": base_name,
                        "case_name": str(item["case_name"]),
                        "method": method,
                        **metrics,
                    }
                )
    summaries = {
        method: aggregate_metrics(method_rows)
        for method, method_rows in metrics_by_method.items()
    }
    return rows, summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    extra_fields = sorted({key for row in rows for key in row} - set(fields))
    fields.extend(extra_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summaries: dict[str, dict[str, float]]) -> None:
    rows = [{"method": method, **metrics} for method, metrics in summaries.items()]
    write_csv(path, rows)


def plot_loss_curve(train_losses: list[float], valid_losses: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    ax.plot(range(1, len(valid_losses) + 1), valid_losses, label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_one_epoch(
    model: CNNDetailRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    active_threshold: float,
    peak_preserve_weight: float,
    topk_amplitude_weight: float,
    peak_quantile: float,
    topk_fraction: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        base = batch["base"].to(device, non_blocking=True)
        sparse = batch["sparse"].to(device, non_blocking=True)
        mask = batch["observed_mask"].to(device, non_blocking=True)
        delta = model(x)
        refined = apply_observed_constraint(base + delta, sparse, mask)
        loss = refiner_loss(
            refined,
            target,
            delta,
            mask,
            active_threshold,
            peak_preserve_weight=peak_preserve_weight,
            topk_amplitude_weight=topk_amplitude_weight,
            peak_quantile=peak_quantile,
            topk_fraction=topk_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def validate_loss(
    model: CNNDetailRefiner,
    loader: DataLoader,
    device: torch.device,
    active_threshold: float,
    peak_preserve_weight: float,
    topk_amplitude_weight: float,
    peak_quantile: float,
    topk_fraction: float,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            base = batch["base"].to(device, non_blocking=True)
            sparse = batch["sparse"].to(device, non_blocking=True)
            mask = batch["observed_mask"].to(device, non_blocking=True)
            delta = model(x)
            refined = apply_observed_constraint(base + delta, sparse, mask)
            loss = refiner_loss(
                refined,
                target,
                delta,
                mask,
                active_threshold,
                peak_preserve_weight=peak_preserve_weight,
                topk_amplitude_weight=topk_amplitude_weight,
                peak_quantile=peak_quantile,
                topk_fraction=topk_fraction,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot lightweight CNN detail refiner on ResShift200ep Bohai Vz outputs."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--base-pred-dir", type=Path, default=DEFAULT_BASE_PRED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=2400)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--active-threshold", type=float, default=0.005)
    parser.add_argument("--peak-preserve-weight", type=float, default=0.0)
    parser.add_argument("--topk-amplitude-weight", type=float, default=0.0)
    parser.add_argument("--peak-quantile", type=float, default=0.9)
    parser.add_argument("--topk-fraction", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    splits = split_samples(args.dataset_root)
    split_info = {
        "train_cases": list(TRAIN_CASES),
        "valid_cases": list(VALID_CASES),
        "eval_cases": list(EVAL_CASES),
        "train_samples": len(splits["train"]),
        "valid_samples": len(splits["valid"]),
        "eval_samples": len(splits["eval"]),
        "note": (
            "Pilot split from original test cases. Use only as diagnostic; "
            "final experiment must use original train/valid/test base predictions."
        ),
    }
    (args.output_dir / "split_cases.json").write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"Split info: {split_info}")
    log("Computing input feature statistics from CNN train cases")
    input_mean, input_std = compute_input_stats(
        splits["train"], args.dataset_root, args.base_pred_dir
    )
    log(f"input_mean={input_mean.tolist()}")
    log(f"input_std={input_std.tolist()}")

    train_ds = BohaiCNNRefinerDataset(
        splits["train"],
        args.dataset_root,
        args.base_pred_dir,
        input_mean,
        input_std,
        mode="train",
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
    )
    valid_ds = BohaiCNNRefinerDataset(
        splits["valid"],
        args.dataset_root,
        args.base_pred_dir,
        input_mean,
        input_std,
        mode="valid",
        patch_size=args.patch_size,
    )
    eval_ds = BohaiCNNRefinerDataset(
        splits["eval"],
        args.dataset_root,
        args.base_pred_dir,
        input_mean,
        input_std,
        mode="eval",
        patch_size=args.patch_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNNDetailRefiner(
        in_channels=5,
        features=args.features,
        num_blocks=args.num_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_valid = math.inf
    train_losses: list[float] = []
    valid_losses: list[float] = []
    best_path = args.output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.active_threshold,
            args.peak_preserve_weight,
            args.topk_amplitude_weight,
            args.peak_quantile,
            args.topk_fraction,
        )
        valid_loss = validate_loss(
            model,
            valid_loader,
            device,
            args.active_threshold,
            args.peak_preserve_weight,
            args.topk_amplitude_weight,
            args.peak_quantile,
            args.topk_fraction,
        )
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_mean": input_mean,
                    "input_std": input_std,
                    "args": vars(args),
                    "epoch": epoch,
                    "valid_loss": valid_loss,
                },
                best_path,
            )
        log(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6g} valid_loss={valid_loss:.6g} "
            f"best_valid={best_valid:.6g} time={time.time() - start:.1f}s"
        )

    plot_loss_curve(train_losses, valid_losses, args.output_dir / "loss_curve.png")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows, summaries = evaluate_model(
        model,
        eval_ds,
        device,
        args.output_dir / "predictions_eval",
        args.active_threshold,
    )
    write_csv(args.output_dir / "per_frame_metrics.csv", rows)
    write_summary_csv(args.output_dir / "metrics_summary.csv", summaries)
    log(f"Saved best checkpoint: {best_path}")
    log(f"Saved eval predictions: {args.output_dir / 'predictions_eval'}")
    log(f"Saved summary metrics: {args.output_dir / 'metrics_summary.csv'}")
    for method, metrics in summaries.items():
        log(
            f"{method}: rmse={metrics.get('rmse', float('nan')):.6g} "
            f"active_missing_rmse={metrics.get('active_missing_rmse', float('nan')):.6g} "
            f"ssim={metrics.get('ssim', float('nan')):.6g} "
            f"peak_ratio={metrics.get('peak_ratio', float('nan')):.6g} "
            f"p99={metrics.get('p99_abs_error', float('nan')):.6g} "
            f"max={metrics.get('max_abs_error', float('nan')):.6g}"
        )


if __name__ == "__main__":
    main()

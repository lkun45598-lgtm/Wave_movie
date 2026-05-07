from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


COMPONENT_LABELS = ("Vx", "Vy", "Vz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Bohai Sea SR evaluation figures from test_samples.npz."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("/data1/user/lz/wave_movie/testouts/EDSR_PGN/test_samples.npz"),
        help="Path to test_samples.npz containing lr/sr/hr arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data1/user/lz/wave_movie/testouts/EDSR_PGN/figs_eval"),
        help="Directory for generated figures.",
    )
    parser.add_argument("--model-name", default="EDSR", help="Model label used in plots.")
    parser.add_argument("--max-samples", type=int, default=4, help="Max samples to draw.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser.parse_args()


def as_text_array(value: np.ndarray | str | bytes) -> list[str]:
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return [item.decode("utf-8")]
        return [str(item)]
    out: list[str] = []
    for item in arr.ravel():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input npz: {path}")
    data = dict(np.load(path, allow_pickle=True))
    for key in ("lr", "sr", "hr"):
        if key not in data:
            raise KeyError(f"{path} does not contain required array {key!r}")
        data[key] = np.asarray(data[key], dtype=np.float32)
        if data[key].ndim != 4:
            raise ValueError(f"{key} must be [N,H,W,C], got {data[key].shape}")
    if data["sr"].shape != data["hr"].shape:
        raise ValueError(f"sr and hr shape mismatch: {data['sr'].shape} vs {data['hr'].shape}")
    return data


def select_lr_display_channels(lr: np.ndarray, target_channels: int) -> np.ndarray:
    if lr.ndim != 4:
        raise ValueError(f"lr must be [N,H,W,C], got {lr.shape}")
    if target_channels <= 0:
        raise ValueError(f"target_channels must be positive, got {target_channels}")

    lr_channels = lr.shape[-1]
    if lr_channels == target_channels:
        return lr
    if lr_channels % target_channels != 0:
        raise ValueError(
            f"Cannot map LR channels to target channels: {lr_channels} is not "
            f"a multiple of {target_channels}"
        )

    temporal_window = lr_channels // target_channels
    if temporal_window % 2 == 0:
        raise ValueError(
            f"Temporal LR channel count implies an even window ({temporal_window}); "
            "cannot identify a unique center frame."
        )

    center = temporal_window // 2
    start = center * target_channels
    return lr[..., start : start + target_channels]


def upsample_lr(lr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(lr).permute(0, 3, 1, 2)
    up = F.interpolate(tensor, size=target_hw, mode="bicubic", align_corners=False)
    return up.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


def masked_flat(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return arr.reshape(-1)
    bool_mask = np.asarray(mask).astype(bool)
    while bool_mask.ndim < arr.ndim:
        bool_mask = np.expand_dims(bool_mask, axis=-1)
    bool_mask = np.broadcast_to(bool_mask, arr.shape)
    return arr[bool_mask]


def metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    p = masked_flat(pred, mask).astype(np.float64)
    t = masked_flat(target, mask).astype(np.float64)
    diff = p - t
    denom = np.linalg.norm(t) + 1e-12
    pred_norm = np.linalg.norm(p) + 1e-12
    return {
        "rfne": float(np.linalg.norm(diff) / denom),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "acc": float(np.dot(p, t) / (pred_norm * denom)),
        "bias": float(np.mean(diff)),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def vector_magnitude(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(arr[..., :3] ** 2, axis=-1)).astype(np.float32)


def coordinate_extent(data: dict[str, np.ndarray], key_x: str = "lon_hr", key_y: str = "lat_hr") -> tuple[list[float], str, str]:
    if key_x in data and key_y in data:
        x = np.asarray(data[key_x], dtype=np.float64)
        y = np.asarray(data[key_y], dtype=np.float64)
        x_km = (x - np.nanmin(x)) / 1000.0
        y_km = (y - np.nanmin(y)) / 1000.0
        return [
            float(np.nanmin(x_km)),
            float(np.nanmax(x_km)),
            float(np.nanmin(y_km)),
            float(np.nanmax(y_km)),
        ], "X offset (km)", "Y offset (km)"
    hr = data["hr"]
    return [0.0, float(hr.shape[2] - 1), 0.0, float(hr.shape[1] - 1)], "X index", "Y index"


def coord_vectors(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str, str]:
    extent, xlabel, ylabel = coordinate_extent(data)
    hr = data["hr"]
    x = np.linspace(extent[0], extent[1], hr.shape[2])
    y = np.linspace(extent[2], extent[3], hr.shape[1])
    return x, y, xlabel, ylabel


def add_image(
    ax: plt.Axes,
    arr: np.ndarray,
    *,
    cmap: str,
    extent: list[float],
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.image.AxesImage:
    im = ax.imshow(
        arr,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    return im


def safe_sym_limit(*arrays: np.ndarray, percentile: float = 99.5) -> float:
    vals = np.concatenate([np.ravel(np.abs(a[np.isfinite(a)])) for a in arrays])
    if vals.size == 0:
        return 1.0
    limit = float(np.percentile(vals, percentile))
    return max(limit, 1e-12)


def safe_pos_limit(*arrays: np.ndarray, percentile: float = 99.5) -> float:
    vals = np.concatenate([np.ravel(np.abs(a[np.isfinite(a)])) for a in arrays])
    if vals.size == 0:
        return 1.0
    limit = float(np.percentile(vals, percentile))
    return max(limit, 1e-12)


def save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    mask: np.ndarray | None,
    out_dir: Path,
    model_name: str,
    dpi: int,
) -> dict[str, object]:
    per_sample = []
    for idx in range(hr.shape[0]):
        per_sample.append(
            {
                "sample": idx,
                "bicubic": metrics(lr_up[idx], hr[idx], mask),
                model_name: metrics(sr[idx], hr[idx], mask),
            }
        )

    metric_names = ("rfne", "rmse", "mae", "acc")
    x = np.arange(hr.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, name in zip(axes.ravel(), metric_names):
        bicubic_vals = [row["bicubic"][name] for row in per_sample]
        sr_vals = [row[model_name][name] for row in per_sample]
        width = 0.36
        ax.bar(x - width / 2, bicubic_vals, width, color="#d9822b", label="Bicubic")
        ax.bar(x + width / 2, sr_vals, width, color="#1764ab", label=model_name)
        ax.set_title(name.upper())
        ax.set_xlabel("Sample index")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25)
        if name != "acc":
            ax.set_ylabel("Lower is better")
        else:
            ax.set_ylabel("Higher is better")
        ax.legend()
    fig.suptitle(f"Bohai Sea SR Quality Per Saved Test Sample - {model_name}", fontsize=18)
    save_fig(fig, out_dir / "fig1_per_sample_metrics.png", dpi)

    summary = {
        "num_samples": int(hr.shape[0]),
        "metrics": per_sample,
        "mean": {
            "bicubic": {
                name: float(np.mean([row["bicubic"][name] for row in per_sample]))
                for name in per_sample[0]["bicubic"]
            },
            model_name: {
                name: float(np.mean([row[model_name][name] for row in per_sample]))
                for name in per_sample[0][model_name]
            },
        },
    }
    return summary


def plot_component_frames(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    data: dict[str, np.ndarray],
    selected: list[int],
    out_dir: Path,
    model_name: str,
    dpi: int,
) -> None:
    extent, xlabel, ylabel = coordinate_extent(data)
    for c, label in enumerate(COMPONENT_LABELS):
        nrows = len(selected)
        fig = plt.figure(figsize=(22, 4.6 * nrows), constrained_layout=True)
        gs = fig.add_gridspec(
            nrows,
            7,
            width_ratios=[1, 1, 1, 0.045, 1, 1, 0.045],
            wspace=0.08,
        )
        for row, idx in enumerate(selected):
            vmax = safe_sym_limit(hr[idx, ..., c], sr[idx, ..., c], lr_up[idx, ..., c])
            err_lr = np.abs(lr_up[idx, ..., c] - hr[idx, ..., c])
            err_sr = np.abs(sr[idx, ..., c] - hr[idx, ..., c])
            vmax_err = safe_pos_limit(err_lr, percentile=99.0)

            axes = [fig.add_subplot(gs[row, col]) for col in (0, 1, 2, 4, 5)]
            cbar_field = fig.add_subplot(gs[row, 3])
            cbar_err = fig.add_subplot(gs[row, 6])

            field_items = [
                ("Bicubic", lr_up[idx, ..., c]),
                (model_name, sr[idx, ..., c]),
                ("Ground Truth", hr[idx, ..., c]),
            ]
            last_im = None
            for ax, (title, arr) in zip(axes[:3], field_items):
                last_im = add_image(
                    ax,
                    arr,
                    cmap="RdBu_r",
                    extent=extent,
                    title=title,
                    vmin=-vmax,
                    vmax=vmax,
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel if ax is axes[0] else "")
            assert last_im is not None
            fig.colorbar(last_im, cax=cbar_field, label=f"{label} value")

            err_items = [("|Bicubic - HR|", err_lr), (f"|{model_name} - HR|", err_sr)]
            last_err = None
            for ax, (title, arr) in zip(axes[3:], err_items):
                last_err = add_image(
                    ax,
                    arr,
                    cmap="Reds",
                    extent=extent,
                    title=f"{title}\nmean={arr.mean():.4g}",
                    vmin=0,
                    vmax=vmax_err,
                )
                ax.set_xlabel(xlabel)
            assert last_err is not None
            fig.colorbar(last_err, cax=cbar_err, label="Absolute error")
            axes[0].text(
                0.02,
                0.95,
                f"sample {idx}",
                transform=axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=13,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
        fig.suptitle(f"{label} Spatial Field Comparison - {model_name}", fontsize=18)
        save_fig(fig, out_dir / f"fig2_frames_{label.lower()}.png", dpi)


def plot_vector_magnitude(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    data: dict[str, np.ndarray],
    out_dir: Path,
    model_name: str,
    dpi: int,
) -> int:
    extent, xlabel, ylabel = coordinate_extent(data)
    mag_hr = vector_magnitude(hr)
    mag_sr = vector_magnitude(sr)
    mag_lr = vector_magnitude(lr_up)
    sample_idx = int(np.argmax(mag_hr.reshape(mag_hr.shape[0], -1).max(axis=1)))

    vmax = safe_pos_limit(mag_hr[sample_idx], mag_sr[sample_idx], mag_lr[sample_idx])
    err_lr = np.abs(mag_lr[sample_idx] - mag_hr[sample_idx])
    err_sr = np.abs(mag_sr[sample_idx] - mag_hr[sample_idx])
    vmax_err = safe_pos_limit(err_lr, percentile=99.0)

    fig = plt.figure(figsize=(22, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.045, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, col]) for col in (0, 1, 2, 4, 5)]
    cbar_field = fig.add_subplot(gs[0, 3])
    cbar_err = fig.add_subplot(gs[0, 6])

    last_im = None
    for ax, (title, arr) in zip(
        axes[:3],
        [("Bicubic |V|", mag_lr[sample_idx]), (f"{model_name} |V|", mag_sr[sample_idx]), ("HR |V|", mag_hr[sample_idx])],
    ):
        last_im = add_image(ax, arr, cmap="turbo", extent=extent, title=title, vmin=0, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ax is axes[0] else "")
    assert last_im is not None
    fig.colorbar(last_im, cax=cbar_field, label="Vector magnitude")

    last_err = None
    for ax, (title, arr) in zip(axes[3:], [("|Bicubic - HR|", err_lr), (f"|{model_name} - HR|", err_sr)]):
        last_err = add_image(
            ax,
            arr,
            cmap="Reds",
            extent=extent,
            title=f"{title}\nmean={arr.mean():.4g}",
            vmin=0,
            vmax=vmax_err,
        )
        ax.set_xlabel(xlabel)
    assert last_err is not None
    fig.colorbar(last_err, cax=cbar_err, label="Magnitude absolute error")
    fig.suptitle(f"Vector Magnitude Map, sample {sample_idx} - {model_name}", fontsize=18)
    save_fig(fig, out_dir / "fig3_vector_magnitude_map.png", dpi)
    return sample_idx


def plot_spatial_profiles(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    data: dict[str, np.ndarray],
    sample_idx: int,
    out_dir: Path,
    model_name: str,
    dpi: int,
) -> None:
    x, y, xlabel, ylabel = coord_vectors(data)
    mag = vector_magnitude(hr[sample_idx : sample_idx + 1])[0]
    iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)

    fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    for c, label in enumerate(COMPONENT_LABELS):
        ax = axes[0, c]
        ax.plot(x, hr[sample_idx, iy, :, c], color="black", lw=2.2, label="Ground Truth")
        ax.plot(x, lr_up[sample_idx, iy, :, c], color="#d9822b", lw=1.5, ls="--", label="Bicubic")
        ax.plot(x, sr[sample_idx, iy, :, c], color="#1764ab", lw=1.7, label=model_name)
        ax.set_title(f"{label}, horizontal profile at Y index {iy}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)

        ax = axes[1, c]
        ax.plot(y, hr[sample_idx, :, ix, c], color="black", lw=2.2, label="Ground Truth")
        ax.plot(y, lr_up[sample_idx, :, ix, c], color="#d9822b", lw=1.5, ls="--", label="Bicubic")
        ax.plot(y, sr[sample_idx, :, ix, c], color="#1764ab", lw=1.7, label=model_name)
        ax.set_title(f"{label}, vertical profile at X index {ix}")
        ax.set_xlabel(ylabel)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if c == 2:
            ax.legend(loc="best", fontsize=11)
    fig.suptitle(f"Spatial Cross-Sections Through Max |V| Point, sample {sample_idx}", fontsize=18)
    save_fig(fig, out_dir / "fig4_spatial_profiles.png", dpi)


def plot_correlation_scatter(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    out_dir: Path,
    model_name: str,
    dpi: int,
) -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for c, label in enumerate(COMPONENT_LABELS):
        h = hr[..., c].reshape(-1)
        l = lr_up[..., c].reshape(-1)
        s = sr[..., c].reshape(-1)
        n = min(25000, h.size)
        idx = rng.choice(h.size, size=n, replace=False)
        h_s, l_s, sr_s = h[idx], l[idx], s[idx]
        lim = safe_sym_limit(h_s, l_s, sr_s, percentile=99.8)
        corr_l = np.corrcoef(h_s, l_s)[0, 1]
        corr_s = np.corrcoef(h_s, sr_s)[0, 1]
        rmse_l = float(np.sqrt(np.mean((l_s - h_s) ** 2)))
        rmse_s = float(np.sqrt(np.mean((sr_s - h_s) ** 2)))
        ax = axes[c]
        ax.scatter(h_s, l_s, s=4, alpha=0.18, color="#d9822b", label=f"Bicubic corr={corr_l:.3f}, rmse={rmse_l:.3g}")
        ax.scatter(h_s, sr_s, s=4, alpha=0.18, color="#1764ab", label=f"{model_name} corr={corr_s:.3f}, rmse={rmse_s:.3g}")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.1, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(label)
        ax.set_xlabel("Ground Truth")
        if c == 0:
            ax.set_ylabel("Prediction")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Pixel-Wise Correlation Scatter", fontsize=18)
    save_fig(fig, out_dir / "fig5_corr_scatter.png", dpi)


def radial_average_spectrum(arr: np.ndarray, *, amplitude: bool) -> tuple[np.ndarray, np.ndarray]:
    n, h, w, cnum = arr.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = np.linspace(0.0, float(rr.max()), 90)
    centers = 0.5 * (bins[:-1] + bins[1:])
    spec = np.zeros((centers.size, cnum), dtype=np.float64)

    for c in range(cnum):
        accum = np.zeros_like(centers)
        count = 0
        for i in range(n):
            field = arr[i, ..., c].astype(np.float64)
            field = field - np.mean(field)
            fft = np.fft.fft2(field)
            values = np.abs(fft)
            if not amplitude:
                values = values**2 / (h * w)
            for b in range(centers.size):
                m = (rr >= bins[b]) & (rr < bins[b + 1])
                if np.any(m):
                    accum[b] += float(values[m].mean())
            count += 1
        spec[:, c] = accum / max(count, 1)
    keep = centers > 0
    return centers[keep], spec[keep]


def plot_spectrum(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    out_dir: Path,
    model_name: str,
    dpi: int,
    *,
    amplitude: bool,
) -> None:
    freqs, hr_spec = radial_average_spectrum(hr, amplitude=amplitude)
    _, sr_spec = radial_average_spectrum(sr, amplitude=amplitude)
    _, lr_spec = radial_average_spectrum(lr_up, amplitude=amplitude)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for c, label in enumerate(COMPONENT_LABELS):
        ax = axes[c]
        ax.loglog(freqs, hr_spec[:, c] + 1e-30, color="black", lw=2.1, label="Ground Truth")
        ax.loglog(freqs, lr_spec[:, c] + 1e-30, color="#d9822b", lw=1.5, ls="--", label="Bicubic")
        ax.loglog(freqs, sr_spec[:, c] + 1e-30, color="#1764ab", lw=1.8, label=model_name)
        ax.axvspan(0.125, float(freqs.max()), color="red", alpha=0.10, label="Beyond 4x LR Nyquist")
        ax.set_title(label)
        ax.set_xlabel("Radial spatial frequency (cycles / HR pixel)")
        if c == 0:
            ax.set_ylabel("Mean |FFT|" if amplitude else "Mean PSD")
        ax.grid(which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    if amplitude:
        title = "Radial Amplitude Spectrum Comparison"
        fname = "fig8_amplitude_spectrum.png"
    else:
        title = "Radial Power Spectral Density Comparison"
        fname = "fig6_psd_compare.png"
    fig.suptitle(title, fontsize=18)
    save_fig(fig, out_dir / fname, dpi)


def plot_gradient_error(
    lr_up: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    data: dict[str, np.ndarray],
    sample_idx: int,
    out_dir: Path,
    model_name: str,
    dpi: int,
) -> None:
    extent, xlabel, ylabel = coordinate_extent(data)

    def grad_mag(mag: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(mag)
        return np.sqrt(gx**2 + gy**2).astype(np.float32)

    g_hr = grad_mag(vector_magnitude(hr[sample_idx : sample_idx + 1])[0])
    g_sr = grad_mag(vector_magnitude(sr[sample_idx : sample_idx + 1])[0])
    g_lr = grad_mag(vector_magnitude(lr_up[sample_idx : sample_idx + 1])[0])
    vmax = safe_pos_limit(g_hr, g_sr, g_lr)
    e_lr = np.abs(g_lr - g_hr)
    e_sr = np.abs(g_sr - g_hr)
    vmax_err = safe_pos_limit(e_lr, percentile=99.0)

    fig = plt.figure(figsize=(22, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.045, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, col]) for col in (0, 1, 2, 4, 5)]
    cbar_field = fig.add_subplot(gs[0, 3])
    cbar_err = fig.add_subplot(gs[0, 6])

    last_im = None
    for ax, (title, arr) in zip(axes[:3], [("Bicubic grad |V|", g_lr), (f"{model_name} grad |V|", g_sr), ("HR grad |V|", g_hr)]):
        last_im = add_image(ax, arr, cmap="magma", extent=extent, title=title, vmin=0, vmax=vmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ax is axes[0] else "")
    assert last_im is not None
    fig.colorbar(last_im, cax=cbar_field, label="Gradient magnitude")

    last_err = None
    for ax, (title, arr) in zip(axes[3:], [("|Bicubic - HR|", e_lr), (f"|{model_name} - HR|", e_sr)]):
        last_err = add_image(
            ax,
            arr,
            cmap="Reds",
            extent=extent,
            title=f"{title}\nmean={arr.mean():.4g}",
            vmin=0,
            vmax=vmax_err,
        )
        ax.set_xlabel(xlabel)
    assert last_err is not None
    fig.colorbar(last_err, cax=cbar_err, label="Gradient absolute error")
    fig.suptitle(f"Spatial Detail Gradient Error, sample {sample_idx}", fontsize=18)
    save_fig(fig, out_dir / "fig7_spatial_gradient_error.png", dpi)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
        }
    )

    data = load_npz(args.npz)
    lr = data["lr"]
    sr = data["sr"]
    hr = data["hr"]
    if sr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels for Vx/Vy/Vz, got {sr.shape[-1]}")

    lr_display = select_lr_display_channels(lr, target_channels=hr.shape[-1])
    lr_up = upsample_lr(lr_display, target_hw=hr.shape[1:3])
    mask = data.get("mask_hr")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 4:
            mask = mask[0, ..., 0]

    n_select = min(args.max_samples, hr.shape[0])
    selected = list(range(n_select))

    summary = {
        "input_npz": str(args.npz),
        "output_dir": str(args.out_dir),
        "model_name": args.model_name,
        "lr_shape": list(lr.shape),
        "lr_display_shape": list(lr_display.shape),
        "lr_up_shape": list(lr_up.shape),
        "sr_shape": list(sr.shape),
        "hr_shape": list(hr.shape),
        "dyn_vars": as_text_array(data.get("dyn_vars", np.array(COMPONENT_LABELS))),
        "filename": as_text_array(data.get("filename", np.array([]))),
    }

    summary["quality"] = plot_metric_bars(lr_up, sr, hr, mask, args.out_dir, args.model_name, args.dpi)
    plot_component_frames(lr_up, sr, hr, data, selected, args.out_dir, args.model_name, args.dpi)
    sample_idx = plot_vector_magnitude(lr_up, sr, hr, data, args.out_dir, args.model_name, args.dpi)
    plot_spatial_profiles(lr_up, sr, hr, data, sample_idx, args.out_dir, args.model_name, args.dpi)
    plot_correlation_scatter(lr_up, sr, hr, args.out_dir, args.model_name, args.dpi)
    plot_spectrum(lr_up, sr, hr, args.out_dir, args.model_name, args.dpi, amplitude=False)
    plot_gradient_error(lr_up, sr, hr, data, sample_idx, args.out_dir, args.model_name, args.dpi)
    plot_spectrum(lr_up, sr, hr, args.out_dir, args.model_name, args.dpi, amplitude=True)

    with open(args.out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved Bohai SR evaluation figures to: {args.out_dir}")
    for path in sorted(args.out_dir.glob("fig*.png")):
        print(f"  {path.name}")
    print("  metrics_summary.json")


if __name__ == "__main__":
    main()

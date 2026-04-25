#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


TMP_CACHE_DIR = Path("/tmp") / f"pyvista_wave_cache_{os.getuid()}"
TMP_MPL_DIR = TMP_CACHE_DIR / "matplotlib"
TMP_SHADER_CACHE_DIR = TMP_CACHE_DIR / "mesa_shader_cache"
TMP_MPL_DIR.mkdir(parents=True, exist_ok=True)
TMP_SHADER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_MPL_DIR))
os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(TMP_SHADER_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE_DIR))

COMPONENTS = ("Total", "X", "Y", "Z")
DEFAULT_DATASET_ROOT = Path("/data/Bohai_Sea/To_ZGT_wave_movie")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "visualizations"


def load_pyvista() -> "pv":
    try:
        import pyvista as pv  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未找到 pyvista。请使用带 pyvista 的环境运行，"
            "例如 /home/lz/miniconda3/envs/pytorch/bin/python save_wave_movie.py ..."
        ) from exc
    return pv


def pick_scalar_name(mesh: "pv.DataSet") -> str:
    candidates = [
        name for name in mesh.array_names if name.strip().lower() != "material id"
    ]
    if not candidates:
        raise ValueError(f"没有可用于显示的标量数组: {mesh.array_names}")
    return candidates[0]


def snapshot_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def resolve_snapshot_files(base_dir: Path) -> list[Path]:
    files = sorted(base_dir.glob("AVS_movie_*.inp"), key=snapshot_sort_key)
    if not files:
        raise FileNotFoundError(f"在 {base_dir} 下没有找到 AVS_movie_*.inp 文件")
    return files


def select_snapshot_files(
    files: list[Path],
    start: int,
    end: int,
    step: int,
    frame: int | None,
) -> list[Path]:
    total_frames = len(files)

    if start < 1 or start > total_frames:
        raise IndexError(f"--start 超出范围: {start}，当前共有 {total_frames} 帧")
    if end < 1 or end > total_frames:
        raise IndexError(f"--end 超出范围: {end}，当前共有 {total_frames} 帧")
    if start > end:
        raise ValueError(f"--start ({start}) 不能大于 --end ({end})")
    if step < 1:
        raise ValueError("--step 必须 >= 1")

    if frame is not None:
        if frame < 1 or frame > total_frames:
            raise IndexError(f"--frame 超出范围: {frame}，当前共有 {total_frames} 帧")
        return [files[frame - 1]]

    selected = files[start - 1 : end : step]
    if not selected:
        raise ValueError("筛选后没有可处理的帧。")
    return selected


def compute_plane_camera(points: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    right = vh[0]
    view_up = vh[1]
    normal = vh[2]

    if normal[2] < 0:
        normal = -normal
    if view_up[2] < 0:
        view_up = -view_up
    if np.dot(np.cross(right, view_up), normal) < 0:
        view_up = -view_up

    diagonal = np.linalg.norm(np.ptp(points, axis=0))
    distance = max(diagonal * 1.5, 1.0)
    position = center + normal * distance

    return tuple(position), tuple(center), tuple(view_up)


def apply_view(plotter: "pv.Plotter", mesh: "pv.DataSet", view: str) -> None:
    if view == "xy":
        plotter.view_xy()
        plotter.enable_parallel_projection()
        return
    if view == "xz":
        plotter.view_xz()
        plotter.enable_parallel_projection()
        return
    if view == "yz":
        plotter.view_yz()
        plotter.enable_parallel_projection()
        return
    if view == "isometric":
        plotter.view_isometric()
        return

    plotter.camera_position = compute_plane_camera(np.asarray(mesh.points))


def expand_flat_clim(clim: tuple[float, float]) -> tuple[float, float]:
    lower, upper = clim
    if lower != upper:
        return lower, upper
    padding = abs(lower) * 0.01 or 1.0
    return lower - padding, upper + padding


def compute_global_clim(files: list[Path]) -> tuple[float, float]:
    pv = load_pyvista()
    global_min = float("inf")
    global_max = float("-inf")

    for file_path in files:
        mesh = pv.read(str(file_path))
        scalar_name = pick_scalar_name(mesh)
        values = np.asarray(mesh.get_array(scalar_name), dtype=np.float64)
        global_min = min(global_min, float(values.min()))
        global_max = max(global_max, float(values.max()))

    return expand_flat_clim((global_min, global_max))


def compute_percentile_clim(
    files: list[Path],
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    pv = load_pyvista()
    values_list = []
    for file_path in files:
        mesh = pv.read(str(file_path))
        scalar_name = pick_scalar_name(mesh)
        values = np.asarray(mesh.get_array(scalar_name), dtype=np.float64).ravel()
        values_list.append(values)

    merged = np.concatenate(values_list)
    lower = float(np.percentile(merged, lower_percentile))
    upper = float(np.percentile(merged, upper_percentile))
    if lower == upper:
        lower = float(merged.min())
        upper = float(merged.max())
    return expand_flat_clim((lower, upper))


def build_default_png_path(
    output_dir: Path,
    case_name: str,
    component: str,
    frame_file: Path,
) -> Path:
    return output_dir / f"{case_name}_{component}_{frame_file.stem}.png"


def build_default_gif_path(
    output_dir: Path,
    case_name: str,
    component: str,
    selected_files: list[Path],
    fps: float,
) -> Path:
    fps_text = str(int(fps)) if float(fps).is_integer() else str(fps).replace(".", "p")
    return (
        output_dir
        / f"{case_name}_{component}_{selected_files[0].stem}_to_{selected_files[-1].stem}_{fps_text}fps.gif"
    )


def resolve_output_request(
    request: str | None,
    auto_path: Path,
) -> Path | None:
    if request is None:
        return None

    request_text = str(request).strip()
    if request_text.lower() == "none":
        return None
    if request_text.lower() == "auto":
        auto_path.parent.mkdir(parents=True, exist_ok=True)
        return auto_path

    path = Path(request_text)
    if not path.is_absolute():
        path = auto_path.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def can_write_gif() -> bool:
    return importlib.util.find_spec("imageio") is not None


def render_png(
    frame_file: Path,
    output_path: Path,
    window_size: tuple[int, int],
    cmap: str,
    clim: tuple[float, float],
    view: str,
    show_edges: bool,
) -> Path:
    pv = load_pyvista()
    mesh = pv.read(str(frame_file))
    scalar_name = pick_scalar_name(mesh)

    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.disable_anti_aliasing()
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        lighting=False,
        scalar_bar_args={"title": scalar_name.strip() or scalar_name},
    )
    plotter.add_text(frame_file.name, font_size=14, color="black")
    apply_view(plotter, mesh, view)
    plotter.screenshot(str(output_path))
    plotter.close()
    return output_path


def render_gif(
    selected_files: list[Path],
    output_path: Path,
    window_size: tuple[int, int],
    fps: float,
    cmap: str,
    clim: tuple[float, float],
    view: str,
    show_edges: bool,
) -> Path:
    pv = load_pyvista()
    first_frame = pv.read(str(selected_files[0]))
    scalar_name = pick_scalar_name(first_frame)

    if scalar_name not in first_frame.point_data:
        raise ValueError(
            f"当前字段 {scalar_name!r} 不在 point_data 里，"
            "这版脚本只处理点标量时序。"
        )

    base_mesh = first_frame.copy(deep=True)
    pv.global_theme.multi_samples = 0
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.disable_anti_aliasing()
    plotter.set_background("white")
    plotter.add_mesh(
        base_mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        lighting=False,
        scalar_bar_args={"title": scalar_name.strip() or scalar_name},
    )
    plotter.add_text("Generating GIF...", name="hint", position="upper_left", font_size=10)
    apply_view(plotter, base_mesh, view)
    plotter.open_gif(str(output_path), fps=max(1, int(round(fps))))

    for index, file_path in enumerate(selected_files, start=1):
        frame_mesh = pv.read(str(file_path))
        current_scalar_name = pick_scalar_name(frame_mesh)

        if frame_mesh.n_points != base_mesh.n_points:
            raise ValueError(
                f"{file_path.name} 的点数与首帧不同: "
                f"{frame_mesh.n_points} != {base_mesh.n_points}"
            )
        if not np.allclose(frame_mesh.points, base_mesh.points):
            raise ValueError(
                f"{file_path.name} 的几何与首帧不同。"
                "这版脚本按固定几何、更新点标量的方式保存。"
            )

        base_mesh.point_data[scalar_name] = frame_mesh.point_data[current_scalar_name]
        plotter.add_text(
            f"{file_path.name} ({index}/{len(selected_files)})",
            name="frame_label",
            position="lower_left",
            font_size=12,
        )
        plotter.render()
        plotter.write_frame()

    plotter.close()
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 Bohai 数据目录结构导出波场单帧 PNG 和 GIF。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"数据集根目录，默认: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--case",
        dest="case_name",
        default="S1.ABAZ",
        help="样本目录名，例如 S1.ABAZ。",
    )
    parser.add_argument(
        "--component",
        choices=COMPONENTS,
        default="Total",
        help="分量目录，默认 Total。",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="保存 PNG 时使用的单帧序号（1-based）。默认使用 --start 对应帧。",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="GIF 起始帧序号（1-based），默认 1。",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="GIF 结束帧序号（1-based），默认到最后一帧。",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="GIF 步长，默认每帧都保存。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="GIF 帧率，默认 8。",
    )
    parser.add_argument(
        "--cmap",
        default="turbo",
        help="颜色表，默认 turbo。",
    )
    parser.add_argument(
        "--view",
        choices=("auto", "xy", "xz", "yz", "isometric"),
        default="auto",
        help="视角。auto 会自动正视面网格，默认 auto。",
    )
    parser.add_argument(
        "--clim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="手动指定固定颜色范围，例如 --clim 0 0.5。",
    )
    parser.add_argument(
        "--clim-mode",
        choices=("percentile", "global"),
        default="percentile",
        help="颜色范围模式：percentile 或 global。",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="percentile 模式下的下百分位，默认 1。",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.9,
        help="percentile 模式下的上百分位，默认 99.9。",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="输出图片宽度，默认 1600。",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="输出图片高度，默认 900。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--save-png",
        nargs="?",
        const="auto",
        default="auto",
        metavar="PATH",
        help="保存单帧 PNG。默认启用并自动命名；传 none 禁用。",
    )
    parser.add_argument(
        "--save-gif",
        nargs="?",
        const="auto",
        default="none",
        metavar="PATH",
        help="保存 GIF。默认不保存；传 --save-gif 可自动命名。",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="显示网格边线。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps 必须 > 0")
    if not 0 <= args.lower_percentile < args.upper_percentile <= 100:
        raise ValueError("百分位参数必须满足 0 <= lower < upper <= 100")

    base_dir = args.dataset_root / args.case_name / args.component
    if not base_dir.exists():
        raise FileNotFoundError(f"目录不存在: {base_dir}")

    all_files = resolve_snapshot_files(base_dir)
    end = len(all_files) if args.end is None else args.end
    selected_files = select_snapshot_files(
        files=all_files,
        start=args.start,
        end=end,
        step=args.step,
        frame=None,
    )

    png_frame_number = args.start if args.frame is None else args.frame
    png_file = select_snapshot_files(
        files=all_files,
        start=args.start,
        end=end,
        step=args.step,
        frame=png_frame_number,
    )[0]

    if args.clim is not None:
        clim = expand_flat_clim((float(args.clim[0]), float(args.clim[1])))
        clim_description = "manual"
    elif args.clim_mode == "global":
        clim = compute_global_clim(selected_files)
        clim_description = "global"
    else:
        clim = compute_percentile_clim(
            selected_files,
            args.lower_percentile,
            args.upper_percentile,
        )
        clim_description = (
            f"percentile [{args.lower_percentile:.3g}, {args.upper_percentile:.3g}]"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = resolve_output_request(
        args.save_png,
        build_default_png_path(args.output_dir, args.case_name, args.component, png_file),
    )
    gif_path = resolve_output_request(
        args.save_gif,
        build_default_gif_path(
            args.output_dir,
            args.case_name,
            args.component,
            selected_files,
            args.fps,
        ),
    )

    if gif_path is not None and not can_write_gif():
        raise ModuleNotFoundError(
            "当前环境没有安装 imageio，无法写 GIF。"
            "请先执行: pip install imageio 或 conda install -n pytorch imageio"
        )

    window_size = (args.width, args.height)

    print(f"输入目录: {base_dir}")
    print(f"选中帧数: {len(selected_files)} / {len(all_files)}")
    print(f"单帧 PNG: {png_file.name}")
    print(f"颜色范围: [{clim[0]:.6g}, {clim[1]:.6g}]")
    print(f"颜色范围模式: {clim_description}")
    print(f"视角: {args.view}")
    print(f"输出目录: {args.output_dir}")

    if png_path is not None:
        png_written = render_png(
            frame_file=png_file,
            output_path=png_path,
            window_size=window_size,
            cmap=args.cmap,
            clim=clim,
            view=args.view,
            show_edges=args.show_edges,
        )
        print(f"PNG 已保存: {png_written}")

    if gif_path is not None:
        gif_written = render_gif(
            selected_files=selected_files,
            output_path=gif_path,
            window_size=window_size,
            fps=args.fps,
            cmap=args.cmap,
            clim=clim,
            view=args.view,
            show_edges=args.show_edges,
        )
        print(f"GIF 已保存: {gif_written}")


if __name__ == "__main__":
    main()

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from save_wave_movie import (
    apply_plain_axis_format,
    build_default_gif_path,
    build_default_png_path,
    build_frame_title,
    cells_to_triangles,
    compute_font_sizes,
    project_points_for_view,
    resolve_snapshot_files,
    select_snapshot_files,
    snapshot_sort_key,
)


class SaveWaveMovieTests(unittest.TestCase):
    def test_snapshot_sort_key_orders_numerically(self) -> None:
        names = [
            Path("AVS_movie_000010.inp"),
            Path("AVS_movie_000002.inp"),
            Path("AVS_movie_000001.inp"),
        ]

        ordered = sorted(names, key=snapshot_sort_key)

        self.assertEqual(
            [path.name for path in ordered],
            [
                "AVS_movie_000001.inp",
                "AVS_movie_000002.inp",
                "AVS_movie_000010.inp",
            ],
        )

    def test_resolve_snapshot_files_filters_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            (base_dir / "AVS_movie_000003.inp").write_text("")
            (base_dir / "AVS_movie_000001.inp").write_text("")
            (base_dir / "AVS_movie_000002.inp").write_text("")
            (base_dir / "notes.txt").write_text("")

            files = resolve_snapshot_files(base_dir)

        self.assertEqual(
            [path.name for path in files],
            [
                "AVS_movie_000001.inp",
                "AVS_movie_000002.inp",
                "AVS_movie_000003.inp",
            ],
        )

    def test_select_snapshot_files_by_range(self) -> None:
        files = [Path(f"AVS_movie_{index:06d}.inp") for index in range(1, 7)]

        selected = select_snapshot_files(files, start=2, end=6, step=2, frame=None)

        self.assertEqual(
            [path.name for path in selected],
            [
                "AVS_movie_000002.inp",
                "AVS_movie_000004.inp",
                "AVS_movie_000006.inp",
            ],
        )

    def test_select_snapshot_files_single_frame_overrides_range(self) -> None:
        files = [Path(f"AVS_movie_{index:06d}.inp") for index in range(1, 7)]

        selected = select_snapshot_files(files, start=1, end=6, step=1, frame=5)

        self.assertEqual([path.name for path in selected], ["AVS_movie_000005.inp"])

    def test_build_default_png_path(self) -> None:
        output_dir = Path("/tmp/visualizations")
        frame = Path("AVS_movie_000050.inp")

        png_path = build_default_png_path(output_dir, "S1.ABAZ", "Total", frame)

        self.assertEqual(
            png_path,
            output_dir / "S1.ABAZ_Total_AVS_movie_000050.png",
        )

    def test_build_default_gif_path(self) -> None:
        output_dir = Path("/tmp/visualizations")
        selected = [
            Path("AVS_movie_000001.inp"),
            Path("AVS_movie_000100.inp"),
        ]

        gif_path = build_default_gif_path(output_dir, "S1.ABAZ", "Total", selected, 8.0)

        self.assertEqual(
            gif_path,
            output_dir / "S1.ABAZ_Total_AVS_movie_000001_to_AVS_movie_000100_8fps.gif",
        )

    def test_build_frame_title(self) -> None:
        title = build_frame_title("S1.ABAZ", "Total", Path("AVS_movie_000050.inp"))

        self.assertEqual(title, "S1.ABAZ | Total | AVS_movie_000050")

    def test_cells_to_triangles_splits_quads(self) -> None:
        cells = np.array([4, 0, 1, 2, 3, 4, 3, 2, 4, 5], dtype=np.int64)

        triangles = cells_to_triangles(cells)

        expected = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [3, 2, 4],
                [3, 4, 5],
            ],
            dtype=np.int64,
        )
        self.assertTrue(np.array_equal(triangles, expected))

    def test_project_points_for_view_converts_to_km(self) -> None:
        points = np.array(
            [
                [1000.0, 2000.0, 3000.0],
                [4000.0, 5000.0, 6000.0],
            ]
        )

        x_values, y_values, x_label, y_label = project_points_for_view(points, "yz")

        self.assertTrue(np.allclose(x_values, np.array([2.0, 5.0])))
        self.assertTrue(np.allclose(y_values, np.array([3.0, 6.0])))
        self.assertEqual(x_label, "y (km)")
        self.assertEqual(y_label, "z (km)")

    def test_apply_plain_axis_format_uses_plain_ticks(self) -> None:
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)

        apply_plain_axis_format(ax, "x (km)", "y (km)", label_size=17, tick_size=15)

        self.assertEqual(ax.get_xlabel(), "x (km)")
        self.assertEqual(ax.get_ylabel(), "y (km)")
        self.assertFalse(ax.xaxis.get_major_formatter().get_useOffset())
        self.assertFalse(ax.yaxis.get_major_formatter().get_useOffset())

    def test_compute_font_sizes_scales_with_height(self) -> None:
        small = compute_font_sizes((1200, 900))
        large = compute_font_sizes((1200, 1800))

        self.assertGreaterEqual(large["title"], small["title"])
        self.assertGreaterEqual(large["label"], small["label"])
        self.assertGreaterEqual(large["tick"], small["tick"])
        self.assertGreaterEqual(large["colorbar_label"], small["colorbar_label"])
        self.assertGreaterEqual(large["colorbar_tick"], small["colorbar_tick"])


if __name__ == "__main__":
    unittest.main()

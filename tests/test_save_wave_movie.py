import tempfile
import unittest
from pathlib import Path

from save_wave_movie import (
    build_default_gif_path,
    build_default_png_path,
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


if __name__ == "__main__":
    unittest.main()

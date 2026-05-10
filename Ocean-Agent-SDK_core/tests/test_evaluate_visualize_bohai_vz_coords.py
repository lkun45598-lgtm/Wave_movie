from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_visualize_bohai_vz as eval_vz  # noqa: E402


def test_coordinate_extent_defaults_to_offset_km_like_diagnostic_plots(tmp_path: Path) -> None:
    static_hr = tmp_path / "static_variables" / "hr"
    static_hr.mkdir(parents=True)
    np.save(static_hr / "00_lon_rho.npy", np.array([[10_000.0, 20_000.0]], dtype=np.float32))
    np.save(static_hr / "10_lat_rho.npy", np.array([[-50_000.0, -30_000.0]], dtype=np.float32))

    extent, xlabel, ylabel = eval_vz.load_coordinate_extent(tmp_path)

    assert extent == [0.0, 10.0, 0.0, 20.0]
    assert xlabel == "X offset (km)"
    assert ylabel == "Y offset (km)"


def test_coordinate_extent_can_use_avs_absolute_km_like_wave_movie_plots(tmp_path: Path) -> None:
    static_hr = tmp_path / "static_variables" / "hr"
    static_hr.mkdir(parents=True)
    np.save(static_hr / "00_lon_rho.npy", np.array([[10_000.0, 20_000.0]], dtype=np.float32))
    np.save(static_hr / "10_lat_rho.npy", np.array([[-50_000.0, -30_000.0]], dtype=np.float32))

    extent, xlabel, ylabel = eval_vz.load_coordinate_extent(tmp_path, coordinate_mode="absolute")

    assert extent == [10.0, 20.0, -50.0, -30.0]
    assert xlabel == "x (km)"
    assert ylabel == "y (km)"


def test_lonlat_source_projects_to_avs_utm60s_km_position() -> None:
    x_km, y_km = eval_vz.project_lonlat_to_avs_xy_km(
        longitude=175.455693,
        latitude=-41.455012,
        utm_zone=60,
    )

    assert x_km == np.float64(x_km)
    assert y_km == np.float64(y_km)
    assert abs(x_km - 371.0167) < 0.05
    assert abs(y_km - -4590.4206) < 0.05


def test_source_locations_parse_case_name_and_metadata(tmp_path: Path) -> None:
    info_path = tmp_path / "info.txt"
    info_path.write_text(
        "\n".join(
            [
                "############## 震源位置信息 ##############",
                "WRRZ 175.455693 -41.455012",
                "TTTZ 174.304150 -35.339980",
            ]
        ),
        encoding="utf-8",
    )

    locations = eval_vz.load_source_locations(info_path)

    assert set(locations) == {"WRRZ", "TTTZ"}
    assert eval_vz.source_name_from_case("S1_WRRZ") == "WRRZ"
    assert eval_vz.source_name_from_case("S1.TTTZ") == "TTTZ"


def test_source_marker_can_be_returned_in_offset_coordinate_frame(tmp_path: Path) -> None:
    static_hr = tmp_path / "static_variables" / "hr"
    static_hr.mkdir(parents=True)
    np.save(static_hr / "00_lon_rho.npy", np.array([[12_482.844, 680_228.3]], dtype=np.float32))
    np.save(static_hr / "10_lat_rho.npy", np.array([[-4_800_530.0, -3_769_130.2]], dtype=np.float32))
    source_locations = {"WRRZ": (175.455693, -41.455012)}

    marker = eval_vz.source_marker_for_case(
        "S1_WRRZ",
        source_locations,
        utm_zone=60,
        coordinate_mode="offset",
        dataset_root=tmp_path,
    )

    assert marker is not None
    x_km, y_km = marker
    assert abs(x_km - 358.53) < 0.05
    assert abs(y_km - 210.11) < 0.05

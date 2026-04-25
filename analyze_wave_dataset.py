#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


COMPONENTS = ("Total", "X", "Y", "Z")
FRAME_PATTERN = re.compile(r"AVS_movie_(\d{6})\.inp$")


def is_hidden_artifact(path: Path) -> bool:
    return path.name.startswith("._")


def parse_known_metadata(info_path: Path) -> dict[str, object]:
    if not info_path.exists():
        return {}

    lines = info_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    metadata: dict[str, object] = {}
    declared_entries: list[dict[str, object]] = []
    in_location_section = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if "震源位置信息" in line:
            in_location_section = True
            continue

        if in_location_section:
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    longitude = float(parts[1])
                    latitude = float(parts[2])
                except ValueError:
                    continue
                declared_entries.append(
                    {"name": name, "longitude": longitude, "latitude": latitude}
                )
            continue

        if "震源类型：" in line and "主频：" in line:
            left, right = line.split("主频：", 1)
            metadata["source_type"] = left.split("震源类型：", 1)[1].strip()
            metadata["dominant_frequency"] = right.strip()
            continue

        if "震源深度：" in line:
            metadata["source_depth"] = line.split("震源深度：", 1)[1].strip()
            continue

        if "地区：" in line:
            metadata["region"] = line.split("地区：", 1)[1].strip()
            continue

        if "总时长：" in line:
            metadata["total_duration"] = line.split("总时长：", 1)[1].strip()
            continue

        if "总帧数：" in line:
            metadata["total_frames_declared"] = line.split("总帧数：", 1)[1].strip()
            continue

        if "采样步长：" in line:
            metadata["time_step"] = line.split("采样步长：", 1)[1].strip()
            continue

        if "速度分量：" in line:
            component_text = line.split("速度分量：", 1)[1].strip()
            metadata["declared_components"] = [
                item.strip() for item in component_text.split("、") if item.strip()
            ]

    metadata["declared_location_entries"] = declared_entries
    return metadata


def classify_case_name(case_name: str) -> str:
    short_name = case_name.replace("S1.", "", 1)
    if short_name.startswith("NZ"):
        return "NZ_indexed"
    if short_name.startswith("R") and len(short_name) >= 2 and short_name[1].isdigit():
        return "R_indexed"
    if short_name.startswith("L") and len(short_name) >= 2 and short_name[1].isdigit():
        return "L_indexed"
    return "named"


def iter_case_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not is_hidden_artifact(path)
    )


def real_inp_files(component_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(component_dir.iterdir()):
        if not path.is_file():
            continue
        if is_hidden_artifact(path):
            continue
        if FRAME_PATTERN.match(path.name):
            files.append(path)
    return files


def frame_ids(paths: list[Path]) -> list[int]:
    ids: list[int] = []
    for path in paths:
        match = FRAME_PATTERN.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    return ids


def geometry_signature(inp_path: Path) -> tuple[int, int, int, int, int, str]:
    with inp_path.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="ignore").split()
        nnode, nelem, nnode_data, nelem_data, nmodel_data = map(int, header[:5])
        digest = hashlib.md5()
        for _ in range(nnode + nelem):
            digest.update(handle.readline())
    return nnode, nelem, nnode_data, nelem_data, nmodel_data, digest.hexdigest()


def parse_inp_geometry(inp_path: Path) -> dict[str, object]:
    with inp_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().split()
        nnode, nelem, nnode_data, nelem_data, nmodel_data = map(int, header[:5])

        x_min = y_min = z_min = float("inf")
        x_max = y_max = z_max = float("-inf")
        for _ in range(nnode):
            parts = handle.readline().split()
            x_value = float(parts[1])
            y_value = float(parts[2])
            z_value = float(parts[3])
            x_min = min(x_min, x_value)
            x_max = max(x_max, x_value)
            y_min = min(y_min, y_value)
            y_max = max(y_max, y_value)
            z_min = min(z_min, z_value)
            z_max = max(z_max, z_value)

        element_types = Counter()
        for _ in range(nelem):
            parts = handle.readline().split()
            if len(parts) >= 3:
                element_types[parts[2]] += 1

        data_header = handle.readline().split()
        field_labels = handle.readline().strip()

    return {
        "nnode": nnode,
        "nelem": nelem,
        "nnode_data": nnode_data,
        "nelem_data": nelem_data,
        "nmodel_data": nmodel_data,
        "coordinate_ranges": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_min, z_max],
        },
        "element_types": dict(element_types),
        "data_header": data_header,
        "field_labels": field_labels,
    }


def analyze_dataset(dataset_root: Path) -> dict[str, object]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    info_path = dataset_root / "数据信息.txt"
    metadata = parse_known_metadata(info_path)
    case_dirs = iter_case_dirs(dataset_root)

    hidden_artifact_files = 0
    valid_inp_files = 0
    valid_inp_total_bytes = 0
    component_presence = Counter()
    case_name_groups = Counter()
    complete_cases: list[str] = []
    partial_cases: list[dict[str, object]] = []
    empty_cases: list[str] = []
    sample_inp: Path | None = None

    for root_item in dataset_root.iterdir():
        if root_item.is_file() and is_hidden_artifact(root_item):
            hidden_artifact_files += 1

    for case_dir in case_dirs:
        case_name_groups[classify_case_name(case_dir.name)] += 1
        real_component_dirs = sorted(
            path.name
            for path in case_dir.iterdir()
            if path.is_dir() and not is_hidden_artifact(path)
        )

        component_counts: dict[str, int] = {}
        component_frame_ranges: dict[str, list[int] | None] = {}
        total_real_frames = 0

        for child in case_dir.iterdir():
            if child.is_file() and is_hidden_artifact(child):
                hidden_artifact_files += 1

        for component_name in real_component_dirs:
            component_presence[component_name] += 1
            component_dir = case_dir / component_name

            for child in component_dir.iterdir():
                if child.is_file() and is_hidden_artifact(child):
                    hidden_artifact_files += 1

            files = real_inp_files(component_dir)
            file_count = len(files)
            component_counts[component_name] = file_count
            total_real_frames += file_count

            if files:
                ids = frame_ids(files)
                component_frame_ranges[component_name] = [min(ids), max(ids)]
                valid_inp_files += file_count
                valid_inp_total_bytes += sum(path.stat().st_size for path in files)
                if sample_inp is None:
                    sample_inp = files[0]
            else:
                component_frame_ranges[component_name] = None

        all_components_present = sorted(component_counts) == list(COMPONENTS)
        full_component_counts = all(
            component_counts.get(component_name, 0) == 100 for component_name in COMPONENTS
        )

        if total_real_frames == 0:
            empty_cases.append(case_dir.name)
        elif all_components_present and full_component_counts:
            complete_cases.append(case_dir.name)
        else:
            partial_cases.append(
                {
                    "case": case_dir.name,
                    "component_counts": component_counts,
                    "component_frame_ranges": component_frame_ranges,
                }
            )

    if sample_inp is None:
        raise RuntimeError("No valid .inp files found in the dataset.")

    geometry_summary = parse_inp_geometry(sample_inp)
    reference_signature = geometry_signature(sample_inp)

    consistent_geometry_cases = 0
    mismatched_geometry_cases: list[str] = []
    for case_name in complete_cases:
        candidate = dataset_root / case_name / "Total" / "AVS_movie_000001.inp"
        if not candidate.exists():
            mismatched_geometry_cases.append(case_name)
            continue
        if geometry_signature(candidate) == reference_signature:
            consistent_geometry_cases += 1
        else:
            mismatched_geometry_cases.append(case_name)

    declared_names = {
        entry["name"] for entry in metadata.get("declared_location_entries", [])
    }
    actual_short_names = {case_dir.name.replace("S1.", "", 1) for case_dir in case_dirs}

    complete_case_count = len(complete_cases)
    frames_per_complete_case = 100
    component_count = len(COMPONENTS)

    return {
        "dataset_root": str(dataset_root),
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key != "declared_location_entries"
        },
        "declared_location_entry_count": len(metadata.get("declared_location_entries", [])),
        "case_summary": {
            "case_dirs": len(case_dirs),
            "complete_cases": complete_case_count,
            "partial_cases": len(partial_cases),
            "empty_cases": len(empty_cases),
            "empty_case_names": empty_cases,
            "partial_case_details": partial_cases,
            "case_name_groups": dict(case_name_groups),
            "component_presence": dict(component_presence),
        },
        "file_summary": {
            "valid_inp_files": valid_inp_files,
            "valid_inp_total_bytes": valid_inp_total_bytes,
            "hidden_artifact_files": hidden_artifact_files,
        },
        "data_dimensions": {
            "logical_shape": {
                "cases": complete_case_count,
                "components": component_count,
                "frames": frames_per_complete_case,
                "nodes": geometry_summary["nnode"],
            },
            "mesh_topology": {
                "elements": geometry_summary["nelem"],
                "element_types": geometry_summary["element_types"],
            },
            "per_file_header": {
                "nnode_data": geometry_summary["nnode_data"],
                "nelem_data": geometry_summary["nelem_data"],
                "nmodel_data": geometry_summary["nmodel_data"],
                "data_header": geometry_summary["data_header"],
                "field_labels": geometry_summary["field_labels"],
            },
            "coordinate_ranges": geometry_summary["coordinate_ranges"],
        },
        "consistency_checks": {
            "geometry_consistent_across_complete_cases": len(mismatched_geometry_cases) == 0,
            "geometry_checked_cases": complete_case_count,
            "geometry_mismatched_cases": mismatched_geometry_cases,
            "declared_not_present_in_dataset": sorted(declared_names - actual_short_names),
            "present_but_not_declared": sorted(actual_short_names - declared_names),
        },
        "sample_file": str(sample_inp),
    }


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def print_report(summary: dict[str, object]) -> None:
    metadata = summary["metadata"]
    case_summary = summary["case_summary"]
    file_summary = summary["file_summary"]
    data_dimensions = summary["data_dimensions"]
    consistency = summary["consistency_checks"]
    logical_shape = data_dimensions["logical_shape"]
    mesh_topology = data_dimensions["mesh_topology"]
    per_file_header = data_dimensions["per_file_header"]

    print(f"Dataset: {summary['dataset_root']}")
    print()
    print("Metadata")
    print(f"  region: {metadata.get('region', 'N/A')}")
    print(f"  source_type: {metadata.get('source_type', 'N/A')}")
    print(f"  dominant_frequency: {metadata.get('dominant_frequency', 'N/A')}")
    print(f"  source_depth: {metadata.get('source_depth', 'N/A')}")
    print(f"  total_duration: {metadata.get('total_duration', 'N/A')}")
    print(f"  declared_frames: {metadata.get('total_frames_declared', 'N/A')}")
    print(f"  time_step: {metadata.get('time_step', 'N/A')}")
    print(f"  declared_components: {metadata.get('declared_components', [])}")
    print(f"  declared_location_entries: {summary['declared_location_entry_count']}")
    print()
    print("Cases")
    print(f"  case_dirs: {case_summary['case_dirs']}")
    print(f"  complete_cases: {case_summary['complete_cases']}")
    print(f"  partial_cases: {case_summary['partial_cases']}")
    print(f"  empty_cases: {case_summary['empty_cases']}")
    if case_summary["empty_case_names"]:
        print(f"  empty_case_names: {case_summary['empty_case_names']}")
    print(f"  case_name_groups: {case_summary['case_name_groups']}")
    print(f"  component_presence: {case_summary['component_presence']}")
    print()
    print("Files")
    print(f"  valid_inp_files: {file_summary['valid_inp_files']}")
    print(
        "  valid_inp_total_size: "
        f"{format_bytes(file_summary['valid_inp_total_bytes'])}"
    )
    print(f"  hidden_artifact_files: {file_summary['hidden_artifact_files']}")
    print()
    print("Data Dimensions")
    print(
        "  logical_shape: "
        f"cases={logical_shape['cases']} × "
        f"components={logical_shape['components']} × "
        f"frames={logical_shape['frames']} × "
        f"nodes={logical_shape['nodes']}"
    )
    print(
        "  mesh_topology: "
        f"elements={mesh_topology['elements']}, "
        f"element_types={mesh_topology['element_types']}"
    )
    print(
        "  per_file_header: "
        f"nnode_data={per_file_header['nnode_data']}, "
        f"nelem_data={per_file_header['nelem_data']}, "
        f"nmodel_data={per_file_header['nmodel_data']}, "
        f"data_header={per_file_header['data_header']}, "
        f"field_labels={per_file_header['field_labels']!r}"
    )
    print(f"  coordinate_ranges: {data_dimensions['coordinate_ranges']}")
    print()
    print("Consistency")
    print(
        "  geometry_consistent_across_complete_cases: "
        f"{consistency['geometry_consistent_across_complete_cases']}"
    )
    print(f"  geometry_checked_cases: {consistency['geometry_checked_cases']}")
    print(
        "  declared_not_present_in_dataset: "
        f"{len(consistency['declared_not_present_in_dataset'])}"
    )
    print(f"  present_but_not_declared: {consistency['present_but_not_declared']}")
    print()
    print(f"Sample file: {summary['sample_file']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze wave movie dataset structure and dimensions."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="/data/Bohai_Sea/To_ZGT_wave_movie",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text report.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = analyze_dataset(Path(args.dataset_root))

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

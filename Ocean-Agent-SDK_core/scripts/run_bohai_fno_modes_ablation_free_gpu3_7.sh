#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/user/lz/wave_movie"
PYTHON="/home/lz/miniconda3/envs/pytorch/bin/python"
MAIN="${ROOT}/Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/main.py"
SRC_DIR="${ROOT}/Ocean-Agent-SDK_core/configs/bohai_xyz_4x/fno_modes_ablation"
RUN_DIR="${SRC_DIR}/runtime_free_gpu3_7"

mkdir -p "${RUN_DIR}"

make_config() {
  local src="$1"
  local dst="$2"
  local log_dir="$3"
  "${PYTHON}" - "$src" "$dst" "$log_dir" <<'PY'
from pathlib import Path
import sys
import yaml

src, dst, log_dir = map(Path, sys.argv[1:])
with src.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg.setdefault("train", {})
cfg["train"]["device"] = 0
cfg["train"]["device_ids"] = [0]
cfg["train"]["distribute"] = False

cfg.setdefault("log", {})
cfg["log"]["log_dir"] = str(log_dir)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_one() {
  local gpu="$1"
  local session="$2"
  local config="$3"
  local log_dir="$4"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "skip existing session: ${session}"
    return
  fi

  mkdir -p "${log_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${gpu} '${PYTHON}' '${MAIN}' --config '${config}' 2>&1 | tee '${log_dir}/train_stdout.log'"
  echo "started ${session} on GPU${gpu}: ${config}"
}

make_config \
  "${SRC_DIR}/bohai_vz_point_4x_fno2d_modes_baseline_gpu1.yaml" \
  "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_baseline_gpu3.yaml" \
  "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Baseline_GPU3"

make_config \
  "${SRC_DIR}/bohai_vz_point_4x_fno2d_modes_medium_gpu2.yaml" \
  "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_medium_gpu4.yaml" \
  "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Medium_GPU4"

make_config \
  "${SRC_DIR}/bohai_vz_point_4x_fno2d_modes_high_gpu3.yaml" \
  "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_high_gpu5.yaml" \
  "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_High_GPU5"

make_config \
  "${SRC_DIR}/bohai_vz_point_4x_fno2d_modes_highfront_gpu4.yaml" \
  "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_highfront_gpu6.yaml" \
  "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_HighFront_GPU6"

make_config \
  "${SRC_DIR}/bohai_vz_point_4x_temporal5_fno2d_residual_hf_modes_high_gpu7.yaml" \
  "${RUN_DIR}/bohai_vz_point_4x_temporal5_fno2d_residual_hf_modes_high_gpu7.yaml" \
  "/data1/user/lz/wave_movie/testouts/FNO2d_Temporal5ResidualHF_POINT_Modes_High_GPU7"

run_one 3 bohai_fno_free_g3_baseline "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_baseline_gpu3.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Baseline_GPU3"
sleep 20
run_one 4 bohai_fno_free_g4_medium "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_medium_gpu4.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Medium_GPU4"
sleep 20
run_one 5 bohai_fno_free_g5_high "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_high_gpu5.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_High_GPU5"
sleep 20
run_one 6 bohai_fno_free_g6_highfront "${RUN_DIR}/bohai_vz_point_4x_fno2d_modes_highfront_gpu6.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_HighFront_GPU6"
sleep 20
run_one 7 bohai_fno_free_g7_temporal "${RUN_DIR}/bohai_vz_point_4x_temporal5_fno2d_residual_hf_modes_high_gpu7.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_Temporal5ResidualHF_POINT_Modes_High_GPU7"

tmux list-sessions | grep 'bohai_fno_free_' || true

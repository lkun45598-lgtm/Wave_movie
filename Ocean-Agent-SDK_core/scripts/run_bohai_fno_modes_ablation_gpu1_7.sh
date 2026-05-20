#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/user/lz/wave_movie"
PYTHON="/home/lz/miniconda3/envs/pytorch/bin/python"
MAIN="${ROOT}/Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/main.py"
CONFIG_DIR="${ROOT}/Ocean-Agent-SDK_core/configs/bohai_xyz_4x/fno_modes_ablation"

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
}

run_one 1 bohai_fno_modes_g1 "${CONFIG_DIR}/bohai_vz_point_4x_fno2d_modes_baseline_gpu1.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Baseline_GPU1"
sleep 15
run_one 2 bohai_fno_modes_g2 "${CONFIG_DIR}/bohai_vz_point_4x_fno2d_modes_medium_gpu2.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Medium_GPU2"
sleep 15
run_one 3 bohai_fno_modes_g3 "${CONFIG_DIR}/bohai_vz_point_4x_fno2d_modes_high_gpu3.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_High_GPU3"
sleep 15
run_one 4 bohai_fno_modes_g4 "${CONFIG_DIR}/bohai_vz_point_4x_fno2d_modes_highfront_gpu4.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_HighFront_GPU4"
sleep 15
run_one 5 bohai_fno_modes_g5 "${CONFIG_DIR}/bohai_vz_point_4x_fno2d_modes_aggressive_gpu5.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_Aggressive_GPU5"
sleep 15
run_one 6 bohai_fno_modes_g6 "${CONFIG_DIR}/bohai_vz_point_4x_fno2d_modes_high_width96_gpu6.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_POINT_Modes_HighWidth96_GPU6"
sleep 15
run_one 7 bohai_fno_modes_g7 "${CONFIG_DIR}/bohai_vz_point_4x_temporal5_fno2d_residual_hf_modes_high_gpu7.yaml" "/data1/user/lz/wave_movie/testouts/FNO2d_Temporal5ResidualHF_POINT_Modes_High_GPU7"

tmux list-sessions | grep 'bohai_fno_modes_g' || true

#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/user/lz/wave_movie/Ocean-Agent-SDK_core"
TRAIN_DIR="${ROOT}/scripts/ocean-SR-training-masked"
PYTHON="/home/lz/miniconda3/envs/pytorch/bin/python"
TORCHRUN="/home/lz/miniconda3/envs/pytorch/bin/torchrun"

CONFIGS=(
  "${ROOT}/configs/bohai_xyz_4x/bohai_vz_point_4x_temporal5_fno2d_residual_hf_pgn_ddp_gpu12.yaml"
  "${ROOT}/configs/bohai_xyz_4x/bohai_vz_point_4x_temporal5_edsr_residual_pgn_ddp_gpu12.yaml"
)

export CUDA_VISIBLE_DEVICES=6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl_bohai_vz_point_temporal5_gpu67

mkdir -p "${MPLCONFIGDIR}"
cd "${TRAIN_DIR}"

for config in "${CONFIGS[@]}"; do
  log_dir="$("${PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["log"]["log_dir"])' "${config}")"
  mkdir -p "${log_dir}"
  echo "[$(date --iso-8601=seconds)] START ${config}" | tee -a "${log_dir}/launch.log"
  "${TORCHRUN}" --standalone --nproc_per_node=2 main_ddp.py --config "${config}" 2>&1 | tee -a "${log_dir}/launch.log"
  echo "[$(date --iso-8601=seconds)] END ${config}" | tee -a "${log_dir}/launch.log"
done

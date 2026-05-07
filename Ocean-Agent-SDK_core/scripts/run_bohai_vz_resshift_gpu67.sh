#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/user/lz/wave_movie/Ocean-Agent-SDK_core"
TRAIN_DIR="${ROOT}/scripts/ocean-SR-training-masked"
CONFIG="${ROOT}/configs/bohai_xyz_4x/bohai_vz_4x_resshift_pgn_ddp_gpu67.yaml"
PYTHON="/home/lz/miniconda3/envs/pytorch/bin/python"
TORCHRUN="/home/lz/miniconda3/envs/pytorch/bin/torchrun"

export CUDA_VISIBLE_DEVICES=6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl_bohai_vz_resshift_gpu67

mkdir -p "${MPLCONFIGDIR}"
cd "${TRAIN_DIR}"

log_dir="$("${PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["log"]["log_dir"])' "${CONFIG}")"
mkdir -p "${log_dir}"
echo "[$(date --iso-8601=seconds)] START ${CONFIG}" | tee -a "${log_dir}/launch.log"
"${TORCHRUN}" --standalone --nproc_per_node=2 main_ddp.py --config "${CONFIG}" 2>&1 | tee -a "${log_dir}/launch.log"
echo "[$(date --iso-8601=seconds)] END ${CONFIG}" | tee -a "${log_dir}/launch.log"

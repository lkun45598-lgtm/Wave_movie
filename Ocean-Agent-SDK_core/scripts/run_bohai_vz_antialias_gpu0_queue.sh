#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/user/lz/wave_movie/Ocean-Agent-SDK_core"
TRAIN_DIR="${ROOT}/scripts/ocean-SR-training-masked"
PYTHON="/home/lz/miniconda3/envs/pytorch/bin/python"
TORCHRUN="/home/lz/miniconda3/envs/pytorch/bin/torchrun"

CONFIGS=(
  "${ROOT}/configs/bohai_xyz_4x/bohai_vz_antialias_4x_edsr_pgn_gpu0.yaml"
  "${ROOT}/configs/bohai_xyz_4x/bohai_vz_antialias_4x_fno2d_pgn_gpu0.yaml"
)

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl_bohai_vz_antialias_gpu0
export WAIT_MAX_USED_MB="${WAIT_MAX_USED_MB:-2000}"
export WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"

WAIT_GPU_INDEX=0

mkdir -p "${MPLCONFIGDIR}"
cd "${TRAIN_DIR}"

wait_for_gpu() {
  while true; do
    local status
    local used_mb
    status="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
    used_mb="$(
      echo "${status}" |
        awk -F, -v gpu="${WAIT_GPU_INDEX}" '$1 + 0 == gpu { gsub(/ /, "", $2); print $2 }'
    )"

    if [[ -n "${used_mb}" && "${used_mb}" -le "${WAIT_MAX_USED_MB}" ]]; then
      echo "[$(date --iso-8601=seconds)] GPU0 ready for training"
      return 0
    fi

    echo "[$(date --iso-8601=seconds)] waiting for GPU0 memory <= ${WAIT_MAX_USED_MB} MiB"
    echo "${status}"
    sleep "${WAIT_POLL_SECONDS}"
  done
}

wait_for_gpu

for config in "${CONFIGS[@]}"; do
  log_dir="$("${PYTHON}" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["log"]["log_dir"])' "${config}")"
  mkdir -p "${log_dir}"
  echo "[$(date --iso-8601=seconds)] START ${config}" | tee -a "${log_dir}/launch.log"
  "${TORCHRUN}" --standalone --nproc_per_node=1 main_ddp.py --config "${config}" 2>&1 | tee -a "${log_dir}/launch.log"
  echo "[$(date --iso-8601=seconds)] END ${config}" | tee -a "${log_dir}/launch.log"
done

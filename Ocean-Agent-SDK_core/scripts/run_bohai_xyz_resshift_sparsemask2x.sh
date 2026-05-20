#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SESSION="${2:-bohai_xyz_resshift_g${GPU}}"
POST_SESSION="${SESSION}_postrun"

ROOT_DIR="/data1/user/lz/wave_movie"
PYTHON_BIN="/home/lz/miniconda3/envs/pytorch/bin/python"
MAIN="${ROOT_DIR}/Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/main.py"
CONFIG_PATH="${ROOT_DIR}/Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_xyz_sparsemask_2x_resshift_mse_aux_200ep_gpu0.yaml"
RUN_DIR="${ROOT_DIR}/testouts/Resshift_SparseMask2xObserved_XYZ_MSEAux_200ep"
TRAIN_STDOUT="${RUN_DIR}/train_stdout.log"
POSTRUN_LOG="${RUN_DIR}/postrun.log"

mkdir -p "${RUN_DIR}"

session_exists_exact() {
  local session_name="$1"
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fxq "${session_name}"
}

if session_exists_exact "${SESSION}"; then
  echo "training session already exists: ${SESSION}"
else
  tmux new-session -d -s "${SESSION}" \
    "cd '${ROOT_DIR}' && \
     export CUDA_VISIBLE_DEVICES='${GPU}' && \
     export OMP_NUM_THREADS=8 && \
     export MKL_NUM_THREADS=8 && \
     export OPENBLAS_NUM_THREADS=8 && \
     export NUMEXPR_NUM_THREADS=8 && \
     export NUMEXPR_MAX_THREADS=8 && \
     export MPLCONFIGDIR=/tmp/matplotlib-lz && \
     export PYTHONUNBUFFERED=1 && \
     '${PYTHON_BIN}' '${MAIN}' --config '${CONFIG_PATH}' 2>&1 | tee '${TRAIN_STDOUT}'"
  echo "started training session ${SESSION} on GPU${GPU}"
fi

if session_exists_exact "${POST_SESSION}"; then
  echo "postrun session already exists: ${POST_SESSION}"
else
  tmux new-session -d -s "${POST_SESSION}" \
    "cd '${ROOT_DIR}' && \
     mkdir -p '${RUN_DIR}' && \
     exec > >(tee -a '${POSTRUN_LOG}') 2>&1 && \
     echo '[postrun] waiting for ${SESSION}' && \
     while tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fxq '${SESSION}'; do sleep 120; done && \
     echo '[postrun] training session finished' && \
     if [ ! -f '${RUN_DIR}/best_model.pth' ]; then echo 'ERROR: missing best_model.pth'; exit 1; fi && \
     export CUDA_VISIBLE_DEVICES='${GPU}' && \
     export OMP_NUM_THREADS=8 && \
     export MKL_NUM_THREADS=8 && \
     export OPENBLAS_NUM_THREADS=8 && \
     export NUMEXPR_NUM_THREADS=8 && \
     export NUMEXPR_MAX_THREADS=8 && \
     export MPLCONFIGDIR=/tmp/matplotlib-lz && \
     export PYTHONUNBUFFERED=1 && \
     '${PYTHON_BIN}' '${MAIN}' --mode predict --config '${CONFIG_PATH}' && \
     echo '[postrun] prediction finished' && \
     find '${RUN_DIR}/predictions' -maxdepth 1 -type f -name '*_sr.npy' | wc -l"
  echo "started postrun session ${POST_SESSION}"
fi

tmux list-sessions | grep -E "^(${SESSION}|${POST_SESSION}):" || true

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/user/lz/wave_movie"
PYTHON_BIN="/home/lz/miniconda3/envs/pytorch/bin/python"
CONFIG_PATH="${ROOT_DIR}/Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_resshift_source_conditioned_e20_gpu0.yaml"
TRAIN_SESSION="${1:-bohai_e20_source}"

RUN_DIR="${ROOT_DIR}/testouts/Resshift_SourceConditioned_E20_Vz"
PRED_DIR="${RUN_DIR}/predictions"
EVAL_DIR="${ROOT_DIR}/testouts/Resshift_SourceConditioned_E20_eval"
BASELINE_PRED_DIR="${ROOT_DIR}/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions"
POSTRUN_LOG="${RUN_DIR}/postrun.log"

mkdir -p "${RUN_DIR}" "${EVAL_DIR}"
exec > >(tee -a "${POSTRUN_LOG}") 2>&1

echo "[$(date --iso-8601=seconds)] postrun started"
echo "root=${ROOT_DIR}"
echo "config=${CONFIG_PATH}"
echo "train_session=${TRAIN_SESSION}"
echo "run_dir=${RUN_DIR}"
echo "eval_dir=${EVAL_DIR}"

session_exists_exact() {
  local session_name="$1"
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fxq "${session_name}"
}

while session_exists_exact "${TRAIN_SESSION}"; do
  echo "[$(date --iso-8601=seconds)] waiting for tmux session ${TRAIN_SESSION} to finish..."
  sleep 120
done

echo "[$(date --iso-8601=seconds)] training tmux session finished"

if [[ ! -f "${RUN_DIR}/best_model.pth" ]]; then
  echo "ERROR: best checkpoint not found: ${RUN_DIR}/best_model.pth"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lz}"
export PYTHONUNBUFFERED=1

cd "${ROOT_DIR}"

echo "[$(date --iso-8601=seconds)] running predict"
"${PYTHON_BIN}" \
  Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/main.py \
  --mode predict \
  --config "${CONFIG_PATH}"

if [[ ! -d "${PRED_DIR}" ]]; then
  echo "ERROR: prediction directory not found: ${PRED_DIR}"
  exit 1
fi

PRED_COUNT="$(find "${PRED_DIR}" -maxdepth 1 -type f -name '*_sr.npy' | wc -l)"
echo "[$(date --iso-8601=seconds)] prediction files: ${PRED_COUNT}"

echo "[$(date --iso-8601=seconds)] running evaluation and every-10-frame figures"
"${PYTHON_BIN}" \
  Ocean-Agent-SDK_core/scripts/evaluate_visualize_bohai_vz.py \
  --dataset-root /data/Bohai_Sea/process_data_sparsemask_2x \
  --component Vz \
  --lr-component Vz_interp \
  --baseline-label Interp \
  --output-root "${EVAL_DIR}" \
  --model "SourceConditionedE20=${PRED_DIR}" \
  --model "ResShift200ep=${BASELINE_PRED_DIR}" \
  --case S1_TTTZ \
  --frame-start 10 \
  --frame-end 100 \
  --frame-step 10 \
  --dpi 200

echo "[$(date --iso-8601=seconds)] postrun completed"
echo "predictions=${PRED_DIR}"
echo "evaluation=${EVAL_DIR}"

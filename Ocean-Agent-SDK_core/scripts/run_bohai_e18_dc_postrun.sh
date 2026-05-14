#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/user/lz/wave_movie"
PY="/home/lz/miniconda3/envs/pytorch/bin/python"
RUN_DIR="${ROOT}/testouts/Resshift_SparseMask2xObserved_Vz_DataConsistency_E18"
PRED_DIR="${RUN_DIR}/predictions"
PRED_LOG="${RUN_DIR}/predict.log"
EVAL_DIR="${ROOT}/testouts/Resshift_DataConsistency_E18_eval"
DIAG_DIR="${ROOT}/testouts/bohai_vz_e18_data_consistency_diagnostics"
BASELINE_PRED_DIR="${ROOT}/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions"

mkdir -p "${RUN_DIR}" "${EVAL_DIR}" "${DIAG_DIR}"

echo "[postrun] waiting for E18 predictions..."
while true; do
  count="$(find "${PRED_DIR}" -maxdepth 1 -name '*_sr.npy' 2>/dev/null | wc -l)"
  if grep -q '"event": "predict_end"' "${PRED_LOG}" 2>/dev/null || [ "${count}" -eq 1200 ]; then
    echo "[postrun] prediction complete: ${count}/1200"
    break
  fi
  if ! tmux has-session -t bohai_e18_dc 2>/dev/null; then
    echo "[postrun] prediction tmux session ended before completion: ${count}/1200" >&2
    tail -80 "${PRED_LOG}" >&2 || true
    exit 1
  fi
  echo "[postrun] prediction progress: ${count}/1200"
  sleep 60
done

echo "[postrun] running full-frame evaluation and every-10-frame figures..."
MPLCONFIGDIR=/tmp/matplotlib-lz "${PY}" \
  "${ROOT}/Ocean-Agent-SDK_core/scripts/evaluate_visualize_bohai_vz.py" \
  --dataset-root /data/Bohai_Sea/process_data_sparsemask_2x \
  --component Vz \
  --output-root "${EVAL_DIR}" \
  --model "E18_DataConsistency=${PRED_DIR}" \
  --model "ResShift200ep=${BASELINE_PRED_DIR}" \
  --lr-component Vz_interp \
  --baseline-label Interp \
  --case S1_TTTZ \
  --frame-start 10 \
  --frame-end 100 \
  --frame-step 10 \
  --dpi 200

echo "[postrun] running failure diagnostics..."
MPLCONFIGDIR=/tmp/matplotlib-lz "${PY}" \
  "${ROOT}/Ocean-Agent-SDK_core/scripts/diagnose_bohai_vz_failure_modes.py" \
  --output-dir "${DIAG_DIR}" \
  --model "DirectActiveMissing=${ROOT}/testouts/Temporal3DUNet_SparseMask2xHardActiveMissing_Vz_Test/predictions" \
  --model "ResShift200ep=${BASELINE_PRED_DIR}" \
  --model "E18_DataConsistency=${PRED_DIR}"

echo "[postrun] done"
echo "[postrun] eval: ${EVAL_DIR}"
echo "[postrun] diagnostics: ${DIAG_DIR}"

#!/usr/bin/env bash
set -euo pipefail

expected=1200
log_path="/data1/user/lz/wave_movie/testouts/vz_single_eval/resshift_auto_eval.log"
prediction_dir="/data1/user/lz/wave_movie/testouts/Resshift_PGN_Vz/full_predict_run/predictions"
script_path="/data1/user/lz/wave_movie/Ocean-Agent-SDK_core/scripts/evaluate_visualize_bohai_vz.py"
python_bin="/home/lz/miniconda3/envs/pytorch/bin/python"
predict_config="bohai_vz_4x_resshift_pgn_predict_gpu7.yaml"

mkdir -p "$(dirname "$log_path")"
echo "watcher_start $(date -Is)" > "$log_path"

while true; do
    if [[ -d "$prediction_dir" ]]; then
        count="$(find "$prediction_dir" -name '*_sr.npy' | wc -l | tr -d ' ')"
    else
        count="0"
    fi

    echo "$(date -Is) count=${count}/${expected}" >> "$log_path"

    if [[ "$count" -ge "$expected" ]]; then
        MPLCONFIGDIR="/tmp/mpl_bohai_vz_eval_after_resshift" \
            "$python_bin" "$script_path" >> "$log_path" 2>&1
        echo "watcher_done $(date -Is)" >> "$log_path"
        break
    fi

    if ! pgrep -f "$predict_config" >/dev/null; then
        echo "predict_process_not_found $(date -Is)" >> "$log_path"
        exit 2
    fi

    sleep 60
done

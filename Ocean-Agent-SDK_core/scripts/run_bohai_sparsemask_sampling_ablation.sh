#!/usr/bin/env bash
set -euo pipefail

cd /data1/user/lz/wave_movie

log_dir="/data1/user/lz/wave_movie/testouts/sampling_ablation"
mkdir -p "${log_dir}"

common_args=(
  --source-root /data/Bohai_Sea
  --scale 2
  --downsample-method sparse_mask
  --sparse-mask-seed 42
  --workers 8
  --overwrite
)

/usr/bin/time -f 'elapsed=%E' \
  /home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/preprocess_bohai_wave_xyz.py \
  "${common_args[@]}" \
  --sparse-mask-pattern random \
  --output-root /data/Bohai_Sea/process_data_sparsemask_2x_random25 \
  2>&1 | tee "${log_dir}/preprocess_random25.log"

/usr/bin/time -f 'elapsed=%E' \
  /home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/preprocess_bohai_wave_xyz.py \
  "${common_args[@]}" \
  --sparse-mask-pattern jittered \
  --output-root /data/Bohai_Sea/process_data_sparsemask_2x_jittered25 \
  2>&1 | tee "${log_dir}/preprocess_jittered25.log"

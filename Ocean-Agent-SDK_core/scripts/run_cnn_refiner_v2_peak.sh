#!/usr/bin/env bash
set -euo pipefail

cd /data1/user/lz/wave_movie

output_dir="/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot_v2_Peak"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8
export MPLCONFIGDIR=/tmp/matplotlib-lz
export PYTHONUNBUFFERED=1

/home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py \
  --epochs 80 \
  --batch-size 32 \
  --patch-size 64 \
  --samples-per-epoch 2400 \
  --features 32 \
  --num-blocks 8 \
  --peak-preserve-weight 0.5 \
  --topk-amplitude-weight 0.5 \
  --peak-quantile 0.9 \
  --topk-fraction 0.05 \
  --output-dir "${output_dir}" \
  --device cuda \
  2>&1 | tee "${output_dir}/train.log"

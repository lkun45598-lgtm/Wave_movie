#!/usr/bin/env bash
set -euo pipefail

repo_root="/data1/user/lz/wave_movie/Ocean-Agent-SDK_core"
train_dir="$repo_root/scripts/ocean-SR-training-masked"
python_bin="/home/lz/miniconda3/envs/pytorch/bin/python"
torchrun_bin="/home/lz/miniconda3/envs/pytorch/bin/torchrun"
config_dir="$repo_root/configs/bohai_xyz_4x"
launch_log_dir="/data1/user/lz/wave_movie/testouts/_launch_logs"

mkdir -p "$launch_log_dir"
cd "$train_dir"

export CUDA_VISIBLE_DEVICES=1,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl_bohai_vz_point_gpu13
mkdir -p "$MPLCONFIGDIR"

run_model() {
    local name="$1"
    local config_path="$2"
    local log_path="$launch_log_dir/${name}.nohup.log"

    echo "[$(date -Is)] start ${name}" | tee -a "$log_path"
    "$torchrun_bin" --standalone --nproc_per_node=2 main_ddp.py --config "$config_path" 2>&1 | tee -a "$log_path"
    echo "[$(date -Is)] end ${name}" | tee -a "$log_path"
}

run_model \
    "bohai_vz_point_edsr_pgn_gpu13" \
    "$config_dir/bohai_vz_point_4x_edsr_pgn_ddp_gpu13.yaml"

run_model \
    "bohai_vz_point_fno2d_pgn_gpu13" \
    "$config_dir/bohai_vz_point_4x_fno2d_pgn_ddp_gpu13.yaml"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将CMEMS NC文件转换为NPY格式并按7:2:1划分"""
import numpy as np
import xarray as xr
import os
from glob import glob

NC_DIR = '/data/OceanSR/copernicus_uv_data/avg'
OUTPUT_BASE = '/data/OceanSR/copernicus_uv_data/avg'
VARIABLES = ['uo', 'vo']
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

print("步骤1: 读取所有NC文件...")
nc_files = sorted(glob(os.path.join(NC_DIR, 'cmems_gba_avg_*.nc')))
print(f"找到 {len(nc_files)} 个文件")

# 读取所有数据
all_data = []
all_times = []

for nc_file in nc_files:
    year = os.path.basename(nc_file).split('_')[-1].replace('.nc', '')
    ds = xr.open_dataset(nc_file)

    # 提取数据 (time, lat, lon, channels)
    data = np.stack([ds[var].values for var in VARIABLES], axis=-1)
    all_data.append(data)
    all_times.extend(ds['time'].values)

    ds.close()
    print(f"  {year}: {data.shape[0]} 天")

# 合并
all_data = np.concatenate(all_data, axis=0)
print(f"\n总数据形状: {all_data.shape}")  # (10591, 100, 100, 2)

# 划分
n_total = all_data.shape[0]
n_train = int(n_total * TRAIN_RATIO)
n_valid = int(n_total * VALID_RATIO)

train_data = all_data[:n_train]
valid_data = all_data[n_train:n_train+n_valid]
test_data = all_data[n_train+n_valid:]

print(f"\n划分结果:")
print(f"  Train: {train_data.shape[0]} 样本")
print(f"  Valid: {valid_data.shape[0]} 样本")
print(f"  Test: {test_data.shape[0]} 样本")

# 保存（按日期命名）
splits = [
    ('train', train_data, all_times[:n_train]),
    ('valid', valid_data, all_times[n_train:n_train+n_valid]),
    ('test', test_data, all_times[n_train+n_valid:])
]

for split, data, times in splits:
    split_dir = os.path.join(OUTPUT_BASE, split)
    os.makedirs(split_dir, exist_ok=True)

    for i in range(data.shape[0]):
        date_str = str(times[i])[:10].replace('-', '')  # YYYYMMDD
        output_file = os.path.join(split_dir, f'{date_str}.npy')
        np.save(output_file, data[i])

    print(f"  ✓ {split}: {data.shape[0]} 文件")

print(f"\n完成！输出: {OUTPUT_BASE}")

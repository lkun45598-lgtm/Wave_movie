#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMEMS 全球数据预处理脚本
功能：区域裁剪 → Cubic 插值到 100×100 → 深度平均 → 保存为 NC
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置参数 ========================
# 输入路径
SOURCE_DIR = '/data/CMEMS_GLOBAL_MULTIYEAR_PHY_001_030_daily_full/daily_my'
REF_LON_PATH = '/data/OceanSR/GBA_uv_data/static/lon.npy'
REF_LAT_PATH = '/data/OceanSR/GBA_uv_data/static/lat.npy'

# 输出路径
OUTPUT_BASE = '/data/OceanSR/copernicus_uv_data/avg'

# 时间范围
START_YEAR = 1994
END_YEAR = 2022

# 目标分辨率
TARGET_SIZE = 100  # 100×100

# 变量列表
VARIABLES = ['uo', 'vo']

# ======================== 加载参考经纬度 ========================
print("=" * 80)
print("步骤 1: 加载参考经纬度网格")
print("=" * 80)

lon_ref = np.load(REF_LON_PATH)
lat_ref = np.load(REF_LAT_PATH)

print(f"参考网格形状: {lon_ref.shape}")
print(f"经度范围: [{lon_ref.min():.4f}°E, {lon_ref.max():.4f}°E]")
print(f"纬度范围: [{lat_ref.min():.4f}°N, {lat_ref.max():.4f}°N]")

# 从 400×400 采样到 100×100
lon_target = lon_ref[::4, ::4]  # 每隔 4 个点取一个
lat_target = lat_ref[::4, ::4]

print(f"\n目标网格形状: {lon_target.shape}")
print(f"目标经度范围: [{lon_target.min():.4f}°E, {lon_target.max():.4f}°E]")
print(f"目标纬度范围: [{lat_target.min():.4f}°N, {lat_target.max():.4f}°N]")

# 定义裁剪范围（扩大一点以保证覆盖）
lon_min, lon_max = lon_target.min() - 0.5, lon_target.max() + 0.5
lat_min, lat_max = lat_target.min() - 0.5, lat_target.max() + 0.5

print(f"\n裁剪范围（带缓冲）:")
print(f"  经度: [{lon_min:.4f}°E, {lon_max:.4f}°E]")
print(f"  纬度: [{lat_min:.4f}°N, {lat_max:.4f}°N]")

# ======================== 创建输出目录 ========================
os.makedirs(OUTPUT_BASE, exist_ok=True)
print(f"\n输出目录: {OUTPUT_BASE}")

# ======================== 处理函数 ========================
def process_single_file(nc_file, year):
    """处理单个 NC 文件"""
    print(f"\n{'='*80}")
    print(f"处理 {year} 年数据: {os.path.basename(nc_file)}")
    print(f"{'='*80}")

    # 打开数据集
    ds = xr.open_dataset(nc_file)
    print(f"原始维度: {dict(ds.sizes)}")

    # 步骤 1: 区域裁剪（减少内存占用）
    print("\n步骤 1: 区域裁剪...")
    ds_crop = ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max)
    )
    print(f"裁剪后维度: {dict(ds_crop.sizes)}")

    # 步骤 2: 深度平均
    print("\n步骤 2: 深度平均...")
    ds_depth_avg = ds_crop.mean(dim='depth', keep_attrs=True)
    print(f"深度平均后维度: {dict(ds_depth_avg.sizes)}")

    # 步骤 3: 插值到目标网格（逐时间步处理以节省内存）
    print("\n步骤 3: 插值到 100×100 网格...")
    n_time = ds_depth_avg.sizes['time']

    # 获取源网格（裁剪后的）
    lon_src = ds_depth_avg['longitude'].values
    lat_src = ds_depth_avg['latitude'].values

    # 准备输出数组
    results = {}
    for var in VARIABLES:
        results[var] = np.zeros((n_time, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

    # 逐时间步插值
    for t in tqdm(range(n_time), desc=f"插值 {year}"):
        for var in VARIABLES:
            # 提取当前时间步数据 (lat, lon)
            data = ds_depth_avg[var].isel(time=t).values

            # 创建插值器（cubic）
            interp_func = RegularGridInterpolator(
                (lat_src, lon_src),
                data,
                method='cubic',
                bounds_error=False,
                fill_value=np.nan
            )

            # 准备目标点（需要展平后插值）
            lat_flat = lat_target.ravel()
            lon_flat = lon_target.ravel()
            points = np.column_stack([lat_flat, lon_flat])

            # 插值
            interp_data = interp_func(points)

            # 重塑为 100×100
            results[var][t] = interp_data.reshape(TARGET_SIZE, TARGET_SIZE)

    # 步骤 4: 保存为 NC 文件
    print(f"\n步骤 4: 保存 {year} 年数据...")
    output_file = os.path.join(OUTPUT_BASE, f'cmems_gba_avg_{year}.nc')

    # 创建新的 xarray Dataset
    ds_out = xr.Dataset(
        data_vars={
            var: (['time', 'lat', 'lon'], results[var], ds_crop[var].attrs)
            for var in VARIABLES
        },
        coords={
            'time': ds_depth_avg['time'],
            'lon': (['lat', 'lon'], lon_target),
            'lat': (['lat', 'lon'], lat_target),
        },
        attrs={
            'description': f'CMEMS data interpolated to GBA region (100×100), depth-averaged',
            'source': os.path.basename(nc_file),
            'interpolation_method': 'cubic',
            'original_resolution': f'{len(lat_src)}×{len(lon_src)}',
            'target_resolution': f'{TARGET_SIZE}×{TARGET_SIZE}',
            'processing_date': str(np.datetime64('now')),
        }
    )

    # 保存
    ds_out.to_netcdf(output_file)
    print(f"✓ 已保存: {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file) / 1024**2:.2f} MB")

    # 清理内存
    ds.close()
    ds_crop.close()
    ds_depth_avg.close()
    ds_out.close()

    return output_file

# ======================== 主处理流程 ========================
def main():
    print("\n" + "=" * 80)
    print("开始批量处理")
    print("=" * 80)

    processed_files = []

    for year in range(START_YEAR, END_YEAR + 1):
        nc_file = os.path.join(
            SOURCE_DIR,
            f'uv_fulldepth_{year}-01-01_to_{year}-12-31.nc'
        )

        if not os.path.exists(nc_file):
            print(f"\n⚠️  文件不存在: {nc_file}")
            continue

        try:
            output_file = process_single_file(nc_file, year)
            processed_files.append(output_file)
        except Exception as e:
            print(f"\n❌ 处理 {year} 年数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ======================== 汇总信息 ========================
    print("\n" + "=" * 80)
    print("处理完成汇总")
    print("=" * 80)
    print(f"成功处理文件数: {len(processed_files)}")
    print(f"输出目录: {OUTPUT_BASE}")
    print(f"\n处理后的文件列表:")
    for f in processed_files:
        print(f"  - {os.path.basename(f)}")

    # ======================== 验证总时间步数 ========================
    print("\n" + "=" * 80)
    print("验证时间步数")
    print("=" * 80)

    total_time_steps = 0
    for nc_file in sorted(processed_files):
        ds = xr.open_dataset(nc_file)
        n_time = ds.sizes['time']
        total_time_steps += n_time
        print(f"{os.path.basename(nc_file)}: {n_time} 个时间步")
        ds.close()

    print(f"\n总时间步数: {total_time_steps}")
    print(f"预期时间步数: 10591（1994-2022，部分年份 365 天，闰年 366 天）")

    if total_time_steps == 10591:
        print("✓ 时间步数验证通过！")
    else:
        print(f"⚠️  时间步数不匹配，差异: {10591 - total_time_steps}")

    print("\n" + "=" * 80)
    print("下一步：使用 ocean_sr_preprocess_full 将这些 NC 文件转换为 NPY 格式")
    print("=" * 80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMEMS 数据预处理 - 192核并行版本
按年份+时间块并行处理，充分利用多核资源
"""
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import os
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# 配置参数
SOURCE_DIR = '/data/CMEMS_GLOBAL_MULTIYEAR_PHY_001_030_daily_full/daily_my'
REF_LON_PATH = '/data/OceanSR/GBA_uv_data/static/lon.npy'
REF_LAT_PATH = '/data/OceanSR/GBA_uv_data/static/lat.npy'
OUTPUT_BASE = '/data/OceanSR/copernicus_uv_data/avg'
START_YEAR = 1994
END_YEAR = 2022
TARGET_SIZE = 100
VARIABLES = ['uo', 'vo']
N_WORKERS = 192

# 加载参考网格（全局变量，避免重复加载）
print("加载参考网格...")
lon_ref = np.load(REF_LON_PATH)
lat_ref = np.load(REF_LAT_PATH)
lon_target = lon_ref[::4, ::4]
lat_target = lat_ref[::4, ::4]
lon_min, lon_max = lon_target.min() - 0.5, lon_target.max() + 0.5
lat_min, lat_max = lat_target.min() - 0.5, lat_target.max() + 0.5

print(f"目标网格: {lon_target.shape}")
print(f"裁剪范围: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

os.makedirs(OUTPUT_BASE, exist_ok=True)

def process_time_chunk(args):
    """处理单个时间块"""
    year, time_start, time_end, chunk_id = args

    nc_file = os.path.join(SOURCE_DIR, f'uv_fulldepth_{year}-01-01_to_{year}-12-31.nc')

    try:
        # 打开数据集
        ds = xr.open_dataset(nc_file)

        # 区域裁剪
        ds_crop = ds.sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
            time=slice(time_start, time_end)
        )

        # 深度平均
        ds_avg = ds_crop.mean(dim='depth', keep_attrs=True)

        # 获取源网格
        lon_src = ds_avg['longitude'].values
        lat_src = ds_avg['latitude'].values
        n_time = ds_avg.sizes['time']

        # 插值
        results = {var: np.zeros((n_time, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
                   for var in VARIABLES}

        lat_flat = lat_target.ravel()
        lon_flat = lon_target.ravel()
        points = np.column_stack([lat_flat, lon_flat])

        for t in range(n_time):
            for var in VARIABLES:
                data = ds_avg[var].isel(time=t).values
                interp_func = RegularGridInterpolator(
                    (lat_src, lon_src), data, method='cubic',
                    bounds_error=False, fill_value=np.nan
                )
                results[var][t] = interp_func(points).reshape(TARGET_SIZE, TARGET_SIZE)

        ds.close()
        ds_crop.close()
        ds_avg.close()

        return (year, chunk_id, results, ds_avg['time'].values)

    except Exception as e:
        return (year, chunk_id, None, str(e))

def main():
    print(f"\n使用 {N_WORKERS} 个进程并行处理\n")

    # 生成任务列表：每年分成7个时间块
    tasks = []
    for year in range(START_YEAR, END_YEAR + 1):
        nc_file = os.path.join(SOURCE_DIR, f'uv_fulldepth_{year}-01-01_to_{year}-12-31.nc')
        if not os.path.exists(nc_file):
            continue

        # 读取时间维度
        with xr.open_dataset(nc_file) as ds:
            times = ds['time'].values
            n_time = len(times)

        # 分成7块
        chunk_size = n_time // 7
        for i in range(7):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 6 else n_time
            tasks.append((year, start_idx, end_idx, i))

    print(f"总任务数: {len(tasks)}")

    # 并行处理
    with Pool(N_WORKERS) as pool:
        results = pool.map(process_time_chunk, tasks)

    # 合并结果
    print("\n合并结果...")
    year_data = {}
    for year, chunk_id, data, time_or_err in results:
        if data is None:
            print(f"  年份 {year} 块 {chunk_id} 失败: {time_or_err}")
            continue

        if year not in year_data:
            year_data[year] = []
        year_data[year].append((chunk_id, data, time_or_err))

    # 保存每年的数据
    for year in sorted(year_data.keys()):
        chunks = sorted(year_data[year], key=lambda x: x[0])

        # 合并所有块
        all_data = {var: [] for var in VARIABLES}
        all_times = []

        for _, data, times in chunks:
            for var in VARIABLES:
                all_data[var].append(data[var])
            all_times.append(times)

        # 拼接
        merged_data = {var: np.concatenate(all_data[var], axis=0) for var in VARIABLES}
        merged_times = np.concatenate(all_times)

        # 保存为NC
        output_file = os.path.join(OUTPUT_BASE, f'cmems_gba_avg_{year}.nc')
        ds_out = xr.Dataset(
            data_vars={var: (['time', 'lat', 'lon'], merged_data[var]) for var in VARIABLES},
            coords={
                'time': merged_times,
                'lon': (['lat', 'lon'], lon_target),
                'lat': (['lat', 'lon'], lat_target),
            }
        )
        ds_out.to_netcdf(output_file)
        print(f"  ✓ {year}: {len(merged_times)} 时间步")
        ds_out.close()

    print(f"\n完成！输出目录: {OUTPUT_BASE}")

if __name__ == '__main__':
    main()

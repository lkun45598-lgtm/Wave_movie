#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CMEMS数据预处理 - 多进程版本（按年份并行）"""
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import os
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SOURCE_DIR = '/data/CMEMS_GLOBAL_MULTIYEAR_PHY_001_030_daily_full/daily_my'
REF_LON_PATH = '/data/OceanSR/GBA_uv_data/static/lon.npy'
REF_LAT_PATH = '/data/OceanSR/GBA_uv_data/static/lat.npy'
OUTPUT_BASE = '/data/OceanSR/copernicus_uv_data/avg'
START_YEAR = 1994
END_YEAR = 2022
TARGET_SIZE = 100
VARIABLES = ['uo', 'vo']
N_WORKERS = 29  # 29年数据，每年一个进程

# 全局加载参考网格
lon_ref = np.load(REF_LON_PATH)
lat_ref = np.load(REF_LAT_PATH)
lon_target = lon_ref[::4, ::4]
lat_target = lat_ref[::4, ::4]
lon_min, lon_max = lon_target.min() - 0.5, lon_target.max() + 0.5
lat_min, lat_max = lat_target.min() - 0.5, lat_target.max() + 0.5

os.makedirs(OUTPUT_BASE, exist_ok=True)

def process_year(year):
    """处理单年数据"""
    nc_file = os.path.join(SOURCE_DIR, f'uv_fulldepth_{year}-01-01_to_{year}-12-31.nc')

    if not os.path.exists(nc_file):
        return f"{year}: 文件不存在"

    try:
        # 打开并裁剪
        ds = xr.open_dataset(nc_file)
        ds_crop = ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))

        # 深度平均
        ds_avg = ds_crop.mean(dim='depth', keep_attrs=True)

        # 获取源网格
        lon_src = ds_avg['longitude'].values
        lat_src = ds_avg['latitude'].values
        n_time = ds_avg.sizes['time']

        # 准备插值
        results = {var: np.zeros((n_time, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
                   for var in VARIABLES}

        lat_flat = lat_target.ravel()
        lon_flat = lon_target.ravel()
        points = np.column_stack([lat_flat, lon_flat])

        # 逐时间步插值
        for t in range(n_time):
            for var in VARIABLES:
                data = ds_avg[var].isel(time=t).values
                data = np.nan_to_num(data, nan=0.0)  # 将NaN替换为0（陆地区域）
                interp_func = RegularGridInterpolator(
                    (lat_src, lon_src), data, method='cubic',
                    bounds_error=False, fill_value=0.0
                )
                results[var][t] = interp_func(points).reshape(TARGET_SIZE, TARGET_SIZE)

        # 保存
        output_file = os.path.join(OUTPUT_BASE, f'cmems_gba_avg_{year}.nc')
        ds_out = xr.Dataset(
            data_vars={var: (['time', 'lat', 'lon'], results[var]) for var in VARIABLES},
            coords={'time': ds_avg['time'], 'lon': (['lat', 'lon'], lon_target),
                    'lat': (['lat', 'lon'], lat_target)}
        )
        ds_out.to_netcdf(output_file)

        ds.close()
        ds_crop.close()
        ds_avg.close()
        ds_out.close()

        return f"{year}: ✓ {n_time}天"

    except Exception as e:
        return f"{year}: ✗ {str(e)}"

if __name__ == '__main__':
    print(f"使用 {N_WORKERS} 进程并行处理 {START_YEAR}-{END_YEAR}")

    years = list(range(START_YEAR, END_YEAR + 1))

    with Pool(N_WORKERS) as pool:
        results = pool.map(process_year, years)

    print("\n处理结果:")
    for r in results:
        print(f"  {r}")

    print(f"\n完成！输出: {OUTPUT_BASE}")

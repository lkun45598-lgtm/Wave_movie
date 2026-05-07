# Wave_movie

用于分析 Bohai Sea 波场快照数据、导出 PNG/GIF 可视化，并记录 Bohai 波场超分辨率实验代码与轻量结果。

## Files

- `analyze_wave_dataset.py`：扫描数据集目录，统计样本数、分量、帧数、节点数和网格信息。
- `save_wave_movie.py`：按 `S1.* / Total|X|Y|Z / AVS_movie_*.inp` 结构导出单帧 PNG 或时序 GIF。
- `tests/test_save_wave_movie.py`：覆盖帧排序、筛选逻辑和默认输出命名。
- `docs/superpowers/specs/2026-04-25-wave-visualization-layout-design.md`：当前可视化排版设计说明。
- `Ocean-Agent-SDK_core/`：Bohai wavefield SR 训练、推理、数据处理、模型配置和测试代码。
- `docs/bohai-vz-sparsemask-2x-summary.md`：当前 Vz sparse-mask 2x 实验结论和复现实验命令。
- `results/bohai_vz_sparsemask2x_compare_residual_active_missing/`：轻量指标汇总、对比曲线和代表性可视化图。

## Usage

分析数据集：

```bash
python analyze_wave_dataset.py /data/Bohai_Sea/To_ZGT_wave_movie
```

导出单帧 PNG：

```bash
python save_wave_movie.py \
  --dataset-root /data/Bohai_Sea/To_ZGT_wave_movie \
  --case S1.ABAZ \
  --component Total \
  --frame 50 \
  --save-png
```

导出完整 GIF：

```bash
python save_wave_movie.py \
  --dataset-root /data/Bohai_Sea/To_ZGT_wave_movie \
  --case S1.ABAZ \
  --component Total \
  --start 1 \
  --end 100 \
  --fps 8 \
  --save-gif
```

## Notes

- 默认数据集根目录是 `/data/Bohai_Sea/To_ZGT_wave_movie`。
- 可视化输出默认写到 `visualizations/`，该目录已在 `.gitignore` 中排除，不会推送到仓库。
- SR 训练数据默认位于 `/data/Bohai_Sea/process_data_sparsemask_2x`，完整预测、checkpoint 和 17GB 级 `testouts/` 不纳入 Git；仓库只保留可复现实验配置和轻量结果摘要。

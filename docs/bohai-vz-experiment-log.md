# Bohai Vz Sparse Reconstruction Experiment Log

本文档用于记录 Bohai Vz 稀疏重建实验的过程、结果和下一步判断。后续每次实验都应追加到这里，避免只保留零散日志和图片。

## 记录规范

每次新实验按以下模板追加：

```markdown
## E?? - 实验名称

- 日期：
- 目的：
- 数据：
- 模型/配置：
- 训练命令：
- 推理命令：
- 评估命令：
- 输出目录：
- 关键指标：
- 代表图：
- 观察现象：
- 结论：
- 下一步：
```

指标优先记录：

```text
RMSE / MAE / RFNE / ACC / SSIM / peak_ratio_mean
p99_abs_error_mean / p99_abs_error_max / max_abs_error
missing RMSE / active-missing RMSE / inactive RMSE
hard-case frame list
```

可视化优先保存：

```text
Interp / SR / HR / |Interp-HR| / |SR-HR|
每隔 10 帧图
典型 hard case 图
per-frame metric 曲线
case-level metric 曲线
```

## 当前任务定义

- 原始数据：`/data/Bohai_Sea/To_ZGT_wave_movie`
- 当前训练数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- 任务类型：同网格稀疏重建 / inpainting，不是严格几何 2x SR
- HR target：`Vz`
- 单帧 LR 输入：`Vz_sparse`, `Vz_interp`, `mask_observed`
- 时序窗口：5 帧，`t-2, t-1, t, t+1, t+2`
- 输入通道数：`5 x 3 = 15`
- 输出：中心帧 `Vz(t)`
- 网格：`200 x 150`
- 观测点比例：25%
- 缺失点比例：75%

中心帧通道约定：

```text
channel 6 = Vz_sparse(t)
channel 7 = Vz_interp(t)
channel 8 = mask_observed(t)
```

当前主线判断：

```text
模型整体能恢复大尺度波前和能量分布；
但局部高频结构、强波前细节、相位结构仍会丢失或错位。
下一步应从 residual/high-frequency target、结构型 loss、temporal consistency 入手。
```

## E00 - 数据构造：sparse-mask 2x Vz

- 日期：2026-05-07 至 2026-05-10
- 目的：把 Bohai AVS 波场快照处理成 OceanNPY 框架可训练的数据。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- HR：`test/hr/Vz/*.npy`
- LR：`test/lr/Vz_sparse`, `test/lr/Vz_interp`, `test/lr/mask_observed`
- 输出目录：`/data/Bohai_Sea/process_data_sparsemask_2x`

处理逻辑：

```text
Vz_sparse = 规则 2x 稀疏采样点保留，其余位置置 0
Vz_interp = 对稀疏采样点做二维线性插值回 full grid
mask_observed = 观测点为 1，缺失点为 0
```

关键数据统计：

```text
HR shape = 200 x 150
test frames = 1200
observed points per frame = 7500
observed fraction = 0.25
missing fraction = 0.75
```

结论：

```text
当前任务本质是同分辨率网格上的稀疏补全；
不能简单称为传统 low-res grid -> high-res grid 的 2x 超分。
```

## E01 - LR 构造诊断：2x sparse-mask vs 4x sparse-mask

- 日期：2026-05-09
- 目的：确认 LR 构造是否是效果差的主要原因，并比较 2x 与 4x 稀疏观测难度。
- 输出目录：`/data1/user/lz/wave_movie/testouts/lr_sampling_diagnostics`
- 结果文件：`/data1/user/lz/wave_movie/testouts/lr_sampling_diagnostics/lr_interp_metrics_2x_vs_4x.json`
- 代表图：
  - `/data1/user/lz/wave_movie/testouts/lr_sampling_diagnostics/vz_lr_sampling_2x_S1_TTTZ_000050.png`
  - `/data1/user/lz/wave_movie/testouts/lr_sampling_diagnostics/vz_lr_sampling_4x_S1_TTTZ_000050.png`

关键指标：

| LR 构造 | observed fraction | RMSE | MAE | RFNE | ACC | p99 mean | peak ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse-mask 2x interp | 0.2500 | 0.03730 | 0.01057 | 1.05308 | 0.30167 | 0.11150 | 0.85319 |
| sparse-mask 4x interp | 0.0617 | 0.04162 | 0.01460 | 1.17005 | 0.07893 | 0.13620 | 0.66413 |

观察现象：

```text
2x 插值已经能给出大尺度轮廓，但高频结构和局部相位明显损失；
4x 观测比例太低，峰值和相关性明显更差。
```

结论：

```text
当前优先继续做 2x sparse-mask；
4x 可作为后续更难任务，不应作为当前主实验。
```

## E02 - Temporal3DUNet：Direct vs Explicit Residual

- 日期：2026-05-08
- 目的：比较直接预测完整 `Vz` 与显式预测 `HR - Vz_interp` 残差哪个更稳。
- 输出目录：`/data1/user/lz/wave_movie/testouts/bohai_vz_sparsemask2x_compare_residual_active_missing`
- 结果文件：`model_summary_vz.csv`
- 代表图：输出目录下 `figs_eval_every10_vz_maxerr/`

配置：

```text
Direct: Vz_sr = f(Vz_sparse, Vz_interp, mask)
Residual: Vz_sr = Vz_interp(t) + f(Vz_sparse, Vz_interp, mask)
loss_mask_mode = active_missing
temporal_window = 5
```

关键指标：

| Method | RMSE | MAE | RFNE | ACC | max error | SSIM | peak ratio | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.03730 | 0.01057 | 1.05308 | 0.30167 | 13.366 | 0.7384 | 0.8532 | baseline |
| Direct | 0.02261 | 0.00859 | 0.63841 | 0.77041 | 8.662 | 0.7914 | 0.8893 | 39.38% |
| Residual | 0.02278 | 0.00893 | 0.64322 | 0.76757 | 9.308 | 0.7950 | 0.8943 | 38.92% |

观察现象：

```text
Direct 的 RMSE/RFNE/ACC 更好；
Residual 的 SSIM 和 peak ratio 略高，但主误差指标不占优。
```

结论：

```text
普通 residual target 没有明显优于 direct；
但这不等于 residual 思路无效，可能需要改成 high-frequency residual，而不是完整场残差。
```

## E03 - 输入消融：InterpGuided vs SparseOnly

- 日期：2026-05-08
- 目的：判断 `Vz_interp` 这个插值场是否对模型有帮助，还是只用 `Vz_sparse + mask` 更合理。
- 输出目录：`/data1/user/lz/wave_movie/testouts/bohai_vz_sparsemask2x_compare_sparseonly`
- 结果文件：`model_summary_vz.csv`

关键指标：

| Method | RMSE | MAE | RFNE | ACC | max error | SSIM | peak ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| InterpGuided | 0.02261 | 0.00859 | 0.63841 | 0.77041 | 8.662 | 0.7914 | 0.8893 |
| SparseOnly | 0.02262 | 0.00881 | 0.63849 | 0.77016 | 8.741 | 0.7639 | 0.8861 |

观察现象：

```text
SparseOnly 的数值接近 InterpGuided，但 SSIM 明显更差；
说明插值场虽然会带来平滑偏差，但仍提供了重要的大尺度结构先验。
```

结论：

```text
后续不建议完全丢掉 Vz_interp；
更合理的方式是把 Vz_interp 当 baseline，让模型学习结构化残差。
```

## E04 - Loss 消融：Observed vs MixedLoss

- 日期：2026-05-09
- 目的：比较 observed/active-missing 约束与更复杂 mixed loss 的效果。
- 输出目录：`/data1/user/lz/wave_movie/testouts/Temporal3DUNet_MixedLoss_vs_Observed_eval`
- 结果文件：`model_summary_vz.csv`

关键指标：

| Method | RMSE | MAE | RFNE | ACC | max error | SSIM | peak ratio | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Observed | 0.01971 | 0.00799 | 0.55640 | 0.83153 | 8.385 | 0.8218 | 0.9124 | 47.16% |
| MixedLoss | 0.02392 | 0.00849 | 0.67533 | 0.73797 | 8.471 | 0.8032 | 0.8759 | 35.87% |

观察现象：

```text
当前 mixed loss 版本没有带来收益；
Observed/active-missing 训练更稳。
```

结论：

```text
不是“loss 越复杂越好”；
下一步需要更有针对性的结构型 loss，而不是简单混合更多区域权重。
```

## E05 - ResShift：MSEAux Direct vs ResidualMSEAux

- 日期：2026-05-09
- 目的：测试条件扩散 ResShift 是否比 Temporal3DUNet 更适合稀疏重建，并比较 direct/residual。
- 输出目录：`/data1/user/lz/wave_movie/testouts/Resshift_MSEAux_vs_Residual_eval`
- 结果文件：`model_summary_vz.csv`

ResShift 输入/条件：

```text
diffusion y = channel 7 = Vz_interp(t)
condition lq = 15 channels = 5 frames x [Vz_sparse, Vz_interp, mask_observed]
output = center-frame Vz
steps = 15
predict_type = xstart
```

关键指标：

| Method | RMSE | MAE | RFNE | ACC | max error | SSIM | peak ratio | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DirectMSEAux | 0.01942 | 0.00788 | 0.54825 | 0.84175 | 10.359 | 0.8255 | 0.9404 | 47.94% |
| ResidualMSEAux | 0.01969 | 0.00803 | 0.55595 | 0.83596 | 6.720 | 0.8197 | 0.9376 | 47.21% |

观察现象：

```text
DirectMSEAux 的平均指标更好；
ResidualMSEAux 的 max error 更低，但 RMSE/SSIM 不占优。
```

结论：

```text
ResShift 比早期 Temporal3DUNet 更强；
但普通 residual 仍不是最终答案。
后续应测试 high-frequency residual 或结构约束残差。
```

## E06 - ResShift 训练轮数：50ep vs 200ep

- 日期：2026-05-10
- 目的：确认 ResShift 继续训练到 200 epoch 是否有效。
- 训练输出：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep`
- 预测输出：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions`
- 评估输出：`/data1/user/lz/wave_movie/testouts/Resshift_50ep_vs_200ep_eval`
- 训练日志：`train.log`
- 推理日志：`predict.log`
- 结果文件：`model_summary_vz.csv`

训练结果：

```text
actual_epochs = 200
best_epoch = 199
early_stopped = false
final patch-valid RMSE = 0.03641
final patch-test RMSE = 0.04461
full prediction files = 1200 / 1200
```

关键指标：

| Method | RMSE | MAE | RFNE | ACC | max error | p99 mean | SSIM | peak ratio | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.03730 | 0.01057 | 1.05308 | 0.30167 | 13.366 | 0.11150 | 0.7384 | 0.8532 | baseline |
| Temporal3DUNet | 0.01971 | 0.00799 | 0.55640 | 0.83153 | 8.385 | 0.06366 | 0.8218 | 0.9124 | 47.16% |
| ResShift50ep | 0.01942 | 0.00788 | 0.54825 | 0.84175 | 10.359 | 0.06485 | 0.8255 | 0.9404 | 47.94% |
| ResShift200ep | 0.01583 | 0.00683 | 0.44684 | 0.89708 | 10.668 | 0.05419 | 0.8564 | 0.9719 | 57.57% |

观察现象：

```text
200ep 明显优于 50ep，当前是综合最强模型；
但 max_abs_error 没有同步下降，说明仍有局部极端结构错误。
```

结论：

```text
ResShift200ep 是当前主结果；
后续改进目标不应只追求 overall RMSE，而应针对局部结构、p99 error、active-missing 和 hard cases。
```

## E07 - Hard Case 诊断：WRRZ early frames

- 日期：2026-05-10
- 目的：解释 ResShift200ep 的 `max_abs_error` 为什么仍然较大。
- 结果文件：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/vz_single_per_frame_metrics.csv`
- 诊断图：`/data1/user/lz/wave_movie/testouts/Resshift200ep_WRRZ_extreme_eval/figs_sharp_source_fixed`

典型 hard cases：

```text
S1_WRRZ_000001 max_abs_error = 10.6677, p99 = 0.00826, rmse = 0.07129
S1_WRRZ_000011 p99_abs_error = 0.12406, max = 0.62165, rmse = 0.02524
S1_WRRZ_000013 p99_abs_error = 0.12142
S1_WRRZ_000017 p99_abs_error = 0.11632
```

观察现象：

```text
max_abs_error 最大的帧不一定是大面积失败，有些是孤立点或初始峰值；
p99_abs_error 更能反映真实结构性 hard case。
```

结论：

```text
后续评估要同时看 max_abs_error 和 p99_abs_error；
不能只用单点 max error 判断模型整体。
```

## E08 - 可视化坐标与点源标注修正

- 日期：2026-05-10
- 目的：修正可视化坐标和点源位置混乱问题。
- 修改文件：`Ocean-Agent-SDK_core/scripts/evaluate_visualize_bohai_vz.py`
- 测试文件：`Ocean-Agent-SDK_core/tests/test_evaluate_visualize_bohai_vz_coords.py`
- 默认可视化输出：`X/Y offset (km)`，不画点源，保持当前论文/汇报图风格
- 可选模式：`--coordinate-mode absolute --show-source-marker`

验证：

```text
pytest evaluate/coordinate tests: 9 passed
py_compile passed
```

新图目录：

```text
/data1/user/lz/wave_movie/testouts/Resshift200ep_TTTZ_like_image/figs_eval_every10_vz
```

结论：

```text
默认图按 offset 风格展示即可；
点源坐标用于诊断时再打开，不作为普通对比图默认元素。
```

## 当前结论汇总

目前最好的主结果是：

```text
ResShift200ep
RMSE = 0.01583
MAE = 0.00683
RFNE = 0.44684
ACC = 0.89708
SSIM = 0.8564
peak_ratio = 0.9719
RMSE reduction vs Interp = 57.57%
```

但主要问题仍然是：

```text
整体结构和波前大体正确；
局部细节不是偏一点，而是高频结构/相位结构可能直接消失或错位。
```

当前 ResShift 方法准确表述：

```text
2D conditional ResShift diffusion
+ 5-frame temporal channels as condition
+ center-frame Vz_interp as diffusion degraded input
+ center-frame Vz output
```

它不是严格的：

```text
3D spatiotemporal diffusion
sequence-to-sequence temporal model
physics-constrained wave equation model
```

## E13 - Temporal3DUNet Mixed Boundary Loss

- 日期：2026-05-11
- 状态：completed
- 目的：测试 active-missing 局部结构错误是否来自 missing 区域和 observed/missing 边界约束不足。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- 模型/配置：`Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_mixed_boundary_e13_gpu0.yaml`
- 输出目录：`/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xMixedBoundary_E13_Vz`
- 训练设备：GPU0

固定部分：

```text
sampling = regular25
model = Temporal3DUNet
temporal_window = 5
epochs = 50
train_batchsize = 64
patch_size = 64
```

只改 loss：

```text
L = active_missing_l1
  + 0.2 * missing_l1
  + 0.05 * inactive_missing_l1
  + 0.2 * boundary_gradient_l1
```

说明：

```text
原计划中的 temporal_derivative_l1 暂不加入。
原因是当前 OceanNPY temporal 样本只返回中心帧 HR，不能严格计算
HR(t)-HR(t-1)。若要做真实 temporal derivative loss，需要 E13b 中
扩展 temporal HR target 或改成 sequence-to-sequence 输出。
```

训练前验证：

```text
新增 boundary_gradient_l1 单测先失败：0.0 != 6.0
实现后定向测试：8 passed
相关定向测试：20 passed
py_compile: loss.py/base.py passed
```

训练结果：

```text
actual_epochs = 50
best_epoch = 50
early_stopped = false
training_time = 0h 7m 30s
final patch-valid RMSE = 0.016198
final patch-valid SSIM = 0.792861
final patch-test RMSE = 0.022464
final patch-test SSIM = 0.754593
full prediction files = 1200 / 1200
```

全测试集评估：

| Method | RMSE | MAE | RFNE | ACC | max error | p99 mean | SSIM | peak ratio | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 13.366 | 0.111502 | 0.738434 | 0.853189 | baseline |
| E13 MixedBoundary | 0.024240 | 0.008423 | 0.684312 | 0.729250 | 8.552 | 0.075654 | 0.804381 | 0.867044 | 35.02% |

区域指标：

| Method | Missing RMSE | Active-Missing RMSE | Inactive-Missing RMSE | Observed RMSE |
|---|---:|---:|---:|---:|
| Interp | 0.043074 | 0.060749 | 0.006650 | 0.000000 |
| DirectActiveMissing baseline | 0.026113 | 0.036494 | 0.006359 | 0.000000 |
| E13 MixedBoundary | 0.027990 | 0.039392 | 0.005017 | 0.000000 |

输出文件：

```text
train log: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xMixedBoundary_E13_Vz/train.log
postrun log: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xMixedBoundary_E13_Vz/postrun.log
predictions: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xMixedBoundary_E13_Vz/predictions
every-10 figures: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xMixedBoundary_E13_Vz/figs_eval_every10_vz_maxerr
summary: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_MixedBoundary_E13_eval/model_summary_vz.csv
region metrics: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_MixedBoundary_E13_eval/region_metrics_vz.json
```

观察现象：

```text
E13 相比插值仍有明显提升，但弱于原 DirectActiveMissing baseline。
SSIM 高于 Direct baseline，inactive-missing RMSE 下降，说明背景伪影被压低。
但 missing RMSE、active-missing RMSE、ACC、peak ratio 均变差，
说明 mixed missing + boundary loss 把模型推向更平滑/保守的解，
没有解决主要的活跃波前结构恢复问题。
```

结论：

```text
E13 是负向/控制实验。
当前局部结构错误不能靠简单增加 missing/inactive/boundary 权重解决。
下一步不应继续加大 boundary loss，而应回到 active-missing 主目标，
在不损伤 active wavefront 的前提下引入更有针对性的 peak/phase/high-frequency
或真正 sequence-level temporal 约束。
```

## E14 - Temporal3DUNet Active-Missing High-Frequency Loss

- 日期：2026-05-11
- 状态：completed
- 目的：测试结构约束是否应该只作用在 active-missing 波前区域，而不是 E13 那样扩散到全 missing/background/boundary。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- 模型/配置：`Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_active_hf_e14_gpu0.yaml`
- 输出目录：`/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xActiveHF_E14_Vz`
- 训练设备：GPU0

固定部分：

```text
sampling = regular25
model = Temporal3DUNet
temporal_window = 5
epochs = 50
train_batchsize = 64
patch_size = 64
```

只改 loss：

```text
L = active_missing_l1
  + 0.1 * active_missing_gradient_l1
  + 0.025 * active_missing_laplacian_l1
  + 1.0 * active_missing_peak_l1
```

实现说明：

```text
配置中仍使用 loss_mask_mode=active_missing。
l1、gradient_l1、peak_l1 沿用 DirectActiveMissing baseline 的 active-missing mask。
新增 active_missing_laplacian_l1 是独立区域项，只作用在未观测且 |HR| > 0.005 的点。
不加入 E13 的 missing_l1、inactive_missing_l1、boundary_gradient_l1。
```

训练前验证：

```text
新增 active_missing_gradient_l1 / active_missing_laplacian_l1 单测先失败：
0.0 != 6.0, 0.0 != 2.0
实现后定向测试：10 passed
```

结果文件：

```text
train log: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xActiveHF_E14_Vz/train.log
postrun log: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xActiveHF_E14_Vz/postrun.log
predictions: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xActiveHF_E14_Vz/predictions
every-10 figures: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xActiveHF_E14_Vz/figs_eval_every10_vz_maxerr
summary: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_ActiveHF_E14_eval/model_summary_vz.csv
region metrics: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_ActiveHF_E14_eval/region_metrics_vz.json
```

全图指标：

| Method | RMSE | MAE | RFNE | ACC | max error | p99 mean | SSIM | peak ratio | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 13.366 | 0.111502 | 0.738434 | 0.853189 | baseline |
| DirectActiveMissing | 0.022614 | 0.008590 | 0.638410 | 0.770408 | 8.662 | 0.071912 | 0.791413 | 0.889274 | 39.38% |
| E14 ActiveHF | 0.022820 | 0.009094 | 0.644203 | 0.765999 | 8.837 | 0.072846 | 0.789032 | 0.886153 | 38.83% |

区域指标：

| Method | Missing RMSE | Active-Missing RMSE | Inactive-Missing RMSE | Observed RMSE |
|---|---:|---:|---:|---:|
| Interp | 0.043074 | 0.060749 | 0.006650 | 0.000000 |
| DirectActiveMissing | 0.026113 | 0.036494 | 0.006359 | 0.000000 |
| E14 ActiveHF | 0.026350 | 0.036685 | 0.007165 | 0.000000 |

观察现象：

```text
E14 相比插值仍有明显提升；
但相比 DirectActiveMissing baseline，RMSE/RFNE/ACC/SSIM/peak ratio 均略差。
active-missing RMSE 只略差，但 inactive-missing RMSE 明显变差。
说明 active-only Laplacian/二阶高频约束没有修复局部结构丢失，反而增加了背景区域的小误差。
```

结论：

```text
E14 是负向/控制实验。
把结构约束限制在 active-missing 区域比 E13 稳，但仍未优于原始 active-missing baseline。
当前问题不应继续靠简单加大 Laplacian/空间导数 loss 解决；
更可能需要针对相位/时间传播/物理条件的约束，例如序列级 temporal consistency、峰值/相位指标，或引入源位置/介质等条件。
```

## E15 - Failure Mode Diagnostics: LR / Peak / Spectrum

- 日期：2026-05-11
- 状态：completed
- 目的：不训练新模型，只诊断恢复差到底来自 LR 插值、峰值/相位位置、频谱能量还是模型修正方式。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- 脚本：`Ocean-Agent-SDK_core/scripts/diagnose_bohai_vz_failure_modes.py`
- 输出目录：`/data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics`
- 对比对象：`Interp`, `DirectActiveMissing`, `E14_ActiveHF`, `ResShift200ep`

运行命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib-lz /home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/diagnose_bohai_vz_failure_modes.py \
  --output-dir /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics
```

输出文件：

```text
per-frame metrics: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/per_frame_failure_diagnostics.csv
summary: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/summary_failure_diagnostics.csv
summary json: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/summary_failure_diagnostics.json
pair deltas: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/model_pair_delta_summary.json
E14 worse cases: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/e14_worse_than_direct_active_missing_top50.csv
figure: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/fig1_failure_mode_summary.png
figure: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/fig2_peak_recovery_scatter.png
figure: /data1/user/lz/wave_movie/testouts/bohai_vz_e15_failure_diagnostics/fig3_spectral_retention_boxplot.png
```

关键 per-frame 平均指标：

| Method | RMSE | Active-Missing RMSE | p99 error | target peak ratio | active-missing ACC | high-power ratio |
|---|---:|---:|---:|---:|---:|---:|
| Interp | 0.033689 | 0.106997 | 0.111502 | 0.504299 | 0.035433 | 0.157990 |
| DirectActiveMissing | 0.021125 | 0.071497 | 0.072015 | 0.634177 | 0.616672 | 0.592702 |
| E14 ActiveHF | 0.021346 | 0.071675 | 0.072846 | 0.634974 | 0.610278 | 0.605556 |
| ResShift200ep | 0.015087 | 0.046115 | 0.054186 | 0.856433 | 0.806288 | 0.882172 |

说明：

```text
这里的 RMSE / active-missing RMSE 是 per-frame 平均，不是把所有像素全局加权后的 RMSE。
per-frame 平均会更突出事件初期和少量 active points 的困难帧，因此数值会大于全局区域 RMSE。
```

pairwise 结论：

| 对比 | 指标 | mean delta | better fraction |
|---|---|---:|---:|
| E14 - Direct | active-missing RMSE | +0.000178 | 0.263 |
| E14 - Direct | RMSE | +0.000221 | 0.188 |
| E14 - Direct | p99 error | +0.000831 | 0.286 |
| E14 - Direct | high-power ratio | +0.012854 | 0.698 |
| ResShift200ep - Direct | active-missing RMSE | -0.025381 | 0.998 |
| ResShift200ep - Direct | high-power ratio | +0.289469 | 1.000 |

观察现象：

```text
1. Vz_interp 的 high-power ratio 只有约 0.158，说明 LR/interp 基线已经严重丢失高频能量。
2. Direct/E14 把 high-power ratio 提升到约 0.59/0.61，但仍明显低于 HR。
3. E14 的 high-power ratio 略高于 Direct，但 active-missing RMSE、p99、ACC 反而变差。
4. ResShift200ep 的 high-power ratio 约 0.882，target peak ratio 约 0.856，active-missing ACC 约 0.806，明显更接近 HR。
5. 最困难帧集中在各 case 的第 1-2 帧，例如 `S1_WRRZ_000001`, `S1_WCZ_000001`, `S1_WATZ_000001`。
6. 在这些早期强事件帧，Direct/E14 在 HR peak 位置的幅值经常只恢复到真实峰值的 3%-10%，不是简单边缘不够锐。
```

结论：

```text
E15 支持当前判断：本质问题不是“再加空间高频 loss 就能解决”。
LR/interp 已经丢掉或扭曲了大量高频/峰值/相位信息；
Direct/E14 能降低平均误差，但主要是平滑修正，不能可靠恢复早期强波前峰值位置和局部结构。
E14 只是略微增加频谱能量，却没有提高结构正确性，因此不是本质改进。
ResShift200ep 更好，说明更强的条件生成/迭代修正确实有帮助，但仍应继续针对早期强事件、时间传播一致性和物理条件做改进。
```

下一步：

```text
优先不要继续叠加单帧空间导数 loss。
建议下一轮只测一个本质假设：引入真实 temporal consistency / sequence-to-sequence 监督，
验证是否能改善第 1-2 帧强波前和 active-missing phase/peak recovery。
```

## 下一步实验计划

## E16 - Temporal Consistency Minimal Test

- 日期：2026-05-11
- 状态：completed
- 目的：验证“恢复差主要来自缺失区域的时间传播/相位不确定”这个假设，而不是继续叠加单帧空间高频 loss。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- 模型/配置：`Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_temporal_consistency_e16_gpu0.yaml`
- 输出目录：`/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xTemporalConsistency_E16_Vz`
- 训练设备：GPU0

固定部分：

```text
sampling = regular25
model = Temporal3DUNet
temporal_window = 5
epochs = 50
train_batchsize = 64
patch_size = 64
main center-frame loss = DirectActiveMissing baseline loss
```

只新增 temporal supervision：

```text
data.temporal_supervision_window = 3
optimize.temporal_consistency_weight = 0.1
```

loss 形式：

```text
L = center_frame_active_missing_loss
  + 0.1 * mean |(SR[t+1]-SR[t]) - (HR[t+1]-HR[t])|
```

实现原则：

```text
模型结构不改，仍输出单帧 Vz。
训练时同一个模型对 t-1, t, t+1 三个中心样本分别推理，
再用预测序列和 HR 序列的一阶时间差分计算 temporal consistency。
评估和预测仍按中心帧输出，不改变已有推理流程。
```

判断标准：

```text
重点不只看 full RMSE；
必须看 active-missing RMSE、target peak ratio、active-missing ACC、
high-power ratio、hard frames 尤其 S1_WRRZ_000001 / S1_WCZ_000001 / S1_WATZ_000001。
如果第 1-2 帧强波前没有改善，说明单纯 temporal consistency 也不足，
下一步才考虑 source/medium conditioning 或 data-consistency diffusion。
```

结果路径：

```text
predictions:
/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xTemporalConsistency_E16_Vz/predictions

every-10-frame figures:
/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xTemporalConsistency_E16_Vz/figs_eval_every10_vz_maxerr

full-frame evaluation:
/data1/user/lz/wave_movie/testouts/Temporal3DUNet_TemporalConsistency_E16_eval

failure diagnostics:
/data1/user/lz/wave_movie/testouts/bohai_vz_e16_failure_diagnostics
```

完整 full-frame 指标：

| Model | RMSE | MAE | RFNE | ACC | SSIM | Peak ratio | p99 abs error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 0.738434 | 0.853189 | 0.111502 |
| E16 TemporalConsistency | 0.022676 | 0.008950 | 0.640146 | 0.769104 | 0.793289 | 0.888497 | 0.072297 |

相对插值：

```text
RMSE reduction = 39.21%
RFNE reduction = 39.21%
ACC delta = +0.4674
```

与 DirectActiveMissing baseline 对比：

| Model | RMSE mean | Missing RMSE | Active-missing RMSE | Inactive-missing RMSE | Active-missing ACC | Target peak ratio | High-power ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| DirectActiveMissing | 0.021125 | 0.024393 | 0.071497 | 0.006855 | 0.616672 | 0.634177 | 0.592702 |
| E16 TemporalConsistency | 0.021211 | 0.024492 | 0.071408 | 0.007334 | 0.612879 | 0.635954 | 0.613189 |
| ResShift200ep | 0.015087 | 0.017420 | 0.046115 | 0.007896 | 0.806288 | 0.856433 | 0.882172 |

结论：

```text
E16 相比插值仍然明显更好，但相比 DirectActiveMissing 没有本质提升。
active-missing RMSE 只从 0.071497 到 0.071408，变化极小；
target peak ratio 只从 0.634177 到 0.635954，几乎不变；
high-power ratio 从 0.592702 到 0.613189，有小幅提升，但结构正确性没有同步改善；
inactive-missing RMSE 反而从 0.006855 变差到 0.007334。

因此，当前这种“模型仍只输出中心帧，只在训练阶段加 t-1/t/t+1 一阶差分一致性”的做法，
不足以解决强波前结构缺失和峰值相位恢复问题。
下一步不应继续单独加大 temporal_consistency_weight，
而应该改成真正的 sequence-to-sequence 预测，或引入更明确的源位置/传播条件/data-consistency 约束。
```

补充诊断：

```text
用户观察到的问题是正确的：当前结果在高频细节上缺少结构一致性，同时部分帧有伪影。

定量检查显示：
- E16 high-power ratio = 0.613189，只比 DirectActiveMissing 的 0.592702 略高，仍远低于 HR 目标的 1.0；
- E16 active-missing RMSE = 0.071408，与 DirectActiveMissing 的 0.071497 基本持平；
- E16 inactive-missing high_lap_ratio ≈ 10.37，高于 DirectActiveMissing 的 ≈ 7.39，说明背景/近零区域高频伪影被放大；
- patch seam RMSE / non-seam RMSE ≈ 1.10，说明 patch 拼接边界有轻微影响，但不是主要根因。

根因判断：
当前问题不是“模型没有生成高频能量”，而是“生成的高频没有对齐真实波前结构”。
active_missing loss 忽略 inactive missing 区域，使近零背景缺少约束；
单帧中心输出的 temporal consistency 只能约束一阶时间差分，不能恢复被稀疏采样破坏的相位/峰值位置。

因此下一步应把问题分成两类分别处理：
1. 高频结构缺失：需要 sequence-to-sequence 输出或引入源/介质/传播条件，而不是单帧后验加时间差分；
2. 伪影：需要对 inactive/missing 区域加入弱的高频抑制或背景一致性约束，但不能像 E13 那样大幅增加全 missing loss，否则会牺牲 active wavefront。
```

## E17 - Temporal3DUNet Sequence Output Minimal Test

- 日期：2026-05-11
- 状态：completed
- 目的：验证“结构缺失来自只监督中心帧、时序传播约束太弱”这个假设。
- 配置：`Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_sequence_e17_gpu0.yaml`
- 输出目录：`/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xSeqOut_E17_Vz`
- 设备：GPU0

固定部分：

```text
dataset = /data/Bohai_Sea/process_data_sparsemask_2x
sampling = regular25 sparse mask
model = Temporal3DUNet
temporal input window = 5
temporal supervision window = 3
patch_size = 64
epochs = 50
base loss = E16/DirectActiveMissing 的 active_missing + gradient_l1 + peak_l1
```

唯一核心变化：

```text
E16:
input t-2..t+2 -> output center frame t
训练阶段额外对 t-1/t/t+1 分别推理后算 temporal consistency

E17:
input t-2..t+2 -> output [t-1, t, t+1] 三帧
一次 forward 直接输出 3 个时间通道
训练 loss 对三帧都算 active-missing reconstruction
再保留一阶 temporal consistency
预测/评估仍只取中心帧 t，与旧结果可比
```

对应硬约束：

```text
output t-1 uses Vz_sparse channel 3 and mask channel 5
output t   uses Vz_sparse channel 6 and mask channel 8
output t+1 uses Vz_sparse channel 9 and mask channel 11
```

判断标准：

```text
如果 E17 的 active-missing ACC、target peak ratio、high-power ratio 明显优于 E16/Direct，
说明“只监督中心帧”确实是结构缺失的重要原因。

如果 E17 仍然没有改善，下一步不应继续扩大时序 loss，
而应转向源/介质/传播条件或数据一致性扩散。
```

完成情况：

```text
训练完成：yes
best checkpoint: /data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xSeqOut_E17_Vz/best_model.pth
预测完成：1200 / 1200
预测目录：/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xSeqOut_E17_Vz/predictions
每隔 10 帧图：/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xSeqOut_E17_Vz/figs_eval_every10_vz_maxerr
全图评估：/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SeqOut_E17_eval
failure diagnostics：/data1/user/lz/wave_movie/testouts/bohai_vz_e17_failure_diagnostics
```

全图评估：

| Model | RMSE | MAE | RFNE | ACC | SSIM | Peak Ratio | p99 Abs Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 0.738434 | 0.853189 | 0.111502 |
| E17 SeqOut | 0.025275 | 0.009469 | 0.713520 | 0.701829 | 0.784444 | 0.875813 | 0.079555 |

同口径 failure diagnostics：

| Model | RMSE | Missing RMSE | Active-Missing RMSE | Inactive-Missing RMSE | Active-Missing ACC | Target Peak Ratio | High-Power Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.033689 | 0.038901 | 0.106997 | 0.007150 | 0.035433 | 0.504299 | 0.157990 |
| DirectActiveMissing | 0.021125 | 0.024393 | 0.071497 | 0.006855 | 0.616672 | 0.634177 | 0.592702 |
| E16 TemporalConsistency | 0.021211 | 0.024492 | 0.071408 | 0.007334 | 0.612879 | 0.635954 | 0.613189 |
| E17 SeqOut | 0.023379 | 0.026995 | 0.077853 | 0.007668 | 0.547263 | 0.530505 | 0.521522 |
| ResShift200ep | 0.015087 | 0.017420 | 0.046115 | 0.007896 | 0.806288 | 0.856433 | 0.882172 |

结论：

```text
E17 相比插值仍然有明显提升，但没有超过 DirectActiveMissing / E16，
并且在 active-missing ACC、target peak ratio、high-power ratio 上明显退化。

这说明“只监督中心帧”不是当前结构缺失的主要瓶颈。
简单把 Temporal3DUNet 改成一次输出 t-1/t/t+1，并不能自动恢复高频波前结构。

下一步不建议继续堆更复杂的时序输出。
更合理的方向是：
1. 以 ResShift200ep 作为当前强基线；
2. 针对局部结构缺失和伪影，测试 source/medium conditioning、data consistency、physics/gradient continuity loss；
3. 如果继续做扩散，应避免纯图像式 diffusion，改成带观测点一致性和物理/频域约束的 conditional refinement。
```

## E18 - ResShift Data-Consistency Projection

- 日期：2026-05-13
- 状态：completed
- 目的：验证“ResShift 局部结构缺失/伪影是否来自扩散采样过程中没有持续满足稀疏观测一致性”。
- 配置：`Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_resshift_data_consistency_e18_gpu0.yaml`
- 输出目录：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_DataConsistency_E18`
- 预测 tmux：`bohai_e18_dc`
- postrun tmux：`bohai_e18_dc_postrun`
- postrun 脚本：`Ocean-Agent-SDK_core/scripts/run_bohai_e18_dc_postrun.sh`
- 设备：物理 GPU2（`CUDA_VISIBLE_DEVICES=2`，配置内仍显示 `device: 0`）

核心假设：

```text
当前 ResShift200ep 只在最终输出后做 sparse known hard constraint。
如果扩散采样过程中 pred_xstart 在观测点附近已经形成不一致结构，
最后再硬贴回观测点可能造成局部网格感、边界不连续或伪影。

E18 在每一步 p_mean_variance 的 pred_xstart 层加入 data-consistency projection：
pred_xstart[observed] = Vz_sparse[observed]

这样不直接改 noisy state x_t，避免 clean 观测值和扩散噪声尺度不匹配。
```

固定部分：

```text
dataset = /data/Bohai_Sea/process_data_sparsemask_2x
base checkpoint = /data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/best_model.pth
model = ResShift
steps = 15
temporal input window = 5
sampling = regular25 sparse mask
training = none for first test; inference-only controlled comparison
```

代码改动：

```text
新增 NCHW sparse projection helper:
apply_sparse_known_constraint_nchw()

新增 ResShift inference denoised_fn:
ResshiftTrainer._build_data_consistency_denoised_fn()

新增配置项:
model.resshift_data_consistency: true
model.resshift_data_consistency_strength: 1.0
```

测试：

```text
新增测试：Ocean-Agent-SDK_core/tests/test_resshift_data_consistency.py

定向验证：
/home/lz/miniconda3/envs/pytorch/bin/python -m pytest \
  Ocean-Agent-SDK_core/tests/test_resshift_data_consistency.py \
  Ocean-Agent-SDK_core/tests/test_resshift_loss_norm.py \
  Ocean-Agent-SDK_core/tests/test_temporal_ocean_npy.py -q

结果：16 passed, 1 warning
```

自动评估输出：

```text
E18 predictions:
/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_DataConsistency_E18/predictions

E18 full-frame eval:
/data1/user/lz/wave_movie/testouts/Resshift_DataConsistency_E18_eval

E18 diagnostics:
/data1/user/lz/wave_movie/testouts/bohai_vz_e18_data_consistency_diagnostics
```

判断标准：

```text
如果 active-missing RMSE 下降、active-missing ACC 上升、high-power ratio 更接近 1，
且每隔 10 帧图中伪影减少，说明采样过程 data consistency 有效。

如果 observed 周围更连续但 high-power/peak 没改善，
说明 data consistency 主要解决边界不连续，不解决强波前结构缺失。

如果指标变差或图像变硬/棋盘感增强，
说明 hard projection 太强，下一步改成 strength=0.5~0.7 的 soft projection。
```

完成情况：

```text
预测完成：1200 / 1200
预测目录：/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_DataConsistency_E18/predictions
每隔 10 帧图：/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_DataConsistency_E18/figs_eval_every10_vz_maxerr
全图评估：/data1/user/lz/wave_movie/testouts/Resshift_DataConsistency_E18_eval
failure diagnostics：/data1/user/lz/wave_movie/testouts/bohai_vz_e18_data_consistency_diagnostics
```

全图评估：

| Model | RMSE | MAE | RFNE | ACC | SSIM | Peak Ratio | p99 Abs Error | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 0.738434 | 0.853189 | 0.111502 | 13.366163 |
| ResShift200ep | 0.015829 | 0.006835 | 0.446843 | 0.897084 | 0.856351 | 0.971896 | 0.054186 | 10.667669 |
| E18 DataConsistency | 0.015829 | 0.006835 | 0.446845 | 0.897082 | 0.856349 | 0.971941 | 0.054186 | 10.538763 |

同口径 failure diagnostics：

| Model | RMSE | Missing RMSE | Active-Missing RMSE | Active-Missing ACC | Target Peak Ratio | High-Power Ratio | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| DirectActiveMissing | 0.021125 | 0.024393 | 0.071497 | 0.616672 | 0.634177 | 0.592702 | 0.617714 |
| ResShift200ep | 0.015087 | 0.017420 | 0.046115 | 0.806288 | 0.856433 | 0.882172 | 0.398256 |
| E18 DataConsistency | 0.015088 | 0.017422 | 0.046152 | 0.806269 | 0.856401 | 0.882249 | 0.398566 |

逐帧 E18 - ResShift200ep 差值：

| Metric | Mean Delta | Better Fraction |
|---|---:|---:|
| RMSE | +0.00000101 | 0.474 |
| Active-Missing RMSE | +0.00003681 | 0.478 |
| p99 Abs Error | +0.00000034 | 0.502 |
| Max Abs Error | +0.00031022 | 0.379 |
| Active-Missing ACC | -0.00001804 | 0.507 |
| Target Peak Ratio | -0.00003163 | 0.264 |
| High-Power Ratio | +0.00007716 | 0.676 |

结论：

```text
E18 没有带来实质改进。

Hard data-consistency projection 在 pred_xstart 层不会破坏结果，
但相对 ResShift200ep 几乎只是数值级扰动。
高频能量 high-power ratio 有极小提升，但 active-missing RMSE、ACC、peak ratio、
p99/max error 都没有稳定改善。

这说明当前 ResShift 的主要瓶颈不在“最终才贴回观测点”。
观测点一致性已经不是主导误差来源；
真正困难仍然是未观测 active wavefront 的结构/相位/峰值重建。

下一步不建议继续只调 hard/soft projection strength。
更应该转向引入物理/传播条件、源位置/时间条件、大感受野或针对 active wavefront 的结构监督。
```

### P01 - High-Frequency Residual ResShift

目的：

```text
解决“整体 OK，但局部结构没了”的问题。
```

建议形式：

```text
baseline = Vz_interp(t)
target_residual = HR - baseline
high_freq_target = target_residual - smooth(target_residual)
SR = baseline + predicted_residual
```

新增 loss：

```text
active_missing L1
gradient/Sobel loss
Laplacian loss
FFT high-frequency loss
peak/top-k loss
```

判断标准：

```text
不只看 RMSE；
重点看 p99_abs_error、active-missing RMSE、局部结构可视化、hard-case frames。
```

## E09 - P01 启动：ResShift High-Frequency Residual 50ep

- 日期：2026-05-10
- 状态：running
- 目的：测试 residual ResShift + 高频/二阶结构约束是否能改善“整体 OK 但局部结构消失”的问题。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- 模型/配置：`Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_resshift_hf_residual_50ep_gpu0.yaml`
- 输出目录：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_HFResidual_50ep`
- 训练设备：GPU0
- 训练 tmux：`bohai_hf_residual`
- postrun tmux：`bohai_hf_residual_postrun`
- postrun 脚本：`Ocean-Agent-SDK_core/scripts/run_bohai_hf_residual_postrun.sh`
- postrun 日志：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_HFResidual_50ep/postrun.log`
- 计划评估输出：`/data1/user/lz/wave_movie/testouts/Resshift_HFResidual_50ep_eval`

配置重点：

```text
predict_type = residual
diffusion y = center-frame Vz_interp, channel 7
condition lq = 15 channels = 5 frames x [Vz_sparse, Vz_interp, mask_observed]
loss_mask_mode = active_missing
resshift_aux_loss_weight = 0.75
```

结构型辅助 loss：

```text
l1 = 1.0
gradient_l1 = 0.25
laplacian_l1 = 0.1
fft_hf_l1 = 0.05
fft_cutoff = 0.08
peak_l1 = 2.0
peak_boost = 6.0
observed_l1 = 0.2
```

训练前验证：

```text
MaskedCompositeSRLoss 新增 laplacian_l1 单测通过
ResShift residual/loss 相关测试：8 passed
配置解析通过
GPU0 空闲，启动前显存占用约 934 MB
```

备注：

```text
estimate_memory.py 对当前 ResShift 15 通道条件输入不适配，
dry-run 构造了错误通道数并失败；这不是训练配置错误。
已有相同 ResShift 主体配置完成过 50/200ep 训练，本实验只增加结构型 aux loss。
```

当前运行记录：

```text
2026-05-10 08:59 UTC 已启动 postrun watcher。
watcher 会等待 bohai_hf_residual 训练结束后自动执行：
1. predict 生成全测试集 SR npy；
2. evaluate_visualize_bohai_vz.py 生成 Interp / HFResidual50ep / ResShift200ep 对比指标；
3. 生成 S1_TTTZ 每 10 帧诊断图，frame 10-100。
```

## E10 - CNN Detail Refiner Pilot on ResShift200ep Base

- 日期：2026-05-10
- 状态：completed
- 目的：测试“ResShift 恢复大结构 + 轻量 CNN 修复局部细节残差”的 two-stage 方案是否有价值。
- 数据：`/data/Bohai_Sea/process_data_sparsemask_2x`
- ResShift base：`/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions`
- 脚本：`Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py`
- 测试：`Ocean-Agent-SDK_core/tests/test_cnn_detail_refiner_pilot.py`
- 计划：`docs/superpowers/plans/2026-05-10-cnn-detail-refiner-pilot.md`
- 输出目录：`/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot`
- 训练 tmux：`bohai_cnn_refiner_pilot`
- 训练设备：GPU1

重要说明：

```text
这是快速诊断 pilot，不是最终无泄漏论文实验。
原因：当前只已有 ResShift200ep 的 original-test 全量预测。
因此本实验在 original-test 内按 case 重新切分 CNN train/valid/eval，
用于判断 CNN 是否具备修复 ResShift 残差细节的能力。
最终正式实验需要先生成 original train/valid/test 的 ResShift base 预测。
```

CNN case split：

```text
train cases = S1_TTTZ, S1_TWVZ, S1_URZ, S1_UTU, S1_WATZ, S1_WCZ, S1_WHTZ, S1_WIAZ
valid cases = S1_WLCZ, S1_WTAZ
eval cases  = S1_WRRZ, S1_WTVZ
```

模型形式：

```text
input = [SR_base, Vz_interp, Vz_sparse, mask_observed, SR_base - Vz_interp]
CNN predicts Δ_detail
SR_refined = SR_base + Δ_detail
observed points are hard-constrained back to Vz_sparse
```

预期判断：

```text
如果 CNNRefined 在 eval cases 上降低 active-missing RMSE、p99_abs_error、
max_abs_error 或 peak_ratio 更接近 1，说明 second-stage detail refiner 值得继续。
如果只变锐但 max error 变大，说明存在幻觉细节，需要换 loss 或加 temporal/physics 约束。
```

运行结果：

```text
epochs = 80
best valid loss = 0.0371264 at epoch 17
后续 epoch valid loss 上升，说明轻量 CNN 很快过拟合。
```

Eval cases 汇总指标：

| Method | RMSE | MAE | RFNE | ACC | SSIM | active-missing RMSE | p99 error | max error | peak ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.038015 | 0.011527 | 1.018285 | 0.349780 | 0.745803 | 0.129245 | 0.124029 | 1.260390 | 0.861046 |
| ResShift200ep | 0.017868 | 0.007404 | 0.533909 | 0.836213 | 0.862851 | 0.068541 | 0.062523 | 0.570479 | 0.959240 |
| CNNRefined | 0.017256 | 0.006923 | 0.513031 | 0.843517 | 0.869254 | 0.066172 | 0.060193 | 0.554138 | 0.934098 |

Case-level 观察：

```text
S1_WRRZ:
ResShift200ep RMSE 0.022699 -> CNNRefined 0.022042
active-missing RMSE 0.091138 -> 0.088698
SSIM 0.929158 -> 0.933266
peak ratio 0.946650 -> 0.931747

S1_WTVZ:
ResShift200ep RMSE 0.013037 -> CNNRefined 0.012471
active-missing RMSE 0.045943 -> 0.043646
SSIM 0.796545 -> 0.805243
peak ratio 0.971830 -> 0.936448
```

输出文件：

```text
metrics: /data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/metrics_summary.csv
per-frame: /data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/per_frame_metrics.csv
eval predictions: /data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/predictions_eval
every-10-frame figures: /data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/figs_eval_every10_vz_maxerr
WRRZ comparison: /data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/eval_WRRZ
WTVZ comparison: /data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/eval_WTVZ
```

结论：

```text
轻量 CNN detail refiner 方向有效，但当前版本更像误差抑制器，而不是峰值增强器。
它在 held-out eval cases 上稳定降低 RMSE / active-missing RMSE / p99 / max error，
并提升 SSIM；但 peak ratio 下降，说明它会进一步压低峰值。

因此该方向值得继续，但下一版需要加入 peak-preserving loss 或约束 residual 不压峰值。
正式论文实验还需要生成 original train/valid/test 的 ResShift base 预测，不能只用 original-test pilot split。
```

## E11 - CNN Detail Refiner v2: Peak-Preserving Loss

时间：2026-05-11

目的：

```text
在 E10 的轻量 CNN refiner 基础上加入峰值保持约束，
测试是否能解决 E10 虽然降低 RMSE 但压低 peak ratio 的问题。
```

配置：

```text
base model = ResShift200ep predictions
CNN input = [SR_base, Vz_interp, Vz_sparse, mask_observed, SR_base - Vz_interp]
CNN output = delta_detail
SR_refined = SR_base + delta_detail
observed hard constraint = true
epochs = 80
batch_size = 32
patch_size = 64
samples_per_epoch = 2400
features = 32
num_blocks = 8
peak_preserve_weight = 0.5
topk_amplitude_weight = 0.5
peak_quantile = 0.9
topk_fraction = 0.05
GPU = 1
```

输出目录：

```text
/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot_v2_Peak
```

训练状态：

```text
完成。
best valid loss = 0.0756797 at epoch 73
epoch 41-42 出现一次 loss spike，但 best checkpoint 没有被覆盖，后续恢复正常。
```

Eval cases 汇总指标：

| Method | RMSE | MAE | RFNE | ACC | SSIM | active-missing RMSE | p99 error | max error | peak ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Interp | 0.038015 | 0.011527 | 1.018285 | 0.349780 | 0.745803 | 0.129245 | 0.124029 | 1.260390 | 0.861046 |
| ResShift200ep | 0.017868 | 0.007404 | 0.533909 | 0.836213 | 0.862851 | 0.068541 | 0.062523 | 0.570479 | 0.959240 |
| CNNRefined v1 | 0.017256 | 0.006923 | 0.513031 | 0.843517 | 0.869254 | 0.066172 | 0.060193 | 0.554138 | 0.934098 |
| CNNRefined v2 Peak | 0.019752 | 0.008555 | 0.597084 | 0.820608 | 0.835571 | 0.072126 | 0.070261 | 0.586707 | 1.027801 |

每隔 10 帧可视化：

```text
/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot_v2_Peak/figs_eval_every10_vz_maxerr
```

对比评估输出：

```text
WRRZ comparison:
/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot_v2_Peak/eval_WRRZ

WTVZ comparison:
/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot_v2_Peak/eval_WTVZ
```

结论：

```text
v2 没有超过 E10 v1，也没有超过 ResShift200ep。

峰值保持约束确实把 peak ratio 从 v1 的 0.934 拉到 1.028，
但代价是 RMSE、MAE、active-missing RMSE、SSIM、p99 error、max error 全部变差。
这说明当前 peak-preserving loss 权重过强或形式过硬，
模型倾向于把局部幅值推高，而不是恢复正确的细节结构。

保留该实验作为负面对照。
下一步不建议继续加大 peak loss；
更合理的是做弱峰值约束或 gated residual，只在高置信未观测波前区域修正，
并加入 temporal consistency / gradient continuity，而不是单独追峰值幅度。
```

## E12 - Sampling Pattern Ablation: Regular vs Random vs Jittered 25%

时间：2026-05-11

目的：

```text
验证规则 2x 采样是否因为 aliasing 导致恢复困难。
固定观测比例 25%、固定 Vz 任务和 Temporal3DUNet 结构，
只改变 sparse mask 形态：regular / random / jittered。
```

实现改动：

```text
preprocess_bohai_wave_xyz.py 新增:
--sparse-mask-pattern regular|random|jittered
--sparse-mask-seed 42

regular:
每隔 2 个网格点采样，保持原始 sparse-mask 2x 行为。

random:
在 full grid 上固定随机选择 25% 点。

jittered:
每个 2x2 block 内随机选择 1 个点，固定 mask，观测比例严格 25%。
```

数据输出：

```text
regular25:
/data/Bohai_Sea/process_data_sparsemask_2x

random25:
/data/Bohai_Sea/process_data_sparsemask_2x_random25

jittered25:
/data/Bohai_Sea/process_data_sparsemask_2x_jittered25
```

数据生成状态：

```text
random25: 完成，11500 samples，split = 9200 / 1100 / 1200，用时 7:28
jittered25: 完成，11500 samples，split = 9200 / 1100 / 1200，用时 5:12
```

插值基线指标：

| Sampling | Obs Ratio | Interp RMSE | Interp MAE | Interp RFNE | Interp ACC | Missing RMSE | Active-Missing RMSE | SSIM | Peak Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| regular25 | 0.2500 | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 0.043074 | 0.060749 | 0.738434 | 0.853189 |
| random25 | 0.2500 | 0.037510 | 0.011187 | 1.058907 | 0.304889 | 0.043312 | 0.060926 | 0.710589 | 0.834891 |
| jittered25 | 0.2500 | 0.037487 | 0.010970 | 1.058264 | 0.304398 | 0.043286 | 0.060936 | 0.724077 | 0.840622 |

Temporal3DUNet 50ep full-frame SR 指标：

| Sampling | SR RMSE | SR MAE | SR RFNE | SR ACC | Missing RMSE | Active-Missing RMSE | SSIM | Peak Ratio | RMSE Reduction vs Interp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| regular25 | 0.022614 | 0.008590 | 0.638410 | 0.770408 | 0.026113 | 0.036494 | 0.791413 | 0.889274 | 39.38% |
| random25 | 0.024025 | 0.008976 | 0.678241 | 0.735575 | 0.027742 | 0.038838 | 0.793488 | 0.863038 | 35.95% |
| jittered25 | 0.023178 | 0.008896 | 0.654322 | 0.757857 | 0.026763 | 0.037347 | 0.799101 | 0.874360 | 38.17% |

输出文件：

```text
interp metrics:
/data1/user/lz/wave_movie/testouts/sampling_ablation/interp_metrics_regular_random_jittered25.csv

sr metrics:
/data1/user/lz/wave_movie/testouts/sampling_ablation/sr_metrics_regular_random_jittered25_temporal3dunet.csv

regular eval:
/data1/user/lz/wave_movie/testouts/sampling_ablation/eval_regular25_temporal3dunet

random eval:
/data1/user/lz/wave_movie/testouts/sampling_ablation/eval_random25_temporal3dunet

jittered eval:
/data1/user/lz/wave_movie/testouts/sampling_ablation/eval_jittered25_temporal3dunet

random every-10-frame figures:
/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xRandom25ActiveMissing_Vz_Test/figs_eval_every10_vz_maxerr

jittered every-10-frame figures:
/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xJittered25ActiveMissing_Vz_Test/figs_eval_every10_vz_maxerr
```

结论：

```text
这次实验没有支持“random/jittered 采样会让当前任务更容易”的假设。

Interp 层面：
regular25 的 RMSE、MAE、missing RMSE、active-missing RMSE、SSIM、peak ratio 都最好。
random/jittered 的 scattered interpolation 反而损失更多局部结构。

SR 层面：
regular25 的 RMSE 和 active-missing RMSE 最好；
jittered25 的 SSIM 略高，但 RMSE / active-missing RMSE / peak ratio 仍不如 regular25；
random25 整体最弱。

因此当前效果不好不能简单归因于“规则 2x mask 比 random/jittered 更难”。
更可能的主因仍然是：
1. 25% 稀疏观测本身对强波前/高频相位信息不足；
2. 插值先验会平滑并改变局部结构；
3. 当前 loss 更偏平均误差，缺少 temporal / gradient continuity / physics consistency。

下一步优先级应转向 loss 和物理/时序约束，而不是继续只改 mask 形态。
```

### P02 - Temporal Consistency Loss

目的：

```text
让模型不仅恢复单帧，还恢复合理的时间传播。
```

建议 loss：

```text
L_dt = |(SR_t - SR_{t-1}) - (HR_t - HR_{t-1})|
```

判断标准：

```text
每隔 10 帧图中波前结构是否连续；
per-frame RMSE 是否减少尖峰；
hard-case early frames 是否改善。
```

### P03 - Larger Patch / Full-Frame Fine-Tune

目的：

```text
解决 64x64 patch 对全局波前上下文不足的问题。
```

建议：

```text
patch_size = 96 或 128
后期 full-frame fine-tune
full-frame validation 选 checkpoint
```

判断标准：

```text
波前曲率和环状结构是否更完整；
不是只看局部纹理是否变锐。
```

## 维护要求

每次训练前先登记：

```text
实验编号
配置文件
训练输出目录
预期验证指标
```

每次训练后补齐：

```text
训练是否完成
best epoch
预测文件数量
model_summary
hard-case 可视化路径
结论：保留 / 放弃 / 作为对照
```

如果某个实验失败，也要记录失败原因，不能只保留成功实验。

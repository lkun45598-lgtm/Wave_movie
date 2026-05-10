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

## 下一步实验计划

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

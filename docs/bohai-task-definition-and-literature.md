# Bohai Vz 稀疏波场重建：任务定义、文献对齐与下一步改进

本文档用于在继续训练前统一三件事：

```text
1. 你的任务到底是什么；
2. 领域里类似问题通常怎么处理；
3. 当前方法的具体不足是什么，以及下一步只针对哪个问题改。
```

## 1. 当前任务的准确定义

当前 Bohai Vz 任务更准确应定义为：

```text
Sparse wavefield reconstruction / wavefield interpolation / wavefield completion
```

而不是普通自然图像意义上的：

```text
single-image super-resolution
```

当前数据流是：

```text
HR target:
Vz(t), shape = 200 x 150

LR / condition:
Vz_sparse(t), Vz_interp(t), mask_observed(t)

temporal context:
t-2, t-1, t, t+1, t+2

model output:
center-frame Vz(t)
```

中心帧通道约定：

```text
channel 6 = Vz_sparse(t)
channel 7 = Vz_interp(t)
channel 8 = mask_observed(t)
```

当前任务本质是：

```text
在同一个 200 x 150 网格上，给定 25% 观测点和插值先验，恢复完整 Vz 波场。
```

因此它不是严格的几何 2x SR：

```text
不是 100 x 75 -> 200 x 150
而是 200 x 150 sparse/incomplete -> 200 x 150 complete
```

论文或汇报中建议使用的表述：

```text
We formulate the problem as a sparse wavefield reconstruction task on a fixed spatial grid.
The model receives sparse particle-velocity observations, an interpolated prior field,
and the observation mask, and reconstructs the complete Vz wavefield at the center frame.
```

中文表述：

```text
本文将该问题定义为固定空间网格上的稀疏波场重建任务。
模型以稀疏粒子速度观测、插值先验场和观测掩码为条件，
恢复中心时刻完整的 Vz 波场。
```

## 2. 这个方向与文献的对应关系

### 2.1 地震数据重建通常处理的是 missing traces / interpolation

地震数据处理中，空间采样不完整很常见，目标通常是从缺失道、稀疏接收点或不规则采样中恢复完整数据。Scientific Reports 2020 的工作明确讨论了 irregularly and regularly missing data reconstruction，并指出传统方法依赖地震数据的先验性质，而深度学习方法通常用 encoder-decoder / U-Net 类结构从不完整输入恢复完整数据。

这与当前 Bohai 任务的对应关系是：

```text
seismic missing trace interpolation:
missing traces -> complete seismic gather

Bohai Vz sparse reconstruction:
missing grid points -> complete Vz wavefield
```

对应到你的数据：

```text
missing traces / sparse receivers  ~=  mask_observed = 0 的网格点
complete seismic gather            ~=  HR Vz
pre-interpolated input             ~=  Vz_interp
```

结论：

```text
你的任务放在 sparse reconstruction / interpolation 框架下是合理的。
但不能只套自然图像 SR 的 PSNR/SSIM 逻辑，必须关注 missing 区域、active wavefront 和物理连续性。
```

### 2.2 稀疏输入 + mask + 预插值先验是合理的，但插值先验也会带来偏差

一些 seismic reconstruction 方法会先构造 pre-interpolated input，再让网络修正；也有工作指出预插值会影响最终性能。Scientific Reports 2020 提到若训练输入依赖初始 bicubic pre-interpolation，这个预插值过程会影响最终方法表现。

你的当前输入：

```text
[Vz_sparse, Vz_interp, mask_observed]
```

是合理的，因为：

```text
Vz_sparse 提供真实观测点；
mask_observed 告诉模型哪里是真观测、哪里是缺失；
Vz_interp 提供大尺度背景和初始连续场。
```

但它也有风险：

```text
Vz_interp 会平滑尖峰；
Vz_interp 可能改变局部波前宽度；
模型容易学成“插值修正器”，而不是真正的物理重建器。
```

这正好符合当前现象：

```text
大尺度结构能恢复；
局部强波前、高频相位、峰值细节仍然不足。
```

### 2.3 全波场重建领域也会用 synthetic HR -> sparse/LR -> reconstruct 的流程

Lamb wave full wavefield super-resolution 工作中，常见流程是用高分辨率 synthetic wavefield 构造 low-resolution / sparse measurement，再训练 DNN 恢复完整波场，并用实验数据验证真实场景可行性。

这与你的思路是对齐的：

```text
传统模拟 / 数值方法产生 HR wavefield；
人为构造稀疏观测或低分辨率观测；
训练模型学习从低成本观测恢复完整波场；
后续再讨论真实观测迁移。
```

因此你的整体设定是合理的。需要注意的是：

```text
LR 构造方式必须明确对应真实应用场景；
如果未来声称用于真实传感器场景，需要加入更真实的 mask / sensor layout；
如果当前只是验证方法可行性，regular 2x sparse grid 可以作为受控 benchmark。
```

### 2.4 物理约束在 sparse wavefield reconstruction 中很重要

Physics-informed full wavefield reconstruction 工作通常不是只用 L1/MSE，而是把波动方程或弹性动力学方程放入 loss，作为稀疏观测到完整波场的正则约束。

文献里的核心思想是：

```text
稀疏观测下，数据项不足以唯一确定完整波场；
物理方程、时间传播、空间梯度连续性可以作为约束；
这些约束能减少无物理意义的局部结构和背景伪影。
```

当前 Bohai 代码还没有真正做到这一点：

```text
已有 temporal window = 5；
但没有显式 temporal consistency loss；
没有 wave equation residual；
没有速度模型 / 介质参数 / 源位置条件；
没有频谱或波数域约束。
```

因此当前模型更接近：

```text
data-driven interpolation refinement
```

而不是：

```text
physics-consistent wavefield reconstruction
```

### 2.5 扩散模型有人做，但重点是 data consistency，不是简单换成扩散

近年的 seismic diffusion reconstruction 工作通常把 diffusion 用作数据先验，并在采样过程中结合 measurement operator 或 known trace consistency。比如 SeisDDRM 使用预训练 diffusion prior 和线性 degradation operator 做 interpolation / denoising；还有 DDIM / score-based seismic interpolation 工作会在 reverse sampling 中反复融合已知 trace 信息。

这对当前项目的启示是：

```text
扩散模型不是简单“输入 sparse，输出 HR”就能自动解决问题；
如果要做扩散，关键是每一步 sampling 都要满足 observed data consistency；
否则扩散可能生成看似合理但局部结构不真实的波场。
```

因此扩散模型暂时不应作为下一步主线。只有当我们明确需要概率先验或多解采样时，再考虑：

```text
data-consistency diffusion
posterior sampling
known-observation projection
```

## 3. 当前实验已经证明了什么

### 3.1 规则 2x 采样不是当前主要瓶颈

E12 已经比较：

```text
regular25
random25
jittered25
```

观测比例都固定为 25%，模型固定为 Temporal3DUNet 50ep。

主要结果：

| Sampling | Interp RMSE | Interp Active-Missing RMSE | SR RMSE | SR Active-Missing RMSE | SR SSIM | SR Peak Ratio |
|---|---:|---:|---:|---:|---:|---:|
| regular25 | 0.037303 | 0.060749 | 0.022614 | 0.036494 | 0.791413 | 0.889274 |
| random25 | 0.037510 | 0.060926 | 0.024025 | 0.038838 | 0.793488 | 0.863038 |
| jittered25 | 0.037487 | 0.060936 | 0.023178 | 0.037347 | 0.799101 | 0.874360 |

结论：

```text
random/jittered 没有改善主要误差；
regular25 在 RMSE 和 active-missing RMSE 上仍最好；
因此不能把当前结构恢复不足简单归因于规则 2x mask。
```

### 3.2 模型能降低平均误差，但还没解决局部结构

当前最基础 direct Temporal3DUNet：

```text
Interp RMSE = 0.037303
Direct SR RMSE = 0.022614
RMSE reduction = 39.38%
```

这说明模型不是没学到东西。

但 active-missing 区域仍然较难：

```text
Interp active-missing RMSE = 0.060749
Direct active-missing RMSE = 0.036494
```

虽然有明显改善，但图像上仍可能出现：

```text
强波前被平滑；
局部峰值偏低；
相位或波前位置错位；
有些细节不是偏差，而是结构直接消失。
```

### 3.3 CNN refiner 的结果说明不能只追单个指标

CNN refiner v1：

```text
RMSE / active-missing RMSE / SSIM 改善；
peak ratio 下降。
```

CNN refiner v2 peak-preserving：

```text
peak ratio 被推到 1.0278；
但 RMSE / active-missing RMSE / SSIM 变差。
```

结论：

```text
硬追 peak ratio 会制造错误结构；
局部细节恢复需要更合理的区域约束和物理/时序约束，
不能只靠 peak loss 或后处理 CNN。
```

### 3.4 ResShift 当前更强，但问题没有本质消失

当前较强结果是 ResShift200ep：

```text
ResShift200ep RMSE = 0.015829
ACC = 0.89708
SSIM = 0.85635
Peak Ratio = 0.97190
```

它显著强于基础 Temporal3DUNet，但仍存在：

```text
局部结构缺失；
强波前细节恢复不稳定；
一些 hard frame 的局部最大误差仍大。
```

因此下一步不应只说“换更强模型”，而应先问：

```text
到底是哪类错误没有被训练目标约束到？
```

## 4. 当前方法的具体不足

### 4.1 当前 loss 与真实关注目标不完全一致

现在主要关注：

```text
active_missing L1
peak L1
gradient L1
```

但实际希望恢复的是：

```text
missing 区域整体合理；
active wavefront 准确；
inactive 背景不乱长伪影；
observed-missing 边界连续；
时间传播连续；
高频/波数能量合理。
```

当前 loss 没有充分覆盖：

```text
inactive missing 区域；
observed/missing 交界；
temporal derivative；
frequency / wavenumber consistency。
```

### 4.2 输入有 temporal window，但训练目标仍是单帧

当前 temporal window = 5：

```text
t-2, t-1, t, t+1, t+2
```

但监督目标主要是中心帧：

```text
SR(t) -> HR(t)
```

这会导致：

```text
模型可以利用邻帧信息；
但没有被强制学习正确的时间变化；
SR(t+1)-SR(t) 是否符合 HR(t+1)-HR(t) 没有约束。
```

这对波场尤其重要，因为波前是随时间传播的结构，不是独立图像。

### 4.3 评估仍需要更聚焦 hard region

全图 RMSE 会被大量近零区域和 observed hard constraint 稀释。当前更应该主看：

```text
missing RMSE
active-missing RMSE
inactive-missing RMSE
observed-missing boundary error
temporal derivative error
peak ratio
peak location error
wavefront-band RMSE
high-frequency spectral error
```

其中最优先的是：

```text
active-missing RMSE
temporal derivative error
observed-missing boundary gradient error
peak ratio + peak location error
```

### 4.4 当前任务还没有真实场景闭环

如果最终目标是用于真实稀疏观测或更低成本传统计算，后续必须说明：

```text
LR 是真实观测稀疏？
还是低分辨率传统模拟？
还是高分辨率模拟的人工降采样？
```

当前实验属于：

```text
controlled synthetic degradation benchmark
```

也就是：

```text
用传统/数值模拟生成 HR；
人为构造 sparse LR；
训练模型恢复 HR；
用于验证方法可行性和误差模式。
```

这个是合理起点，但不能直接声称已经解决真实传感器场景。

## 5. 后续改进原则

后续不能过度设计。每次只针对一个明确问题改。

### 不建议下一步做的事

暂时不建议：

```text
继续换 random/jittered mask；
继续堆 CNN refiner；
直接把扩散作为主线；
同时加入 temporal + spectral + physics + diffusion + refiner；
只凭 PSNR/SSIM 判断效果。
```

原因：

```text
mask 消融已经说明采样形态不是当前主瓶颈；
CNN refiner 已经证明单纯后处理会压峰值或推错峰值；
扩散若不做 data consistency，容易生成不稳定局部结构；
多因素同时改会导致实验不可解释。
```

### 下一步应该只针对一个问题

当前最具体的问题应定义为：

```text
模型已经能降低平均误差，但 active-missing 波前区域和 observed-missing 边界的局部结构仍不稳定。
```

对应的最小改进先分两步做，避免为了加入时序项而一次性扩大数据接口：

```text
E13a:
在 regular25 主数据上，引入 mixed regional loss 和 observed/missing boundary gradient loss，
验证是否改善 active-missing RMSE、missing RMSE 和边界连续性。

E13b:
若 E13a 有效，再改数据接口或模型输出，加入真正的 temporal derivative consistency。
```

## 6. 建议的下一轮最小实验

### 实验名称

```text
E13 - Regular25 Temporal3DUNet with Mixed Boundary Loss
```

### 实验目的

```text
验证“当前局部结构恢复不足是否主要来自 loss 没有约束 inactive/missing 区域和 observed/missing 边界”。
```

### 固定不变的部分

```text
dataset = /data/Bohai_Sea/process_data_sparsemask_2x
sampling = regular25
model = Temporal3DUNet
temporal_window = 5
input = [Vz_sparse, Vz_interp, mask_observed]
output = Vz(t)
epochs = 与已有 baseline 对齐，先 50 epoch
```

### 只改的部分

只改 loss，不换模型、不换采样、不加 refiner：

```text
L = active_missing_l1
  + 0.2 * all_missing_l1
  + 0.05 * inactive_missing_l1
  + 0.2 * boundary_gradient_l1
```

解释：

```text
active_missing_l1:
继续关注真正有波动且缺失的区域。

all_missing_l1:
避免模型只关注高能区域，忽略一般缺失点。

inactive_missing_l1:
抑制近零区域伪影，但权重要小，避免模型过度变平。

boundary_gradient_l1:
约束 observed 点和 missing 点交界处的空间连续性，减少网格感和断裂。

temporal_derivative_l1:
暂不在 E13 中加入。当前训练样本只返回中心帧 HR，严格的
SR(t)-SR(t-1) 与 HR(t)-HR(t-1) 对齐需要 temporal HR target
或 sequence-to-sequence 输出，不能用现有中心帧接口伪造。
```

### 判断标准

该实验不是只看 full RMSE。通过标准应是：

```text
active-missing RMSE 不高于 baseline；
missing RMSE 不高于 baseline；
inactive missing artifact 不增加；
每隔 10 帧图中波前结构更连续；
peak ratio 不明显低于 baseline；
max error 不显著上升。
```

如果只出现：

```text
SSIM 上升但 active-missing RMSE 变差；
peak ratio 更接近 1 但 max error 上升；
图像更锐但结构位置错；
```

则不算有效改进。

## 7. 后续路线图

### P1：先做 loss 对齐

优先级最高：

```text
mixed regional loss
boundary gradient continuity
```

原因：

```text
当前问题是“优化目标没有准确约束你关心的错误”；
先改 loss 比换模型更可解释。
```

Temporal derivative consistency 作为 E13b，不在当前中心帧 HR 接口下硬加。

### P2：再做模型结构

如果 E13 有效，再考虑：

```text
larger patch / full-frame fine-tune
Temporal3DUNet 输出多帧
FNO / neural operator 类结构
```

### P3：最后再考虑 diffusion

只有当我们需要生成式先验或多解恢复时，再考虑：

```text
data-consistency diffusion
posterior sampling
known-observation projection at every sampling step
```

不建议继续做普通 conditional diffusion，因为当前问题不是“模型形式不够高级”，而是：

```text
已知观测一致性、波前连续性、时间传播规律没有被充分约束。
```

## 8. 可直接写进论文/汇报的当前结论

```text
This task is formulated as sparse wavefield reconstruction on a fixed spatial grid,
rather than conventional image super-resolution. The current interpolation-guided
models substantially reduce global reconstruction error, indicating that the sparse
observations and interpolated prior provide useful large-scale wavefield information.
However, the remaining errors concentrate in active missing wavefront regions, where
local phase, peak amplitude, and high-frequency structures are not sufficiently
constrained by standard regression losses. Sampling-pattern ablations show that
random or jittered 25% masks do not resolve this issue, suggesting that the next
improvement should focus on physics- and time-aware objectives rather than merely
changing the sparse mask pattern.
```

中文版本：

```text
本文任务被定义为固定空间网格上的稀疏波场重建，而非传统图像超分。
当前插值引导模型已经能够显著降低全局误差，说明稀疏观测和插值先验
能够提供有效的大尺度波场信息。然而，剩余误差主要集中在活跃缺失波前区域，
表现为局部相位、峰值幅度和高频结构恢复不足。采样方式消融表明，
random 或 jittered 25% mask 并不能解决该问题，因此下一步改进应优先关注
物理和时序一致性的训练目标，而不是单纯改变稀疏采样形态。
```

## 9. 参考文献和链接

- Deep learning for irregularly and regularly missing data reconstruction, Scientific Reports, 2020.  
  https://www.nature.com/articles/s41598-020-59801-x

- Recent advances on deep residual learning based seismic data reconstruction: an overview, Journal of Geophysics and Engineering, 2025.  
  https://academic.oup.com/jge/article/22/3/854/8110702

- Deep learning super-resolution for the reconstruction of full wavefield of Lamb waves, Mechanical Systems and Signal Processing, 2023.  
  https://www.sciencedirect.com/science/article/pii/S0888327022009463

- Physics-informed deep learning for scattered full wavefield reconstruction from a sparse set of sensor data, Structural Health Monitoring, 2024.  
  https://journals.sagepub.com/doi/10.1177/14759217231202547

- Seismic data interpolation using deeply supervised U-Net++ with natural seismic training sets, Geophysical Prospecting, 2023.  
  https://www.earthdoc.org/content/journals/10.1111/1365-2478.13307

- Seismic interpolation via multi-scale HU-Net, Geoenergy Science and Engineering, 2023.  
  https://www.sciencedirect.com/science/article/pii/S2949891023000441

- Seismic denoising diffusion restoration model for seismic data processing, Engineering Applications of Artificial Intelligence, 2025.  
  https://www.sciencedirect.com/science/article/pii/S0952197625010693

- Seismic Data Interpolation via Denoising Diffusion Implicit Models with Coherence-corrected Resampling, arXiv, 2023.  
  https://arxiv.org/abs/2307.04226

- Seismic trace interpolation via score-based diffusion model with wavelet convolution, Journal of Applied Geophysics, 2025.  
  https://www.sciencedirect.com/science/article/pii/S092698512500309X

# Bohai Vz 4x Point FNO Modes Ablation

Purpose: test whether increasing FNO Fourier modes improves high-frequency structure recovery for Vz point-sampled 4x super-resolution.

GPU mapping:

| GPU | Config | Main change |
| --- | --- | --- |
| 1 | `bohai_vz_point_4x_fno2d_modes_baseline_gpu1.yaml` | baseline modes `[15,12,9,9,9]` |
| 2 | `bohai_vz_point_4x_fno2d_modes_medium_gpu2.yaml` | moderate modes |
| 3 | `bohai_vz_point_4x_fno2d_modes_high_gpu3.yaml` | high safe modes |
| 4 | `bohai_vz_point_4x_fno2d_modes_highfront_gpu4.yaml` | higher full-resolution modes, safe upsample modes |
| 5 | `bohai_vz_point_4x_fno2d_modes_aggressive_gpu5.yaml` | aggressive full-resolution modes, lower LR |
| 6 | `bohai_vz_point_4x_fno2d_modes_high_width96_gpu6.yaml` | high modes plus wider network |
| 7 | `bohai_vz_point_4x_temporal5_fno2d_residual_hf_modes_high_gpu7.yaml` | temporal residual FNO with high modes |

The 4x point LR grid is approximately `50 x 37`, so the upsample-layer `modes2` must stay at or below `19`. These configs keep the last `modes2` value within that limit because `FNO2d` uses `modes2[-1]` inside `SpectralUpsampleConv2d`.

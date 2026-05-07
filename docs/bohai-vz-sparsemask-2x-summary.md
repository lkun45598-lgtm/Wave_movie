# Bohai Vz Sparse-Mask 2x Experiment Summary

This note summarizes the current Bohai Sea Vz sparse-mask super-resolution experiment.

## Dataset

- Raw/processed data root used locally: `/data/Bohai_Sea/process_data_sparsemask_2x`
- Target component: `Vz`
- Input channels per frame: `Vz_sparse`, `Vz_interp`, `mask_observed`
- Temporal window: 5 frames, using `t-2, t-1, t, t+1, t+2`
- Spatial shape: `200 x 150`
- Observed sparse points: 25% of grid
- Missing points: 75% of grid

The model uses the interpolated field as a large-scale baseline and the sparse observation mask to distinguish observed values from missing values.

## Current Main Configs

Direct/interpolation-guided reconstruction:

```bash
Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_active_missing_test_gpu0.yaml
```

Explicit residual reconstruction:

```bash
Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_residual_active_missing_test_gpu0.yaml
```

The direct model predicts the target field directly:

```text
Vz_sr = f(Vz_sparse, Vz_interp, mask)
```

The explicit residual model predicts a correction on top of the center-frame interpolated baseline:

```text
Vz_sr = Vz_interp(t) + f(Vz_sparse, Vz_interp, mask)
```

For the 5-frame input, the center-frame channels are:

```text
6: Vz_sparse(t)
7: Vz_interp(t)
8: mask_observed(t)
```

## Main Result

Full-frame test evaluation covers 1200 frames.

| Method | RMSE | MAE | RFNE | ACC | Max Error | SSIM | RMSE Reduction vs Interp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interp baseline | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 13.366 | 0.7384 | baseline |
| Direct | 0.022614 | 0.008590 | 0.638410 | 0.770408 | 8.662 | 0.7914 | 39.38% |
| Explicit residual | 0.022785 | 0.008935 | 0.643216 | 0.767572 | 9.308 | 0.7950 | 38.92% |

Interpretation:

- Direct reconstruction is the current primary method because it has lower RMSE, RFNE, and max error.
- Explicit residual reconstruction slightly improves SSIM and peak ratio, but it does not improve the main pointwise error metrics.
- The direct model can be interpreted as an implicit residual-like method because `Vz_interp` is already part of the input.

## Where The Model Still Fails

Region-level analysis shows that full-field RMSE hides the hardest part of the task because observed sparse points are constrained to match the input exactly.

| Method | Missing RMSE | Active-Missing RMSE | Inactive RMSE |
|---|---:|---:|---:|
| Interp baseline | 0.043074 | 0.060749 | 0.005763 |
| Direct | 0.026113 | 0.036494 | 0.005511 |
| Explicit residual | 0.026309 | 0.036666 | 0.006031 |

Current failure modes:

- Strong wave peaks are still underestimated. Direct mean peak ratio is about `0.889`.
- The hardest frames are often event-initial frames, for example `S1_WRRZ_000001`, `S1_WCZ_000001`, and `S1_WATZ_000001`.
- Some frames have worse local maximum error than interpolation, even though average RMSE improves.
- Near-zero inactive regions can contain small model-induced artifacts.

## Reproduction Commands

Train the explicit residual Temporal3DUNet on GPU0:

```bash
CUDA_VISIBLE_DEVICES=0 /home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/main.py \
  --config Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_residual_active_missing_test_gpu0.yaml
```

Predict full-frame test outputs:

```bash
CUDA_VISIBLE_DEVICES=0 /home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/main.py \
  --mode predict \
  --config Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_residual_active_missing_test_gpu0.yaml
```

Evaluate direct vs residual:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python \
  Ocean-Agent-SDK_core/scripts/evaluate_visualize_bohai_vz.py \
  --dataset-root /data/Bohai_Sea/process_data_sparsemask_2x \
  --component Vz \
  --output-root /data1/user/lz/wave_movie/testouts/bohai_vz_sparsemask2x_compare_residual_active_missing \
  --model Direct=/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xHardActiveMissing_Vz_Test/predictions \
  --model Residual=/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xHardResidualActiveMissing_Vz_Test/predictions \
  --lr-component Vz_interp \
  --baseline-label Interp \
  --case S1_TTTZ \
  --frame-start 10 \
  --frame-end 100 \
  --frame-step 10 \
  --dpi 200
```

## Included Lightweight Results

The repository includes lightweight summaries and representative figures under:

```text
results/bohai_vz_sparsemask2x_compare_residual_active_missing/
```

Included files:

- `model_summary_vz.csv`
- `average_error_energy_vz.csv`
- `run_summary_vz.json`
- `comparison_per_frame_metrics_vz.png`
- `comparison_case_metrics_vz.png`
- `average_error_energy_vz_legend_bottom.png`
- Representative Direct and Residual diagnostic frames for `S1_TTTZ_000010`, `S1_TTTZ_000050`, and `S1_TTTZ_000090`

Full predictions, checkpoints, logs, and large `testouts/` outputs are intentionally excluded from Git.

## Next Experiments

- Compare LR construction methods: sparse-only, nearest interpolation, linear interpolation, anti-aliased/downsampled baselines, and residual variants.
- Track primary metrics on `missing`, `active-missing`, peak ratio, peak location error, and temporal consistency instead of relying only on full-field RMSE.
- Add stronger peak-aware and temporal-consistency losses if the focus is event-initial frames and strong wavefront reconstruction.

# Current Bohai Vz Sparse Reconstruction Experiments

This folder contains a Git-trackable snapshot of the later/current Bohai Vz
sparse reconstruction experiments. The older `resshift200ep_every10/` folder is
kept as a historical best baseline, but the folders here are the current
experiment set for external review.

## Metric Summary

All metrics below are computed on the same sparse-mask 2x fixed-grid Vz task.

| Method | RMSE | MAE | RFNE | ACC | SSIM | Peak Ratio | RMSE Reduction vs Interp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interp baseline | 0.037303 | 0.010568 | 1.053080 | 0.301673 | 0.738434 | 0.853189 | baseline |
| E13 MixedBoundary | 0.024240 | 0.008423 | 0.684312 | 0.729250 | 0.804381 | 0.867044 | 35.02% |
| E14 ActiveHF | 0.022820 | 0.009094 | 0.644203 | 0.765999 | 0.789032 | 0.886153 | 38.83% |
| E16 TemporalConsistency | 0.022676 | 0.008950 | 0.640146 | 0.769104 | 0.793289 | 0.888497 | 39.21% |
| E17 SequenceOutput | 0.025275 | 0.009469 | 0.713520 | 0.701829 | 0.784444 | 0.875813 | 32.24% |
| ResShift200ep | 0.015829 | 0.006835 | 0.446843 | 0.897084 | 0.856351 | 0.971896 | 57.57% |
| E18 DataConsistency | 0.015829 | 0.006835 | 0.446845 | 0.897082 | 0.856349 | 0.971941 | 57.57% |

Interpretation:

- `ResShift200ep` remains the strongest main model by global metrics.
- `E18 DataConsistency` is effectively tied with `ResShift200ep`, so the extra
  sampling-step projection did not materially improve the already constrained
  output.
- `E16 TemporalConsistency` is the best Temporal3DUNet-family variant among the
  later supervised experiments, but it is still behind ResShift.
- `E17 SequenceOutput` is a controlled negative result in this setup.

## Figure Folders

```text
e13_mixed_boundary/
e14_active_hf/
e16_temporal_consistency_every10/
e17_sequence_output_every10/
e18_data_consistency_every10/
sampling_ablation/
cnn_refiner_v2_peak/
```

The every-10 folders contain per-frame visual comparisons with:

```text
Interp, SR, HR target, |Interp-HR|, |SR-HR|
```

The color plots use fixed field/error bars in each figure. Error panels include
P99 error and max error in the title.

## Sampling Ablation

The sampling ablation compares three 25% observed-point patterns:

| Sampling | Interp RMSE | SR RMSE | SR Active-Missing RMSE | SR ACC | SR SSIM |
|---|---:|---:|---:|---:|---:|
| regular25 | 0.037303 | 0.022614 | 0.036494 | 0.770408 | 0.791413 |
| random25 | 0.037510 | 0.024025 | 0.038838 | 0.735575 | 0.793488 |
| jittered25 | 0.037487 | 0.023178 | 0.037347 | 0.757857 | 0.799101 |

In this controlled test, changing the sparse sampling pattern alone did not fix
the high-frequency/structure loss problem.

## CNN Refiner Pilot

The CNN refiner is a post-processing pilot trained on top of ResShift200ep
outputs. On the two evaluated cases:

| Method | RMSE | MAE | RFNE | ACC | SSIM | Peak Ratio |
|---|---:|---:|---:|---:|---:|---:|
| ResShift200ep | 0.019524 | 0.007404 | 0.485723 | 0.877263 | 0.862851 | 0.959240 |
| CNNRefinedV1 | 0.018896 | 0.006923 | 0.470104 | 0.882946 | 0.869254 | 0.934098 |
| CNNRefinedV2 | 0.021311 | 0.008555 | 0.530201 | 0.864398 | 0.835571 | 1.027801 |

Interpretation: a conservative refiner can slightly improve RMSE/SSIM, but the
peak-forcing variant can overcorrect and degrade the field. It should remain a
pilot, not the main result.

## Next Targeted Experiment

`E20 ResShift SourceConditioned` is configured but not trained yet.

It keeps the current ResShift sparse-mask setup but changes the condition input
from 15 channels to 22 channels:

```text
5 frames x [Vz_sparse, Vz_interp, mask_observed]
+ x_norm, y_norm, z_norm, t_norm, source_dx_norm, source_dy_norm, source_r_norm
```

The goal is to test whether explicit source/time/space conditioning improves
active-missing wavefront structure and peak location recovery. This is a
targeted test of the current failure mode, not a broad model sweep.

Config:

```text
Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_resshift_source_conditioned_e20_gpu0.yaml
```

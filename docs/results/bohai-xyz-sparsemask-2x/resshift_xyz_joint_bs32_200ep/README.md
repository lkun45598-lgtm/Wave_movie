# XYZ-joint ResShift sparse-mask 2x result

Run:

- `Resshift_SparseMask2xObserved_XYZ_MSEAux_BS32_200ep`

Source evaluation directory before archiving:

- `/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_XYZ_MSEAux_BS32_200ep/xyz_eval`

Key configuration:

- Dataset root: `/data/Bohai_Sea/process_data_sparsemask_2x`
- Dynamic target variables: `Vx`, `Vy`, `Vz`
- LR condition variables:
  `Vx_sparse`, `Vx_interp`, `Vy_sparse`, `Vy_interp`,
  `Vz_sparse`, `Vz_interp`, `mask_observed`
- Temporal window: 5 frames
- Input condition channels: `5 x 7 = 35`
- Output channels: 3 (`Vx`, `Vy`, `Vz`)
- Spatial shape: `200 x 150`
- Patch size: 64
- Batch size: 32
- Epochs: 200
- ResShift diffusion steps: 15

Summary metrics:

| Scope | Interp RMSE | XYZ-ResShift RMSE | RMSE reduction | XYZ-ResShift ACC |
| --- | ---: | ---: | ---: | ---: |
| XYZ all | 0.02686 | 0.01005 | 62.6% | 0.921 |
| XYZ missing | 0.03102 | 0.01161 | 62.6% | 0.893 |
| XYZ active-missing | 0.03851 | 0.01435 | 62.8% | 0.894 |
| Vz all | 0.03730 | 0.01261 | 66.2% | 0.935 |
| Vz active-missing | 0.06075 | 0.01987 | 67.3% | 0.919 |

Artifact layout:

- `model_summary_xyz.csv`: aggregate XYZ and per-component metrics.
- `model_summary_vz_compare.csv`: Vz comparison with interpolation and prior
  single-Vz ResShift result.
- `per_frame_metrics.csv`: per-frame metrics.
- `xyz_metrics.json`: full metric dump.
- `comparison_*`: metric curves and case-level summary plots.
- `figs_every10_vx`, `figs_every10_vy`, `figs_every10_vz`: every-10-frame
  component visualizations.
- `figs_every10_magnitude`: every-10-frame vector magnitude visualizations.


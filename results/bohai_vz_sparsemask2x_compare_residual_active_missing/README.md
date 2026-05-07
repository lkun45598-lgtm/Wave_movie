# Bohai Vz Sparse-Mask 2x Results

This directory contains lightweight result artifacts for the Direct vs Explicit Residual Temporal3DUNet comparison.

The full local run outputs are excluded from Git because they contain checkpoints, logs, and 1200 full-frame prediction files. See `docs/bohai-vz-sparsemask-2x-summary.md` for the experiment setup, metrics, and reproduction commands.

Key files:

- `model_summary_vz.csv`: full-test summary metrics for interpolation baseline, Direct, and Explicit Residual.
- `comparison_per_frame_metrics_vz.png`: per-frame RMSE/RFNE/ACC/max-error comparison.
- `comparison_case_metrics_vz.png`: per-case metric comparison.
- `average_error_energy_vz.csv`: average radial-frequency error energy.
- `average_error_energy_vz_legend_bottom.png`: spectral error comparison figure.
- `diagnostics_direct/`: representative Direct diagnostic frames.
- `diagnostics_residual/`: representative Explicit Residual diagnostic frames.

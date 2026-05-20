# Bohai XYZ sparse-mask 2x reconstruction results

This directory stores selected evaluation artifacts for the Bohai Sea
three-component sparse wavefield reconstruction experiments.

The current main result is:

- `resshift_xyz_joint_bs32_200ep/`: XYZ-joint ResShift reconstruction trained on
  the sparse-mask 2x dataset with batch size 32 for 200 epochs.

Task definition:

- HR target: full-field `[Vx, Vy, Vz]` on a `200 x 150` grid.
- Sparse observation: 25% regular grid observations generated from the HR field,
  with unobserved locations zero-filled.
- Baseline: linear interpolation from sparse samples to the full field.
- Model input: 5 temporal frames of
  `[Vx_sparse, Vx_interp, Vy_sparse, Vy_interp, Vz_sparse, Vz_interp, mask_observed]`,
  i.e. 35 condition channels.
- Model output: center-frame full-field `[Vx, Vy, Vz]`.
- Evaluation focuses on all, observed, missing, and active-missing regions.


# Bohai E16 Temporal Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal, controlled temporal-consistency experiment for Bohai Vz sparse wavefield reconstruction.

**Architecture:** Keep the existing Temporal3DUNet center-frame model unchanged. Extend `OceanNPY` only when `temporal_supervision_window` is enabled so each training sample can also return neighboring center samples. Extend `BaseTrainer` to run the same model on the small temporal supervision sequence and add a first-difference consistency loss.

**Tech Stack:** PyTorch, existing OceanNPY dataset layout, existing masked trainer, YAML configs, pytest.

---

### Task 1: Dataset Temporal Supervision

**Files:**
- Modify: `Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/datasets/OceanNPY.py`
- Test: `Ocean-Agent-SDK_core/tests/test_temporal_ocean_npy.py`

- [ ] Add a failing test showing `temporal_supervision_window=3` returns `(x, y, x_seq, y_seq)` without crossing case boundaries.
- [ ] Implement optional supervision indices in `_build_temporal_split`.
- [ ] Add `temporal_supervision_indices` support to `OceanNPYDatasetBase`.
- [ ] Crop `x_seq` and `y_seq` with the same patch as `x` and `y`.

### Task 2: Trainer Temporal Consistency Loss

**Files:**
- Modify: `Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/trainers/base.py`
- Test: `Ocean-Agent-SDK_core/tests/test_residual_hf_training.py`

- [ ] Add a failing test for `_temporal_consistency_loss`.
- [ ] Add a batch parser that supports `(x, y)`, `(x, y, mask)`, `(x, y, x_seq, y_seq)`, and `(x, y, mask, x_seq, y_seq)`.
- [ ] Add config keys under `optimize`: `temporal_consistency_weight` and `temporal_consistency_observed_channel`.
- [ ] During training only, if the weight is positive and sequence tensors exist, infer all sequence frames, compute `L1((SR[t+1]-SR[t]) - (HR[t+1]-HR[t]))`, and add it to the center-frame loss.

### Task 3: E16 Config And Log

**Files:**
- Create: `Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_temporal_consistency_e16_gpu0.yaml`
- Modify: `docs/bohai-vz-experiment-log.md`

- [ ] Copy the DirectActiveMissing config as the controlled baseline.
- [ ] Add `temporal_supervision_window: 3`.
- [ ] Add `temporal_consistency_weight: 0.1`.
- [ ] Use a new output directory under `/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xTemporalConsistency_E16_Vz`.
- [ ] Record hypothesis, config, output paths, and expected evaluation criteria in the log.

### Task 4: Verification

**Commands:**
- `pytest Ocean-Agent-SDK_core/tests/test_temporal_ocean_npy.py::TemporalOceanNPYTest::test_temporal_supervision_window_returns_neighboring_center_samples -q`
- `pytest Ocean-Agent-SDK_core/tests/test_residual_hf_training.py::ResidualHighFrequencyTrainingTest::test_temporal_consistency_loss_matches_neighbor_differences -q`
- `pytest Ocean-Agent-SDK_core/tests/test_residual_hf_training.py Ocean-Agent-SDK_core/tests/test_temporal_ocean_npy.py Ocean-Agent-SDK_core/tests/test_temporal3d_unet.py -q`
- `python -m py_compile Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/datasets/OceanNPY.py Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/trainers/base.py`
- `git diff --check`

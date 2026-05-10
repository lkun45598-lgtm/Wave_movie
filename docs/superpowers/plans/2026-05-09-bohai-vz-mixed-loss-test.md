# Bohai Vz Mixed Loss Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, testable mixed sparse loss experiment for Bohai Vz sparse reconstruction.

**Architecture:** Keep the existing sparse-mask dataset and Temporal3DUNet model unchanged. Add optional loss terms inside `MaskedCompositeSRLoss` so configs can weight active-missing, all-missing, and inactive-missing regions independently.

**Tech Stack:** PyTorch, pytest/unittest, YAML training configs, existing `main.py --mode train` pipeline.

---

### Task 1: Mixed Sparse Loss Terms

**Files:**
- Modify: `Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/utils/loss.py`
- Modify: `Ocean-Agent-SDK_core/tests/test_residual_hf_training.py`

- [ ] **Step 1: Write the failing test**

Add tests proving `active_missing_l1`, `missing_l1`, and `inactive_missing_l1` compute on the intended sparse regions using the observed mask from the LR input.

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/lz/miniconda3/envs/pytorch/bin/python -m pytest Ocean-Agent-SDK_core/tests/test_residual_hf_training.py -q`

Expected: FAIL because the new mixed sparse terms are not implemented.

- [ ] **Step 3: Implement minimal loss support**

Add helper masks inside `MaskedCompositeSRLoss`:
- observed mask from `input_tensor[..., observed_mask_channel]`
- missing mask = base mask and not observed
- active mask = `abs(target) > active_threshold`
- inactive mask = not active

- [ ] **Step 4: Run tests**

Run: `/home/lz/miniconda3/envs/pytorch/bin/python -m pytest Ocean-Agent-SDK_core/tests/test_residual_hf_training.py Ocean-Agent-SDK_core/tests/test_temporal_ocean_npy.py -q`

Expected: PASS.

### Task 2: Bohai Small Experiment Config

**Files:**
- Create: `Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_mixed_loss_test_gpu0.yaml`

- [ ] **Step 1: Create config**

Copy the existing `Temporal3DUNet_SparseMask2xObserved_Vz` config and change only:
- `log_dir` / `saving_path` to `Temporal3DUNet_SparseMask2xMixedLoss_Vz_Test`
- `epochs` to a small test run
- `eval_freq` to a practical interval
- loss terms to include `active_missing_l1`, `missing_l1`, `inactive_missing_l1`, `gradient_l1`, `peak_l1`, and `fft_hf_l1`
- `loss_mask_mode` to `static`, because mixed terms build their own sparse masks

- [ ] **Step 2: Preflight config**

Run: `/home/lz/miniconda3/envs/pytorch/bin/python Ocean-Agent-SDK_core/scripts/ocean-SR-training-masked/check_output_shape.py --config Ocean-Agent-SDK_core/configs/bohai_xyz_2x/bohai_vz_sparsemask_2x_temporal3dunet_mixed_loss_test_gpu0.yaml`

Expected: output shape compatible.

### Task 3: Launch Test Training

**Files:**
- Runtime output: `/data1/user/lz/wave_movie/testouts/Temporal3DUNet_SparseMask2xMixedLoss_Vz_Test`

- [ ] **Step 1: Start tmux training on GPU0**

Run the existing training entrypoint in conda env `pytorch`, with CPU thread limits and `CUDA_VISIBLE_DEVICES=0`.

- [ ] **Step 2: Monitor first epochs**

Check `train.log` for `training_error`, `epoch_train`, and first validation event. If loss is finite and validation appears, continue. If it fails, stop and inspect the traceback.

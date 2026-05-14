# Sparse Mask Sampling Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare regular, random, and jittered 25% sparse observation masks for Bohai Vz reconstruction while keeping the model and training protocol fixed.

**Architecture:** Extend the existing Bohai preprocessing script with a `--sparse-mask-pattern` option. Keep regular behavior unchanged, add fixed random and fixed jittered masks, and evaluate them through the same `Vz_sparse`, `Vz_interp`, `mask_observed` data layout.

**Tech Stack:** Python, NumPy, SciPy scattered interpolation, pytest, existing OceanNPY training/evaluation scripts.

---

### Task 1: Sampling Pattern Support

**Files:**
- Modify: `Ocean-Agent-SDK_core/scripts/preprocess_bohai_wave_xyz.py`
- Modify: `Ocean-Agent-SDK_core/tests/test_bohai_preprocess_downsample.py`

- [ ] Write failing tests for reproducible random masks, jittered one-point-per-block masks, and full-grid interpolation from jittered sparse observations.
- [ ] Run `pytest Ocean-Agent-SDK_core/tests/test_bohai_preprocess_downsample.py -q` and confirm the new tests fail before implementation.
- [ ] Add `regular`, `random`, and `jittered` sparse mask patterns to preprocessing.
- [ ] Add `--sparse-mask-pattern` and `--sparse-mask-seed` CLI options.
- [ ] Run the preprocessing tests again and confirm they pass.

### Task 2: Data Generation

**Files:**
- Create: `Ocean-Agent-SDK_core/scripts/run_bohai_sparsemask_sampling_ablation.sh`

- [ ] Generate `/data/Bohai_Sea/process_data_sparsemask_2x_random25`.
- [ ] Generate `/data/Bohai_Sea/process_data_sparsemask_2x_jittered25`.
- [ ] Confirm train/valid/test split counts match the regular dataset.

### Task 3: Quick Training And Evaluation

**Files:**
- Create configs under `Ocean-Agent-SDK_core/configs/bohai_xyz_2x/`
- Modify: `docs/bohai-vz-experiment-log.md`

- [ ] Train a lightweight Temporal3DUNet baseline for random25 and jittered25.
- [ ] Predict full test frames.
- [ ] Evaluate Interp and SR with `evaluate_visualize_bohai_vz.py`.
- [ ] Record metrics and every-10-frame visualization paths in the experiment log.

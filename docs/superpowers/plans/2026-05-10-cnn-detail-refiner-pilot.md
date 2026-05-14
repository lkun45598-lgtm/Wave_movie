# CNN Detail Refiner Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a lightweight CNN can repair local details in the existing ResShift200ep Bohai Vz sparse reconstruction output.

**Architecture:** Use the existing ResShift200ep test predictions as `SR_base`, split the 12 original test cases into case-level CNN train/valid/eval subsets, and train a small residual CNN to predict `HR - SR_base`. This is a fast diagnostic pilot, not the final leakage-free experiment; a final experiment must regenerate `SR_base` for original train/valid/test splits.

**Tech Stack:** PyTorch, NumPy, Matplotlib, existing Bohai sparse-mask dataset layout, existing `evaluate_visualize_bohai_vz.py` for downstream comparison.

---

### File Structure

- Create `Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py`
  - Standalone pilot trainer for `CNNDetailRefiner`.
  - Reads HR/LR arrays from `/data/Bohai_Sea/process_data_sparsemask_2x/test`.
  - Reads ResShift base predictions from `/data1/user/lz/wave_movie/testouts/Resshift_SparseMask2xObserved_Vz_MSEAux_200ep/predictions`.
  - Trains on case-level subsets to avoid frame-level leakage within the pilot.
  - Writes refined predictions, metrics, split metadata, and loss curves.
- Create `Ocean-Agent-SDK_core/tests/test_cnn_detail_refiner_pilot.py`
  - Tests residual CNN shape, observed-point hard constraint, case split behavior, and masked metric computation.
- Modify `docs/bohai-vz-experiment-log.md`
  - Add a new experiment entry documenting the pilot setup and result path.

### Task 1: Pilot Script Core

**Files:**
- Create: `Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py`
- Test: `Ocean-Agent-SDK_core/tests/test_cnn_detail_refiner_pilot.py`

- [ ] **Step 1: Add tests for reusable helpers**

Test these functions/classes:

```python
def test_apply_observed_constraint_preserves_sparse_points():
    pred = torch.zeros(1, 4, 4, 1)
    sparse = torch.full((1, 4, 4, 1), 7.0)
    mask = torch.zeros(1, 4, 4, 1)
    mask[:, ::2, ::2, :] = 1.0
    constrained = apply_observed_constraint(pred, sparse, mask)
    assert torch.equal(constrained[:, ::2, ::2, :], sparse[:, ::2, ::2, :])
    assert torch.equal(constrained[:, 1::2, 1::2, :], pred[:, 1::2, 1::2, :])
```

```python
def test_cnn_detail_refiner_shape():
    model = CNNDetailRefiner(in_channels=5, features=8, num_blocks=2)
    x = torch.randn(2, 5, 16, 16)
    y = model(x)
    assert y.shape == (2, 1, 16, 16)
```

- [ ] **Step 2: Implement minimal helpers**

Implement:

```python
def apply_observed_constraint(pred, sparse, observed_mask):
    return torch.where(observed_mask > 0.5, sparse, pred)
```

Implement `CNNDetailRefiner` as a no-downsample residual CNN:

```text
Conv2d(in_channels -> features)
SiLU
N x ResidualBlock(features)
Conv2d(features -> 1)
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lz /home/lz/miniconda3/envs/pytorch/bin/python -m pytest Ocean-Agent-SDK_core/tests/test_cnn_detail_refiner_pilot.py -q
```

Expected: helper tests pass.

### Task 2: Dataset and Pilot Split

**Files:**
- Modify: `Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py`
- Test: `Ocean-Agent-SDK_core/tests/test_cnn_detail_refiner_pilot.py`

- [ ] **Step 1: Add case-level split function**

Use deterministic default splits:

```text
train cases = S1_TTTZ, S1_TWVZ, S1_URZ, S1_UTU, S1_WATZ, S1_WCZ, S1_WHTZ, S1_WIAZ
valid cases = S1_WLCZ, S1_WTAZ
eval cases  = S1_WRRZ, S1_WTVZ
```

Rationale:

```text
WRRZ remains an eval hard case.
WTVZ gives a second held-out case.
Frames from the same case never cross CNN train/valid/eval.
```

- [ ] **Step 2: Implement `BohaiCNNRefinerDataset`**

For each sample, load:

```text
HR: test/hr/Vz/{base}.npy
SR_base: ResShift200ep/predictions/{base}_sr.npy
Vz_interp: test/lr/Vz_interp/{base}.npy
Vz_sparse: test/lr/Vz_sparse/{base}.npy
mask_observed: test/lr/mask_observed/{base}.npy
```

Build input channels:

```text
[SR_base, Vz_interp, Vz_sparse, mask_observed, SR_base - Vz_interp]
```

Target is `HR`.

- [ ] **Step 3: Add random patch sampling for training**

Use `patch_size=64` for train mode; use full frame for valid/eval.

### Task 3: Training, Prediction, and Metrics

**Files:**
- Modify: `Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py`

- [ ] **Step 1: Add masked loss**

Train on final refined field:

```text
delta = model(input)
sr_refined = sr_base + delta
sr_refined = observed hard constraint(sr_refined, Vz_sparse, mask_observed)
loss = active_missing_l1 + 0.2 * missing_l1 + 0.2 * gradient_l1 + 0.1 * laplacian_l1 + 0.05 * residual_l2
```

- [ ] **Step 2: Add eval metrics**

Compute for Interp, ResShift base, and CNN refined:

```text
rmse, mae, rfne, acc, ssim, peak_ratio, p99_abs_error, max_abs_error
```

Also compute missing-region and active-missing-region RMSE.

- [ ] **Step 3: Save outputs**

Write:

```text
/data1/user/lz/wave_movie/testouts/CNNRefiner_ResShift200epBase_Vz_Pilot/
  best_model.pth
  train.log
  split_cases.json
  metrics_summary.csv
  per_frame_metrics.csv
  predictions_eval/*.npy
```

### Task 4: Launch Pilot Run

**Files:**
- Runtime output only.

- [ ] **Step 1: Run tests**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lz /home/lz/miniconda3/envs/pytorch/bin/python -m pytest Ocean-Agent-SDK_core/tests/test_cnn_detail_refiner_pilot.py -q
```

- [ ] **Step 2: Start training in tmux**

Run on GPU1 unless occupied:

```bash
tmux new-session -d -s bohai_cnn_refiner_pilot "cd /data1/user/lz/wave_movie && CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMEXPR_MAX_THREADS=8 MPLCONFIGDIR=/tmp/matplotlib-lz PYTHONUNBUFFERED=1 /home/lz/miniconda3/envs/pytorch/bin/python Ocean-Agent-SDK_core/scripts/train_cnn_detail_refiner_bohai_vz.py --epochs 80 --batch-size 32 --patch-size 64 --device cuda"
```

- [ ] **Step 3: Inspect result**

Read:

```text
metrics_summary.csv
per_frame_metrics.csv
predictions_eval/
```

Conclusion rule:

```text
Keep this direction if CNN refined improves active-missing RMSE, p99_abs_error, or peak_ratio on held-out eval cases without increasing max_abs_error substantially.
Discard or redesign if it only improves train cases or creates sharper but wrong structures.
```

### Task 5: Experiment Log

**Files:**
- Modify: `docs/bohai-vz-experiment-log.md`

- [ ] **Step 1: Add E10 entry**

Record:

```text
E10 - CNN detail refiner pilot on ResShift200ep base
Purpose: test two-stage detail refinement
Status: running/completed
Output path
Split cases
Metrics summary
Representative figures
Conclusion
```

### Self-Review

- Spec coverage: The plan covers the user request to train a lightweight CNN on ResShift-restored fields for detail repair.
- Placeholder scan: No unresolved placeholders are left; all paths, split cases, and output names are concrete.
- Scope check: This plan is a pilot. The final leakage-free version is intentionally out of scope because generating ResShift base predictions for original train/valid/test may take many hours.

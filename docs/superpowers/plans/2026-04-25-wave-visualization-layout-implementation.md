# Wave Visualization Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `save_wave_movie.py` 的 PNG/GIF 输出改成稳定的科研版式，统一坐标轴、标题和右侧竖直色标，并重新生成最终图像。

**Architecture:** 保留现有数据选择与颜色范围逻辑，用 `pyvista` 读取网格，用 `matplotlib` 负责稳定的 2D 绘图与 GIF 帧输出。新增纯函数处理四边形拆三角形、坐标投影和坐标轴格式化，便于单元测试。

**Tech Stack:** Python, NumPy, Matplotlib, imageio, PyVista

---

### Task 1: Add failing tests for layout helpers

**Files:**
- Modify: `tests/test_save_wave_movie.py`
- Test: `tests/test_save_wave_movie.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement minimal helpers**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Switch rendering to Matplotlib scientific layout

**Files:**
- Modify: `save_wave_movie.py`
- Test: `tests/test_save_wave_movie.py`

- [ ] **Step 1: Add mesh projection, triangle extraction, and fixed layout helpers**
- [ ] **Step 2: Replace PNG renderer with Matplotlib layout**
- [ ] **Step 3: Replace GIF renderer with fixed-frame Matplotlib animation**
- [ ] **Step 4: Keep CLI behavior and output naming stable**

### Task 3: Verify and regenerate deliverables

**Files:**
- Modify: `save_wave_movie.py`
- Test: `tests/test_save_wave_movie.py`

- [ ] **Step 1: Run unit tests**
- [ ] **Step 2: Render one PNG and one short GIF to check layout stability**
- [ ] **Step 3: Render full GIF plus key PNG frames**
- [ ] **Step 4: Report output paths and any remaining caveats**

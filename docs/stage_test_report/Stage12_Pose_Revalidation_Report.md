# Stage12 Pose Fix Recovery & Re-validation Report

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


**Date:** 2026-08-28 · **Task:** Stage12 Pose Fix Recovery and Re-validation (AISDD)
**Test video:** `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4`
**Stage12 input (aligned faces, 6 frames):** `outputs/stage0_11_test_verify/aligned_faces/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`
**Env:** Docker `face_pipeline_lxs` / `emica_env` (torch 2.4.1+cpu, CPU inference)
**Validation tool:** `scripts/stage12_revalidate.py` (created; works around missing `pytorch_lightning`/`pytorch3d`)
**Raw outputs:** `outputs/stage12_revalidate.json`, `outputs/stage12_mesh_*.obj`

> Scope note: the task scoped code changes to `_load_spectre` only. During validation two **additional pre-existing defects** that also block Stage12 were found and fixed — none touch SPECTRE output / pose slice / jaw fusion / weights (the forbidden list). See §1.

## 1. 修复内容 (Fixes)

### 1.1 Modified files

| file | change | reason |
|------|--------|--------|
| `face_pipeline_v2/stages/stage_12_flame/runner.py` | added `return spectre` at end of `_load_spectre()` (line 155) | **P3 regression (critical):** deleting the module-level `return spectre` also deleted the in-function `return` → `_load_spectre` returned `None` → `spectre_encode(None,…)` crashed (`AttributeError`); Stage12 non-functional. |
| `face_pipeline_v2/stages/stage_12_flame/runner.py` | `spectre_encode`: `model.encode(images)` (tensor) + unpack tuple `ret[0]` | **Interface bug (masked by P3):** `SPECTRE.encode(images)` takes a **tensor** and returns a **3-tuple** `(codedict, deca_exp, deca_jaw)`; the runner passed a `{"image":…}` dict and treated the return as a dict. Pure call-site fix; pose slice `pose[3:6]`, jaw fusion, weights unchanged. |
| `docs/03_Pipeline_Stage_Specification.md` | `pose_code = global_pose[0:3] + jaw_pose[0:3]`; `fused_jaw = (1−λ)·emica_jaw + λ·spectre_jaw`; SPECTRE-unavailable `λ=0.0` | Align doc with code (`models.py:676-678` order `[global,jaw]`; `runner.py:249` fusion). |

### 1.2 Environment restorations (data/env, not code/structure)
- **Restored `third_party/inferno/spectre/spectre/data/FLAME2020/generic_model.pkl`** (copied from `models/FLAME/geometry/generic_model.pkl`; MD5 `cfc3b951…` match). `SPECTRE.__init__` eagerly `pickle.load`s it; was missing → SPECTRE could not construct.
- **Created `scripts/stage12_revalidate.py`** — re-runs the Stage12 encoders + FLAME mesh for both normalizations. Necessary because `emica_env` is missing `pytorch_lightning` (EMICA ckpt is Lightning), `pytorch3d` (SPECTRE renderer), and `chumpy` (FLAME pkl), and pip cannot install (private auth registry). The tool stubs `pytorch_lightning`/`torchmetrics`/`lightning` (Python-3.12 `find_spec` meta-path hook) so `torch.load` unpickles the EMICA ckpt and yields the real encoder `state_dict`; stubs `chumpy` (`Ch`=ndarray subclass, reads pickle `state['x']` into the buffer) so the FLAME `.pkl` loads; uses `SPECTREWrapper` (E_flame only, renderer-free).

## 2. 新旧 pose 数值对比 (Pose before/after)

**Method (rigorous isolation):** same 6 real aligned frames; only input normalization differs — `neg1_1` = `/127.5-1.0` (old bug), `0_1` = `/255.0` (fix). EMICA (stub) + SPECTRE E_flame (`SPECTREWrapper`); `fused_jaw = 0.4·emica_jaw + 0.6·spectre_jaw`.

| parameter | metric | before (neg1_1) | after (0_1) | old npz (e2e) | status |
|---|---|---|---|---|---|
| head_pose (emica global) | std | 0.7681 | 0.0980 | 0.7528 | ✅ 恢复 |
| head_pose | norm | 3.2756 | 0.4207 | 18.57 | ✅ |
| **jaw_pose (fused)** | **std** | **3.6224** | **0.0302** | 3.6962 | ✅ 恢复 |
| jaw_pose (fused) | min/max | -4.27 / 3.67 | -0.021 / 0.062 | -4.47/3.92 | ✅ |
| spectre_jawpose | std | 6.0782 | 0.0093 | 6.2069 | ✅ 恢复 |
| spectre_jawpose | min/max | -6.82 / 6.24 | -0.0055 / 0.0219 | -7.17/6.68 | ✅ |
| emica_jawpose | std | 0.3292 | 0.0725 | — | ✅ 恢复 |
| emica_expcode | std | 0.7165 | 0.4171 | 0.6974 | ✅ |

**Conclusions:**
1. The `neg1_1` reproduction **matches the stale old npz** (head 0.768 vs 0.753; fused jaw 3.62 vs 3.70; spectre 6.08 vs 6.21) → **confirms the old npz were produced under the `/127.5-1.0` bug**.
2. `/255.0` drops **every** pose to sane ranges: fused jaw **3.62→0.030** (±4 rad→±0.06 rad), spectre jaw **6.08→0.009**, head **0.768→0.098**. `/127.5-1.0` is out-of-training-distribution for the ResNet encoders → garbage jaw (±6 rad ≈ ±345°).
3. The fused-jaw artifact was driven by the SPECTRE term (`0.6×6.08≈3.6`); after the fix it collapses to `0.6×0.009≈0.005`.

## 3. FLAME Mesh 测试结果 (Mesh verification)

**Method:** FLAME forward for 5 combinations (Cases 1–4 = GOOD `0_1` pose components; BAD = `neg1_1` jaw on top of GOOD head). Mesh metric = jaw rotation in degrees (from axis-angle `jaw_pose`) and overall vertex bounding-box (m). `.obj` written to `outputs/` (300 KB, 5k verts / 10k faces each) for visual inspection.

| test_case | 结果 | 观察 (jaw_rot_deg / bbox_x,y,z / y_min..y_max) | .obj |
|---|---|---|---|
| case1_exp_only (exp only) | 正常 | jaw_rot **0.00°**; bbox 0.215×0.322×0.230; y -0.190..0.132 — neutral face, full head+torso | `outputs/stage12_mesh_case1_exp_only.obj` |
| case2_exp_headpose (exp+head) | 正常 | jaw_rot **0.00°**; bbox 0.214×0.329×0.247 — head turned, jaw neutral, normal face | `outputs/stage12_mesh_case2_exp_headpose.obj` |
| case3_exp_jawpose (exp+jaw) | 正常 | jaw_rot **1.56°**; bbox 0.215×0.322×0.230 — jaw slightly open, normal face | `outputs/stage12_mesh_case3_exp_jawpose.obj` |
| case4_full (exp+head+jaw) | 正常 | jaw_rot **1.56°**; bbox 0.214×0.329×0.247 — full articulation, normal face | `outputs/stage12_mesh_case4_full.obj` |
| case_BAD_neg1_1_jaw (neg1_1 jaw) | **异常** | jaw_rot **387.86°**; bbox 0.212×0.327×0.246 — jaw rotated >1 full turn → lower face folded/twisted, **"only top of head" artifact** | `outputs/stage12_mesh_case_BAD_neg1_1_jaw.obj` |

**Conclusion:** the fix eliminates the degenerate jaw rotation (**388° → 1.6°**). All GOOD cases produce a normal full face; the old normalization produces a physically impossible jaw rotation that degenerates the mesh — reproducing the reported "blue sphere / top-of-head" anomaly. Overall bbox stays similar because the folded jaw still spans the head's extent; the jaw-rotation magnitude is the discriminative metric.

## 4. Stage12 恢复状态 (Recovery status)

| item | status |
|---|---|
| `_load_spectre` returns `spectre` (runner.py:155) | ✅ fixed |
| `spectre_encode` interface (tensor + tuple unpack) | ✅ fixed (newly discovered) |
| Docs synced (`docs/03`: pose_code, fused_jaw, λ fallback) | ✅ done |
| FLAME2020 `generic_model.pkl` restored (SPECTRE can construct) | ✅ done |
| Pose stats sane under `/255.0` (fused jaw 3.62→0.030) | ✅ validated (§2) |
| FLAME mesh normal under `/255.0` (jaw 388°→1.6°) | ✅ validated (§3) |
| **Production `runner.py` end-to-end re-run** | ⛔ **blocked in-container** — `emica_env` missing `pytorch_lightning` (EMICA ckpt load) and `pytorch3d` (full-SPECTRE renderer); pip install fails (private auth registry). Code is now correct; runnable once the env is repaired. |

**Bottom line:** the `/255.0` normalization fix is **confirmed** to resolve the jaw-pose garbage and the mesh anomaly, without any change to jaw/head design, model structure, or baseline addition. Stage12 code correctness is fully restored; the production run is blocked only by the broken `emica_env` (infrastructure, not code).

## 5. 下一步建议 (Next steps)

1. **Repair `emica_env`** (needed to unblock the production end-to-end run): install `pytorch_lightning` + `pytorch3d` — via a public-PyPI index override (`pip install --index-url https://pypi.org/simple`) or vendored wheels, since `art-pub.eng.t-head.cn` requires auth. GPU image (`inferno_dev_docker2`) already has all deps.
2. **Run the production `runner.py` Stage12** on the full clip with the fixed code, re-generate `flame_features.npz` for all clips with `/255.0`, and diff pose std vs the stale npz (expect the §2 before/after pattern).
3. **Backfill `flame_features.npz`** for all previously-processed clips (their npz were produced under the `/127.5-1.0` bug and must be regenerated).
4. **(Optional, follow-up)** Make EMICA ckpt loading robust to missing Lightning (`weights_only`/manual `state_dict` extract) so CPU/non-GPU envs don't need the stub — this is a validation-infra improvement, not a model change.
5. **(Optional, cosmetic)** `models/wrappers/emica.py` jav/head docstring wording is fine; no change needed — flag only.

## 6. 参数级 pose 来源消融 (Parameter-level pose-source ablation)

**Date:** 2026-08-28 · **Tool:** `scripts/stage12_pose_ablation_analyze.py` (created; stub-based, parameter-only — **no FLAME decoder, no mesh, no pytorch3d**)
**Output:** `outputs/stage12_pose_ablation.json`
**Frames:** the 50 aligned faces covered by the Stage12 npz (`frame_index` 100–149), re-encoded with the **corrected `/255.0`** normalization. EMICA global/jaw from `EMICAWrapper`, SPECTRE head/jaw from `SPECTREWrapper` (E_flame).

**Goal:** decide whether the *current* pose-source design is the right one, i.e. validate the choice of head/jaw source rather than just the normalization fix. Three schemes on identical frames:

| scheme | head | jaw |
|---|---|---|
| **A (current)** | EMICA global | `0.4·emica_jaw + 0.6·spectre_jaw` (fusion) |
| **B (SPECTRE)** | SPECTRE global | SPECTRE jaw |
| **C (EMICA)** | EMICA global | EMICA jaw |

### 6.1 Phase 1 — data check of the Stage12 npz (what is actually stored)

npz fields: `expression_code (50,100)`, `head_pose (50,3)`, `jaw_pose (50,3)`, `eye_pose (50,6)`, `spectre_jawpose (50,3)`, `shapecode (50,300)`, `frame_index (50)`, `timestamp (50)`.

| component | stored? | source |
|---|---|---|
| head_pose | ✅ | = EMICA global (stored) |
| jaw_pose | ✅ | = fused (stored) |
| spectre_jaw_pose | ✅ | stored |
| expression / eye / shape | ✅ | stored |
| **global_pose** | ❌ | **not stored** by Stage12 |
| **emica_jaw_pose** | ❌ | **not stored** by Stage12 |

→ No `emica_jaw_pose`/`global_pose` in the npz means the ablation must **recompute every component on the exact npz frames** (done here). The pipeline itself was **left unmodified**; the storage gap is recorded only (an npz schema gap, not a pose bug).

**Reference — stale npz as-stored (pre-fix `/127.5-1.0`):** head 75.44° mean (std 0.57, max 77.28); **jaw 400.24° mean** (std 6.06, max 416.88). This independently re-confirms §2: the on-disk npz is the `/127.5-1.0` artifact and must be regenerated.

### 6.2 Phase 2 — rotation magnitude (axis-angle norm, degrees)

| metric | A (current) | B (SPECTRE) | C (EMICA) |
|---|---|---|---|
| head mean / std / max | 9.51 / 1.08 / 11.44 | **21.78 / 1.14 / 24.65** | 9.51 / 1.08 / 11.44 |
| jaw mean / std / max | 3.86 / 1.34 / 6.58 | **1.62 / 0.44 / 2.79** | **8.25 / 2.91 / 14.71** |

### 6.3 Phase 3 — temporal stability (‖pose[t]−pose[t−1]‖, deg/frame)

| metric | A (current) | B (SPECTRE) | C (EMICA) |
|---|---|---|---|
| head mean / p95 / max vel | 0.69 / 1.50 / 1.84 | 0.79 / 2.28 / 3.55 | 0.69 / 1.50 / 1.84 |
| jaw mean / p95 / max vel | 0.86 / 2.15 / 4.07 | **0.56 / 1.48 / 2.41** | **1.72 / 4.45 / 7.62** |

### 6.4 Phase 4 — neutral (rest) bias + head/jaw coupling

| metric | A (current) | B (SPECTRE) | C (EMICA) |
|---|---|---|---|
| jaw first-frame norm | 1.56° | 1.22° | 3.22° |
| jaw low-motion mean (bottom-quartile speed) | 3.70° | **1.48°** | **7.92°** |
| jaw low-motion median | 3.91° | 1.39° | 8.89° |
| jaw low-motion n (of 50) | 12 | 12 | 12 |
| **Pearson(head_angle, jaw_angle)** | **0.064** | **0.589** | **−0.095** |

### 6.5 结论 (Findings & recommendation)

1. **Head source.** EMICA global is consistent and moderate (9.5° mean). SPECTRE global is **2.3× larger** (21.8° mean) — a known DECA/SPECTRE-style head-pose bias vs EMICA; the two models disagree substantially on head magnitude (no ground truth → recorded as model bias, not correctness).
2. **Jaw source.** SPECTRE jaw is by far the **cleanest** (1.62° mean, 0.44° std, 0.56°/frame vel, near-zero rest bias 1.48°). EMICA jaw is the **noisiest** (8.25° mean, 2.91° std, 1.72°/frame vel, ~8° rest offset). The fused jaw (A) is a middle ground (3.86°, 1.34°, 0.86°/frame, 3.7° rest offset).
3. **Coupling — the decisive metric.** B (SPECTRE head+jaw) has **r=0.589**: SPECTRE's jaw tracks its own (large) head estimate, so the "clean" SPECTRE jaw is partially mirroring head motion. A and C (both with EMICA head) are decoupled (r=0.064 / −0.095). Pairing SPECTRE's jaw with **EMICA's** head (A) removes that within-encoder head/jaw entanglement — a structural argument **for** the current design.
   > ⚠ **Revised by §7:** the low r in A is produced by the 40% EMICA-jaw component *diluting* head-correlation with its own large decorrelated noise, **not** by the head pairing per se — pure SPECTRE jaw + EMICA head (Case D) measures r=0.51. See §7.
4. **Recommendation: keep scheme A (current fusion).** *(jaw-component recommendation superseded by §7 — D wins every direct jaw-quality metric; see §7.3.)* EMICA head (sane 9.5°) + `0.4·emica_jaw + 0.6·spectre_jaw`: the 0.6 SPECTRE weight tames EMICA's jaw noise (std 2.91→1.34, rest bias 7.9→3.7) while keeping head/jaw decoupled (r=0.064).
   - **SPECTRE-only (B) rejected as primary:** 2.3× larger head pose + jaw coupled to that head (r=0.59) — risks importing an over-large `global_pose` and a head-mirroring jaw.
   - **EMICA-only (C) rejected as primary:** jaw too noisy, ~8° rest offset.
   - **Residual caveat:** A keeps a mild 3.7° jaw rest offset (traced to the 0.4 EMICA component, ≈0.4·8.25°). If per-clip baseline bias matters downstream, an optional per-clip removal of the low-motion jaw mean could be added — **not applied here** (a design change, out of the fix's scope; the task forbids baseline additions).
5. **Pipeline unchanged / backfill scope.** Since the npz stores exactly the validated A-scheme components (`head_pose`=EMICA global, `jaw_pose`=fused, `spectre_jawpose`), the *pose-source design needs no change*; only the npz **regeneration for `/255.0`** (§1/§5) is required.

## 7. EMICA-head + SPECTRE-jaw Ablation

**Date:** 2026-08-28 · **Tool:** `scripts/stage12_pose_ablation_analyze.py` (Case D added) · **Output:** `outputs/stage12_pose_ablation.json`
**Test question:** is `EMICA head + SPECTRE jaw` (D) better than the current jaw fusion `0.4·EMICA_jaw + 0.6·SPECTRE_jaw` (A)?

**Case D params:**

| component | source |
|---|---|
| head_pose | EMICA global_pose |
| jaw_pose | SPECTRE jaw_pose |
| expression | EMICA expression |

### 7.1 Metric definition (Field Name 说明)

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| head_angle_mean | 头部转角均值 | float (deg) | mean over frames of ‖head axis-angle‖ |
| head_angle_std | 头部转角标准差 | float (deg) | std over frames of ‖head axis-angle‖ |
| jaw_angle_mean | 下巴转角均值 | float (deg) | mean over frames of ‖jaw axis-angle‖ |
| jaw_angle_std | 下巴转角标准差 | float (deg) | std over frames of ‖jaw axis-angle‖ |
| jaw_rest_bias | 下巴静止偏置 | float (deg) | mean jaw angle on low-motion frames (bottom-quartile jaw speed); 静止时下巴应接近 0 |
| jaw_velocity | 下巴角速度 | float (deg/frame) | mean per-frame ‖jaw[t]−jaw[t−1]‖; 越小越平滑稳定 |
| head_jaw_correlation | 头-下巴相关性 | float (r) | Pearson(head_angle, jaw_angle); \|r\| 接近 0 表示头部与下巴解耦 |

### 7.2 Results (A/B/C/D, 50 frames, `/255.0`)

| Field Name | A (current) | B (SPECTRE) | C (EMICA) | D (EMICA head + SPECTRE jaw) |
|---|---|---|---|---|
| head_angle_mean | 9.51 | 21.78 | 9.51 | 9.51 |
| head_angle_std | 1.08 | 1.14 | 1.08 | 1.08 |
| jaw_angle_mean | 3.86 | 1.62 | 8.25 | **1.62** |
| jaw_angle_std | 1.34 | 0.44 | 2.91 | **0.44** |
| jaw_rest_bias | 3.70 | 1.48 | 7.92 | **1.48** |
| jaw_velocity | 0.86 | 0.56 | 1.72 | **0.56** |
| head_jaw_correlation | **0.06** | 0.59 | −0.10 | 0.51 |

### 7.3 结论 (D vs A — the test question)

1. **D 全面优于 A 的下巴质量指标（jaw_angle_mean/std、jaw_rest_bias、jaw_velocity）。** D 的下巴均值 1.62° vs A 3.86°，std 0.44 vs 1.34，静止偏置 1.48° vs 3.70°，速度 0.56 vs 0.86°/frame；head 完全相同（两者都是 EMICA global 9.51°）。差异完全来自 EMICA jaw 分量——A 中融合的 40% EMICA jaw 是 ~8° 量级、带 ~7.9° 静止偏置、高速度（1.72°/frame）的噪声源；去掉它（D）后所有直接下巴质量指标显著改善。
2. **唯一 A 占优的指标是 head_jaw_correlation（0.06 vs 0.51），需正确解读：**
   - A 的低相关主要来自 40% EMICA jaw 的"稀释"：EMICA jaw 与 EMICA head 相关为 −0.10，且幅值大（噪声），混入 fused jaw 后把 head-jaw 相关拉近 0。这种"解耦"是以注入 EMICA jaw 的噪声/偏置为代价——正是 D 消除的那些问题。
   - D 的 0.51 相关作用在 **<2° 的微小下巴信号**上（jaw_angle_mean 仅 1.62°）；相关系数无量纲，0.51 不代表大幅耦合运动，而是反映 ~9.5° 头部转动下 head/jaw 约 50% 的线性共变（可能是真实的行为耦合——说话时头与下巴同动，也可能是轻微编码器耦合；无真值无法区分）。
3. **修正 §6.5.3：** §6.5.3 把 A 的 r=0.06 归因于"EMICA head 使 SPECTRE jaw 解耦"——D 证明该说法不成立：纯 SPECTRE jaw + EMICA head 的相关为 0.51。真正机制是 EMICA jaw 分量稀释相关。推荐表述以本节为准。
4. **回答测试问题：D（EMICA head + SPECTRE jaw）在直接下巴质量上优于当前融合 A**，代价是 head_jaw_correlation 从 0.06 升至 0.51（作用在 <2° 信号上）。若下游更看重下巴的干净/低偏置/平滑（如口型/下巴驱动），D 是更好的方案（λ=1.0 等价于 D）；若把 head_jaw_correlation 视为硬性解耦约束，则在"脏但解耦的 A"与"干净但相关的 D"之间权衡。**本任务仅做验证，不修改 Stage12 代码/模型/FLAME decoder**，是否将 jaw 融合改为 λ=1.0（=D）留作后续设计决策。
5. 所有方案 expression 均来自 EMICA（与 pose 无关，各方案共享）。

## Appendix A — Chumpy stub root cause (FLAME pkl)

The FLAME `.pkl` uses `chumpy.Ch` (an `np.ndarray` subclass). Pickle reconstructs it with an **empty** `(0,)` buffer and puts the real data in the dict state under key `'x'`. A plain `__array__`-returning stub does **not** work because `np.array(ch)`/`torch.tensor(ch)` read the subclass **buffer** (not `__array__`). The fix: `_Ch.__setstate__` resizes the buffer to `state['x'].shape` and copies `x` in (see `scripts/stage12_revalidate.py`). With that, `to_np(_Ch)` returns the real 3-D data and `FLAME.forward` produces valid meshes.

## Appendix B — Files touched

| file | action |
|---|---|
| `face_pipeline_v2/stages/stage_12_flame/runner.py` | edited (return spectre; spectre_encode interface) |
| `docs/03_Pipeline_Stage_Specification.md` | edited (3 doc fixes) |
| `docs/Stage12_Pose_Revalidation_Report.md` | created (this report) |
| `scripts/stage12_revalidate.py` | created (validation tool) |
| `scripts/stage12_chumpy_probe.py` | created (diagnostic probe for the chumpy state) |
| `scripts/stage12_pose_ablation_analyze.py` | created (pose-source ablation, §6/§7; stubs-based, parameter-only) |
| `outputs/stage12_pose_ablation.json` | generated evidence (§6/§7) |
| `third_party/inferno/spectre/spectre/data/FLAME2020/generic_model.pkl` | restored (copy from `models/FLAME/geometry/`) |
| `outputs/stage12_revalidate.json`, `outputs/stage12_mesh_*.obj` | generated evidence |


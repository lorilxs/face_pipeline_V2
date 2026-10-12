# Stage12 SPECTRE Jaw Final Adoption Report（验证报告）

> **⚠️ 已被取代 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案
> （FLAME mesh 重建验证）。本报告保留作为历史记录；最终方案见
> `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。

**日期:** 2026-08-28
**范围:** 对「head=EMICA global / jaw=SPECTRE / expr=EMICA」（方案 D）的正式小范围验证与采用评估。
**关键约束:** **本阶段未修改任何生产代码**（`runner.py`/`processor.py`/`stage.py`/`config.yaml` 均未改动）；D 输出以 `outputs/stage12_pose_validation/` 预览产物形式验证。

---

## 1. 修改原因

当前 Stage12 的下巴为融合值 `jaw_pose = 0.4·EMICA_jaw + 0.6·SPECTRE_jaw`（λ=0.6）。ablation 表明 40% EMICA jaw 分量是主要噪声/偏置源（EMICA jaw 均值 ≈8.25°、静止偏置 ≈7.92°、速度 ≈1.72°/frame），把干净、低偏置、平滑的 SPECTRE jaw 稀释。方案 D（直接取 SPECTRE jaw）在**所有直接下巴质量指标**上优于当前方案 A。

## 2. 修改前后公式

| | 公式 |
|---|---|
| 修改前 | `jaw_pose = 0.4·EMICA_jaw + 0.6·SPECTRE_jaw`；`head_pose = EMICA_global`；`expression = EMICA` |
| 修改后 | `jaw_pose = SPECTRE_jaw`（等价 λ=1.0）；`head_pose = EMICA_global`（不变）；`expression = EMICA`（不变）|

## 3. 数据统计（7 字段，A vs D，同 50 帧，`/255.0`）

工具：`scripts/stage12_pose_source_d_preview.py`（与 `stage12_pose_ablation_analyze.py` 同方法、同帧、同归一化，重编码 EMICA + SPECTRE wrapper）。
输出：`outputs/stage12_pose_validation/stage12_pose_source_d_preview.json`。

| Field Name | 中文名称 | A（当前 fusion） | D（EMICA head + SPECTRE jaw） | 判定 |
|---|---|---|---|---|
| head_angle_mean | 头部转角均值 (deg) | 9.5145 | 9.5145 | 相同（同源）|
| head_angle_std | 头部转角标准差 (deg) | 1.0768 | 1.0768 | 相同 |
| jaw_angle_mean | 下巴转角均值 (deg) | 3.8603 | **1.6192** | D 优 |
| jaw_angle_std | 下巴转角标准差 (deg) | 1.3447 | **0.4402** | D 优 |
| jaw_rest_bias | 下巴静止偏置 (deg) | 3.6992 | **1.4795** | D 优 |
| jaw_velocity | 下巴角速度 (deg/frame) | 0.8594 | **0.5559** | D 优 |
| head_jaw_correlation | 头-下巴相关性 (r) | **0.0643** | 0.5106 | A 优（见 §5 解读）|

> **复现性：** 本次验证重编码得到的 A/D 指标与既有 `outputs/stage12_pose_ablation.json` **逐项完全一致（max_abs_diff = 0.0）**，证明同一帧集合、同一修正归一化下结果可复现。

## 4. 异常检查（方案 D 输出）

| 检查项 | 要求 | 结果 | 状态 |
|---|---|---|---|
| 无 NaN | NaN count = 0 | 0 | ✅ |
| 无 Inf | Inf count = 0 | 0 | ✅ |
| jaw rotation < 1 rad | max ‖jaw‖ < 1.0 | 0.0487 rad（2.79°） | ✅ |
| head rotation | max ‖head‖ | 11.44° | ✅（正常量级）|
| 帧数一致 | = 源 npz 50 | 50 | ✅ |
| 时间连续 | frame_index 连续递增 | 100..149，差=1 | ✅ |
| 时间戳单调 | timestamp 非降 | ✅ | ✅ |

## 5. Ablation 结果（A/B/C/D 全景，来自 `stage12_pose_ablation.json`）

| Field Name | A | B (SPECTRE) | C (EMICA) | **D** |
|---|---|---|---|---|
| head_angle_mean | 9.51 | 21.78 | 9.51 | 9.51 |
| head_angle_std | 1.08 | 1.14 | 1.08 | 1.08 |
| jaw_angle_mean | 3.86 | 1.62 | 8.25 | **1.62** |
| jaw_angle_std | 1.34 | 0.44 | 2.91 | **0.44** |
| jaw_rest_bias | 3.70 | 1.48 | 7.92 | **1.48** |
| jaw_velocity | 0.86 | 0.56 | 1.72 | **0.56** |
| head_jaw_correlation | **0.06** | 0.59 | −0.10 | 0.51 |

结论（承 `Stage12_Pose_Revalidation_Report.md §7.3`）：
1. D 全面优于 A 的直接下巴质量指标；差异完全来自去掉 40% EMICA jaw 噪声分量。
2. 唯一 A 占优的 `head_jaw_correlation`：A 的低相关是 40% EMICA jaw **噪声稀释**所致，非真正“解耦”；D 的 0.51 作用于 **<2° 的微小下巴信号**（scale-free r），不表示大幅耦合运动。
3. D 等价 λ=1.0。

## 6. 接口兼容验证（Stage12 → Stage17 端到端）

对 D 预览 npz（生产 schema）执行完整读取 + 装配：

| 验证 | 结果 |
|---|---|
| `read_stage12_outputs(D 预览 npz)` | 50 条 `FLAMEFeature`，expcode=100 / posecode=6 / jawpose=3 / globalpose=3 / spectre_jawpose=3 全部通过 ✅ |
| `build_training_sample`（Stage17，配 Stage13 gaze=50 + Stage14 audio=200）| T=50, valid_length=50, visual_feature `[50,112]`, dropout=0 ✅ |
| visual_feature 布局 | expression[0:100] + pose[100:106] + gaze[106:112]，pose 块首帧 = [global(3), jaw(3)] ✅ |

→ **Stage12→Stage17 数据接口完全兼容；Stage17/FLAME decoder 零改动。**

## 7. 风险分析

| 风险 | 评估 |
|---|---|
| SPECTRE 成为 jaw 硬依赖 | 现状管线已依赖 SPECTRE；`runner.py` 对 SPECTRE 缺失回退 zeros 的路径需在正式采用时改为显式失败（见 Change Plan §5）|
| head_jaw_correlation 0.06→0.51 | 作用于 <2° 信号，非大幅耦合；若下游视其为硬性解耦约束，则应保留 A（权衡）|
| 存量 npz 为 pre-fix 产物 | **必须先**用 `/255.0` 管线重生成 npz 再采用；D 预览值不能由存量 `spectre_jawpose` 直接代入（其含 ~±7 rad 偏移）|
| 回归 | 选项 A（λ=1.0 配置）可一行回滚 |

## 8. 是否建议永久采用

**建议（有条件地）采用方案 D**：`head = EMICA global`、`jaw = SPECTRE jaw`、`expression = EMICA`，不再融合 EMICA jaw。

前提条件（务必满足）：
1. 用修正 `/255.0` 管线重新生成 `flame_features.npz`（并重跑 Stage13/14/17 保持一致）。
2. 采用前明确 SPECTRE 缺失时的降级/失败策略（避免静默全零 jaw）。
3. 更新 `flame_metadata.json` 记录 `jaw_fusion_lambda: 1.0` 与 `jaw_source: spectre`。
4. 下游若以 `head_jaw_correlation` 作为硬性解耦约束，则维持 A；否则 D 为更优选择。

**采用方式（二选一）：** 选项 A（`config.yaml:260` `jaw_fusion_lambda: 0.6 → 1.0`，零代码改动，推荐）或选项 B（`runner.py`+`processor.py` 直接取 `spectre_jaw`）。详见 `Stage12_Pose_Source_Change_Plan.md`。

**本阶段交付（无生产代码变更）：**
- `scripts/stage12_pose_source_d_preview.py`（新增验证工具）
- `outputs/stage12_pose_validation/`（D 预览 npz / A 预览 npz / D 分量 npy / 验证 json）
- `docs/Stage12_Pose_Source_Review.md` / `docs/Stage12_Pose_Source_Change_Plan.md`（本报告前置）

# Stage12 Pose Source Review（检查报告）

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


**日期:** 2026-08-28
**任务:** 正式评估将 Stage12 pose 来源永久改为「head=EMICA global / jaw=SPECTRE / expr=EMICA」（方案 D）之前，先完成只读代码链路检查与数据契约核对。
**范围:** 本报告仅检查，不修改任何生产代码。

---

## 1. 结论概述

- Stage12 的下巴融合 `jaw_pose = (1−λ)·EMICA_jaw + λ·SPECTRE_jaw`（λ=0.6）实现于 **两处**：`runner.py:255-256`（subprocess/Docker 主路径）与 `processor.py:93-96`（direct 模式）。
- 方案 D 在代码层面 **等价于 λ=1.0**：`fused_jaw = (1−1.0)·emica_jaw + 1.0·spectre_jaw = spectre_jaw`。因此修改可以是 **纯配置一行**（`config.yaml` 的 `jaw_fusion_lambda: 0.6 → 1.0`），也可在 `stage_12_flame/` 内硬改。
- 输出 npz 字段名/维度 **不变**；Stage17 只消费 `expcode[100]` + `posecode[6]`（=global[3]+jaw[3]）的 **维度**，不关心 jaw 值的来源 → **Stage12→Stage17 接口完全兼容**。
- **关键存量问题（必须先解决）:** 当前 `outputs/stage12_flame/**/flame_features.npz` 是 pre-fix（`/127.5-1.0`）产物，其 `spectre_jawpose` 含 ~±7 rad 分量偏移，**不可**直接作为 D 的下巴来源；任何采用都必须用修正后的 `/255.0` 管线重新生成 npz（与 `Stage12_Pose_Revalidation_Report.md §5` 的结论一致）。

---

## 2. 代码链路检查（Phase 1，只读）

### 2.1 Stage12 主路径（Docker subprocess）— `face_pipeline_v2/stages/stage_12_flame/runner.py`

| 组件 | 位置 | 当前逻辑 |
|---|---|---|
| EMICA global pose 来源 | `runner.py:58-117` `_load_emica` → `run_stage12:245` | `emica_global = emica_out["globalpose"]`（DECA encoder，axis-angle [3]）|
| EMICA jaw pose 来源 | `runner.py:244` | `emica_jaw = emica_out["jawpose"]`（DECA encoder，axis-angle [3]）|
| SPECTRE jaw pose 来源 | `runner.py:247-253` | `spectre_jaw = spectre_out["pose"][3:6]`（SPECTRE E_flame 6D pose 的 jaw 半段）|
| **jaw fusion（修改点）** | **`runner.py:255-256`** | **`fused_jaw = (1.0 - λ)·emica_jaw + λ·spectre_jaw`** |
| expression 来源 | `runner.py:241` | `emica_exp = emica_out["expcode"]`（EMICA expression [100]）|
| 结果装配 | `runner.py:262-266` | `jaw_pose=fused_jaw`, `head_pose=emica_global`, `spectre_jawpose=spectre_jaw` |
| CLI 入口 | `runner.py:371-372` | `--jaw_fusion_lambda` 默认 `0.6` |

### 2.2 Stage12 direct 模式 — `processor.py`

| 组件 | 位置 | 当前逻辑 |
|---|---|---|
| **jaw fusion（修改点）** | **`processor.py:93-96`** | `fused_jaw = (1.0 - λ)·emica_jaw + λ·spectre_jaw` |
| FLAMEFeature 装配 | `processor.py:99-110` | `jawpose=fused_jaw`, `globalpose=emica_global`, `spectre_jawpose=spectre_jaw` |

> `processor.py` 在 `stage.py` 的 `_process_direct()`（`stage.py:172-192`）中调用，仅在 `runtime_mode=direct` 时使用；当前生产配置为 `subprocess`。

### 2.3 配置与管线接线

| 文件 | 位置 | 内容 |
|---|---|---|
| `configs/config.yaml` | `:251-264` | `stage12.jaw_fusion_lambda: 0.6`（当前唯一 λ 来源）|
| `stage.py` | `:72-77` | 从 config 读 `jaw_fusion_lambda`（默认 0.6），传给 subprocess/direct 两路 |
| `integration.py` | `:89` | `invoke_stage12_runner` 把 λ 作为 `--jaw_fusion_lambda` 传入 runner |
| `integration.py` | `:122-189` | `read_stage12_outputs`：npz → `FLAMEFeature`（`jaw_pose→jawpose`, `head_pose→globalpose`, `expression_code→expcode`, `spectre_jawpose→spectre_jawpose`）|

### 2.4 结论（Phase 1）

- **jaw fusion 只有两处实现**（`runner.py:255-256`、`processor.py:93-96`），且都按统一公式，λ 由 `config.yaml` 驱动。改 jaw 来源的**最小触点**是 `config.yaml:260` 一行（λ=1.0）。
- EMICA jaw 是 DECA encoder 前向的副产品，无法在不改 encoder 配置的前提下“关掉”；λ=1.0 下它只是不被用于 `jaw_pose`，仍可作为审计字段保留。

---

## 3. FLAME 数据契约检查（Phase 2）

### 3.1 `flame_features.npz` 字段（当前，实测）

| 字段 | 维度 | 来源 | 修改后 |
|---|---|---|---|
| `expression_code` | `[T,100]` | EMICA expression | 不变 |
| `head_pose` | `[T,3]` | EMICA global pose | 不变 |
| `jaw_pose` | `[T,3]` | fused（当前）/ SPECTRE（D）| **仅值变，维度不变** |
| `spectre_jawpose` | `[T,3]` | SPECTRE raw jaw（审计）| 不变（=D 的 jaw_pose 值）|
| `eye_pose` | `[T,6]` | zeros（Stage13 占位）| 不变 |
| `shapecode` | `[T,300]` | zeros（MICA 未加载）| 不变 |
| `frame_index` | `[T]` | 帧号 | 不变 |
| `timestamp` | `[T]` | 时间戳 | 不变 |

### 3.2 下游契约

- `FLAMEFeature`（`data/models.py:609-678`）：`expcode[100]`, `jawpose[3]`, `globalpose[3]`, `posecode` 属性 = `globalpose(3)+jawpose(3)` [6]。
- Stage17（`stage_17_fusion/processor.py:231-249`）：只用 `f.expcode`（校验 100 维）与 `f.posecode`（校验 6 维）→ `visual_feature = [expression(100), pose(6), gaze(6)] = [T,112]`。
- Stage17 读取：`stage_17_fusion/runner.py:114` 用 `read_stage12_outputs`（Stage12 官方读函数），字段映射见 §2.3。
- FLAME decoder 输入：`posecode = [global(3), jaw(3)]`，顺序固定（`models.py:676-678`）。

### 3.3 结论（Phase 2）

修改 jaw 来源后：
1. ✅ **输出维度不变**（npz 全部字段 shape/dtype 不变）；
2. ✅ **下游 Stage17 不需要修改**（只校验维度，不校验值来源）；
3. ✅ **FLAME decoder 输入格式保持一致**（posecode 仍为 [global, jaw]，仅 jaw 半段值来源变化）；
4. ✅ **不需要改 Stage13/14/17、FLAME decoder、Docker 环境**。

---

## 4. 关键发现

1. **存量 npz 为 pre-fix artifact。** `outputs/stage12_flame/**/flame_features.npz` 的 `spectre_jawpose` 各列均值 ≈ [−6.90, −6.92, 6.40] rad，而修正 `/255.0` 重编码的 SPECTRE jaw 各列均值 ≈ [0.007, 0.019, −0.006] rad、全局 max|Δ|=7.30 rad、逐列相关 ≈ [−0.14, 0.40, −0.11]。二者**几乎无关**——存量值是 `/127.5-1.0` 归一化 bug 的产物。
   → **含义：不能拿存量 npz 的 `spectre_jawpose` 直接替到 `jaw_pose`；任何 D 采用都必须用修正管线重跑并重生成 npz。**
2. **D 等价 λ=1.0，修改触点极小**（见 §2.4）。
3. **方案 D 引入新的硬依赖：** `runner.py:247-253` 在 SPECTRE 输出缺失时回退 `spectre_jaw = zeros(3)`；若采用 D 且 SPECTRE 不可用，jaw 将退化为全零（静默），需在变更计划中明确降级策略。

---

## 5. 影响范围（初步）

| 范围 | 影响 |
|---|---|
| Stage12 `runner.py` / `processor.py` | 若走 λ=1.0 配置方案：**零代码改动**；若走硬改：仅 fusion 2 行 |
| `configs/config.yaml` | 1 行（`jaw_fusion_lambda: 0.6 → 1.0`）|
| `flame_metadata.json` | `jaw_fusion_lambda` 字段随配置变化，建议补 `jaw_source` 说明 |
| Stage17 / 下游 | 无（维度兼容）|
| 存量 npz | **必须用 `/255.0` 重新生成**（影响 Stage13/14/17 下游产物一致性）|
| 指标影响 | `head_jaw_correlation` 0.06 → 0.51（详见验证报告）|

*详细修改方案见 `Stage12_Pose_Source_Change_Plan.md`，验证证据见 `Stage12_SPECTRE_Jaw_Final_Adoption_Report.md`。*

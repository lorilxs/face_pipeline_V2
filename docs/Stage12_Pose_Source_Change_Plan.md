# Stage12 Pose Source Change Plan（修改方案）

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


**日期:** 2026-08-28
**前置:** `Stage12_Pose_Source_Review.md`（代码链路 + 数据契约）、`Stage12_Pose_Revalidation_Report.md §7`（A/B/C/D ablation）、`Stage12_SPECTRE_Jaw_Final_Adoption_Report.md`（小范围验证）。
**状态:** 本方案为**计划**，未应用于生产代码（本阶段仅验证 + 报计划）。

---

## 1. 修改前 / 修改后

### 修改前（现状，λ=0.6）

```python
# runner.py:255-256 / processor.py:93-96（两处相同）
fused_jaw = (1.0 - 0.6) * emica_jaw + 0.6 * spectre_jaw   # = 0.4·EMICA_jaw + 0.6·SPECTRE_jaw
jaw_pose   = fused_jaw
head_pose  = emica_global
expression = emica_exp
```

### 修改后（方案 D）

```python
jaw_pose   = spectre_jaw            # 等价于 fused_jaw with λ=1.0
head_pose  = emica_global           # 不变
expression = emica_exp              # 不变
# spectre_jawpose 仍保留原始 SPECTRE jaw 作为审计字段（= jaw_pose 值）
```

---

## 2. 方案选项

### 选项 A（推荐）：纯配置一行，λ=1.0

`configs/config.yaml:260`：

```yaml
stage12:
  jaw_fusion_lambda: 0.6   # 修改前
  jaw_fusion_lambda: 1.0   # 修改后（= D：jaw = SPECTRE）
```

- 优点：**零代码改动**、最小、可逆（改回 0.6 即回退）、不改动 `stage_12_flame/` 内逻辑；与现有 `--jaw_fusion_lambda` 参数化完全兼容。
- 缺点：λ 语义上仍是“融合”，需要文档/元数据注明实际为纯 SPECTRE；`flame_metadata.json` 记录的 λ 需同步。
- 注意：`config.yaml` 位于 `configs/`，若严格限定只允许动 `stage_12_flame/`，则需走选项 B。

### 选项 B：在 `stage_12_flame/` 内硬改（明确语义）

`runner.py:255-256` 与 `processor.py:93-96`：

```python
# 修改前
fused_jaw = ((1.0 - jaw_fusion_lambda) * emica_jaw
             + jaw_fusion_lambda * spectre_jaw)
# 修改后（直接取 SPECTRE）
fused_jaw = spectre_jaw
```

- 优点：语义直白，代码自解释为“jaw 来源 = SPECTRE”；不受 λ 配置影响。
- 缺点：改动 2 处代码；λ 参数/CLI 选项成为死参数（需清理或保留兼容）；以后想恢复融合要改回。

### 选项 C：wrapper 层提供独立 jaw 接口（改动最大，不推荐）

在 `emica.py` / `spectre.py` 加显式 `get_jaw()` 等接口——对当前需求过度设计；`spectre_jawpose` 审计字段已存在，无需新增接口。

> **建议：选项 A（配置一行）。** 若团队硬性要求“改动必须落在 `stage_12_flame/` 内”，则用选项 B。

---

## 3. 任务预设问题逐条答复

| # | 问题 | 答复 |
|---|---|---|
| 1 | 是否需要删除 EMICA jaw 计算？ | **否。** EMICA jaw 是 DECA encoder 前向的固有输出，无法在不动 encoder 配置的前提下单独关闭；λ=1.0（或选项 B 直接取 SPECTRE）下它只是不再被写入 `jaw_pose`。保留计算（开销可忽略）。|
| 2 | 是否保留 EMICA jaw 作为 debug 输出？ | **建议保留。** 当前 `EMICAWrapper.encode()` 已返回 `jawpose`；`flame_features.npz` 目前**未**存 `emica_jawpose` 字段（`Stage12_Pose_Revalidation_Report.md §1` 已记录该 schema 缺口）。本方案**不新增** npz 字段（避免波及 `read_stage12_outputs`），仅保留内存级审计；如需落盘可后续单独立项。|
| 3 | 是否需要修改 metadata？ | **需要。** `flame_metadata.json` 当前记录 `jaw_fusion_lambda`；采用后应记录 `"jaw_fusion_lambda": 1.0` 并在 `notes` 注明 `"jaw_source": "spectre"`，保证产物自描述。|
| 4 | 是否需要修改 config.yaml？ | **需要。** 选项 A 改 `stage12.jaw_fusion_lambda: 1.0`（`config.yaml:260`）。|

---

## 4. 影响范围分析

| 项 | 影响 |
|---|---|
| npz schema | **不变**（8 字段，shape/dtype 全同）|
| Stage17 | **无需修改**（只消费 expcode[100]+posecode[6] 维度，`processor.py:231-249`）|
| FLAME decoder 输入 | **不变**（posecode = [global(3), jaw(3)]，`models.py:676-678`）|
| Stage13/14 | 无直接代码影响；但**依赖 Stage12 npz 重新生成**保持时间轴一致 |
| Docker 环境 | 不涉及 |
| 存量 npz | **必须用 `/255.0` 修正管线重新生成**（见 §6 迁移步骤）|
| 指标 | 直接下巴质量全面改善；`head_jaw_correlation` 0.06→0.51（见验证报告 §5）|
| 运行时依赖 | D 使 **SPECTRE 成为 jaw 的硬依赖**（见 §5 风险）|

---

## 5. 风险分析与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| SPECTRE 不可用 → jaw 全零 | `runner.py:247-253` 在 SPECTRE 输出缺失时回退 `zeros(3)` | 采用前给 `runner.py` 增加显式失败（抛错）或明确降级策略；Stage12 现状已依赖 SPECTRE，风险等级不变 |
| `head_jaw_correlation` 上升 | 0.06→0.51，作用于 <2° 信号 | 已在 §7.3 解读；若下游把 correlation 当硬性解耦约束，则维持 A |
| 存量 npz 不一致 | pre-fix 产物会污染 Stage13/14/17 下游 | 迁移必须全链路重跑 Stage12→17 |
| 回滚 | — | 选项 A：改回 0.6 即回滚；选项 B：git revert 2 行 |

---

## 6. 迁移步骤（确认后执行）

1. （可选）备份 `configs/config.yaml`。
2. 选项 A：`config.yaml:260` 改 `jaw_fusion_lambda: 1.0`；选项 B：改 `runner.py:255-256` + `processor.py:93-96`。
3. 重跑 Stage12（Docker `face_pipeline_lxs` + `emica_env`，修正 `/255.0`）→ 重新生成 `flame_features.npz`。
4. 用 `scripts/stage12_pose_source_d_preview.py`（或 `stage12_pose_ablation_analyze.py`）复核：D 指标应复现 `jaw_angle_mean≈1.62° / std≈0.44° / rest_bias≈1.48° / velocity≈0.56°`、异常检查全过、Stage17 端到端 T/valid_length 不变。
5. 重跑 Stage13/14/17，确认 `training_sample` 输出与 manifest 一致。
6. 更新 `flame_metadata.json` 的 `jaw_fusion_lambda` + `jaw_source` 说明。
7. 更新本计划状态为“已执行”。

---

## 7. 建议代码 diff（选项 B，未应用）

```diff
--- a/face_pipeline_v2/stages/stage_12_flame/runner.py
+++ b/face_pipeline_v2/stages/stage_12_flame/runner.py
@@ -252,7 +252,7 @@ def run_stage12(
             else:
                 spectre_jaw = np.zeros(3, dtype=np.float32)

-            fused_jaw = ((1.0 - jaw_fusion_lambda) * emica_jaw
-                         + jaw_fusion_lambda * spectre_jaw)
+            # Scheme D: jaw source = SPECTRE jaw (was 0.4*EMICA+0.6*SPECTRE fusion)
+            fused_jaw = spectre_jaw

             result = {
                 "frame_index": face["frame_index"],
                 "timestamp": face["timestamp"],
                 "clip_id": face["clip_id"],
                 "expression_code": emica_exp.tolist(),
                 "jaw_pose": fused_jaw.tolist(),
                 "head_pose": emica_global.tolist(),
                 "eye_pose": [0.0] * 6,
                 "spectre_jawpose": spectre_jaw.tolist(),
                 "shapecode": [0.0] * 300,
             }

--- a/face_pipeline_v2/stages/stage_12_flame/processor.py
+++ b/face_pipeline_v2/stages/stage_12_flame/processor.py
@@ -93,7 +93,8 @@ def extract_flame_parameters(
-            # --- Jaw fusion ---
-            # fused_jaw = (1 - λ) * emica_jaw + λ * spectre_jaw
-            fused_jaw = (1.0 - jaw_fusion_lambda) * emica_jaw + \
-                        jaw_fusion_lambda * spectre_jaw
+            # --- Jaw source: SPECTRE (Scheme D; was 0.4*EMICA+0.6*SPECTRE) ---
+            fused_jaw = spectre_jaw
```

> 若选选项 A，则无以上代码 diff，仅 `config.yaml` 一行。
> **本阶段不应用以上任何 diff；待确认后执行。**

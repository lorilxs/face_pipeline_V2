# Stage12 EMICA Jaw Final Adoption Report

- **日期**: 2026-08-31
- **状态**: ✅ FINAL — `jaw_pose` 正式切换为 **EMICA**，并同步 pipeline 设计、代码、配置、文档、测试与真实数据输出
- **结论**: `jaw_pose = EMICA_jaw_pose`（纯 EMICA，无融合）；`jaw_fusion_lambda` 已废弃；SPECTRE 原始 jaw 以 `spectre_jawpose` debug 字段保留

---

## 1. 修改原因

原生产方案（经 AISDD 批准采用）为 SPECTRE-jaw（Scheme D）：

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| expression_source | 表情来源 | string | EMICA expression |
| head_source | 头部姿态来源 | string | EMICA global_pose |
| jaw_source | 下巴来源 | string | 原 SPECTRE jaw_pose |
| gaze_source | 眼睛来源 | string | MediaPipe gaze |

经过 **FLAME mesh 实际生成验证**：

- **SPECTRE jaw 版本**：嘴部区域存在扭曲（mouth-region distortion）
- **EMICA jaw 版本**：mesh 重建效果更好

因此正式确定：`jaw_pose` 使用 **EMICA 提取结果**。

## 2. SPECTRE jaw vs EMICA jaw mesh 对比结论

对照实验（`outputs/ablation_emica_jaw/` vs 生产，11 个 clip，共 2188 帧）结论：

- SPECTRE jaw 与 EMICA jaw **全局最大差值 0.2327 rad ≈ 13.33°**
- EMICA jaw 具有更大的运动幅度：movement std / 帧间 velocity 均大于 SPECTRE
- mesh 重建：EMICA jaw 版本嘴部区域无 SPECTRE 版本的扭曲，重建效果更好
- 对照验证确认：除 jaw 外，expression / head_pose / gaze / audio 均与生产完全一致（max diff = 0）

## 3. 修改文件列表

**核心代码（Face_Pipeline_V2）：**

| 文件 | 修改 |
|---|---|
| `face_pipeline_v2/stages/stage_12_flame/processor.py` | `fused_jaw = emica_jaw`（无条件）；`jaw_source` 默认 `"emica"`；`jaw_fusion_lambda` 标记 deprecated |
| `face_pipeline_v2/stages/stage_12_flame/runner.py` | `fused_jaw = emica_jaw`（无条件）；`save_outputs` 元数据 `jaw_source=emica`；CLI 默认 `"emica"` |
| `face_pipeline_v2/stages/stage_12_flame/stage.py` | `jaw_source` 默认 `"emica"`；`jaw_fusion_lambda` 兼容 deprecated 标记读取 |
| `face_pipeline_v2/stages/stage_12_flame/integration.py` | `invoke_stage12_runner` 默认 `jaw_source="emica"` |
| `face_pipeline_v2/stages/stage_17_fusion/export.py` | `DEFAULT_SOURCES["jaw_source"] = "emica"` |
| `face_pipeline_v2/data/models.py` | `FLAMEFeature` docstring：jawpose = EMICA（最终方案）；spectre_jawpose = audit 字段 |
| `scripts/stage0_17_e2e_stage12.py` | `--jaw_source` 默认 `"emica"` |
| `scripts/final_mesh_export_validate.py` | Stage12 校验改为 `jaw == emica_jawpose`（`jaw_equals_emica`） |
| `scripts/generate_final_mesh_export_report.py` | 报表字段改为 `jaw==emica` / `jaw_vs_emica_maxdiff` |

**配置：**

| 文件 | 修改 |
|---|---|
| `configs/config.yaml` | `jaw_source: emica`；`jaw_fusion_lambda` → deprecated 标记 |
| `configs/config_dataset_mode.yaml` | 同上 |

**测试：**

| 文件 | 修改 |
|---|---|
| `tests/test_stage12_fusion.py` | 期望改为 `jaw == emica_jaw`（任意 λ） |
| `tests/test_stage12_processor.py` | jaw 测试期望改为 EMICA |
| `tests/test_stage12_pipeline_integration.py` | TestJawFusion 期望改为 EMICA |
| `tests/test_stage17_export.py` | 期望 `jaw_source == "emica"` |

**文档：**

| 文件 | 修改 |
|---|---|
| `docs/03_Pipeline_Stage_Specification.md` | §3.18 输出表 / 设计约束改为 EMICA jaw，保留历史 |
| `docs/02_Data_Model_and_Interface.md` | FLAMEFeature 字段表 / pose_code / jaw 来源改为 EMICA，保留历史 |
| `docs/04_Config_and_Model_Management.md` | config 示例 `jaw_fusion_lambda` → deprecated |
| `docs/01_System_Architecture.md` | Stage12 描述改为 EMICA jaw |
| `docs/Stage17_Multimodal_Export_Report.md` | metadata_check 期望 `jaw=emica` |
| `PROJECT_CONTEXT.md` | Jaw Source (FINAL Adopted) 段 |
| 历史报告（`Stage12_SPECTRE_Jaw_Adoption/Final_Adoption`、`Stage12_17_Final_Regression`、`Stage12_Pose_*`、`Stage0_17_End_to_End`、`Final_Stage0_17_Mesh_Export`、`Stage12_14_DataFlow`、`Stage12_EMICA_Jaw_Ablation` 等） | 顶部加「已被取代 / 已正式采用」标注，保留历史 |

**新增：**

| 文件 | 说明 |
|---|---|
| `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md` | 本报告 |
| `scripts/export_production_multimodal.py` | 生产 Stage17 fusion + export 驱动（EMICA jaw） |

## 4. 最终 FLAME 参数来源说明

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| expression | 表情参数 | float[100] | EMICA expression code |
| head_pose | 头部姿态 | float[3] | EMICA global_pose |
| jaw_pose | 下巴姿态 | float[3] | **EMICA jaw_pose（最终方案）** |
| gaze | 眼睛姿态 | float[6] | MediaPipe gaze |

最终 FLAME feature：`visual_feature_dim = 112`

布局保持不变：

```
expression: [0:100]
head_pose:  [100:103]
jaw_pose:   [103:106]
gaze:       [106:112]
```

`global_pose + jaw_pose` 的 FLAME pose 拼接顺序不变。SPECTRE 原始 jaw 保留为
`spectre_jawpose` debug 字段（审计/对比用），不进入 `jaw_pose`。

## 5. 测试结果

- **单元测试**：`tests/` 全量（排除 `manual/`、`test_scrfd_ort.py` 环境相关错误）：
  **687 passed**；15 个失败均为 stage_05/stage_09 历史无关问题（pre-existing，与本次改动无关）。
  本次改动的 Stage12/Stage17/export/datamodel 相关测试全部通过（64 passed）。
- **真实数据验证**（11 clips，`CNFACE_00001_BV18Qg96YE1W_000000`）：
  - Stage12（EMICA jaw）/ Stage13 / Stage14 / Stage17 全部 rc=0
  - 检查项：expression_shape (T,100) ✅ · head_pose_shape (T,3) ✅ ·
    jaw_pose_shape (T,3) ✅ · gaze_shape (T,6) ✅ · audio_shape (T,768) ✅ ·
    nan_check 无 NaN/Inf ✅ · jaw_source_check `jaw_pose == emica_jawpose`（max diff = 0）✅
- **生产输出已重新生成**：`outputs/stage0_17_e2e/dataset_validation/`（Stage12/13/14/17 fusion）与
  `outputs/stage17_multimodal_export/`（flame/audio/timestamp/metadata 各 11 文件 + manifest.csv），
  metadata 显示 `jaw_source: emica`。

## 6. 回滚方式

若需回滚到 SPECTRE jaw（或旧融合方案）：

1. **回滚代码**：在 `face_pipeline_v2/stages/stage_12_flame/processor.py` 中把
   `fused_jaw = emica_jaw` 改回 `fused_jaw = spectre_jaw`（或恢复 λ 融合公式）。
2. **回滚配置**：`configs/config.yaml` / `config_dataset_mode.yaml` 中
   `jaw_source: emica` → `jaw_source: spectre`，恢复 `jaw_fusion_lambda` 数值。
3. **恢复元数据**：`runner.save_outputs` 中 `jaw_source` 标签同步改回 `spectre`。
4. **重新生成输出**：重跑 Stage12 → Stage13 → Stage14 → Stage17 → export。
5. **回归测试**：更新 `tests/test_stage12_*` 期望并运行 `python3 -m pytest tests/`。

> 注意：本次为正式生产采用，生产输出（`outputs/stage0_17_e2e/dataset_validation/`、
> `outputs/stage17_multimodal_export/`）已按 EMICA jaw 重新生成并被覆盖。

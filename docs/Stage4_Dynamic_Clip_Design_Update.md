# Stage4_Dynamic_Clip_Design_Update.md

> **Status**: 设计文档（Documentation Phase）— 本阶段仅更新设计文档，**未修改** pipeline 代码 / config / stage 实现 / tests。
> **Date**: 2026-08-29
> **Related**: [Stage4_Removal_Impact_Audit.md](./Stage4_Removal_Impact_Audit.md)

---

# 1. Background

审计（`Stage4_Removal_Impact_Audit.md`）确认：

- Stage4 固定 8s window + 4s stride **不是**下游 Stage5-17 的硬性依赖。
- Stage5-17 均支持动态长度 T（`T = len(sequence)`）。
- 真正的结构性依赖是 `ClipInfo` + `clip_id` 契约，而非固定 8s 窗口。

新训练需求：Stage3 输出完整 single-face segment（例如 16 秒），希望整段进入下游：

```
16s segment
   ↓
1 个 clip
   ↓
Stage 5-17
   ↓
FLAME / Gaze / Audio
   ↓
TrainingSample (T=400)
```

---

# 2. Stage4 Redefinition

## 2.1 新定位

将 Stage4 从 **Fixed Window Split Stage（固定窗口切片）** 重新定义为：

**Clip Generation / Clip Adapter Stage（Clip 生成 / Clip 适配阶段）**

Stage4 负责：

- **Clip 生成**（Clip Generation）
- **clip_id 分配**（Clip ID Assignment）
- **Timestamp 保持**（Timestamp Preservation）
- **Downstream Contract 建立**（Establish the downstream contract）

不再描述为「Stage4 固定 8 秒切片」。

---

# 3. Data Contract

## 3.1 ClipInfo 支持动态 duration

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| clip_id | clip唯一ID | string | downstream sample identifier |
| start_time | 起始时间 | float | source video timestamp |
| end_time | 结束时间 | float | source video timestamp |
| duration | 时长 | float | dynamic clip length |
| frame_indices | 帧索引 | List[int] | source frames |

> 现有 `ClipInfo` 字段（`clip_id` / `scene_id` / `video_id` / `start_frame` / `end_frame` / `frame_count` / `duration`）继续保留；
> 强化 `start_time` / `end_time`，并补充 `frame_indices`；
> `duration` 语义由「固定 8s」改为「**动态 clip 长度**」（segment 模式下等于 Scene 整段时长）。

---

# 4. Configuration Design

```yaml
stage_04:
  mode: sliding_window   # sliding_window | segment
  window_size: 8         # sliding_window 模式有效
  stride: 4              # sliding_window 模式有效
  target_fps: 25
```

- `mode: sliding_window`：保持当前行为（8s window / 4s stride），用于传统训练样本生成。
- `mode: segment`：**新增设计**，每个 `SceneInfo`（Stage3 single-face segment）生成 1 个完整 `ClipInfo`（1:1 适配）。

**默认值**：保持当前行为 `sliding_window`，向后兼容。

---

# 5. Data Flow Update

## 5.1 旧数据流

```
Stage 3
  ↓
Stage 4  8s clips（固定窗口切片）
  ↓
Stage 5-17
```

## 5.2 新数据流

```
Stage 3
  ↓
Stage 4
  ├── sliding_window   → 8s window / 4s stride（传统训练样本）
  └── segment          → 整段 1:1 ClipInfo（动态时长）
  ↓
Stage 5-17
```

---

# 6. Design Decision Record

## 6.1 为什么不删除 Stage4

1. **Stage4 承担 Clip Contract 职责**：`ClipInfo` + `clip_id` 结构契约是 Stage5（输入）、Stage7/7.1（duration/start_time）、Stage9.5（clip asset keys）以及 Stage12/13/14/17（per-clip 文件布局）的样本标识基础。Stage4_Removal_Impact_Audit 确认各下游影响级别为 Low/None/Medium，但该契约本身是结构性依赖，删除需大规模重构。
2. **保留旧训练能力**：`sliding_window` 模式维持传统 8s 样本生成，不破坏已有数据集与训练流程。
3. **降低 Pipeline 风险**：保留 Stage 存在与接口，仅增加 `mode` 分支，改动面小、可回退、可 A/B 验证。
4. **支持不同模型需要不同 sequence 长度**：不同下游模型对时间序列长度偏好不同（8s≈200 帧 vs 16s≈400 帧），`mode` 可配置化让同一 pipeline 服务多种长度需求。

---

# 7. Impact & Notes

- **整段过滤语义变化**：segment 模式下，Stage9 硬过滤变为「整段过滤」——任一分段子窗口不达标（pose / offset / quality ratio）将拒绝整个 16s segment，而非仅丢弃某个 8s 子窗口。样本数从 `16s → 3×8s` 变为 `16s → 1×16s`。
- **下游兼容性**：Stage12/13/14/17 已支持动态 T（`T = len(...)`），无需修改即可处理 T≈400。
- **需要与训练同事确认**：16s 整段样本对训练数据量 / 过滤粒度的影响（见上）。

---

# 8. 后续实施清单（本阶段未执行）

- [ ] 代码：`face_pipeline_v2/stages/stage_04_temporal_clip_generation/` 增加 `mode` 分支（segment 模式 emit 整段 `ClipInfo`）。
- [ ] 配置：`configs/config.yaml` Stage4 增加 `mode` 字段（默认 `sliding_window`）。
- [ ] 验证：重跑 `scripts/stage0_17_e2e_validate.py`，确认 Stage17 输出 T=400 样本。
- [ ] 测试：更新 test fixtures（`frame_count` 200→400 / 动态）与 CLI 默认 `--num_frames`。
- [ ] 文档同步：`docs/02_Data_Model_and_Interface.md` §2.6.2、`docs/04_Config_and_Model_Management.md` §4.4.5、`docs/Stage17_Design_Review.md`（8s≈200 → 16s≈400 语义）。

---

# End of Stage4_Dynamic_Clip_Design_Update.md

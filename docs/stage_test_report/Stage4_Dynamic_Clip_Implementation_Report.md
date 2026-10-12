# Stage4_Dynamic_Clip_Implementation_Report.md

> **Status**: 实现完成（Implementation Phase）— 代码 / config / tests 已修改并通过回归。
> **Date**: 2026-08-29
> **Related**: [Stage4_Dynamic_Clip_Design_Update.md](./Stage4_Dynamic_Clip_Design_Update.md)
> **待人工确认**: 完成后**未运行**全 pipeline（`stage0_17_e2e_validate.py`），等待确认后执行。

---

# 1. 修改文件清单

| # | 文件 | 类型 | 变更摘要 |
|---|------|------|----------|
| 1 | `configs/config.yaml` | 修改 | `stage_04` 新增 `mode: sliding_window`（默认，向后兼容） |
| 2 | `face_pipeline_v2/stages/stage_04_temporal_clip_generation/clip_generator.py` | 修改 | 新增纯函数 `generate_segment_clips()`；模块 docstring 更新 |
| 3 | `face_pipeline_v2/stages/stage_04_temporal_clip_generation/stage.py` | 修改 | 读取/校验 `stage_04.mode`，循环中按模式分支；docstring 更新 |
| 4 | `tests/test_stage04_segment_mode.py` | 新增 | 17 个新测试（segment 模式 + 向后兼容） |
| 5 | `docs/Stage4_Dynamic_Clip_Implementation_Report.md` | 新增 | 本报告 |

> 未触碰：Stage5-17、Stage12/13/14 模型、`ClipInfo` 字段定义、`models.py`、`config_loader.py`。

---

# 2. 实现细节

## 2.1 配置（configs/config.yaml）

```yaml
stage_04:
  enabled: true
  mode: sliding_window   # sliding_window | segment (default: sliding_window)
  window_size: 8
  stride: 4
  target_fps: 25
```

- `mode` 默认 `sliding_window` → 行为与改动前完全一致。
- 校验通过 `ConfigLoader.load() + validate()`（`stage_04.mode = sliding_window`）。
- 非法 `mode` 值 → 记录 warning 并回退 `sliding_window`（容错，保证 pipeline 不因配置笔误中断）。

## 2.2 clip_generator.py — 新增 `generate_segment_clips(scene, target_fps)`

Segment 模式 1:1 适配器（纯逻辑，无重采样、无固定窗口）：

```
输入 SceneInfo（filter_result.passed=True）
   │  start_time / end_time / duration  → 原样复制（timestamp 保持）
   │  start_frame / end_frame           → 原样复制（完整帧范围，零重采样）
   │  frame_count                       → end_frame - start_frame（帧连续覆盖）
   ▼
1 个 ClipInfo（clip_id = f"{scene_id}_clip_0000"，稳定、确定性）
```

- 防御逻辑：`target_fps <= 0` → `DataProcessingError`；零时长 / 零帧退化 segment → 返回 `[]`。
- **未新增 `ClipInfo` 字段**：设计文档 §3.1 建议的 `frame_indices` 字段**未实现**（硬性约束「不得改变 ClipInfo 字段」优先）；帧完整性由既有 `start_frame → end_frame → frame_count` 三元组全范围覆盖保证。
- **未写死 8s / 200 frames**：duration / frame_count 全部取自 Scene，动态计算。

## 2.3 stage.py — `mode` 分支

- 新增 `VALID_MODES = ("sliding_window", "segment")` 与 `mode` 读取/校验。
- `process()` 循环中：
  - `mode == "segment"` → `generate_segment_clips(scene, target_fps)`
  - 否则（sliding_window 默认路径）→ `generate_clips(scene, window_size, stride, target_fps)`（原逻辑原样，逐字节兼容）。
- 日志（config / 每 scene 生成数 / 汇总）均增加 `mode` 字段。
- 仅 `passed` 场景参与 clip 生成（segment 模式同样遵守）。

---

# 3. 数据流变化

```
改动前                        改动后
Stage 3                      Stage 3
  ↓                             ↓
Stage 4 固定 8s/4s 切片      Stage 4  Clip Generation / Clip Adapter
  ↓                             ├── sliding_window → 8s/4s（传统，默认，不变）
Stage 5-17                     └── segment        → 整段 1:1（动态时长）
                                  ↓
                               Stage 5-17

16s segment（segment 模式）
  ↓
1 个 ClipInfo（frames=400, duration=16s, T≈400）
  ↓
Stage 5-17 → TrainingSample(T=400)
```

# 4. 测试结果

## 4.1 Stage4 测试（改动核心）

| 测试文件 | 结果 |
|----------|------|
| `tests/test_stage_04_clip_generation.py`（既有 26 项，sliding_window） | ✅ 26 passed |
| `tests/test_stage04_segment_mode.py`（新增 17 项） | ✅ 17 passed |
| **小计** | **43 passed** |

新增测试覆盖（用户要求 5 项 + 回归项）：

1. **8s segment** → 1 个完整 clip（duration=8s, frames=200）。
2. **16s segment** → 1 个完整 clip（duration=16s, frames=400）。
3. **400 帧输入** → clip 保持完整帧范围 `start_frame=0, end_frame=400, frame_count=400`（`frame_count == end-start`，零重采样）。
4. **clip_id 一致性** → 同 scene 跨调用确定性相同（`{scene_id}_clip_0000`）；不同 scene 唯一。
5. **timestamp 连续性** → `start_time/end_time/duration` 原样保持（`end-start == duration`），帧索引 `[start_frame, end_frame)` 连续无缝隙。
6. 额外：metadata 非写死（12.5s @24fps → duration=12.5, frames=300）；退化 segment → `[]`；`target_fps<=0` → 异常；多 scene 1:1；rejected scene 跳过；`validate_output` 通过。

## 4.2 回归测试（Stage5-17 现有 tests）

| 范围 | 结果 |
|------|------|
| Stage5-17（25 个测试文件：05/05.1/05.2/05.3/06/07/07.1_av/07.1_audio/08/09/09.1/09.5/10/11/12_fusion/12_metadata/12_processor/12_pipeline_integration/13_datamodel/13_processor/13_real_data/14/17_datamodel/17_export/17_fusion_processor） | **448 passed**，14 failed（存量，见 §5） |
| Stage0-3 + pipeline 接线（00/00.5/01/02/03/pipeline/interface/enhancement） | ✅ 113 passed |
| config 测试（`test_config.py` / `test_config_loader.py`） | ✅ 5 passed |

> **合计：Stage4 43 + Stage5-17 448 + Stage0-3 113 + config 5 = 609 passed**（未计入 14 个存量失败）。

---

# 5. 向后兼容验证

## 5.1 sliding_window 完全兼容（专项验证）

新增 `TestSlidingWindowBackwardCompat`（4 项）全部通过：

- **默认模式**：config 无 `mode` 键 → 30s scene → 6 个 8s clip（与改动前一致）。
- **显式 `sliding_window`**：输出与旧 `generate_clips()` 逐字段一致（clip_id / start_time / end_time / frame_count / duration）。
- **非法 mode**：`bogus_mode` → warning + 回退 sliding_window（不崩溃）。
- **disabled**：`enabled=false` → 空列表。

原 `generate_clips()` 签名与实现**零改动**，既有 26 项测试原样通过。

## 5.2 存量失败说明（非本次改动引入）

Stage5-17 回归中 14 个失败集中在 `test_stage_05_face_detection.py` / `test_stage_09_quality_fusion.py` / `test_stage_09_1_quality_ranking.py`：

- **与本次改动完全隔离**：这三个测试文件**不导入** `stage_04` 任何模块，也**不加载** `configs/config.yaml`（已逐一核对 import）。
- **经验证为改动前存量失败**：临时移除 config 的 `mode` 行（还原改动前 config）后重跑，**同样 14 failed / 98 passed**，失败集合完全一致。
- 失败根因（均属 Stage5/9/9.1 自身实现与测试预期不匹配，与 clip 生成无关）：
  - `Stage5 face_cropper`：返回 `(Path, bbox)` 元组 vs 测试期望 `Path`；crop 尺寸 150 vs 128；bbox 正则 / face_id 计数不符。
  - `Stage9 / 9.1`：`sync_failed` vs 期望 reason 元组不符；Stage9.1 输出为 list 而测试期望 dict（`TypeError: list indices ... not str`）等。

> 结论：本次 Stage4 双模式改动**未引入任何回归**；14 个失败为独立历史问题，建议另行跟踪。

---

# 6. 未执行事项（等待人工确认）

- [ ] **未运行** `scripts/stage0_17_e2e_validate.py` 全 pipeline（按指令等待人工确认后再执行）。
- [ ] 后续可选：`docs/02_Data_Model_and_Interface.md` §2.6.2 等文档中 8s≈200 → 16s≈400 语义同步（设计文档 §8 遗留项）。
- [ ] 后续可选：确认 16s 整段样本对训练数据量 / 过滤粒度的影响（设计文档 §7）。

---

# End of Stage4_Dynamic_Clip_Implementation_Report.md

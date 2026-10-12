# Stage17 Phase2 — Fusion Core 实现报告

> 阶段: Stage17 Phase2 (三模态融合核心 + 序列化 + 测试)
> 日期: 2026-08-27
> 范围: 时间同步融合装配、样本包读写、独立 runner、配置、单元测试、端到端冒烟
> 前置: Phase1 数据模型 (`TrainingSample` 重构) 已就绪

---

## 1. 修改 / 新增文件

| 文件 | 类型 | 说明 |
|---|---|---|
| `face_pipeline_v2/stages/stage_17_fusion/processor.py` | 重写 | Fusion Core: `build_training_sample()` 三模态时间同步装配 |
| `face_pipeline_v2/stages/stage_17_fusion/integration.py` | 重写 | `save_training_sample()` / `read_training_sample()` / manifest 读写 |
| `face_pipeline_v2/stages/stage_17_fusion/runner.py` | 重写 | 独立 CLI, 经 Stage12/13/14 官方 `read_*` 读入 → processor → integration |
| `face_pipeline_v2/stages/stage_17_fusion/stage.py` | 文档 | BaseStage 骨架 docstring 更新 (process() 仍占位, Phase3 接入 pipeline) |
| `face_pipeline_v2/data/models.py` | 修改 | `TrainingSample.valid_length` 改为 `Optional[int] = None` 哨兵: 显式 0 保留 (全缺失样本), None 自动填 T (兼容 Phase1) |
| `configs/config.yaml` | 修改 | stage17 块: `output_dir: outputs/stage17_fusion`, `fps: 25`, `audio_tolerance: 0.04` |
| `tests/test_stage17_fusion_processor.py` | 新增 | 19 个用例 (布局/对齐/音频匹配/valid_length/序列化/鲁棒性) |

---

## 2. Fusion 算法 (时间同步策略)

对应 `docs/Stage17_Design_Review.md §5`:

1. **主时间轴 = FLAME**: 以 `FLAMEFeature.frame_index` 为锚点, 按 frame_index 升序
   (锚点帧恒存在)。`timestamps` 取自 `FLAMEFeature.timestamp`。
2. **Gaze 精确 join**: `frame_index → GazeFeature` 哈希 join (同帧重复保留首个);
   缺失帧 **forward-fill 上一有效 Gaze**, 无历史则保持 zeros, 并置 `dropout_mask=True`。
3. **Audio nearest-timestamp**: 每个视频帧 timestamp 对 audio 特征做
   `argmin|Δt|`; `|Δt| ≤ audio_tolerance` (默认 0.04s) → 命中该特征;
   否则零向量填充 + `audio_match_idx=-1` + `dropout_mask=True`。
   **不做插值 / 不做特征平均** (插值会引入虚假中间特征)。
4. **valid_length** = 三模态齐全且 |Δt| 合格的帧数 = `count(~dropout_mask)`。
5. **visual_feature [T,112]** = `concat(expcode[100], posecode[6], gaze_vector[6])`,
   子块边界见 `VISUAL_LAYOUT = {expression:(0,100), pose:(100,106), gaze:(106,112)}`。

`audio_match_idx` 记录**原始 audio 输入列表索引** (非排序后索引), 保证可溯源。

---

## 3. 输入 / 输出契约

```
processor.build_training_sample(clip_id, flame_features, gaze_features,
                                audio_features, fps=25, visual_dim=112,
                                audio_dim=768, audio_tolerance=0.04,
                                audio_match="nearest_timestamp", metadata=None)
  → TrainingSample
```

| 字段 | shape / dtype | 说明 |
|---|---|---|
| `visual_feature` | `[T,112] float32` | expcode[100] + pose[6] + gaze[6] |
| `audio_feature` | `[T,768] float32` | Chinese HuBERT; 未命中为零向量 |
| `frame_index` | `[T] int32` | FLAME 锚点帧号 (升序) |
| `timestamps` | `[T] float32` | FLAME 帧时间 (秒) |
| `valid_length` | `int` | 有效帧数 (全缺失样本为 0) |
| `dropout_mask` | `[T] bool` | True = 补齐/缺失帧, 训练侧可丢弃 |
| `audio_match_idx` | `[T] int32` | 原始 audio 输入索引; -1 = 未命中 |
| `metadata` | `dict` | clip_id/sample_id/visual_dim/audio_dim/num_frames/valid_length/fps/audio_backend/audio_match/audio_tolerance/visual_layout/source_stage/n_input + 用户溯源字段 |

### 输出结构 (integration)

```
outputs/stage17_fusion/{clip_id}/
  training_sample.npz   # visual/audio/frame_index/timestamps/audio_match_idx/dropout_mask/valid_length
  metadata.json         # 溯源元数据 (visual_layout 等, JSON 原生 list 形式, round-trip 精确)
outputs/stage17_fusion/manifest.csv   # sample_id,clip_id,video_id,fps,num_frames,valid_length,audio_backend,clip_start,source_audio
```

round-trip 契约: `read_training_sample(save_training_sample(s))` 逐数组/字段**精确一致**。

---

## 4. 测试结果

```
python -m pytest tests/test_stage17_datamodel.py tests/test_stage17_fusion_processor.py tests/test_config_loader.py
→ 45 passed in 0.28s
```

| 测试集 | 用例数 | 覆盖点 |
|---|---|---|
| `test_stage17_datamodel.py` (Phase1 回归) | 21 | 模型 None 哨兵改造后全部通过 |
| `test_stage17_fusion_processor.py` (新增) | 19 | 布局切片 / Gaze forward-fill / Audio nearest (25FPS vs 50Hz, min-delta, 容差闭区间, 超时零填充) / valid_length 语义 / 输入乱序不敏感 / 空输入 / 参数校验 / metadata 契约 / 序列化 round-trip / manifest |
| `test_config_loader.py` (回归) | 5 | stage17 配置改动无破坏 |

关键用例示例:
- `test_nearest_timestamp_25fps_vs_50hz`: 25 FPS 视频帧 vs 50Hz audio, 帧 fi 精确命中
  `i=2·fi` 且 `audio_match_idx` 指向**乱序输入后的原始索引**;
- `test_boundary_tolerance_inclusive`: `|Δt| == tolerance` 判定为命中 (闭区间);
- `test_roundtrip_exact`: 含缺失帧 + 音频超时的样本, 保存/读回逐数组精确一致。

---

## 5. 端到端冒烟 (独立 runner)

构造合成 Stage12/13/14 NPZ (12 视频帧, 11 Gaze 缺 frame3, 24 个 50Hz audio) 后执行:

```
python -m face_pipeline_v2.stages.stage_17_fusion.runner \
    --clip_id smoke_clip --flame_dir s12 --gaze_dir s13 --audio_dir s14 \
    --output_dir out --fps 25
```

结果日志: `flame=12, gaze=11, audio=24 → T=12, valid_length=11, dropout=1, audio_matched=12`
(恰为缺帧的 frame3 被 forward-fill 并标记)。

产物:
```
out/manifest.csv
out/smoke_clip/training_sample.npz
out/smoke_clip/metadata.json
```

read-back 校验: `visual (12,112)`, `audio (12,768)`, `valid_length=11`,
`dropout=[F,F,F,T,F,...]`, `audio_match_idx=[0,2,4,...,22]`, metadata 含全部契约字段。

---

## 6. 已知限制

1. **未接入 pipeline runner**: `stage.py` 的 `process()` 仍为占位 (NotImplementedError);
   按配置调用 processor + integration 属 **Phase3** 范围, 独立 `runner.py` 已可 CLI 使用。
2. **eye_pose 未使用**: `FLAMEFeature.eye_pose` 固定 zeros 且与 gaze 语义不同,
   按设计文档 §4.1 不并入 visual_feature。
3. **无整体 offset 检查**: 设计文档 F5 的「全局音画偏移检测」未实现; 当前逐帧
   nearest + 容差已覆盖局部漂移, 全片系统性偏移 (所有 |Δt| > tolerance → 全部
   dropout) 由 `valid_length=0` + `dropout_mask` 全 True 暴露, 训练侧可据此拒收。
4. **timestamp 全 0 退化**: 若 Stage12 NPZ 无 timestamp (read 回退为 0.0),
   所有视频帧 ts=0 会对同一个 (首个) audio 帧命中, 产生退化对齐。需数据侧保证
   Stage12 产出有效 timestamp (Stage12 F2 修复后即满足)。
5. **`valid_length` 模型语义变更**: 由 `int=0`(自动填 T) 改为 `Optional[int]=None`
   (显式 0 保留)。此变更为向后兼容 (Phase1 测试全绿), 但类型从 `int` 变为
   `Optional[int]`, 依赖默认值 `0` 的调用方需改传 `None` 或显式值。

---

## 7. 后续工作 (Phase3+)

- 将 `stage.py::process()` 接入 pipeline runner, 按 config (stage17 段) 驱动;
- 数据源接入 `AVAlignmentInfo.clip_start` 作为全局 offset 校正 (F5);
- 训练侧消费: 依据 `dropout_mask` / `valid_length` 过滤样本, 按
  `metadata["visual_layout"]` 拆分 expression/pose/gaze 子块。


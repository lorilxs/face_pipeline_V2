# STAGE0_STAGE11_ASSET_AUDIT.md

> **Face_Pipeline_V2 — Stage0-Stage11 数据资产设计审计**
>
> 审计日期: 2026-08-11
> 审计范围: Stage 0 → Stage 11 完整数据流
> 目的: 确认 Stage11 之后是否已形成 Stage12-16 所需的高质量数据资产

---

## 1. 当前最终输出目录结构

```
outputs/
├── videos/                          # Stage 1: 标准化视频 + 音频 (25fps h264, 16kHz mono WAV)
│   └── <video_id>/
│       ├── <video_id>_std.mp4
│       └── <video_id>_audio.wav
│
├── scenes/                          # Stage 2: 场景分割 JSON (唯一在production代码中持久化的stage输出)
│   └── <video_id>_scenes.json
│
├── faces/                           # Stage 5: 人脸crop (256×256 JPG)
│   └── <video_id>/
│       └── <clip_id>/
│           └── crops/
│               └── frame_<idx>_face_<id>.jpg
│
├── aligned_audio/                   # Stage 7.1: 音画同步校正后音频
│   └── <clip_id>/
│       └── <clip_id>_aligned.wav    # ⚠️ 仅当 |offset| > 50ms 时写入
│
├── aligned_faces/                   # Stage 11: 256×256 对齐人脸
│   └── <clip_id>/
│       └── aligned_<frame_idx>_<face_id>.jpg
│
├── validation/                      # 测试/验证脚本输出 (非production pipeline)
│   ├── stage0_8/stage_outputs/       # Stage0-8 JSON序列化输出
│   ├── stage7_*/stage_outputs/       # Stage7专项测试
│   └── ...
│
├── stage10_visualization/           # Stage10 可视化验证输出
├── stage11_visualization/           # Stage11 可视化验证输出
│
├── clips/                           # ⚠️ 空目录 — 未生成clip级视频文件
│   ├── accepted/
│   └── candidates/
│
├── audio/                           # ⚠️ 空目录
├── features/                        # ⚠️ 空目录
├── flame/                           # ⚠️ 空目录
├── gaze/                            # ⚠️ 空目录
├── landmarks/                       # ⚠️ 空目录
├── metadata/                        # ⚠️ 空目录
├── statistics/                      # ⚠️ 空目录
├── alignment/                       # ⚠️ 空目录
└── logs/                            # ⚠️ 空目录
```

---

## 2. 已有资产 (Stage 0 → Stage 11 完整链路)

### 2.1 持久化文件资产 ✅

| # | 资产 | 来源 Stage | 文件格式 | 路径模式 | 状态 |
|---|---|---|---|---|---|
| 1 | **标准化视频** | Stage 1 | MP4 (h264, 25fps) | `videos/<id>/<id>_std.mp4` | ✅ |
| 2 | **标准化音频** | Stage 1 | WAV (16kHz mono) | `videos/<id>/<id>_audio.wav` | ✅ |
| 3 | **场景分割元数据** | Stage 2 | JSON | `scenes/<id>_scenes.json` | ✅ |
| 4 | **人脸Crop图像** | Stage 5 | JPG (256×256) | `faces/<video>/<clip>/crops/frame_*_face_*.jpg` | ✅ |
| 5 | **校正后音频** | Stage 7.1 | WAV (16kHz mono) | `aligned_audio/<clip>/<clip>_aligned.wav` | ⚠️ 条件写入 |
| 6 | **对齐人脸图像** | Stage 11 | JPG (256×256) | `aligned_faces/<clip>/aligned_*_*.jpg` | ✅ |

### 2.2 内存数据资产 (dataclass) ✅

| # | 数据结构 | 来源 Stage | 关键字段 | 每条粒度 |
|---|---|---|---|---|
| 1 | `VideoInfo` | Stage 0 | video_id, original_fps, width, height, duration, total_frames, codec, has_audio | per-video |
| 2 | `StandardizedVideoInfo` | Stage 1 | video_path, audio_path, target_fps(25), resolution, codec | per-video |
| 3 | `AudioInfo` | Stage 1 | audio_path, sample_rate(16000), duration, channels | per-video |
| 4 | `List[SceneInfo]` | Stage 2 | scene_id, video_id, start_frame, end_frame, start_time, end_time, coarse_face_info, filter_result | per-scene |
| 5 | `List[ClipInfo]` | Stage 4 | clip_id, scene_id, video_id, start_frame, end_frame, start_time, end_time, frame_count, duration | per-clip |
| 6 | `List[FaceSequence]` | Stage 5 | clip_id, video_id, frames: Tuple[FrameInfo], identity_result, stability_result | per-clip |
| 7 | `FrameInfo` (in FaceSequence) | Stage 5 | frame_index, clip_id, timestamp, image_path, faces: Tuple[FaceInfo] | per-frame |
| 8 | `FaceInfo` (in FrameInfo) | Stage 5 | frame_index, face_id, bbox(x1,y1,x2,y2), landmark_5(10 floats), confidence, face_crop_path, track_id | per-face |
| 9 | `List[PoseSequence]` | Stage 6 | clip_id, poses: Tuple[PoseInfo{yaw,pitch,roll,confidence}] | per-clip |
| 10 | `List[AVSyncInfo]` | Stage 7 | clip_id, offset(int frames), sync_lse_c, sync_lse_d, median_distance | per-clip |
| 11 | `List[AVAlignmentInfo]` | Stage 7.1 | clip_id, original_audio_path, corrected_audio_path, offset_frames, offset_ms, correction_applied, fps | per-clip |
| 12 | `List[FaceQualitySequence]` | Stage 8 | clip_id, qualities: Tuple[FaceQualityInfo{quality_score, confidence}] | per-clip |
| 13 | `List[QualityMetrics]` | Stage 9 | clip_id, video_id, pose_score, sync_score, quality_score, fusion_score, is_accepted, reject_reasons | per-clip |
| 14 | `List[QualityRankingInfo]` | Stage 9.1 | clip_id, pose_score, sync_score, face_quality_score, weighted_score, quality_level | per-clip |
| 15 | **`List[LandmarkInfo]`** | **Stage 10** | **clip_id, frame_index, face_id, timestamp, landmarks_106(212 floats), image_width, image_height** | **per-face** |
| 16 | **`List[AlignedFaceInfo]`** | **Stage 11** | **clip_id, frame_index, face_id, timestamp, aligned_face_path, affine_matrix(6 floats), original_image_size, aligned_resolution** | **per-face** |

---

## 3. 缺失资产 ⚠️

### 3.1 高优先级缺失

| # | 缺失项 | 影响 Stage | 严重程度 | 说明 |
|---|---|---|---|---|
| **1** | **Clip级视频文件** | Stage 12, 14, 17 | 🔴 **高** | `outputs/clips/` 为空。Stage5/7/11 都只处理crop图像。没有生成 per-clip 标准化视频片段 (如 `<clip_id>_std.mp4`)。Stage12 FLAME和Stage14 Audio都需要clip级视频帧序列。 |
| **2** | **Per-clip音频段** | Stage 14 | 🔴 **高** | Stage 7.1 的音频校正作用于完整视频的 WAV，输出的是 offset-adjusted clip 音频段。但 `AVAlignmentInfo.corrected_audio_path` 仅在 `correction_applied=True` 时有实际文件（\|offset\| > 50ms）。当偏移 ≤ 50ms 时，`corrected_audio_path` 指向不存在的文件。需要：无论是否校正，都生成 per-clip 音频段（或至少复制原始段）。 |
| **3** | **Frame index ↔ Timestamp 映射表** | Stage 12-17 全局 | 🔴 **高** | `FrameInfo` 虽有 `frame_index` + `timestamp`，但仅在内存中。没有持久化的 frame→timestamp 映射文件。Stage12-17 各阶段独立运行时需要此映射做时间对齐。 |
| **4** | **Stage 输出持久化** | Stage 12-17 全局 | 🟡 **中** | 除 Stage 2 外，production pipeline 代码不持久化任何 dataclass 输出。所有 `QualityMetrics`、`LandmarkInfo`、`AlignedFaceInfo` 等仅在内存中传递。Stage 12-17 重新运行时需要重新执行 Stage 0-11。仅测试脚本 (`e2e_stage0_11/run_e2e_test.py`) 有 JSON 序列化。 |

### 3.2 中优先级缺失

| # | 缺失项 | 影响 Stage | 严重程度 | 说明 |
|---|---|---|---|---|
| **5** | **LandmarkInfo 缺少 frame 尺寸** | Stage 11, 12 | 🟡 **中** | `LandmarkInfo.image_width` 和 `image_height` 被硬编码为 0 (Stage 10 processor line 109-110)。Stage 10 无法从 `FaceInfo` 获取原始frame尺寸。需要从 `VideoInfo.width/height` 或 `StandardizedVideoInfo.resolution` 传递。 |
| **6** | **缺少 per-frame 质量标记** | Stage 15, 16 | 🟡 **中** | `QualityMetrics` 是 clip 级别的 (is_accepted + fusion_score)。没有 per-frame 级别的质量标记。Stage 15 统计时需要知道每个 frame 是否被各过滤阶段通过/拒绝。当前只有 clip 级别。 |
| **7** | **缺少 stage 过滤链统计** | Stage 15 | 🟡 **中** | Stage 3 的 `SceneFilterResult` (reject_reasons) 和 Stage 9 的 `QualityMetrics.reject_reasons` 都在内存中，但没有汇总统计信息。Stage 15 需要：每个 stage 过滤了多少 clips/frames、拒绝原因分布。 |
| **8** | **缺少标准化视频的 frame 图像** | Stage 12-14 | 🟡 **中** | `FrameInfo.image_path` 仅在 `stage_05.save_frames=true` 时非空 (默认 false)。Stage 12 FLAME 和 Stage 13 Gaze 如果需要在原图上验证，会缺少原图路径。 |

### 3.3 低优先级/可后续补充

| # | 缺失项 | 影响 Stage | 严重程度 | 说明 |
|---|---|---|---|---|
| **9** | **Video-level metadata JSON** | Stage 16 | 🟢 **低** | 缺少 video_id、duration、resolution、fps 等的持久化 JSON。数据可从 `VideoInfo` 和 `StandardizedVideoInfo` 导出。 |
| **10** | **缺少 clip 之间的时间连续性信息** | Stage 17 | 🟢 **低** | Clip 来自滑动窗口 (8s window, 4s stride)，相邻 clip 有重叠。需要知道 clip-to-clip 的时间重叠关系，用于 Stage 17 融合时处理重复帧。可后续从 `ClipInfo.start_time/end_time` 推导。 |

---

## 4. 后续 Stage12-16 所需输入分析

### Stage 12: FLAME Parameter Extraction
| 需要的输入 | 当前状态 | 可用来源 |
|---|---|---|
| Aligned face images (256×256) | ✅ 已保存 | `AlignedFaceInfo.aligned_face_path` → `outputs/aligned_faces/` |
| 或: face crops + 106 landmarks | ✅ 有 | `FaceInfo.face_crop_path` + `LandmarkInfo.landmarks_106` |
| FLAME model weights | ⚠️ 空目录 | `models/EMICA/` — 需下载 |
| Config: expression_dim=50, pose_dim=6 | ✅ 已有 | `config.yaml` flame section |

### Stage 13: Gaze Parameter Extraction
| 需要的输入 | 当前状态 | 可用来源 |
|---|---|---|
| Aligned face images 或 face crops | ✅ 已保存 | `AlignedFaceInfo.aligned_face_path` 或 `FaceInfo.face_crop_path` |
| Eye region landmarks (来自106点) | ✅ 有 | `LandmarkInfo.landmarks_106` 索引 63-78 (眼区) |
| MediaPipe 或其他 gaze model | ⚠️ 未配置 | `config.yaml` gaze section 指向 MediaPipe，但 wrapper 为空 |

### Stage 14: Audio Feature Extraction (wav2vec2)
| 需要的输入 | 当前状态 | 可用来源 |
|---|---|---|
| Per-clip corrected audio WAV (16kHz) | ⚠️ **条件存在** | `AVAlignmentInfo.corrected_audio_path` — 仅 offset>50ms 时有文件 |
| Audio segment for EACH clip | ⚠️ **缺失** | 当 offset≤50ms 时无文件 |
| wav2vec2 model | ⚠️ 空目录 | `models/wav2vec2/` — 空，wrapper 为空 |

### Stage 15: Quality Statistics
| 需要的输入 | 当前状态 | 可用来源 |
|---|---|---|
| Per-clip QualityMetrics (accepted/rejected + reasons) | ✅ 有 (内存) | `List[QualityMetrics]` |
| Per-scene SceneFilterResult | ✅ 有 (内存) | `List[SceneInfo].filter_result` |
| Per-frame 质量信息 (HyperIQA scores) | ✅ 有 (内存) | `List[FaceQualitySequence]` |
| 统计汇总 | ❌ 需实现 | Stage 15 代码 stub 存在 |

### Stage 16: Data Annotation
| 需要的输入 | 当前状态 | 可用来源 |
|---|---|---|
| Clip 视频文件 | ⚠️ 缺失 | 需要生成 per-clip 视频 |
| Quality tiers (excellent/good/normal) | ✅ 有 (Stage 9.1) | `QualityRankingInfo.quality_level` |
| Audio (用于 Whisper 转录) | ⚠️ 条件存在 | 同 Stage 14 |
| Whisper / FairFace / CLIP models | ❌ 未集成 | config 无配置，wrapper 不存在 |

### Stage 17: Feature Alignment & Fusion → TrainingSample
| 需要的输入 | 当前状态 | 可用来源 |
|---|---|---|
| FLAME features (per-frame) | ❌ 待 Stage 12 | — |
| Gaze features (per-frame) | ❌ 待 Stage 13 | — |
| Audio features (per-frame) | ❌ 待 Stage 14 | — |
| Aligned face path | ✅ 有 | `AlignedFaceInfo.aligned_face_path` |
| Video clip path | ⚠️ 缺失 | 需要 Clip 视频文件 |
| frame_index 对齐 | ✅ 有 (内存) | 所有 dataclass 都有 frame_index |

---

## 5. 需要补充的数据字段

### 5.1 现有 DataClass 需要补充

#### LandmarkInfo (Stage 10 输出)
```python
# 当前:
image_width: int = 0    # ← 硬编码为 0
image_height: int = 0   # ← 硬编码为 0

# 建议: Stage 10 process() 中从 VideoInfo/StandardizedVideoInfo 传入
# 或从 FrameInfo 关联的原始 frame 读取尺寸
```

#### AlignedFaceInfo (Stage 11 输出) — 当前已完整 ✅
```python
# 已有: clip_id, frame_index, face_id, timestamp,
#       aligned_face_path, affine_matrix(6 floats),
#       original_image_size, aligned_resolution
# ✅ 无需补充
```

#### AVAlignmentInfo (Stage 7.1 输出)
```python
# 当前已完整，但 corrected_audio_path 的文件可用性不可靠
# 建议: 增加字段或在 Stage 7.1 中保证无论是否校正都复制音频段
```

### 5.2 需要新增的数据结构

#### ClipAsset (NEW — 建议新增)
```python
@dataclass(frozen=True)
class ClipAsset:
    """Clip级完整资产 — 聚合 Stage0-11 产出，作为 Stage12-16 统一输入."""
    clip_id: str
    video_id: str
    # 视频
    clip_video_path: Path          # clip 级标准化视频片段
    # 音频
    corrected_audio_path: Path     # per-clip 校正音频 (Stage 7.1)
    # 人脸
    face_sequence: FaceSequence    # face crops + bbox
    aligned_faces: Tuple[AlignedFaceInfo, ...]  # 256×256 aligned
    landmarks: Tuple[LandmarkInfo, ...]          # 106 landmarks
    # 质量
    quality_metrics: QualityMetrics              # fusion + acceptance
    quality_ranking: Optional[QualityRankingInfo] # tier
    # 时间
    start_time: float
    end_time: float
    duration: float
    fps: int  # 25
```

#### PerFrameAsset (NEW — 建议新增)
```python
@dataclass(frozen=True)
class PerFrameAsset:
    """单帧聚合资产 — 时间对齐的所有模态数据."""
    frame_index: int
    timestamp: float
    face_crop_path: Path
    aligned_face_path: Path
    landmarks_106: Tuple[float, ...]  # 212 floats
    affine_matrix: Tuple[float, ...]  # 6 floats
    pose: Tuple[float, float, float]  # yaw, pitch, roll
    quality_score: float
```

---

## 6. 建议的数据资产标准结构

```
outputs/
├── videos/                           # Stage 1: 标准化全视频
│   └── <video_id>/
│       ├── <video_id>_std.mp4        # 25fps h264
│       └── <video_id>_audio.wav      # 16kHz mono
│
├── clips/                            # 🆕 建议: per-clip 视频/音频资产
│   └── <clip_id>/
│       ├── <clip_id>_video.mp4       # 🆕 clip级标准化视频 (Stage 4 clip区间)
│       ├── <clip_id>_audio.wav       # 🆕 clip级校正音频 (Stage 7.1, 始终生成)
│       ├── crops/                    # Stage 5: face crops
│       │   └── frame_<idx>_face_<id>.jpg
│       ├── aligned/                  # Stage 11: aligned faces
│       │   └── aligned_<idx>_<id>.jpg
│       └── <clip_id>_assets.json     # 🆕 clip级资产清单
│
├── scenes/                           # Stage 2: 场景分割
│   └── <video_id>_scenes.json
│
├── metadata/                         # 🆕 建议: pipeline 元数据
│   └── <video_id>/
│       ├── video_info.json           # VideoInfo + StandardizedVideoInfo
│       ├── quality_metrics.json      # List[QualityMetrics]
│       ├── quality_ranking.json      # List[QualityRankingInfo]
│       └── stage_stats.json          # 🆕 各stage过滤统计
│
├── features/                         # Stage 12-14: 特征输出 (待实现)
│   └── <clip_id>/
│       ├── flame/                    # Stage 12: FLAME params
│       │   └── frame_<idx>_flame.json
│       ├── gaze/                     # Stage 13: gaze vectors
│       │   └── frame_<idx>_gaze.json
│       └── audio/                    # Stage 14: audio embeddings
│           └── <clip_id>_audio.npy   # (768, T) wav2vec2 features
│
├── statistics/                       # Stage 15: 统计报告
│   └── <video_id>_quality_report.json
│
├── annotations/                      # Stage 16: 标注
│   └── <video_id>_annotations.json
│
└── training_samples/                 # Stage 17: 最终训练样本
    └── <video_id>/
        └── sample_<id>.pt            # 或 .npz / .json
```

---

## 7. 关键发现与建议

### 7.1 🔴 必须修复 (阻塞 Stage 12+)

1. **Stage 7.1 音频文件可用性**: `AVAlignmentInfo.corrected_audio_path` 仅在 `correction_applied=True` 时指向实际文件。当 offset ≤ 50ms 时，文件不存在但路径非空。需要修改 Stage 7.1 使其**始终生成 per-clip 音频文件**（未校正时复制原始音频段）。

2. **缺少 Clip 级视频文件**: `outputs/clips/` 为空。Stage 12 (FLAME) 和 Stage 17 (TrainingSample) 需要 clip 视频。需要新增一个 clip 提取步骤 (Stage 4.1 或在 Stage 11 后)，从标准化视频中裁剪 per-clip 视频段。

### 7.2 🟡 建议修复 (影响数据完整性)

3. **LandmarkInfo 缺少 frame 尺寸**: `image_width/height` 硬编码为 0。需要从 pipeline context 或 VideoInfo 传入。

4. **缺少 per-clip 资产清单 JSON**: 每个 clip 需要一份 JSON 清单，汇总该 clip 的所有资产路径（crop、aligned face、audio、landmark count、quality metrics）。便于 Stage 12-16 独立消费。

5. **Stage 输出持久化**: 建议在 `pipeline_runner.py` 中新增 JSON 序列化，使 `QualityMetrics`、`LandmarkInfo`、`AlignedFaceInfo` 等关键产出可以持久化到磁盘，避免下游阶段重复计算。

### 7.3 🟢 后续优化

6. **PerFrameAsset 聚合**: 建议新增 `PerFrameAsset` dataclass，将同一 frame 的所有模态数据 (face_crop, aligned_face, landmarks, affine, pose, quality) 对齐到一个结构中，简化 Stage 17 融合。

7. **Frame index ↔ Timestamp 查找表**: 建议在 pipeline 早期生成 `frame_index → timestamp` 的 JSON 映射表。

---

## 8. 数据完整性检查清单

| 检查项 | 状态 | 备注 |
|---|---|---|
| Stage11 是否保存 aligned face images | ✅ 是 | `outputs/aligned_faces/<clip_id>/aligned_*_*.jpg`, 256×256 |
| 是否保存 affine_matrix | ✅ 是 | `AlignedFaceInfo.affine_matrix`: 6 floats (2×3 matrix) |
| 是否保存 original_image_size | ✅ 是 | `AlignedFaceInfo.original_image_size`: (height, width) |
| 是否保存 frame_index ↔ timestamp 映射 | ⚠️ 部分 | `FrameInfo` 有映射但仅在内存；`LandmarkInfo`/`AlignedFaceInfo` 有 timestamp |
| Stage7.1 audio 是否作为最终 audio 输出 | ⚠️ 条件 | 仅 offset>50ms 时有物理文件 |
| Stage9 quality metrics 是否保留 | ✅ 内存 | `QualityMetrics` 完整；未持久化到磁盘 |
| Stage 过滤信息是否可供 Stage15 统计 | ✅ 内存 | `SceneFilterResult` + `QualityMetrics.reject_reasons` 可用 |

---

## 9. 总结

**Stage 0-11 已经产生了高质量的人脸数据资产核心**:
- ✅ 标准化视频 + 音频 (25fps / 16kHz)
- ✅ Per-frame 人脸检测 + 256×256 crops
- ✅ 106-point 高精度 landmarks (frame坐标系)
- ✅ 256×256 对齐人脸 + affine matrix (可逆恢复)
- ✅ Per-clip 质量评分 + 融合过滤 + 排序
- ✅ 音画同步检测 + 音频校正

**Stage 12-16 需要在以下方面补充**:
1. **音频**: 确保每个 clip 都有可用的校正音频文件（无论 offset 大小）
2. **视频**: 生成 per-clip 标准化视频片段
3. **元数据持久化**: 将内存中的 QualityMetrics、LandmarkInfo、AlignedFaceInfo 持久化
4. **Frame 对齐**: 确保所有模态可以通过 frame_index 对齐

**整体评价**: 数据资产质量良好，核心人脸数据链路完整。需要补充 clip 级视频/音频文件生成和元数据持久化后，即可支持 Stage 12-17 的独立开发和运行。

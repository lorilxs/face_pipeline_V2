# 05_Data_Flow_and_Output.md

# AI-Friendly Software Design Document (AISDD)

# Part 5

# Data Flow and Output

Version: v1.0

Last Update: 2026-08-03

---

# 5.1 Data Flow and Output Overview

`Face_Pipeline_V2` 采用：

Data-Centric Pipeline Architecture。

---

本章节定义：

- Pipeline内部数据流转关系
- Stage之间输入输出关系
- 中间数据存储规范
- 最终训练数据组织方式
- 多模态数据对齐规则
- Pipeline恢复机制

---

设计目标：

- 明确每个Stage的数据输入和输出
- 保证Video、Face、Audio、FLAME、Gaze严格对应
- 保证最终训练数据来源明确
- 保证所有数据可追踪
- 支持大规模数据生产
- 支持Pipeline断点恢复

---

# 5.2 Data Flow Architecture

`Face_Pipeline_V2`

整体数据流：

| Stage | Input | Output | Description |
|---|---|---|---|
| Stage 0 | Raw Video | VideoInfo | 视频完整性检查 |
| Stage 1 | VideoInfo | StandardizedVideoInfo + AudioInfo | 视频音频标准化 |
| Stage 2 | StandardizedVideoInfo | SceneInfo | 场景切分 |
| Stage 3 | SceneInfo | SceneInfo (with CoarseFaceInfo + SceneFilterResult) | 人脸初筛 |
| Stage 4 | SceneInfo | ClipInfo | 生成候选Clip（sliding_window 时间窗口切片 / segment 整段1:1适配） |
| Stage 5 | CandidateClipInfo | FaceInfo | 人脸定位与裁剪 |
| Stage 5.1 | FaceInfo | TrackingInfo | 人脸跟踪（附加 track_id，不丢弃） |
| Stage 5.2 | TrackingInfo | Filtered FaceSequence List | 单人脸身份硬过滤（拒绝的 sequence 被丢弃） |
| Stage 5.3 | post-5.2 FaceSequence List | Stabilized FaceSequence List | 时序稳定性硬过滤（拒绝的 sequence 被丢弃） |
| Stage 6 | FaceInfo | PoseInfo | 头姿检测 |
| Stage 7 | FaceSequence + AudioInfo + ClipInfo | AVSyncInfo | 音画同步检测（基于face crop） |
| Stage 7.1 | AVSyncInfo + StandardizedVideoInfo | AVAlignmentInfo | 音画同步校正（offset-based audio alignment） |
| Stage 8 | FaceCrop | FaceQualityInfo | 人脸质量检测 |
| Stage 9 | Quality Metrics | AcceptedClipInfo | 质量融合与硬过滤 |
| Stage 9.1 | Quality Metrics | QualityScoreInfo | 质量评分排序 |
| Stage 10 | AcceptedClip FaceCrop | LandmarkInfo | 高精度关键点 |
| Stage 11 | LandmarkInfo | AlignmentInfo | 人脸对齐 |
| Stage 11.5 | AlignmentInfo | FaceValidityInfo | 人脸有效性检查 |
| Stage 12 | AlignmentInfo | FLAMEFeature | FLAME Motion参数（EMICA ExpressionEncoder 100维 expression） |
| Stage 13 | AlignmentInfo | GazeFeature | 视线参数 |
| Stage 14 | AVAlignmentInfo (corrected audio) | AudioFeature | 音频特征（基于Stage 7.1校正音频） |
| Stage 15 | Pipeline Metadata | StatisticsInfo | 质量统计 |
| Stage 16 | Metadata + Models | AnnotationInfo | 数据标注 |
| Stage 17 | All Features | TrainingSample | 最终训练数据 |

---

# 5.3 Data Flow Design Principle

# Principle 1

所有Stage：

必须通过：

Data Object

进行数据传递。

---

禁止：

Stage之间直接访问其他Stage内部变量。

---

# Principle 2

所有Data Object：

采用不可变设计。

---

禁止：

后续Stage修改前序Stage输出。

---

# Principle 3

所有关键中间结果：

必须支持持久化保存。

---

目的：

- Debug
- Resume
- Dataset Analysis

---

# 5.4 Stage Input / Output Mapping


# 5.4.1 Video Processing Data Flow


| Stage | Input | Output |
|---|---|---|
| Stage 0 | Raw Video | VideoInfo |
| Stage 1 | VideoInfo | StandardizedVideoInfo + AudioInfo |
| Stage 2 | StandardizedVideoInfo | SceneInfo |
| Stage 3 | SceneInfo | SceneInfo (with CoarseFaceInfo + SceneFilterResult) |
| Stage 4 | SceneInfo (filtered) | ClipInfo |
| Stage 9 | Quality Metrics | AcceptedClipInfo |

---

## Stage 3 过滤规则


Stage 3 对 Stage 2 输出的 SceneInfo 进行多级过滤。


过滤顺序：


1.

Scene Duration Validation


2.

Face Detection Sampling


3.

Face Count Validation


4.

Confidence Validation


---


Filtering Criteria:


| Rule | 中文名称 | Description |
|---|---|---|
| Duration Check | 时长检查 | 删除无法生成有效Clip的短Scene |
| Face Presence Check | 人脸存在检查 | 删除无人脸Scene |
| Face Count Check | 人脸数量检查 | 删除多人脸Scene |
| Confidence Check | 置信度检查 | 删除低质量检测结果 |


---

说明：

Stage 4输出：

仅表示：

Candidate Clip。

---

Stage 9输出：

表示：

最终用于后续Feature Extraction的Accepted High Quality Clip。

---

# 5.4.2 Face Processing Data Flow


| Stage | Input | Output |
|---|---|---|
| Stage 5 | CandidateClipInfo | FaceInfo |
| Stage 5.1 | FaceInfo | TrackingInfo |
| Stage 5.2 | TrackingInfo | Filtered FaceSequence List (rejected dropped) |
| Stage 5.3 | post-5.2 FaceSequence List | Stabilized FaceSequence List (rejected dropped) |
| Stage 10 | AcceptedClip Face | LandmarkInfo |
| Stage 11 | LandmarkInfo | AlignmentInfo |
| Stage 11.5 | AlignmentInfo | FaceValidityInfo |

---

# 5.4.3 Quality Filtering Data Flow


| Stage | Input | Output |
|---|---|---|
| Stage 6 | FaceInfo | PoseInfo |
| Stage 7 | FaceSequence + AudioInfo + ClipInfo | AVSyncInfo |
| Stage 7.1 | AVSyncInfo + StandardizedVideoInfo | AVAlignmentInfo |
| Stage 8 | FaceCrop | FaceQualityInfo |
| Stage 9 | Pose + Sync + Quality Metrics | AcceptedClipInfo |
| Stage 9.1 | Quality Metrics | QualityScoreInfo |

---

# 5.4.4 Feature Extraction Data Flow


| Stage | Input | Output |
|---|---|---|
| Stage 12 | AlignmentInfo | FLAMEFeature（EMICA） |
| Stage 13 | AlignmentInfo | GazeFeature |
| Stage 14 | CorrectedAudioInfo | AudioFeature |
| Stage 17 | All Features | TrainingSample |

---

# 5.5 Intermediate Data Storage


# 5.5.1 Storage Principle

`Face_Pipeline_V2`

所有重要Stage输出：

支持保存。

---

保存目的：

- Debug
- Quality Analysis
- Pipeline Resume
- Dataset Audit

---

# 5.5.2 Intermediate Data Object Storage


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| stage_name | Stage名称 | str | 当前数据产生的Stage |
| output_path | 输出路径 | Path | 数据保存路径 |
| version | 数据版本 | str | 数据格式版本 |
| timestamp | 时间戳 | str | 数据生成时间 |
| config_hash | 配置Hash | str | 当前配置版本标识 |

---

# 5.6 Output Directory Structure


项目输出目录：

`outputs/`


| Directory | 中文名称 | Description |
|---|---|---|
| videos/ | 视频数据 | 原始和标准化视频 |
| audio/ | 音频数据 | 原始和修正Audio |
| scenes/ | 场景数据 | Scene结果 |
| clips/candidates/ | 候选Clip | Stage 4生成的候选Clip |
| clips/accepted/ | 高质量Clip | Stage 9通过过滤后的Clip |
| faces/ | 人脸数据 | Face Crop |
| landmarks/ | 关键点数据 | Landmark结果 |
| alignment/ | 对齐数据 | 标准化Face |
| flame/ | FLAME数据 | 3D Face参数 |
| gaze/ | Gaze数据 | 视线参数 |
| features/ | 特征数据 | Audio Feature |
| metadata/ | 元数据 | Pipeline信息 |
| statistics/ | 统计数据 | 质量统计 |
| logs/ | 日志 | 运行日志 |

---

# 5.7 Metadata Design


Metadata用于：

记录整个Pipeline处理过程。

---

# 5.7.1 Video Metadata


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| video_id | 视频ID | str | 视频唯一标识 |
| source_path | 来源路径 | Path | 原始视频路径 |
| duration | 视频时长 | float | 视频持续时间 |
| fps | 视频帧率 | float | 视频FPS |
| status | 状态 | str | 当前Pipeline状态 |

---

# 5.7.2 Candidate Clip Metadata


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| clip_id | Clip ID | str | Clip唯一编号 |
| video_id | 视频ID | str | 来源视频 |
| scene_id | Scene ID | str | 来源Scene |
| start_frame | 起始帧 | int | Clip开始位置 |
| end_frame | 结束帧 | int | Clip结束位置 |
| status | 状态 | str | Candidate状态 |

---

# 5.7.3 Accepted Clip Metadata


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| accepted_clip_id | 高质量Clip ID | str | Stage 9输出编号 |
| source_clip_id | 来源Clip ID | str | Stage 4候选Clip |
| quality_status | 质量状态 | str | 是否通过Hard Filter |
| quality_metrics | 质量指标 | dict | Pose/Quality/Sync结果 |
| score | 质量分数 | float | Optional质量评分 |

---

# 5.7.4 Training Sample Metadata


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| sample_id | 样本ID | str | 最终训练样本编号 |
| accepted_clip_id | 高质量Clip ID | str | 来源Stage 9 |
| face_path | 人脸路径 | Path | Face序列路径 |
| flame_path | FLAME路径 | Path | FLAME参数路径 |
| gaze_path | Gaze路径 | Path | Gaze参数路径 |
| audio_feature_path | 音频特征路径 | Path | Audio Feature路径 |
| config_version | 配置版本 | str | 使用配置版本 |

---

# 5.8 Multi-modal Alignment Rules


# 5.8.1 Alignment Principle

最终训练数据：

必须保证：

Video、Face、FLAME、Gaze、Audio

共享同一时间轴。

---

# 5.8.2 Frame Level Alignment


| Data | Alignment Key | Description |
|---|---|---|
| Face Sequence | frame_index | 每帧对应人脸 |
| FLAME | frame_index | 每帧对应3D参数 |
| Gaze | frame_index | 每帧对应视线参数 |

---

# 5.8.3 Audio Alignment


Audio 必须来自：

Stage 7.1 校正后的 Audio：

`AVAlignmentInfo.corrected_audio_path`

---

流程：

Stage 7 (SyncNet Detection)

↓

Stage 7.1 (Audio Alignment Correction)

↓

Corrected Audio

↓

Stage 14 (Audio Feature Extraction)

↓

Audio Feature


---

禁止：

Raw Audio

直接进入：

Stage 14。

---

# 5.8.4 Alignment Validation


Stage 17：

必须检查：

| Validation | Description |
|---|---|
| Frame Count Check | Face/FLAME/Gaze数量一致 |
| Timestamp Check | Audio时间一致 |
| Missing Feature Check | 特征缺失检查 |
| Confidence Check | 数据质量检查 |

---

# 5.9 Final Training Dataset Format


最终训练数据：

来源：

Stage 9之后的Accepted High Quality Clip。

---

| Component | Source | Description |
|---|---|---|
| Video Clip | Stage 9 | 经过质量认证后的高质量视频片段 |
| Face Sequence | Stage 11 | 基于Accepted Clip生成的标准化人脸序列 |
| FLAME Motion Feature | Stage 12 | 基于 Stage 11 提取的 3D 人脸 motion 参数（EMICA ExpressionEncoder 100 维） |
| Gaze Feature | Stage 13 | 基于Stage 11提取的视线参数 |
| Audio Feature | Stage 14 | 基于Stage 7修正Audio提取的Audio Embedding |
| Metadata | Stage 15/16 | 数据统计和标签信息 |

---

# 5.10 Final Training Sample Structure


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| sample_id | 样本ID | str | 唯一样本编号 |
| video_clip | 高质量视频Clip | Path | Stage 9输出视频 |
| face_sequence | 人脸序列 | List[Path] | Stage 11输出Face |
| flame | FLAME参数 | ndarray | 3D Face参数 |
| gaze | Gaze参数 | ndarray | 视线参数 |
| audio_feature | 音频特征 | ndarray | Audio Embedding |
| metadata | 元数据 | dict | 样本描述 |

---

# 5.11 Pipeline Resume Strategy


# 5.11.1 Resume Principle


Pipeline支持：

Stage级恢复。


---

例如：

如果：

Stage 9已经完成。

---

重新运行Pipeline：

无需重新执行：

Stage 0-8。

---

# 5.11.2 Cache Validation


每个Stage输出：

必须记录：


| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| stage_name | Stage名称 | str | 当前Stage |
| version | 版本 | str | Stage版本 |
| config_hash | 配置Hash | str | 配置唯一标识 |
| input_hash | 输入Hash | str | 输入数据标识 |
| timestamp | 时间戳 | str | 创建时间 |

---

# 5.11.3 Cache Invalid Condition


以下情况：

必须重新执行Stage：

- Config变化
- Model变化
- Input变化
- Stage版本变化

---

# 5.12 Data Traceability


最终训练样本：

必须支持反向追踪。


关系：

TrainingSample

↓

AcceptedClipInfo

↓

CandidateClipInfo

↓

SceneInfo

↓

VideoInfo

↓

Raw Video


---

# 5.13 Data Flow Summary


`Face_Pipeline_V2`

核心设计原则：

- Stage 4 生成 Candidate Clip（sliding_window 时间窗口切片 / segment 整段 1:1 适配）
- Stage 5.2 / Stage 5.3 是硬过滤：未通过的 sequence 被丢弃，不进入 Stage 6-9
- Stage 9生成最终Accepted High Quality Clip
- 所有Feature Extraction基于Stage 9通过的数据
- Audio Feature必须来自Stage 7修正后的Audio
- Video与多模态Feature严格同步
- Stage12 跨环境 subprocess 调用（host Python → Docker emica_env）
- 中间结果支持缓存
- 支持大规模训练数据生产

---

# 5.14 Stage12 Data Flow

## 5.14.1 Stage11 → Stage12

```
Stage 10 (InsightFace landmark)
    ↓
Stage 11 (Affine alignment)  →  List[AlignedFaceInfo]
    ↓
Stage 12 (FLAME extraction)  →  List[FLAMEFeature]
```

Stage11 输出 `List[AlignedFaceInfo]` per clip，每个 entry 包含：
- `clip_id`, `frame_index`, `face_id`, `timestamp`
- `aligned_face_path` (256×256 对齐人脸)
- `affine_matrix` (2×3), `original_image_size`, `aligned_resolution`

## 5.14.2 Stage12 Output Format

### NPZ 结构 (`flame_features.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `expression_code` | `[N, 100]` | EMICA ExpressionEncoder 输出 |
| `jaw_pose` | `[N, 3]` | 融合后 jaw pose |
| `head_pose` | `[N, 3]` | EMICA global pose |
| `eye_pose` | `[N, 6]` | zeros(6) placeholder |
| `shapecode` | `[N, 300]` | zeros(300) (MICA 未加载) |
| `frame_index` | `[N]` | 帧编号 |
| `timestamp` | `[N]` | 时间戳(秒) |
| `clip_id` | `[N]` | Clip 标识 |

### Per-Clip Output Directory

```
outputs/stage12_flame/{clip_id}/
├── flame_features.npz
├── flame_features.npy
└── metadata.json
```

## 5.14.3 Stage12 → Stage17

```
FLAMEFeature (Stage12) ────┐
GazeFeature (Stage13) ─────┤
AudioFeature (Stage14) ────┼──→ Stage17 → TrainingSample
Metadata (Stage15/16) ─────┘
```

TrainingSample 中 `flame_feature = expcode[100] + posecode[6] = 106 维`。flatten 操作由 Stage17 完成，Stage12 保持结构化输出。

---

# End of 05_Data_Flow_and_Output.md v1.0
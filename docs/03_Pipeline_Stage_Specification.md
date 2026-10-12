# 03_Pipeline_Stage_Specification.md

# AI-Friendly Software Design Document (AISDD)

# Part 3

# Pipeline Stage Specification

Version: v1.0

Last Update: 2026-08-03

---

# 3.1 Pipeline Stage Overview

`Face_Pipeline_V2` 采用：

Stage-based Pipeline Architecture。

---

整个 Pipeline：

由多个独立 Stage 组成。

每个 Stage：

负责单一功能。

---

设计原则：

- Single Responsibility
- Clear Input / Output
- Config Driven
- Model Abstraction
- Independent Execution
- Reproducible Processing

---

# 3.2 Pipeline Execution Overview


完整执行流程：

`Stage 0`

↓

`Stage 1`

↓

`Stage 2`

↓

`Stage 3`

↓

`Stage 4`

↓

`Stage 5`

↓

根据数据类型选择：

---

High Constraint Talking Face:

`Stage 6`

↓

`Stage 7`

↓

`Stage 7.1`

↓

`Stage 8`

↓

`Stage 9`

---

Open Domain Video:

`Stage 5.1`

↓

`Stage 5.2`

↓

`Stage 5.3`

↓

`Stage 6`

↓

`Stage 7`

↓

`Stage 7.1`

↓

`Stage 8`

↓

`Stage 9`

---

Optional:

`Stage 9.1`

---

之后：

`Stage 10`

↓

`Stage 11`

↓

`Stage 11.5`

↓

`Stage 12-16`

↓

`Stage 17`

---

# 3.3 Stage 0 Video Integrity Check


# 3.3.1 Purpose

检查输入视频：

- 是否损坏
- 是否可以解码
- 是否包含Video Stream
- 是否包含Audio Stream


---

# 3.3.2 Model / Method

Model:

`FFprobe`

---

# 3.3.3 Input


Input:

`Raw Video`

---

Data:

| Field Name | 中文名称 | Type | Description |
|---|---|---|---|
| video_path | 视频路径 | Path | 输入视频文件 |

---

# 3.3.4 Processing Logic


执行：

- 文件存在检查
- Codec检查
- Stream检查
- Frame读取测试


---

# 3.3.5 Output


Output:

`VideoInfo`

---

# 3.3.6 Configuration


| Parameter | Default | Description |
|---|---|---|
| check_video_stream | true | 检查Video Stream |
| check_audio_stream | true | 检查Audio Stream |

---

# 3.4 Stage 1 AV Standardization


# 3.4.1 Purpose


统一：

视频和音频格式。


---

目标：

Video:

- 25 FPS
- Standard Codec


Audio:

- 16kHz
- WAV


---

# 3.4.2 Model / Method


Model:

`FFmpeg`

---

# 3.4.3 Input


Input:

`VideoInfo`

---

# 3.4.4 Processing Logic


执行：

- FPS转换
- Audio Extraction
- Audio Resampling
- Codec转换


---

# 3.4.5 Output


Output:

`StandardizedVideoInfo`

+

`AudioInfo`

---

# 3.4.6 Configuration


| Parameter | Default | Description |
|---|---|---|
| target_fps | 25 | 标准视频FPS |
| audio_sample_rate | 16000 | Audio采样率 |
| audio_format | wav | 音频格式 |

---

# 3.5 Stage 2 Scene Segmentation


# 3.5.1 Purpose


检测：

视频镜头切换。

生成：

连续Scene。


---

# 3.5.2 Model / Method


Model:

`PySceneDetect`

or

`TransNetV2`

---

# 3.5.3 Input


Input:

`StandardizedVideoInfo`

---

# 3.5.4 Processing Logic


分析：

- Frame变化
- Scene Boundary


---

# 3.5.5 Output


Output:

`SceneInfo`

---

# 3.6 Stage 3 Coarse Face Screening


# 3.6.1 Purpose

在进入后续高计算量人脸处理模块之前：

对 Stage 2 输出的 Scene 进行快速有效性筛选。

本阶段主要过滤：

- 过短 Scene
- 无人脸 Scene
- 多人脸 Scene
- 低置信度人脸 Scene

通过低成本检测：

降低后续：

- Face Tracking
- Pose Estimation
- Face Alignment
- Feature Extraction

阶段的计算成本。


---

# 3.6.2 Model


Model:

`SCRFD`

Sampling:

`1 FPS`

---

# 3.6.3 Input

Input:

`List[SceneInfo]`

来自 Stage 2 Scene Segmentation 的所有Scene。

---

# 3.6.4 Processing Logic

## Step 1: 读取Stage 2输出

读取 Stage 2 输出：

`List[SceneInfo]`

---

## Step 2: Scene Duration Validation

检查：

Scene duration

---

过滤规则：

如果：

`scene_duration < minimum_scene_duration`

则：

标记为 `reject`，原因：

`too_short`

---

该步骤优先执行：

避免对无效Scene进行SCRFD推理。

---

## Step 3: Frame Sampling

对于保留Scene：

按照：

`1 FPS`

进行Frame Sampling。

---

## Step 4: SCRFD Face Detection

对采样Frame：

执行：

SCRFD inference。

---

## Step 5: Calculate Per-Scene Metadata

基于SCRFD检测结果：

构建：

`CoarseFaceInfo`：

| Field | Calculation |
|---|---|
| sampled_frame_count | 实际采样帧数 |
| face_detect_ratio | 检测到人脸帧数 / 采样帧数 |
| single_face_ratio | 单人脸帧数 / 检测到人脸帧数 |
| avg_face_confidence | 所有人脸检测平均置信度 |
| max_face_count | 单帧最大人脸数 |

---

构建：

`SceneFilterResult`：

| Field | Calculation |
|---|---|
| passed | 综合判断是否通过 |
| reject_reasons | 拒绝原因列表(支持多个) |

---

reject_reasons枚举：

| Value | Condition |
|---|---|
| too_short | scene_duration < minimum_scene_duration |
| no_face | face_detect_ratio == 0 |
| multi_face | single_face_ratio < threshold |
| low_confidence | avg_face_confidence < threshold |

---

## Step 6: Scene Filtering

根据SceneFilterResult：

过滤Scene。

---

通过Scene：

附加 CoarseFaceInfo + SceneFilterResult。

---

被拒绝Scene：

仍然输出（附加 metadata），由 Stage 15 统计。

---

注意：

Stage 3 不计算：

- 删除Scene数量
- 删除总时长
- 保留比例

不生成：

- Stage3Report
- stage3_statistics.json

---

Stage 15 Quality Statistics：

统一负责所有聚合统计。

---

# 3.6.5 Output

Output:

`List[SceneInfo]`

---

每个SceneInfo携带：

| Field | Type | Description |
|---|---|---|
| coarse_face_info | Optional[CoarseFaceInfo] | SCRFD粗筛检测结果 |
| filter_result | Optional[SceneFilterResult] | 过滤决策(含reject_reasons) |

---

下游Stage职责：

| Stage | Action |
|---|---|
| Stage 4 | 只处理 passed=True 的SceneInfo |
| Stage 15 | 收集所有SceneFilterResult → 聚合统计报告 |

---

# 3.7 Stage 4 Clip Generation / Clip Adapter


# 3.7.1 Purpose


将Scene（Stage3 single-face segment）：

生成 / 适配为Clip。

Stage4 负责：

- Clip 生成（Clip Generation）
- clip_id 分配（Clip ID Assignment）
- Timestamp 保持（Timestamp Preservation）
- Downstream Contract 建立

不再描述为「固定 8 秒切片」。


---

# 3.7.2 Modes


Stage4 支持两种模式：

| Mode | 说明 | 用途 |
|---|---|---|
| `sliding_window` | 固定窗口滑动切片（默认，保持兼容） | 传统训练样本生成 |
| `segment` | 整段 1:1 适配，每个 SceneInfo 生成 1 个完整 ClipInfo（动态时长） | 长 segment（如 16s）→ 单 clip → T≈400 样本 |


---

# 3.7.3 Method (sliding_window)


Method:

`Sliding Window`

---

Parameters:

| Parameter | Default |
|---|---|
| Window Length | 8s |
| Step | 4s |


---

# 3.7.4 Method (segment)


输入：

Stage3 single-face segment（例如 16s）。

输出：

一个完整 clip（1:1 适配）：

| 输入 | 输出 |
|---|---|
| scene_0001, start=0, end=16s | clip_0001, start=0, end=16s |


---

# 3.7.5 Input


Input:

`Candidate Scene`（passed=True 的 SceneInfo）

---

# 3.7.6 Output


Output:

`ClipInfo`

ClipInfo 支持动态 duration（见 `docs/02_Data_Model_and_Interface.md` §2.6.2 与 `docs/Stage4_Dynamic_Clip_Design_Update.md`）。

---

# 3.8 Stage 5 Face Localization & Crop


# 3.8.1 Purpose


逐帧检测：

人脸位置。

并进行：

Face Crop。


---

# 3.8.2 Model


Model:

`SCRFD`

---

# 3.8.3 Input


Input:

`ClipInfo`

---

# 3.8.4 Processing Logic


每帧执行：

- Face Detection
- BBox Extraction
- 5 Point Landmark
- Face Crop


---

# 3.8.5 Output


Output:

`FrameInfo`

+

`FaceInfo`

---

# 3.9 Stage 5 Enhancement Module (Early Hard-Filter Gates)


# 3.9.0 Gating Principle

Stage 5.1/5.2/5.3 是开放域视频在进入昂贵模型推理（Stage 6 6DRepNet / Stage 7 SyncNet / Stage 8 HyperIQA）之前的：

Early Rejection Gates（早期硬过滤门）。

---

原则：Progressive Gating

- 5.1（ByteTrack）只附加 track_id，不丢弃 sequence。
- 5.2（规则，无模型）评估身份一致性；未通过 → 丢弃，不进入 5.3 / 6 / 7 / 8。
- 5.3（规则，无模型）评估时序稳定性；未通过 → 丢弃，不进入 6 / 7 / 8。
- 只有全部通过的 sequence 才进入 Stage 6 - Stage 9。

这样避免无效 clip 消耗 GPU 推理资源（6DRepNet / SyncNet / HyperIQA）。


# 3.9.1 Stage 5.1 Face Tracking


## Purpose


建立：

人物连续轨迹。


---

Model:

`ByteTrack`

---

Input:

`FaceInfo`

---

Output:

`TrackingInfo`

---

# 3.9.2 Stage 5.2 Single Face Identity Hard Filter


## Purpose


筛选：

真正单人Clip（Hard Filter，非 annotation）。


评估：

- 身份一致性（同一 main_track_id）
- 多人脸 / 空帧 / 身份切换比例


---

## Behavior

对每个 FaceSequence：

1. 评估身份一致性（`filter_sequence`）。
2. 生成 `identity_result`（含 `passed` / `reject_reasons`）。
3. `passed=True` → 附加 `identity_result`，继续传递。
4. `passed=False` → 记录 reject reason，丢弃该 sequence。


---

## Output

`List[FaceSequence]`（仅包含通过的 sequence）


---

## Gating

被丢弃的 sequence：

- 不进入 Stage 5.3
- 不进入 Stage 6 / 7 / 8
- 不消耗 GPU 推理资源


---

Method:

`Rule-based`

---

Rules:

- Face Count
- Track Continuity


---

Input:

`TrackingInfo`（Stage 5.1 输出）


---

Configuration（`configs/config.yaml` → `enhancements.stage_05_2_single_face_filter`）：

| Parameter | Default | Description |
|---|---|---|
| enabled | true | 是否启用硬过滤；false 时透传不筛选 |
| max_identity_anomaly_ratio | 0.005 | 异常帧比例阈值（多人脸 / 空帧 / 身份切换均计为异常） |

---

# 3.9.3 Stage 5.3 Temporal Stability Hard Filter


## Purpose


过滤（Hard Filter，非 annotation）：

- BBox跳变
- Landmark异常
- 人脸运动过大


---

## Behavior

对每个 FaceSequence：

1. 分析时序稳定性（`analyze_stability`）。
2. 生成 `stability_result`（含 `passed` / `reject_reasons`）。
3. `passed=True` → 附加 `stability_result`，继续传递。
4. `passed=False` → 记录 reject reason，丢弃该 sequence。


---

## Output

`List[FaceSequence]`（仅包含通过的 sequence）


---

## Gating

被丢弃的 sequence：

- 不进入 Stage 6 / 7 / 8
- 不消耗 GPU 推理资源（6DRepNet / SyncNet / HyperIQA）


---

## Independence

Stage 5.3 不强制依赖 Stage 5.2：

- 若输入 sequence 存在 `identity_result`：使用 `main_track_id` 的 bbox 进行分析。
- 若不存在：使用原 fallback 逻辑（全部 bbox / 单一人脸）。


---

Method:

`Rule-based`

---

Input:

`post-5.2 List[FaceSequence]`（来自 Stage 5.2；若 5.2 禁用则为 Stage 5.1 输出）


---

Output:

`List[FaceSequence]`（仅包含通过的 sequence）


---

Configuration（`configs/config.yaml` → `enhancements.stage_05_3_temporal_stability`）：

| Parameter | Default | Description |
|---|---|---|
| enabled | true | 是否启用硬过滤；false 时透传不筛选 |
| max_center_motion | 15.0 | 人脸中心最大位移（px） |
| max_scale_change_ratio | 0.25 | 最大尺度变化比例 |
| max_stability_anomaly_ratio | 0.05 | 稳定性异常帧比例阈值 |

---

# 3.10 Stage 6 Head Pose Filtering


# 3.10.1 Purpose


过滤：

- 大角度侧脸
- 低头
- 仰头


提高：

FLAME稳定性。


---

# 3.10.2 Model


Model:

`6DRepNet`

---

# 3.10.3 Input


Input:

`FaceInfo`

---

# 3.10.4 Output


Output:

`PoseInfo`

---

# 3.11 Stage 7 A/V Sync Detection


# 3.11.1 Purpose


通过SyncNet计算视频中**人脸运动**和语音之间的同步关系。

核心任务：

- Audio-Visual Synchronization Detection
- Offset Estimation（音频偏移量估计）


---
# 3.11.2 Model


Model:

`SyncNet`

---
# 3.11.3 Input


Input:

`FaceSequence`

+

`AudioInfo`

+

`ClipInfo`


其中：

`FaceSequence` 来自 Stage 5（SCRFD Face Detection & Crop）。

`FaceSequence` 包含：

- `frame_index`：帧编号
- `timestamp`：时间戳
- `face_crop_path`：人脸裁剪图像路径
- `face bbox`：人脸边界框
- `face landmark`：5点关键点

**SyncNet 使用 `face_crop_path` 对应的人脸连续帧作为视觉输入。**

对应关系：

Face Crop → SyncNet Visual Stream

Audio → SyncNet Audio Stream


**禁止使用：**

- 原始完整视频帧（full-frame video）
- 包含背景信息的帧（background frame）


**设计理由：**

1. **增强嘴部区域特征**：face crop 聚焦人脸区域，唇部运动特征更显著
2. **减少背景干扰**：排除背景运动、非人脸区域对 SyncNet 特征提取的干扰
3. **提高音视频同步评价稳定性**：统一输入尺度（crop 256×256 → SyncNet 224×224），避免因视频分辨率差异导致的特征不一致


---
# 3.11.4 Processing Logic


执行流程：

1. 从 `FaceSequence.face_crop_path` 读取人脸连续帧
2. 人脸帧 resize 至 224×224（SyncNet 标准输入尺寸）
3. 从 `AudioInfo.audio_path` 提取 MFCC 音频特征
4. SyncNet 计算唇部视觉特征与音频特征的 pairwise distance
5. 计算 offset（偏移帧数）、LSE-C、LSE-D


---
# 3.11.5 Output


Output:

`AVSyncInfo`

字段：

- `offset`：音频偏移帧数
- `sync_lse_c`：LSE-C = median(distance) - min(distance)
- `sync_lse_d`：LSE-D = min(distance)
- `median_distance`：frame-audio distance 中位数

---

# 3.11.6 Stage 7.1 A/V Sync Correction


# 3.11.6.1 Purpose


基于 Stage 7 SyncNet 输出的 Audio Offset，
对通过质量筛选的视频进行音视频时间轴校正。

生成校正后的 Audio 数据资产，
为后续 Audio Feature Extraction 提供时间准确的输入。

**注意：Stage 7.1 不重新运行 SyncNet。** 只消费 Stage 7 的 offset 输出。


---
# 3.11.6.2 Method


**SyncNet Offset-based Audio Timeline Alignment**


计算 offset 毫秒值：

`offset_ms = offset / fps * 1000`


根据 offset 调整 audio timeline。


---
# 3.11.6.3 Input


来自 Stage 7：

`AVSyncInfo`

字段：

- `clip_id`
- `offset`
- `sync_lse_c`
- `sync_lse_d`
- `median_distance`

以及：

`StandardizedVideoInfo`

字段：

- `video_path`
- `audio_path`
- `fps`


---
# 3.11.6.4 Processing Logic


校正策略：

| offset 范围 | 处理方式 | 说明 |
|---|---|---|
| 0 ~ 50ms | 保持不变 | 小偏移不影响训练质量 |
| 50 ~ 200ms | 自动校正 | 调整 audio timeline |
| > 200ms | 标记 | 由 Stage 9 根据 sync 质量处理 |

处理流程：

1. 读取 `AVSyncInfo.offset` 和 `StandardizedVideoInfo.fps`
2. 计算 `offset_ms = offset / fps * 1000`
3. 根据策略决定是否校正
4. 输出校正后的 audio 文件

保留原始视频不变。输出新的 corrected audio file。

示例：

- original: `clip001.wav`
- 输出: `clip001_aligned.wav`


---
# 3.11.6.5 Output


`AVAlignmentInfo`

字段：

- `clip_id`
- `original_audio_path`
- `corrected_audio_path`
- `offset_frames`
- `offset_ms`
- `correction_applied`
- `fps`

**corrected_audio_path** 作为后续 Stage 14 Audio Feature Extraction 的输入。


---
# 3.11.6.6 Configuration


```yaml
enhancements:
  stage_07_1_av_alignment:
    enabled: true           # 默认开启（音频一致性是基础要求）
    correction_threshold_ms: 50
    max_correction_ms: 200
    save_original_audio: true
```

---


# 3.12 Stage 8 Face Quality Filtering


# 3.12.1 Purpose


过滤：

- 模糊
- 低清晰度
- 低质量Face


---

# 3.12.2 Model


Model:

`HyperIQA`

---

# 3.12.3 Input


Input:

`Face Crop`

---

# 3.12.4 Output


Output:

`FaceQualityInfo`

---

# 3.13 Stage 9 Quality Metrics Fusion & Hard Filtering


# 3.13.1 Purpose


融合：

- Pose
- Face Quality
- A/V Sync


执行：

Hard Filter。


---

# 3.13.2 Method


Method:

`Rule-based Hard Filter`

---

# 3.13.3 Input


Input:

- PoseInfo
- AVSyncInfo
- FaceQualityInfo


---

# 3.13.4 Output


Output:

`QualityFusionInfo`

---

# 3.14 Stage 9.1 Weighted Quality Score Fusion


# 3.14.1 Purpose


Optional Enhancement Module。


用于：

- 数据分级
- 优先级排序
- 高质量样本筛选


---

# 3.14.2 Method


`Weighted Multi-metric Score Fusion`

---

# 3.14.3 Input


Quality Metrics


---

# 3.14.4 Output


`QualityScoreInfo`

---

# 3.15 Stage 10 High Precision Landmark Extraction


# 3.15.1 Purpose


提取：

高精度人脸关键点。


---

# 3.15.2 Model


Model:

- MediaPipe Face Landmarker
- InsightFace 106 Landmark


---

# 3.15.3 Input


`Face Crop`

---

# 3.15.4 Output


`LandmarkInfo`

---

# 3.16 Stage 11 Face Alignment


# 3.16.1 Purpose


通过Landmark：

进行仿射变换。

输出：

256×256标准Face。


---

# 3.16.2 Method


`Landmark-based Affine Transform`

---

# 3.16.3 Input


`LandmarkInfo`

---

# 3.16.4 Output


`AlignmentInfo`

---

# 3.17 Stage 11.5 Face Validity Check


# 3.17.1 Purpose


Optional Enhancement Module。


检查：

- Face有效性
- Landmark质量
- Alignment质量


---

# 3.17.2 Method


`Rule-based Validation`

---

# 3.17.3 Input


`AlignmentInfo`

---

# 3.17.4 Output


`FaceValidityInfo`

---

# 3.18 Stage 12 EMICA FLAME Reconstruction

# 3.18.1 Purpose

提取：

3D Face motion 参数。

使用 **EMICA** 作为主模型，包含：

- **EMICA ExpressionEncoder**：独立训练的 100 维 expression code
- **MICA**（optional）：identity / shape reconstruction → shape_code[300]
- **DECA**（optional）：texture/detail/camera → camera_parameters[N]
- **SPECTRE**（optional）：eye pose + 辅助 FLAME 参数

包括：
- expression motion（表情运动，100 维）
- jaw pose（下颌姿态）
- head pose（头部全局姿态）
- eye pose（眼球姿态，默认 zeros，Stage 13 可替换）


---

# 3.18.2 Architecture

**EMICA ExpressionEncoder** 负责：
- expression motion（100 维，独立训练的 `nn.Linear(2048, 100)`）
- **不是 EMOCA expcode[50] + zero-padding**

**EMICA（DECA 分支）** 负责：
- head/global pose
- jaw pose（最终方案 jaw_pose 来源）
- 注意：DECA 分支不负责 expression 输出

**SPECTRE** 作为辅助模块：
- 输出 eye_pose 和辅助 FLAME 参数（原始 jaw 以 `spectre_jawpose` debug 字段保留）
- **不替代 EMICA ExpressionEncoder 的 expression 空间**
- SPECTRE 的 expression 为 50 维，与 EMICA 100 维完全独立
- **最终方案：jaw_pose 取 EMICA（经 FLAME mesh 重建验证），不采用 SPECTRE jaw**

**Stage12 默认 expression 采用 EMICA ExpressionEncoder 的 100 维输出，不采用 SPECTRE 的 50 维。**


---

# 3.18.3 Models

| Model | Role | Output | Required |
|---|---|---|---|
| **EMICA ExpressionEncoder** | Primary expression encoder | expcode[100] | ✅ 默认开启 |
| **MICA** | Optional shape | shapecode[300] | ❌ 默认关闭 |
| **DECA** | Optional camera/texture | cam[N] | ❌ 默认关闭 |
| **SPECTRE** | Auxiliary eye pose + FLAME params | eye_pose[6], exp[50], pose[6] | ✅ 默认开启 |


---

# 3.18.4 Module Switch Config

```
stage_12:
  emica:
    modules:
      mica:
        enabled: false
      deca:
        enabled: false
      expression_encoder:
        enabled: true
      spectre:
        enabled: true
```

默认：expression_encoder=true, SPECTRE=true, MICA=false, DECA=false


---

# 3.18.5 Input

`AlignmentInfo`（来自 Stage 11）

SPECTRE 额外需要：视频帧序列（时序上下文）


---

# 3.18.6 Processing Flow

```
Aligned Face (per frame)
    ↓
MICA inference (optional) → shape_code[300]
    ↓
DECA inference (optional) → camera_parameters, head_pose, jaw_pose
    ↓
EMICA ExpressionEncoder inference
    ↓
expression_code: [100]  ← 独立训练的 100 维
    ↓
SPECTRE inference (optional)
    ↓
eye_pose: [6], auxiliary exp[50], pose[6]
    ↓
Structured FLAMEFeature
```


---

# 3.18.7 Output

`FLAMEFeature` (per frame)

| Parameter | Dimension | Source |
|---|---|---|
| expression_code | [100] | EMICA ExpressionEncoder |
| pose_code | [6] | global_pose[0:3] + jaw_pose[0:3] |
| jaw_pose | [3] | EMICA (final adopted, no fusion) |
| head_pose | [3] | EMICA |
| eye_pose | Optional[6] | SPECTRE / FLAME default zeros(6) |
| shape_code | Optional[300] | MICA (if enabled) |
| camera_parameters | Optional[N] | DECA (if enabled) |
| source | "EMICA" | 模型来源标识 |


---

# 3.18.8 Design Constraints

1. **expression_code 来源：EMICA ExpressionEncoder，禁止理解为 EMOCA expcode**
   - EMICA ExpressionEncoder: Tensor[100]，独立训练
   - EMOCA expcode: Tensor[50]，不同 latent space
   - 禁止使用 expcode[:50] 截取
   - 禁止 zero-padding EMOCA 50 维到 100 维

2. **禁止假定 Stage12 输出固定 112 维**
   - 112 只是默认配置（expression[100] + pose[6] + eye_pose[6]）下的结果
   - 不是接口约束，下游必须使用字段名访问

3. **flatten 操作禁止在 Stage12 完成**
   - 由 Stage17 负责最终 flatten

4. **eye_pose 当前为 zeros(6)**
   - 未来 Stage 13 gaze estimator 可替换

5. **SPECTRE 的 expression[50] 不参与 Stage12 最终 expression_code**
   - 如需使用 SPECTRE 50 维 expression，需显式配置并记录

6. **jaw_pose 来源（最终方案：纯 EMICA，无融合）**
   ```
   jaw_pose = emica_jaw        # 纯 EMICA jaw，无融合
   ```
   - FLAME mesh 实际重建验证后选定 EMICA jaw（SPECTRE jaw 版本嘴部区域存在扭曲，EMICA 版本重建效果更好）
   - `jaw_fusion_lambda` 已 **废弃（deprecated）**，不再参与计算，仅保留 config 标记与 metadata 审计
   - SPECTRE 原始 jaw 以 `spectre_jawpose` debug 字段保留（审计/对比用）
   - EMICA 不可用时 Stage12 失败（EMICA 是主模型）
   - 历史记录：此前曾测试 SPECTRE jaw（Scheme D）与 λ 加权融合，mesh 验证后最终选择 EMICA jaw

---

# 3.18.9 Runtime Mode

Stage12 运行在 Docker 容器中，通过 subprocess 模式调用：

```
Stage11 (host Python) → AlignedFaceInfo metadata JSON
    ↓
Stage12 Runner (docker exec face_pipeline_lxs)
    ├── source /workspace/emica_env/bin/activate
    └── python face_pipeline_v2/stages/stage_12_flame/runner.py
    ↓
FLAMEFeature outputs → flame_features.npz
```

**原因**: EMICA/SPECTRE 依赖 `inferno`、`pytorch-lightning`、`omegaconf` 等包，仅在 Docker 容器内的 `emica_env` 环境中可用，无法在 host Python 环境导入。

| 环境 | Python | PyTorch | 可用模块 |
|------|--------|---------|---------|
| Host | `/opt/hb/bin/python3` | 2.4.1+cpu | Stage 0-11 (SCRFD, 6DRepNet, etc.) |
| Docker (emica_env) | `/workspace/emica_env/bin/python` | CUDA | Stage 12 (EMICA + SPECTRE) |

---

# 3.18.10 Output File Structure

Per clip, output written to `outputs/stage12_flame/{clip_id}/`:

| File | Format | Content |
|------|--------|---------|
| `flame_features.npz` | NPZ | `{expression_code: [N,100], jaw_pose: [N,3], head_pose: [N,3], eye_pose: [N,6], shapecode: [N,300], frame_index: [N], timestamp: [N], clip_id: [N]}` |
| `flame_features.npy` | NPY | Raw FLAME parameters array |
| `metadata.json` | JSON | Per-frame metadata (frame_index, timestamp, clip_id) |

---

# 3.19 Stage 13 Gaze Parameter Extraction


# 3.19.1 Purpose


提取：

眼球运动参数。


---

# 3.19.2 Model


- MediaPipe Face Landmarker
- Gaze360


---

# 3.19.3 Input


`AlignmentInfo`

---

# 3.19.4 Output


`GazeFeature`

---

# 3.20 Stage 14 Audio Feature Extraction


# 3.20.1 Purpose


提取：

Audio Embedding。


---

# 3.20.2 Model


- HuBERT
- Whisper


---

# 3.20.3 Input


必须输入：

`AVAlignmentInfo.corrected_audio_path`

来源：Stage 7.1 校正后的 Audio

（原始定义 `CorrectedAudioInfo`）

---

禁止：

直接使用：

`Raw Audio`

---

# 3.20.4 Output


`AudioFeature`

---

Feature:

`[T,768]`

---

# 3.21 Stage 15 Quality Statistics


# 3.21.1 Purpose


统计：

Pipeline过滤情况。


---

Method:

`Pandas`

---

Input:

Pipeline Metadata


---

Output:

`StatisticsInfo`

---

# 3.22 Stage 16 Data Annotation


# 3.22.1 Purpose


生成：

数据集画像。


---

Models:

- Whisper
- FairFace
- CLIP


---

Output:

`AnnotationInfo`

---

# 3.23 Stage 17 Parameter Alignment & Fusion


# 3.23.1 Purpose


对齐：

- FLAME
- Gaze
- Audio


生成：

统一Feature。


---

# 3.23.2 Method


`Custom Alignment`

---

# 3.23.3 Input


| Data | Source |
|---|---|
| FLAMEFeature | Stage 12 |
| GazeFeature | Stage 13 |
| AudioFeature | Stage 14 |
| Metadata | Stage 15/16 |

---

# 3.23.4 Output


`TrainingSample`

---

# 3.24 Pipeline Execution Rules


## Rule 1

所有Stage：

必须通过：

标准Data Object通信。


---

## Rule 2

禁止：

Stage之间直接访问内部变量。


---

## Rule 3

所有模型：

必须通过：

Model Manager加载。


---

## Rule 4

所有参数：

必须来自：

`configs/config.yaml`


---

# 3.25 Final Pipeline Output


最终输出：

| Output | Description |
|---|---|
| Video Clip | 高质量视频 |
| Face Sequence | 标准化人脸序列 |
| Corrected Audio | 音画同步修正Audio |
| FLAME | 3D Face参数 |
| Gaze | 视线参数 |
| Audio Feature | Audio Embedding |
| Metadata | 数据描述信息 |

---

# End of 03_Pipeline_Stage_Specification.md v1.0
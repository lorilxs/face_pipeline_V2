# 04_Config_and_Model_Management.md

# AI-Friendly Software Design Document (AISDD)

# Part 4

# Config and Model Management

Version: v1.0

Last Update: 2026-08-03

---

# 4.1 Configuration and Model Management Overview

`Face_Pipeline_V2` 采用：

Configuration Driven Pipeline Architecture。

---

整个 Pipeline：

所有运行参数：

必须通过：

`configs/config.yaml`

统一管理。

---

所有模型：

必须通过：

Model Management System

统一加载和管理。

---

设计目标：

- 消除代码中的Magic Number
- 保证Pipeline可配置
- 保证模型版本可追踪
- 保证运行环境一致
- 支持GPU资源管理
- 支持AI Coding Agent自动生成代码

---

# 4.2 Configuration Management Architecture


配置文件：

统一入口：

`configs/config.yaml`

---

目录结构：

| Path | Purpose |
|---|---|
| configs/config.yaml | Pipeline全部配置 |
| models/ | 模型权重 |
| third_party/ | 第三方代码 |
| outputs/ | Pipeline输出 |
| logs/ | 运行日志 |

---

# 4.3 Configuration Design Principles


# 4.3.1 Single Configuration Source


所有参数：

必须来自：

`config.yaml`

---

禁止：

在代码中直接定义：

- threshold
- path
- model name
- batch size
- device


---

错误：

`threshold = 0.5`

---

正确：

`config.face_detection.confidence_threshold`

---

# 4.3.2 Configuration Hierarchy


配置按照功能划分：

| Category | Responsibility |
|---|---|
| project | 项目信息 |
| video | 视频处理 |
| scene | 场景检测 |
| face | 人脸检测 |
| tracking | 人脸跟踪 |
| pose | 姿态检测 |
| sync | 音画同步 |
| quality | 质量检测 |
| landmark | Landmark |
| alignment | 人脸对齐 |
| flame | FLAME |
| gaze | Gaze |
| audio | Audio Feature |
| runtime | 运行资源 |
| storage | 文件路径 |

---

# 4.4 Configuration Structure


# 4.4.1 Project Configuration


Purpose:

保存项目基础信息。


| Parameter | Type | Description |
|---|---|---|
| project_name | str | 项目名称 |
| version | str | Pipeline版本 |
| environment | str | 运行环境 |

---

Default:

| Parameter | Value |
|---|---|
| project_name | Face_Pipeline_V2 |
| version | v1.0 |

---

# 4.4.2 Video Configuration


Purpose:

控制视频标准化。


| Parameter | Type | Default | Description |
|---|---|---|---|
| target_fps | int | 25 | 标准视频FPS |
| audio_sample_rate | int | 16000 | Audio采样率 |
| resolution | tuple | 256x256 | 标准输入尺寸 |
| codec | str | h264 | 视频编码 |

---

# 4.4.3 Scene Configuration


Purpose:

控制Scene Detection。


| Parameter | Type | Description |
|---|---|---|
| detector | str | Scene检测模型 |
| threshold | float | Scene切分阈值 |
| min_scene_duration | float | 最短Scene长度 |

---

Models:

| Model | Usage |
|---|---|
| PySceneDetect | Scene Detection |
| TransNetV2 | Deep Scene Detection |

---

# 4.4.4 Face Detection Configuration


Purpose:

控制SCRFD检测。


| Parameter | Type | Description |
|---|---|---|
| model | str | SCRFD模型 |
| confidence_threshold | float | 人脸置信度 |
| sampling_rate | int | 初筛采样频率 |
| keep_single_face_only | bool | 是否只保留单人脸 |
| max_allowed_faces | int | 最多人脸数 |
| min_detection_ratio | float | 最小人脸检测比例 |
---


Default:

| Parameter | Value |
|---|---|
| sampling_rate | 1 FPS |
| keep_single_face_only | true |
| max_allowed_faces | 1 |
| min_detection_ratio | 0.8 |

---

# 4.4.4.1 Stage 3 Coarse Face Screening Configuration

Purpose:

控制Stage 3粗筛人脸筛选。

---

| Parameter | Type | Description |
|---|---|---|
| sampling_fps | int | 初筛采样帧率 |
| min_duration | float | 最短Scene时长 |
| min_face_confidence | float | 人脸置信度阈值 |
| min_single_face_ratio | float | 单人脸比例阈值 |
| min_face_detect_ratio | float | 人脸检测比例阈值 |

---

Default:

| Parameter | Value |
|---|---|
| sampling_fps | 1 |
| min_duration | 8 |
| min_face_confidence | 0.5 |
| min_single_face_ratio | 0.8 |
| min_face_detect_ratio | 0.8 |

---

设计原则：

- 所有阈值来自 config.yaml
- Stage 3 不保存任何聚合统计
- 过滤结果以 per-sample metadata 形式嵌入 SceneInfo

---

# 4.4.5 Sliding Window Configuration


Purpose:

控制Clip生成。


| Parameter | Type | Default | Description |
|---|---|---|---|
| window_length | float | 8 | Clip长度秒 |
| step | float | 4 | 滑动步长秒 |

---

# 4.4.6 Tracking Configuration


Purpose:

控制开放域增强模块。


Model:

`ByteTrack`

---

| Parameter | Type | Description |
|---|---|---|
| enabled | bool | 是否启用 |
| track_threshold | float | Track置信度 |
| max_lost_time | int | 最大丢失帧 |

---

说明：

`enabled=true` 时激活对应增强 Stage 的硬过滤（Hard Filter / Gating）行为：

- Stage 5.2：身份不一致的 sequence 被丢弃，不进入 Stage 5.3 / 6 / 7 / 8。
- Stage 5.3：时序不稳定的 sequence 被丢弃，不进入 Stage 6 / 7 / 8。

`enabled=false` 时该 Stage 为纯透传（passthrough）：不附加结果、不丢弃任何 sequence。

---

# 4.4.7 Pose Configuration


Purpose:

控制Head Pose Filtering。


Model:

`6DRepNet`

---

| Parameter | Type | Description |
|---|---|---|
| yaw_threshold | float | 水平旋转限制 |
| pitch_threshold | float | 垂直旋转限制 |
| roll_threshold | float | 平面旋转限制 |

---

# 4.4.8 A/V Sync Configuration


Purpose:

控制音画同步检测。


Model:

`SyncNet`

---

| Parameter | Type | Description |
|---|---|---|
| max_offset | int | 最大允许偏移帧 |
| sync_threshold | float | 同步评分阈值 |
| correction_enable | bool | 是否执行Audio修正 |

---

Default:

| Parameter | Value |
|---|---|
| max_offset | 8 frames |

---

# 4.4.9 Face Quality Configuration


Purpose:

控制HyperIQA。


Model:

`HyperIQA`

---

| Parameter | Type | Description |
|---|---|---|
| quality_threshold | float | 最低质量分 |
| batch_size | int | 推理Batch |

---

# 4.4.10 Landmark Configuration


Purpose:

控制高精度Landmark。


Models:

- MediaPipe Face Landmarker
- InsightFace 106 Landmark


---

| Parameter | Type | Description |
|---|---|---|
| model | str | Landmark模型 |
| num_points | int | Landmark数量 |
| confidence_threshold | float | 关键点阈值 |

---

# 4.4.11 Alignment Configuration


Purpose:

控制Face Alignment。


---

| Parameter | Type | Default | Description |
|---|---|---|---|
| method | str | affine | 对齐方法 |
| output_size | tuple | 256x256 | 输出尺寸 |

---

# 4.4.12 FLAME Configuration


Purpose:

控制 FLAME motion 参数提取。


Models:

- EMICA ExpressionEncoder（Primary expression encoder，100 维）
- SPECTRE（Eye pose + 辅助 FLAME 参数）
- MICA（Optional shape）
- DECA（Optional camera）


---

## Module Switch Config

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

## Model Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| model | str | EMICA | Primary FLAME model |
| device | str | cuda:0 | GPU设备 |
| expression_dim | int | 100 | Expression维度（EMICA ExpressionEncoder 独立训练） |
| pose_dim | int | 6 | Pose维度（jaw[0:3] + head[3:6]） |
| eye_pose_dim | int | 6 | Eye pose维度（SPECTRE / FLAME default zeros） |
| shape_dim | int | 300 | Shape维度（MICA optional） |

**注意：expression_dim=100 是 EMICA ExpressionEncoder 的真实输出维度，不是 EMOCA 的 50 维 padding 到 100。禁止在代码中写死总输出维度（如 112），下游必须使用结构化字段访问。**

---

## Checkpoint Paths

| Model | Checkpoint | Architecture |
|-------|-----------|-------------|
| EMICA | `models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt` | EmicaEncoder (ExpressionEncoder + MicaDecaEncoder) |
| SPECTRE | `third_party/inferno/spectre/spectre/data/deca_model.tar` | DECA (FLAME-based) |
| FLAME | `models/FLAME/geometry/generic_model.pkl` | FLAME 2020 geometry |

---

## Config Structure

```yaml
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
  jaw_source: emica             # FINAL scheme: jaw_pose = EMICA (no fusion)
  jaw_fusion_lambda:
    deprecated: true
    reason: "jaw source switched to EMICA (FLAME mesh validated)"
    legacy_default: 1.0
  batch_size: 8
  device: cuda
  subprocess_timeout: 3600
```

默认：expression_encoder=true, SPECTRE=true, MICA=false, DECA=false

> `jaw_fusion_lambda` 已 **废弃（deprecated）** — 不再参与计算；`jaw_pose` 固定为 EMICA（最终方案，经 FLAME mesh 重建验证）。SPECTRE 原始 jaw 以 `spectre_jawpose` debug 字段保留。

---

## Runtime Environment

Stage12 运行在 Docker 容器中，通过 subprocess 模式调用：

- **Container**: `face_pipeline_lxs`
- **Python env**: `/workspace/emica_env/bin/activate`
- **Mount**: `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2` → `/workspace/Face_Pipeline_V2`
- **Dependencies**: `torch` (CUDA), `pytorch-lightning`, `inferno`, `omegaconf`, `yacs`, `kornia`

EMICA/SPECTRE 无法在 host Python 环境中导入（host 缺少 `inferno`、`omegaconf`、`yacs`）。

---

## Minimal Module Loading

Stage12 采用 encoder-only inference 策略，仅加载编码器所需模块，减少不必要模块加载：

| 模块 | 加载 | 用途 |
|------|------|------|
| ExpressionEncoder (ResNet50) | ✅ 必需 | 输出 expcode[100] |
| DecaEncoder (ResNet50) | ✅ 必需 | 输出 jawpose[3], globalpose[3], cam[3] |
| MicaEncoder | ✅ 必需 | 输出 shapecode[300] |
| SPECTRE E_expression | ✅ 必需 | 输出 SPECTRE jaw[3] |
| FlameShapeModel (FLAME decoder) | ❌ 不加载 | 仅 decode() 需要 |
| FlameLandmarkProjector | ❌ 不加载 | Stage12 不需要渲染 |
| FLAMETex | ❌ 不加载 | 纹理非必需 |

三个编码器 (ExpressionEncoder + DecaEncoder + MicaEncoder) 合计约 73M 参数，均为 Stage12 所必需。

---

## Wrapper Design

Stage12 通过 wrapper 适配 EMICA 和 SPECTRE，不修改 `third_party/` 中的任何源码：

- **EMICAWrapper**: 封装 `EmicaEncoder`，管理 checkpoint 加载和模块开关
- **SPECTREWrapper**: 封装 SPECTRE `E_expression`，仅加载 encoder 部分

Wrapper 核心职责：
1. Checkpoint 加载与验证
2. 模块开关控制（MICA/DECA 按需加载）
3. 输出格式标准化（统一为 FLAMEFeature 字段）

---

# 4.4.13 Gaze Configuration


Purpose:

控制视线提取。


Models:

- MediaPipe
- Gaze360


---

| Parameter | Type | Description |
|---|---|---|
| model | str | Gaze模型 |
| output_dim | int | 输出维度 |

---

Default:

| Parameter | Value |
|---|---|
| output_dim | 6 |

---

# 4.4.14 Audio Feature Configuration


Purpose:

控制Audio Feature Extraction。


Models:

- HuBERT
- Whisper


---

| Parameter | Type | Description |
|---|---|---|
| model | str | Audio模型 |
| sample_rate | int | 输入采样率 |
| feature_dim | int | 输出维度 |

---

Default:

| Parameter | Value |
|---|---|
| feature_dim | 768 |

---

# 4.4.15 Runtime Configuration


Purpose:

控制运行环境。


---

| Parameter | Type | Description |
|---|---|---|
| device | str | GPU/CPU设备 |
| num_workers | int | 并行Worker数量 |
| batch_size | int | 推理Batch |
| enable_parallel | bool | 是否开启并行 |

---

# 4.4.16 Storage Configuration


Purpose:

管理输入输出路径。


---

| Parameter | Type | Description |
|---|---|---|
| input_dir | Path | 输入视频目录 |
| output_dir | Path | 输出目录 |
| model_dir | Path | 模型目录 |
| log_dir | Path | 日志目录 |

---

# 4.5 Model Management Architecture


# 4.5.1 Model Management Principle


所有模型：

禁止：

Stage内部直接加载。


---

统一通过：

`ModelManager`

管理。

---

# 4.5.2 Model Directory Structure


## Third Party Code


Path:

`third_party/`


Structure:

| Directory | Description |
|---|---|
| Aspect | Description |
|---|---|
| **EMICA** | EMICA源码（含 MICA, DECA, ExpressionEncoder, SPECTRE integration） |
| **SPECTRE** | SPECTRE源码（时序 jaw refinement） |
| HyperIQA | HyperIQA源码 |
| SyncNet | SyncNet源码 |
| ByteTrack | ByteTrack源码 |

---

## Model Weights


Path:

`models/`

Structure:

| Directory | Description |
|---|---|
| SCRFD | 人脸检测模型 |
| 6DRepNet | 姿态模型 |
| HyperIQA weights | 质量模型权重 |
| EMICA weights | FLAME模型权重 |
| HuBERT weights | Audio模型权重 |

---

# 4.6 ModelManager Interface


# 4.6.1 Purpose


统一管理：

- 模型加载
- 模型缓存
- GPU分配
- 模型生命周期


---

# 4.6.2 Interface


Method:

`get_model(model_name)`


---

Input:

| Parameter | Type | Description |
|---|---|---|
| model_name | str | 模型名称 |

---

Output:

| Parameter | Type | Description |
|---|---|---|
| model | Object | 已加载模型 |

---

# 4.7 Model Loading Rules


# Rule 1


模型：

首次使用时加载。


---

# Rule 2


同一模型：

禁止重复加载。


---

# Rule 3


模型设备：

必须由：

Runtime Configuration

控制。


---

# Rule 4


模型路径：

必须来自：

`config.yaml`

---

# 4.8 GPU Resource Management


# 4.8.1 Quality Assessment Parallel


以下Stage：

可以并行执行：

| Stage | Model |
|---|---|
| Stage 6 | 6DRepNet |
| Stage 7 | SyncNet |
| Stage 8 | HyperIQA |

---

# 4.8.2 Feature Extraction Parallel


以下Stage：

可以并行执行：

| Stage | Model |
|---|---|
| Stage 12 | EMICA + SPECTRE |
| Stage 13 | Gaze Model |
| Stage 14 | HuBERT |
| Stage 15 | Pandas |
| Stage 16 | Annotation Models |

---

# 4.9 Environment Management


# 4.9.1 Python Package Structure


项目采用：

Python Package Architecture。


---

禁止：

简单Script组织。


---

推荐结构：

| Directory | Responsibility |
|---|---|
| face_pipeline_v2 | 主Package |
| stages | Stage实现 |
| models | Model Wrapper |
| data | Data Model |
| configs | Configuration |
| utils | Common Utility |

---

# 4.9.2 Dependency Isolation


第三方代码：

必须隔离：

`third_party/`


模型权重：

必须隔离：

`models/`


---

# 4.10 Error Handling


# Configuration Error


Examples:

- config.yaml不存在
- 参数缺失
- 类型错误


处理：

Pipeline启动失败。


---

# Model Error


Examples:

- 权重不存在
- 模型加载失败


处理：

记录Log并停止当前任务。


---

# Runtime Error


Examples:

- GPU不可用
- CUDA错误


处理：

Fallback或任务终止。


---

# 4.11 Configuration and Model Management Summary


`Face_Pipeline_V2`

采用：

Configuration Driven + Model Manager Architecture。


---

核心原则：

- 单一配置入口
- 模型统一管理
- GPU资源统一调度
- 无Magic Number
- 环境可复现
- 支持Pipeline扩展


---

最终保证：

每个Stage：

能够通过：

Config

+

ModelManager

+

Data Interface

独立运行。


---

# End of 04_Config_and_Model_Management.md v1.0
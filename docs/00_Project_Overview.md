# 00_Project_Overview.md

# AI-Friendly Software Design Document (AISDD)

# Part 0

# Project Overview

Version: v1.0

Last Update: 2026-08-03

---

# 0.1 Project Introduction

## 0.1.1 Project Name

`Face_Pipeline_V2`

---

## 0.1.2 Project Type

`AI-driven Multi-modal Talking Face Dataset Generation Pipeline`

---

## 0.1.3 Design Objective

Face_Pipeline_V2 是一个自动化高质量 Talking Face 数据生产 Pipeline。

目标：

将原始视频数据：

转换为：

高质量、多模态、时间同步的训练数据。

---

Pipeline 最终生成：

- Standardized Video Clip
- Aligned Face Sequence
- FLAME Parameters
- Gaze Parameters
- Audio Embeddings
- Metadata

---

# 0.1.4 Application Scenario

该系统主要用于：

- Talking Face Generation
- Audio-driven Facial Animation
- 3D Face Reconstruction
- FLAME-based Facial Parameter Learning
- Gaze-aware Avatar Modeling

---

# 0.2 System Goal

## 0.2.1 High-quality Video Filtering

系统需要自动过滤：

- Invalid Video
- Broken Video
- No Face Video
- Multi Face Video
- Low Quality Face Video
- Large Pose Deviation Video
- Audio-Visual Synchronization Error Video

---

## 0.2.2 Face Processing

系统负责：

- Face Detection
- Face Tracking
- Dense Landmark Extraction
- Face Alignment
- Face Sequence Generation

---

## 0.2.3 Multi-modal Feature Extraction

系统负责生成：

- 3D Face Parameters
- Eye Gaze Parameters
- Audio Feature Embeddings
- Dataset Metadata

---

## 0.2.4 Training Dataset Generation

最终输出：

可直接用于模型训练的数据集。

---

所有模态必须保持严格时间一致：

| Data | Requirement |
|---|---|
| Video Frame | Frame index t |
| Face Image | Corresponding Frame t |
| FLAME Parameter | Corresponding Frame t |
| Gaze Parameter | Corresponding Frame t |
| Audio Feature | Corresponding Timestamp t |

---

# 0.3 System Input and Output

# 0.3.1 Input

输入：

`Raw Video`

---

支持来源：

- Internet Video
- Talking Head Video
- Interview Video
- Vlog Video
- Educational Video
- Broadcast Video

---

输入视频可能存在：

- Different FPS
- Different Resolution
- Different Codec
- Different Audio Format
- Different Scene Structure

---

Pipeline 会在前处理阶段：

统一输入格式。

---

# 0.3.2 Output

最终输出：

## Video Data

| Field | Description |
|---|---|
| Video Clip | Standardized video segment |
| FPS | 25 FPS |
| Format | Unified video format |

---

## Face Data

| Field | Description |
|---|---|
| Face Sequence | Aligned face image sequence |
| Resolution | 256 × 256 |
| Format | Image sequence |

---

## FLAME Data

Pipeline 使用：

`EMOCA / EMICA`

提取完整 FLAME 相关参数。

---

提取内容包括：

| Parameter | Description |
|---|---|
| Expression Parameters | Facial expression coefficients |
| Pose Parameters | Head pose coefficients |
| Mouth-related Parameters | Mouth motion related coefficients |
| Other Facial Coefficients | Additional FLAME related features |

---

系统保留：

完整 FLAME 参数表示。

---

当前下游训练任务可选择：

Feature Subset：

`expcode[:50] + posecode[:6]`

维度：

`56`

---

完整 FLAME 表示：

用于：

- Future Model Development
- Feature Expansion
- Alternative Training Strategy

---

## Gaze Data

提取：

| Parameter | Dimension |
|---|---|
| Left Eye Gaze | x,y,z |
| Right Eye Gaze | x,y,z |

总维度：

`6`

---

## Audio Data

Audio Feature:

| Field | Description |
|---|---|
| Model | HuBERT / Whisper |
| Output | Audio Embedding |
| Dimension | T × 768 |

---

## Metadata

包含：

- Video Information
- Clip Information
- Quality Statistics
- Annotation Information
- Processing Status

---

# 0.4 Pipeline Overview

Face_Pipeline_V2 Pipeline 分为：

五个主要处理层。

---

## Layer 1

Video Processing Layer

Stage：

`Stage 0 - Stage 2`

负责：

- Video Validation
- Audio Video Standardization
- Scene Segmentation

---

## Layer 2

Face Screening and Quality Control Layer

Stage：

`Stage 3 - Stage 9.1`

负责：

- Face Screening
- Face Tracking
- Quality Filtering
- Quality Ranking

---

## Layer 3

Face Normalization Layer

Stage：

`Stage 10 - Stage 11.5`

负责：

- Dense Landmark Extraction
- Face Alignment
- Face Validity Verification

---

## Layer 4

Feature Extraction Layer

Stage：

`Stage 12 - Stage 16`

负责：

- FLAME Extraction
- Gaze Extraction
- Audio Feature Extraction
- Dataset Annotation

---

## Layer 5

Feature Alignment and Dataset Generation Layer

Stage：

`Stage 17`

负责：

- Multi-modal Alignment
- Feature Fusion
- Final Dataset Generation

---

# 0.5 Core Design Principles

# 0.5.1 Configuration Driven Design

所有运行参数：

必须来自：

`configs/config.yaml`

---

禁止：

硬编码参数。

---

错误：

`quality_threshold = 40`

---

正确：

`config.quality.threshold`

---

# 0.5.2 Immutable Data Flow

所有 Stage 之间：

必须使用不可变数据结构传递。

---

推荐：

`@dataclass(frozen=True, slots=True)`

---

规则：

- Input Object 不允许直接修改
- Stage 输出新的 Data Object
- 保持数据流可追踪

---

# 0.5.3 Modular Stage Architecture

每一个 Stage：

必须作为独立模块实现。

---

每个 Stage 必须定义：

| Item | Description |
|---|---|
| Input Contract | 输入数据结构 |
| Output Contract | 输出数据结构 |
| Model Dependency | 使用模型 |
| Configuration Dependency | 配置参数 |
| Error Handling | 异常处理 |

---

所有 Stage：

继承统一接口：

`BaseStage`

---

# 0.5.4 Quality Gate Architecture

Pipeline 使用：

渐进式质量筛选策略。

---

原则：

低成本过滤：

优先执行。

---

高成本模型：

只处理：

通过前级筛选的数据。

---

流程：

| Stage Type | Example |
|---|---|
| Early Filtering | FFprobe, FFmpeg, SCRFD |
| Middle Filtering | Pose, Sync, Quality |
| Expensive Extraction | FLAME, Gaze |

---

# 0.5.5 Parallel Processing Design

独立任务：

支持并行执行。

---

Quality Assessment：

| Stage | Task |
|---|---|
| Stage 6 | Head Pose Filtering |
| Stage 7 | Audio Visual Sync Detection |
| Stage 8 | Face Quality Assessment |

完成后：

统一进入：

`Stage 9 Quality Fusion`

---

Feature Extraction：

| Stage | Task |
|---|---|
| Stage 12 | FLAME Extraction |
| Stage 13 | Gaze Extraction |
| Stage 14 | Audio Feature Extraction |
| Stage 15 | Quality Statistics |
| Stage 16 | Data Annotation |

完成后：

进入：

`Stage 17 Feature Alignment and Fusion`

---

# 0.6 Software Architecture Requirement

## 0.6.1 Python Package Structure

项目采用：

Python Package Architecture。

---

Main Package：

`Face_Pipeline_V2`

---

安装方式：

`pip install -e .`

---

代码调用：

`from Face_Pipeline_V2.pipeline.runner import PipelineRunner`

---

# 0.6.2 Project Structure

项目结构：

| Directory | Purpose |
|---|---|
| docs | Software design documents |
| configs | Configuration files |
| models | Model weights |
| third_party | External source code |
| Face_Pipeline_V2 | Main Python package |
| tests | Unit and integration tests |
| scripts | Execution scripts |
| outputs | Pipeline outputs |

---

# 0.7 Technology Stack

## Video Processing

- FFmpeg
- FFprobe

---

## Scene Detection

- PySceneDetect
- TransNetV2

---

## Face Detection

- SCRFD

---

## Face Tracking

- ByteTrack

---

## Head Pose Estimation

- 6DRepNet

---

## Audio Visual Synchronization

- SyncNet

---

## Face Quality Assessment

- HyperIQA

---

## Landmark Extraction

- MediaPipe Face Landmarker
- InsightFace 106 Landmark

---

## FLAME Parameter Extraction

- EMOCA
- EMICA

---

## Gaze Extraction

- MediaPipe Face Landmarker
- Gaze360

---

## Audio Feature Extraction

- HuBERT
- HuBERT
- Whisper

---

# 0.8 Non-functional Requirements

# 0.8.1 Scalability

系统支持：

- Single Video Processing
- Batch Processing
- Large-scale Dataset Generation

---

# 0.8.2 Reproducibility

每个输出样本必须记录：

| Field | Description |
|---|---|
| Pipeline Version | Pipeline version |
| Config Version | Configuration version |
| Model Version | Model version |
| Processing Time | Execution timestamp |

---

# 0.8.3 Fault Tolerance

Pipeline 出现错误时：

必须：

- Record Error Log
- Skip Invalid Sample
- Continue Remaining Processing

---

# 0.8.4 Traceability

每个最终样本：

必须能够追踪：

`Raw Video`

到：

`Final Feature`

完整路径。

---

关系：

| Level | Output |
|---|---|
| Raw Video | Original input |
| Scene | Scene segment |
| Clip | Training clip |
| Face Sequence | Aligned face data |
| Feature Files | Final training features |

---

# 0.9 AISDD Document Structure

本 AISDD 包含：

| Document | Purpose |
|---|---|
| 00_Project_Overview.md | Project objective and system scope |
| 01_System_Architecture.md | Overall architecture and pipeline design |
| 02_Data_Model_and_Interface.md | Data structures and module interfaces |
| 03_Pipeline_Stage_Specification.md | Detailed Stage specification |
| 04_Config_and_Model_Management.md | Configuration and model management |
| 05_Data_Flow_and_Output.md | Storage and output specification |
| 06_Engineering_Guideline.md | Coding standards and implementation rules |

---

# 0.10 Implementation Direction

AISDD 用于：

指导 AI Coding Agent 自动生成工程代码。

---

执行流程：

`AISDD Specification`

↓

`Python Package Implementation`

↓

`Executable Pipeline`

↓

`Training Dataset Generation`

---

实现必须优先保证：

- Correct Data Flow
- Clear Module Boundary
- Reproducibility
- Maintainability
- GPU Efficient Execution

---

# End of 00_Project_Overview.md v1.0
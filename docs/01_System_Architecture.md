# 01_System_Architecture.md

# AI-Friendly Software Design Document (AISDD)

# Part 1

# System Architecture

Version: v1.0

Last Update: 2026-08-03

---

# 1.1 Architecture Overview

`Face_Pipeline_V2` 是一个面向高质量 Talking Face 数据生产的视频处理 Pipeline。

系统采用：

模块化 Stage Architecture

以及：

Configuration Driven Pipeline Design

实现：

原始视频输入

到：

多模态训练数据输出

的自动化处理。

---

整体架构：

| Layer | Stage | Description |
|---|---|---|
| Video Processing Layer | Stage 0 - Stage 2 | 视频验证、标准化、场景切分 |
| Face Screening Layer | Stage 3 - Stage 5 | 人脸初筛、人脸定位、Clip生成 |
| Optional Enhancement Layer (Early Hard-Filter Gates) | Stage 5.1 - Stage 5.3 | 开放域视频增强筛选 + 早期硬过滤（拒绝的 sequence 不进入 Stage 6-9） |
| Quality Assessment Layer | Stage 6 - Stage 9.1 | 质量检测、融合、排序 |
| Face Normalization Layer | Stage 10 - Stage 11.5 | Landmark、对齐、有效性检测 |
| Feature Extraction Layer | Stage 12 - Stage 16 | FLAME、Gaze、Audio、Annotation |
| Fusion Layer | Stage 17 | 多模态参数对齐与融合 |

---

# 1.2 Overall Pipeline Architecture

完整 Pipeline：

`Raw Video`

↓

`Stage 0 Video Integrity Check`

↓

`Stage 1 AV Standardization`

↓

`Stage 2 Scene Segmentation`

↓

`Stage 3 Coarse Face Screening`

↓

`Stage 4 Clip Generation / Clip Adapter (sliding_window | segment)`

↓

`Stage 5 Face Localization & Crop`

↓

根据视频类型选择：

---

## High Constraint Talking Face

例如：

- TED
- Weather Broadcast
- Studio Talking Head

执行：

`Stage 6 - Stage 9`

---

## Open Domain Video

例如：

- Bilibili
- Xiaohongshu
- Vlog

增加：

`Stage 5.1 - Stage 5.3 Enhancement Module`

该模块是 Early Rejection Gates（早期硬过滤门）：

Stage 5.2 / Stage 5.3 中未通过的 sequence 被直接丢弃，

不进入 Stage 6 - Stage 9 的昂贵模型推理。

之后进入：

`Stage 6 - Stage 9`

---

最终：

`Stage 9 / Stage 9.1`

↓

`Stage 10`

↓

`Stage 11`

↓

`Stage 11.5 Optional Face Validity Check`

↓

`Stage 12 - Stage 16`

↓

`Stage 17 Feature Alignment & Fusion`

↓

`Final Training Dataset`

---

# 1.3 Pipeline Execution Flow

## 1.3.1 Main Execution Flow


| Stage | Module | Purpose |
|---|---|---|
| Stage 0 | Video Integrity Check | 检查视频有效性 |
| Stage 1 | AV Standardization | 统一视频音频格式 |
| Stage 2 | Scene Segmentation | 生成连续 Scene |
| Stage 3 | Coarse Face Screening | 快速过滤无效 Scene |
| Stage 4 | Clip Generation / Clip Adapter | 生成 Clip（sliding_window 固定窗口 / segment 整段适配） |
| Stage 5 | Face Localization & Crop | 获取每帧人脸区域 |
| Stage 6 | Head Pose Filtering | 过滤大角度姿态 |
| Stage 7 | A/V Sync Detection | 检测音画同步 |
| Stage 8 | Face Quality Filtering | 检测人脸质量 |
| Stage 9 | Quality Metrics Fusion | 综合质量过滤 |
| Stage 10 | High Precision Landmark | 提取高精度关键点 |
| Stage 11 | Face Alignment | 人脸标准化 |
| Stage 12 | FLAME Motion Extraction | 提取3D人脸motion参数（expression + global pose + jaw pose；jaw = EMICA 最终方案，SPECTRE jaw 仅 debug 保留） |
| Stage 13 | Gaze Extraction | 提取视线参数 |
| Stage 14 | Audio Feature Extraction | 提取音频Embedding |
| Stage 15 | Quality Statistics | 统计Pipeline质量 |
| Stage 16 | Data Annotation | 数据集画像分析 |
| Stage 17 | Parameter Alignment Fusion | 多模态融合 |

---

# 1.4 Layer Architecture

# 1.4.1 Video Processing Layer

## Purpose

负责：

原始视频预处理。

---

包含：

- 视频完整性检查
- 视频格式统一
- 场景检测

---

输入：

`Raw Video`

---

输出：

`Standardized Video`

以及：

`Scene Metadata`

---

# 1.4.2 Face Screening Layer

## Purpose

负责：

降低后续高计算模块的数据规模。

---

主要策略：

先低成本筛选：

再高成本处理。

---

包含：

| Stage | 处理 | Metadata |
|---|---|---|
| Stage 3 | SCRFD粗筛 | CoarseFaceInfo + SceneFilterResult |
| Stage 4 | Clip生成/适配（sliding_window | segment） | ClipInfo |
| Stage 5 | 人脸检测裁剪 | FaceInfo |

---

输入：

`SceneInfo`

---

Stage 3输出：

`SceneInfo`

（每个Scene附加 CoarseFaceInfo + SceneFilterResult）

---

Stage 4输出：

`ClipInfo`

---

# 1.4.3 Open Domain Enhancement Layer (Early Hard-Filter Gates)

## Purpose

针对开放域视频：

增加人物稳定性判断，

并在进入昂贵模型推理（Stage 6/7/8）之前，

对不满足条件的 sequence 进行早期硬过滤（Hard Filter）。

---

该模块为：

Optional Enhancement Module（可通过 `enhancements.*.enabled=false` 整体关闭）。

启用时，Stage 5.2 / Stage 5.3 是 Hard Filter 而非纯 annotation：

未通过检查的 sequence 被直接丢弃，不进入 Stage 6 - Stage 9。

---

包含：

| Stage | Module | Purpose | 硬过滤行为 |
|---|---|---|---|
| Stage 5.1 | Face Tracking | 建立人物连续轨迹 | 不丢弃（仅附加 track_id） |
| Stage 5.2 | Single Face Identity Hard Filter | 保证单人脸身份一致 | 多人脸 / 空帧 / 身份切换比例超标 → 丢弃 |
| Stage 5.3 | Temporal Stability Hard Filter | 过滤运动异常 | BBox 跳变 / 尺度突变 / 运动过大 → 丢弃 |

---

原则：Progressive Gating

- 廉价规则过滤（5.2 / 5.3）先于昂贵 GPU 模型推理（Stage 6 / 7 / 8）。
- 减少无效 clip 消耗 GPU 推理资源。

---

适用：

开放域视频。

---

不影响：

High Constraint Talking Face Pipeline。

---

# 1.4.4 Quality Assessment Layer

## Purpose

对候选 Clip 进行多维质量评估。

---

并行执行：

| Stage | Model | Metric |
|---|---|---|
| Stage 6 | 6DRepNet | Head Pose |
| Stage 7 | SyncNet | Audio Visual Sync |
| Stage 8 | HyperIQA | Face Quality |

---

三个模块完成后：

统一进入：

`Stage 9 Quality Metrics Fusion`

---

# 1.4.5 Face Normalization Layer

## Purpose

生成标准化人脸输入。

---

包含：

| Stage | Function |
|---|---|
| Stage 10 | Dense Landmark Extraction |
| Stage 11 | Face Alignment |
| Stage 11.5 | Face Validity Check |

---

输出：

`Aligned Face Sequence`

---

# 1.4.6 Feature Extraction Layer

## Purpose

生成训练所需多模态参数。

---

并行执行：

| Stage | Output |
|---|---|
| Stage 12 | FLAME Motion Parameters |
| Stage 13 | Gaze Parameters |
| Stage 14 | Audio Embedding |
| Stage 15 | Quality Statistics |
| Stage 16 | Dataset Annotation |

---

# 1.4.7 Fusion Layer

## Purpose

保证所有模态时间一致。

---

输入：

| Data | Source |
|---|---|
| Face Sequence | Stage 11 |
| FLAME | Stage 12 |
| Gaze | Stage 13 |
| Audio Feature | Stage 14 |

---

输出：

`Final Training Data`

---

# 1.5 Parallel Execution Architecture

# 1.5.1 Quality Assessment Parallelism

Stage 6、Stage 7、Stage 8：

独立运行。

---

执行方式：

| GPU Worker | Task |
|---|---|
| Worker 1 | Head Pose |
| Worker 2 | A/V Sync |
| Worker 3 | Face Quality |

---

完成后：

进入：

`Stage 9`

---

# 1.5.2 Feature Extraction Parallelism

Stage 12 - Stage 16：

独立运行。

---

执行方式：

| Worker | Task |
|---|---|
| Worker 1 | FLAME |
| Worker 2 | Gaze |
| Worker 3 | Audio Feature |
| Worker 4 | Statistics |
| Worker 5 | Annotation |

---

完成后：

进入：

`Stage 17`

---

# 1.6 Module Architecture

系统采用：

Python Package Architecture。

---

Main Package：

`Face_Pipeline_V2`

---

核心模块：

| Module | Responsibility |
|---|---|
| pipeline | Pipeline execution |
| stages | Stage implementation |
| data | Data models |
| models | Model wrappers |
| services | Config and resource management |
| utils | Common utilities |

---

# 1.7 Stage Dependency Architecture

Stage之间：

采用：

单向数据流。

---

规则：

- 后置Stage不能修改前置Stage输出
- Stage之间通过 Data Object 通信
- 禁止直接访问其他Stage内部变量

---

数据流：

`Stage Input`

↓

`Stage Processing`

↓

`Stage Output`

↓

`Next Stage`

---

# 1.8 Runtime Architecture

Pipeline运行时：

由以下组件组成：

| Component | Responsibility |
|---|---|
| Pipeline Runner | 调度Stage执行 |
| Pipeline Context | 保存运行环境 |
| Config Manager | 加载配置 |
| Model Manager | 管理模型 |
| Logger | 记录运行状态 |
| Storage Manager | 管理输入输出 |

---

# 1.9 Design Constraints

## Constraint 1

所有参数：

必须来自：

`configs/config.yaml`

---

## Constraint 2

模型权重：

统一管理：

`models/`

---

## Constraint 3

第三方代码：

统一管理：

`third_party/`

---

## Constraint 4

Pipeline输出：

必须保持：

Video

Face

FLAME

Gaze

Audio

之间的时间同步。

---

## Constraint 5

所有Stage：

必须支持：

- Logging
- Error Handling
- Reproducible Execution

---

## Constraint 6 — Stage Processing Metadata Principle

所有Stage：

必须生成Per-Sample Metadata：

附加在输出Data Object上。

---

禁止：

任何Stage内部：

生成聚合统计报告。

---

Stage 15 Quality Statistics：

是唯一负责：

- 跨Stage数据统计
- 过滤比例分析
- Pipeline质量报告

的Stage。

---

例如：

Stage 3 对每个SceneInfo生成：

- CoarseFaceInfo（检测结果）
- SceneFilterResult（过滤决策）

但不统计：

- 删除Scene总数
- 总时长损失

这些由 Stage 15 统一完成。

---

# 1.10 Final Architecture Summary

`Face_Pipeline_V2`

采用：

Layered Pipeline Architecture

---

核心结构：

`Input Video`

↓

`Video Processing`

↓

`Face Filtering`

↓

`Quality Assessment`

↓

`Face Normalization`

↓

`Feature Extraction`

↓

`Feature Fusion`

↓

`Training Dataset`

---

系统目标：

构建一个：

可扩展

可维护

GPU高效

AI可生成代码

的高质量Talking Face数据生产系统。


---

# End of 01_System_Architecture.md v1.0
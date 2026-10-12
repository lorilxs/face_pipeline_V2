# 06_Engineering_Guideline.md

# AI-Friendly Software Design Document (AISDD)

# Part 6

# Engineering Guideline

Version: v1.0

Last Update: 2026-08-03

---

# 6.1 Engineering Guideline Overview

`Face_Pipeline_V2`

采用：

Production-Level Python Pipeline Engineering Architecture。

---

本章节定义：

- Python工程结构规范
- Coding规范
- Module设计规范
- Stage开发规范
- Logging规范
- Error Handling规范
- Testing规范
- Performance优化规范
- AI Coding Agent开发规范

---

设计目标：

- 保证Pipeline长期维护能力
- 保证新增Stage可扩展
- 保证代码结构清晰
- 保证模型和业务逻辑解耦
- 支持Claude等AI Coding Agent自动生成代码

---

# 6.2 Project Engineering Architecture


`Face_Pipeline_V2`

采用：

Python Package Architecture。

---

禁止：

简单Script结构。

---

推荐工程结构：

| Directory | 中文名称 | Field Description |
|---|---|---|
| face_pipeline_v2/ | 主Package | Pipeline核心代码 |
| stages/ | Stage模块 | 每个Pipeline阶段实现 |
| data/ | 数据模型 | Data Object定义 |
| models/ | 模型管理 | Model Wrapper |
| configs/ | 配置管理 | YAML配置 |
| utils/ | 工具模块 | 通用组件 |
| third_party/ | 第三方代码 | 外部依赖 |
| tests/ | 测试代码 | 单元测试和集成测试 |
| scripts/ | 工具脚本 | 执行入口 |

---

# 6.3 Module Design Guideline


# 6.3.1 Module Responsibility


每个Module：

必须：

单一职责。

---

例如：

正确：

| Module | Responsibility |
|---|---|
| FaceDetector | 人脸检测 |
| PoseEstimator | 姿态估计 |
| QualityEvaluator | 质量评估 |
| ModelManager | 模型加载 |

---

禁止：

一个Module包含：

- 模型加载
- 数据处理
- 文件保存
- Pipeline控制

---

# 6.3.2 Dependency Rule


模块依赖方向：
Stage

↓

Service / Model Wrapper

↓

Third Party Model


---

禁止：

Third Party代码：

直接调用Pipeline代码。

---

# 6.4 Stage Development Guideline


# 6.4.1 Stage Interface


每个Stage：

必须实现统一接口。


---

Stage结构：

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| stage_name | Stage名称 | str | 当前Stage名称 |
| input_type | 输入类型 | Type | 输入Data Object类型 |
| output_type | 输出类型 | Type | 输出Data Object类型 |
| config | 配置 | Config | 当前Stage配置 |
| logger | 日志对象 | Logger | Stage日志 |

---

# 6.4.2 Stage Execution Rule


每个Stage：

必须包含：

| Function | Purpose |
|---|---|
| validate_input | 输入检查 |
| process | 核心处理逻辑 |
| validate_output | 输出检查 |
| save_result | 保存结果 |

---

# 6.4.3 Stage Isolation


Stage之间：

禁止共享：

- Global Variable
- Temporary Cache
- Model Instance


---

所有共享资源：

必须通过：

Context

传递。

---

# 6.4.4 Metadata Flow Design

## Principle

每个Stage：

只生成Per-Sample Metadata。

不生成聚合统计。

---

统计职责：

唯一归属：

Stage 15 Quality Statistics。

---

## Metadata Map

| Stage | Metadata | Consumer |
|---|---|---|
| Stage 3 | CoarseFaceInfo + SceneFilterResult | Stage 15 |
| Stage 5 | FaceInfo (per-frame) | Stage 15 |
| Stage 6 | PoseInfo | Stage 15 |
| Stage 7 | AVSyncInfo | Stage 15 |
| Stage 8 | FaceQualityInfo | Stage 15 |
| Stage 15 | 收集所有Metadata | StatisticsInfo |

---

## Design Rule

Stage模块：

不持有全局统计状态。

---

示例：

Stage 3 输出：

每个SceneInfo携带：
- CoarseFaceInfo
- SceneFilterResult

但Stage3内部不统计：
- 删除Scene数量
- 保留比例
- 时长损耗

---

Stage 15：

读取所有SceneFilterResult：

使用Pandas分析：

- 按reject_reason分组
- 计算通过率
- 统计时长损耗
- 生成Pipeline质量报告

---

# 6.5 Data Model Engineering Guideline


# 6.5.1 Immutable Data Object


数据模型：

采用：

Dataclass。


---

要求：

- frozen=True
- 明确Type Annotation
- 禁止动态字段


---

示例规范：

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| video_id | 视频ID | str | 视频唯一标识 |
| frame_index | 帧编号 | int | 当前视频帧编号 |
| timestamp | 时间戳 | float | 时间位置 |

---

# 6.5.2 Data Validation


所有Data Object：

必须提供：

输入合法性检查。

---

检查内容：

- 类型
- 空值
- 范围
- 文件存在性

---

# 6.6 Configuration Engineering Guideline


# 6.6.1 Configuration Driven Design


所有参数：

必须来自：

`config.yaml`

---

禁止：

代码硬编码：

- Threshold
- Path
- Model Name
- Batch Size

---

# 6.6.2 Configuration Access


推荐：

统一访问：

`PipelineContext.config`

---

禁止：

多个模块：

重复读取：

`config.yaml`

---

# 6.7 Model Wrapper Guideline


# 6.7.1 Model Abstraction


所有模型：

必须封装：

Model Wrapper。


---

例如：

| Model | Wrapper |
|---|---|
| SCRFD | SCRFDDetector |
| 6DRepNet | PoseEstimator |
| HyperIQA | QualityEvaluator |
| EMICA | FLAMEExtractor |
| HuBERT | AudioFeatureExtractor |

---

# 6.7.2 Model Loading Rule


模型加载：

统一通过：

`ModelManager`。


---

禁止：

Stage内部：

直接：

load checkpoint。


---

# 6.8 Logging Guideline


# 6.8.1 Logging Principle


所有Stage：

必须记录：

运行状态。


---

日志包含：

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| timestamp | 时间 | str | 日志时间 |
| stage_name | Stage名称 | str | 当前Stage |
| level | 日志等级 | str | INFO/WARNING/ERROR |
| message | 信息 | str | 日志内容 |

---

# 6.8.2 Log Level


支持：

| Level | Usage |
|---|---|
| DEBUG | Debug信息 |
| INFO | 正常流程 |
| WARNING | 非致命问题 |
| ERROR | 错误 |

---

# 6.9 Error Handling Guideline


# 6.9.1 Error Classification


错误分为：

| Type | Description |
|---|---|
| Configuration Error | 配置错误 |
| Model Error | 模型错误 |
| Data Error | 数据错误 |
| Runtime Error | 运行错误 |

---

# 6.9.2 Error Handling Rule


错误处理：

必须：

- 捕获异常
- 记录日志
- 保存状态
- 返回明确错误


---

禁止：

直接：

`except Exception: pass`

---

# 6.10 Performance Optimization Guideline


# 6.10.1 GPU Utilization


GPU Stage：

包括：

| Stage | Model |
|---|---|
| Stage 6 | 6DRepNet |
| Stage 7 | SyncNet |
| Stage 8 | HyperIQA |
| Stage 12 | EMICA + SPECTRE |
| Stage 14 | HuBERT |

---

要求：

- Batch Processing
- Model Reuse
- GPU Memory Management

---

Progressive Gating 原则：

- 廉价规则过滤 Stage（5.2 / 5.3）先于昂贵 GPU 模型推理（Stage 6 / 7 / 8）。
- 过滤 Stage 应减少下游 GPU 推理量：未通过的 sequence 不进入 GPU Stage。
- 目标：避免无效 clip 消耗 GPU 推理资源。

---

# 6.10.2 Parallel Execution


支持并行：

Quality Pipeline：

| Stage | Description |
|---|---|
| Stage 6 | Pose |
| Stage 7 | Sync |
| Stage 8 | Quality |

---

Feature Pipeline：

| Stage | Description |
|---|---|
| Stage 12 | FLAME（EMICA） |
| Stage 13 | Gaze |
| Stage 14 | Audio |
| Stage 15 | Statistics |
| Stage 16 | Annotation |

---

# 6.10.3 Memory Management


要求：

- 避免一次加载全部视频
- 使用Batch处理
- 及时释放GPU Tensor

---
# 6.10.4 Stage 12 EMICA Hard Constraints

Stage 12 使用 **EMICA** 作为 FLAME 参数提取模型，以下为硬约束：

### 禁止行为

1. **禁止将 EMICA expression 理解为 EMOCA expcode**
   - EMICA ExpressionEncoder 输出 100 维，独立训练
   - EMOCA expcode 为 50 维，不同 latent space
   - 两者完全独立，互不依赖

2. **禁止使用 `expcode[:50]` 截取**
   - 从 EMICA 100 维截取前 50 维不具有 EMOCA 语义
   - 不能替代 EMOCA 50 维

3. **禁止 zero-padding EMOCA 50 维到 100 维**
   - 100 维是 EMICA ExpressionEncoder 的真实预测值
   - 不是 `[50]` + `[50]` zero-padding

4. **禁止假设 Stage12 输出固定 112 维**
   - 112 只是默认配置下的结果（expression[100] + pose[6] + eye_pose[6]）
   - 模块开关可改变输出维度
   - 下游必须使用字段访问（如 `feature.expression_code`），禁止索引访问（如 `feature[0:100]`）

### 必须遵循

1. **必须使用结构化字段访问**
   - `FLAMEFeature.expression_code` 而非 `feature[0:100]`
   - `FLAMEFeature.pose_code` 而非 `feature[100:106]`
   - `FLAMEFeature.eye_pose` 而非 `feature[106:112]`

2. **必须通过 config 控制模块开关**
   - `stage_12.emica.modules.expression_encoder.enabled`
   - `stage_12.emica.modules.spectre.enabled`
   - `stage_12.emica.modules.mica.enabled`
   - `stage_12.emica.modules.deca.enabled`

3. **flatten 操作必须由 Stage 17 负责**
   - Stage 12 只输出结构化 FLAMEFeature
   - 禁止在 Stage 12 内部 flatten 成固定维度向量

# 6.11 Testing Guideline


# 6.11.1 Unit Test


每个核心模块：

必须有：

Unit Test。


---

测试：

| Component | Test |
|---|---|
| Data Model | 字段合法性 |
| Config | 参数读取 |
| Model Wrapper | 推理输出 |
| Stage | 输入输出 |

---

## Stage 3 Test Requirements

Stage 3测试必须验证：

1. CoarseFaceInfo生成正确：
   - sampled_frame_count = 实际采样帧数
   - face_detect_ratio正确计算
   - single_face_ratio正确计算
   - avg_face_confidence正确计算
   - max_face_count正确记录

2. SceneFilterResult生成正确：
   - passed正确判断
   - reject_reasons支持多个原因
   - 原因枚举正确：too_short / no_face / multi_face / low_confidence

3. Stage 3不生成：
   - 聚合统计报告
   - stage3_statistics.json
   - Stage3Report

---

# 6.11.2 Integration Test


Pipeline级测试：

验证：

- Stage连接
- 数据传递
- 输出完整性

---

## Stage 3 → Stage 15 集成测试

验证：

Metadata流转完整性：

---

测试点：

1. Stage 3输出的SceneInfo携带CoarseFaceInfo + SceneFilterResult
2. Stage 4只处理passed=True的SceneInfo
3. Stage 15能收集所有SceneFilterResult
4. Stage 15 Pandas分析正确计算通过率/时长损耗
5. reject_reasons分发统计正确

---

# 6.12 Code Style Guideline


# 6.12.1 Naming Convention


| Object | Rule |
|---|---|
| Class | PascalCase |
| Function | snake_case |
| Variable | snake_case |
| Constant | UPPER_CASE |

---

# 6.12.2 Type Annotation


所有函数：

必须声明：

Input Type

和：

Output Type。


---

# 6.12.3 Documentation


所有Public Class：

必须包含：

Docstring。


---

# 6.13 AI Coding Agent Guideline


# 6.13.1 Code Generation Principle


AI Coding Agent：

必须基于：

AISDD文档

生成代码。


---

生成顺序：

| Step | Action |
|---|---|
| 1 | Read Data Model |
| 2 | Read Stage Specification |
| 3 | Implement Interface |
| 4 | Implement Model Wrapper |
| 5 | Add Test |

---

# 6.13.2 Forbidden Behavior


AI Coding Agent禁止：

- 修改Data Structure
- 修改Stage顺序
- 添加未定义依赖
- 引入Magic Number
- 绕过Config


---

# 6.14 Version Management


# 6.14.1 Version Tracking


所有：

- Config
- Model
- Code
- Dataset

必须记录版本。


---

# 6.14.2 Metadata Version


Training Sample：

必须保存：

| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| pipeline_version | Pipeline版本 | str | 当前Pipeline版本 |
| config_version | 配置版本 | str | 配置版本 |
| model_version | 模型版本 | dict | 使用模型版本 |

---

# 6.15 Engineering Summary


`Face_Pipeline_V2`

工程设计原则：

- Python Package Architecture
- Configuration Driven Design
- Model Wrapper Abstraction
- Immutable Data Flow
- Stage Interface Standardization
- Unified Logging
- Error Handling
- GPU Parallel Processing
- Test Driven Development
- AI Coding Friendly Structure
- Subprocess Isolation (跨环境调用)

---

# 6.15.1 Stage12 Subprocess Runtime

## Container & Environment

Stage12 通过 Docker subprocess 运行，与 Stage0-11 的 host Python 环境隔离：

```bash
docker exec -i face_pipeline_lxs bash -c \
  "source /workspace/emica_env/bin/activate && \
   python face_pipeline_v2/stages/stage_12_flame/runner.py \
     --input_metadata /tmp/.../input_metadata.json \
     --output_dir outputs/stage12_flame \
     --emica_checkpoint ... --spectre_checkpoint ..."
```

## Subprocess Flow

1. `Stage12FLAME.process()` 序列化 `AlignedFaceInfo` → JSON metadata
2. `invoke_stage12_runner()` 通过 `docker exec` 调用 runner.py
3. `read_stage12_outputs()` 读取 `flame_features.npz` → `List[FLAMEFeature]`

## Wrapper Design Principle

Stage12 通过 wrapper 适配第三方模型，不修改 `third_party/` 中的任何源码：

- **EMICAWrapper**: 封装 EmicaEncoder，管理 checkpoint 加载和模块开关
- **SPECTREWrapper**: 封装 SPECTRE E_expression，仅加载 encoder 部分

Wrapper 核心原则：
1. 不修改第三方源码
2. 向后兼容（不重命名 FLAMEFeature 现有字段）
3. EMICA 为主模型，SPECTRE 为可选增强

---

# 6.15 Engineering Summary

`Face_Pipeline_V2`

工程设计原则：

- Python Package Architecture
- Configuration Driven Design
- Model Wrapper Abstraction
- Immutable Data Flow
- Stage Interface Standardization
- Unified Logging
- Error Handling
- GPU Parallel Processing
- Test Driven Development
- AI Coding Friendly Structure
- Subprocess Isolation (跨环境调用)


---

# End of 06_Engineering_Guideline.md v1.0
# Face_Pipeline_V2 文档结构

## 核心设计文档 (00-06)

| 编号 | 文件 | 内容 |
|------|------|------|
| 00 | `00_Project_Overview.md` | 项目概述、设计目标、分层架构、Stage 概览 |
| 01 | `01_System_Architecture.md` | 系统架构、数据流、约束条件、序列图 |
| 02 | `02_Data_Model_and_Interface.md` | 数据模型定义、接口规范、Stage12 FLAMEFeature、下游兼容性 |
| 03 | `03_Pipeline_Stage_Specification.md` | 各 Stage 详细规格、Stage12 输入输出、Runtime Mode、输出文件结构 |
| 04 | `04_Config_and_Model_Management.md` | 配置管理、模型管理、Stage12 checkpoint/wrapper/环境 |
| 05 | `05_Data_Flow_and_Output.md` | 数据流、输出格式、Stage12 NPZ 结构、Stage12→Stage17 映射 |
| 06 | `06_Engineering_Guideline.md` | 工程规范、Stage12 subprocess 运行时、wrapper 设计原则 |

## 归档目录

### `archive/stage12/`

Stage12 开发过程中的验证和设计文档，核心设计信息已合并到 00-06 核心设计文档。

| 文件 | 说明 |
|------|------|
| `Stage12_Design_Finalization_Report.md` | Stage12 设计冻结报告 |
| `Stage12_Document_Merge_Plan.md` | 文档合并计划 |
| `Stage12_Downstream_Compatibility_Report.md` | 下游兼容性分析 |
| `Stage12_EMICA_PreIntegration_Report.md` | EMICA 预集成分析 |
| `Stage12_EMICA_Wrapper_Compatibility_Report.md` | EMICA Wrapper 兼容性审查 |
| `Stage12_Full_Pipeline_Runtime_Report.md` | 全 Pipeline 运行时报告 |
| `Stage12_Interface_Design.md` | Stage12 接口设计 |
| `Stage12_Interface_Design_Review.md` | 接口设计审查 |
| `Stage12_Minimal_Inference_Check_Report.md` | 最小推理检查 |
| `Stage12_Minimal_Module_Loading_Analysis.md` | 最小模块加载分析 |
| `Stage12_Model_Interface_Audit.md` | 模型接口审计 |
| `Stage12_Pipeline_Integration_Report.md` | Pipeline 集成报告 |
| `Stage12_Real_Data_Validation_Report.md` | 真实数据验证 |
| `Stage12_Runtime_Environment_Analysis.md` | 运行时环境分析 |

## 文档维护原则

- **00-06**: 核心设计文档，长期维护，保持最新
- **archive/**: 历史验证记录，只读归档，不再修改
- 新增设计信息应合并到 00-06 对应文档，而非创建独立文档
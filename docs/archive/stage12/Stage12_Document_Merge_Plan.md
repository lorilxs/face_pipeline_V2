# Stage12 Document Merge Plan

**Date**: 2025-08-26
**Status**: Pending Review

---

## Classification Summary

| # | File | Lines | Category | Action |
|---|------|-------|----------|--------|
| 1 | Stage12_Interface_Design.md | 196 | A | Merge → 02, 03 |
| 2 | Stage12_Downstream_Compatibility_Report.md | 344 | A | Merge → 02, 03, 05 |
| 3 | Stage12_Model_Interface_Audit.md | 331 | A | Merge → 03, 04 |
| 4 | Stage12_Minimal_Module_Loading_Analysis.md | 411 | A | Merge → 04 |
| 5 | Stage12_EMICA_Wrapper_Compatibility_Report.md | 212 | A | Merge → 04 |
| 6 | Stage12_Design_Finalization_Report.md | 149 | A | Merge → 03 |
| 7 | Stage12_Pipeline_Integration_Report.md | 117 | A | Merge → 03, 04, 05 |
| 8 | Stage12_Full_Pipeline_Runtime_Report.md | 335 | A | Merge → 03, 04, 05 |
| 9 | Stage12_Runtime_Environment_Analysis.md | 100 | A | Merge → 04, 06 |
| 10 | Stage12_EMICA_PreIntegration_Report.md | 781 | B | Archive |
| 11 | Stage12_Interface_Design_Review.md | 227 | B | Archive |
| 12 | Stage12_Minimal_Inference_Check_Report.md | 104 | B | Archive |
| 13 | Stage12_Real_Data_Validation_Report.md | 80 | B | Archive |

---

## Detailed Merge Plan — Category A

### A1. Stage12_Interface_Design.md → 02 + 03

| Section | Content | Target |
|---------|---------|--------|
| 1.1 字段定义 | FLAMEFeature fields, dims, defaults | 02 |
| 1.2 字段汇总表 | Summary table | 02 |
| 2. 数据流 | Stage12→Stage17 diagram | 02 |
| 3. Jaw Fusion | Formula, fallback matrix | 03 |
| 4. 配置接口 | stage12 config structure | 03 |
| 5. 实现约束 | Constraints | 03 |

### A2. Stage12_Downstream_Compatibility_Report.md → 02 + 03 + 05

| Section | Content | Target |
|---------|---------|--------|
| 2.1 FLAMEFeature Dataclass | Current definition with metadata | 02 |
| 2.2 Runner Output Dict | npz keys | 05 |
| 2.3 Subprocess Bridge | integration.py | 03 |
| 3.1 Stage13 Gaze | GazeFeature interface | 02 |
| 3.2 Stage14 Audio | AudioFeature, timestamp req | 02 |
| 3.4 Stage17 TrainingSample | Field mapping | 02 |
| 5. Compatibility Matrix | Complete matrix | 02 |
| 6. Data Flow | Current vs Recommended | 02 |

### A3. Stage12_Model_Interface_Audit.md → 03 + 04

| Section | Content | Target |
|---------|---------|--------|
| 1. EMOCA/EMICA | Encoder data flow, output fields | 03 |
| 2. SPECTRE | Input/output, codedict | 03 |
| 3. MICA | Inference entry, output | 04 |
| 4. DECA | Output fields | 04 |
| 1.5 eye_pose | Audit: zeros | 03 |

### A4. Stage12_Minimal_Module_Loading_Analysis.md → 04

| Section | Content | Target |
|---------|---------|--------|
| 1. 输出定义 | Required vs optional | 04 |
| 2. 依赖映射 | Output→module mapping | 04 |
| 3. 最小加载 | Minimal loading list | 04 |
| 4. EMICA 源码 | Module loading analysis | 04 |
| 5. SPECTRE 源码 | SPECTRE loading | 04 |
| 6. 重复计算 | Dedup analysis | 04 |

### A5. Stage12_EMICA_Wrapper_Compatibility_Report.md → 04

| Section | Content | Target |
|---------|---------|--------|
| 2. Encode 输出 | Output dict, alignment | 04 |
| 3. 不必要模块 | Module tree, MICA/FLAME | 04 |
| 4. Checkpoint | Loading flow, MICA path | 04 |
| 5.3 问题清单 | Known issues | 04 |

### A6. Stage12_Design_Finalization_Report.md → 03

| Section | Content | Target |
|---------|---------|--------|
| 1. Architecture | EMICA+SPECTRE dual model | 03 |
| 2. Tensor Dims | All dimension specs | 03 |
| 4. Jaw Refinement | Formula, λ | 03 |
| 5. Config Design | Config structure | 03 |
| 6. Key Decisions | expcode, eye_pose, flatten, shape | 03 |

### A7. Stage12_Pipeline_Integration_Report.md → 03 + 04 + 05

| Section | Content | Target |
|---------|---------|--------|
| 1. Call Chain | Stage0→12 flow | 03 |
| 2. Runtime | Direct vs subprocess | 03 |
| 3. Model Loading | Pipeline loading | 04 |
| 6. Output Structure | npz/npy format | 05 |
| 7. Limitations | Known limits | 03 |

### A8. Stage12_Full_Pipeline_Runtime_Report.md → 03 + 04 + 05

| Section | Content | Target |
|---------|---------|--------|
| 2. Data Flow | Stage11→12 alignment | 03 |
| 3. Invocation | Driver, config, subprocess | 03 |
| 4. Docker | Container, Python env | 04 |
| 5. Checkpoints | EMICA, SPECTRE paths | 04 |
| 6. Inference | Pipeline, batch | 03 |
| 7. Output | Output files | 05 |

### A9. Stage12_Runtime_Environment_Analysis.md → 04 + 06

| Section | Content | Target |
|---------|---------|--------|
| 1. Launch | Host, Docker, entry | 04 |
| 2. Python Path | sys.path setup | 04 |
| 3. Dependencies | EMICA/SPECTRE deps | 04 |
| 5. Recommendation | Strategy | 06 |

---

## Archive Documents (Category B) → docs/archive/stage12/

| File | Reason |
|------|--------|
| Stage12_EMICA_PreIntegration_Report.md | Pre-integration asset check (historical) |
| Stage12_Interface_Design_Review.md | Design review discussion (historical) |
| Stage12_Minimal_Inference_Check_Report.md | Checkpoint validation (one-time) |
| Stage12_Real_Data_Validation_Report.md | Real data validation (one-time) |
---

## Target 00-06: New Sections to Add

### 02_Data_Model_and_Interface.md
- `## Stage12: FLAMEFeature` — full field definition, timestamp, clip_id
- `## Stage13: GazeFeature` — gaze vector, quaternion interface
- `## Stage14: AudioFeature` — timestamp requirement for sync
- `## Stage17: TrainingSample Field Mapping` — flame_feature, posecode, gaze_feature
- `## Downstream Compatibility Matrix` — field compatibility across stages

### 03_Pipeline_Stage_Specification.md
- `## Stage 12: FLAME Parameter Extraction` — complete spec
  - Architecture: EMICA + SPECTRE dual model
  - Input/Output interfaces
  - Jaw Fusion: formula, λ parameter, fallback matrix
  - Tensor Dimensions: all output dims
  - Key Design Decisions: expcode 100d, eye_pose zeros, flatten, shape_code
  - Config Structure: stage12 config block
  - Pipeline Integration: call chain, direct vs subprocess mode
  - Known Limitations

### 04_Config_and_Model_Management.md
- `## Stage12: EMICA Model` — wrapper, encode() output, checkpoint, MICA sub-model
- `## Stage12: SPECTRE Model` — wrapper, output fields, checkpoint
- `## Stage12: Minimal Module Loading` — dependency map, loading strategy
- `## Stage12: Docker Environment` — container, Python path, dependencies
- `## Stage12: Known Issues` — MICA path, unnecessary module loading

### 05_Data_Flow_and_Output.md
- `## Stage12 Output: npz Structure` — all keys, per-frame format
- `## Stage12 Output: npy Files` — per-clip directory structure
- `## Stage12 Runner Output Dict` — result dict keys and types

### 06_Engineering_Guideline.md
- `## Stage12: Subprocess Runtime Guide` — Docker, environment, entry point
- `## Stage12: Wrapper Design Principles` — EMICAWrapper, SPECTREWrapper patterns

---

## Execution Order

1. Create `docs/archive/stage12/` directory
2. Move 4 archive files (B1-B4) to `docs/archive/stage12/`
3. Merge content from 9 Category A files into target 00-06 files
4. Remove the 9 merged files from `docs/`
5. Create `docs/DOCUMENTATION_STRUCTURE.md`
6. Verify final structure

---

## Post-Merge docs/ Structure

```
docs/
├── 00_Project_Overview.md
├── 01_System_Architecture.md
├── 02_Data_Model_and_Interface.md          ← +Stage12 FLAMEFeature, GazeFeature, TrainingSample mapping
├── 03_Pipeline_Stage_Specification.md      ← +Stage12 full spec (architecture, jaw fusion, config, pipeline)
├── 04_Config_and_Model_Management.md       ← +Stage12 EMICA/SPECTRE models, checkpoint, minimal loading, Docker
├── 05_Data_Flow_and_Output.md              ← +Stage12 output format (npz, npy, runner dict)
├── 06_Engineering_Guideline.md             ← +Stage12 subprocess guide, wrapper patterns
├── Stage12_Document_Merge_Plan.md          ← This plan (temporary)
├── DOCUMENTATION_STRUCTURE.md              ← NEW: documentation overview
└── archive/
    └── stage12/
        ├── Stage12_EMICA_PreIntegration_Report.md
        ├── Stage12_Interface_Design_Review.md
        ├── Stage12_Minimal_Inference_Check_Report.md
        └── Stage12_Real_Data_Validation_Report.md
```
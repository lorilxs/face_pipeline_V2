 # Development Log

---

## 2026-09-01 — Stage10 Landmark Temporal EMA (α=0.7) 消融实验（Mesh 抖动根因验证系列）

### 变更范围
仅新增实验脚本与输出（`scripts/stage10_landmark_temporal_ema_ablation.py`、`scripts/run_single_video_dataset_landmark_ema_ablation.sh`、`outputs/ablation_landmark_ema/`、`docs/stage10_landmark_temporal_ema_ablation/`）。**未修改任何生产 Stage10/Stage12/EMICA/FLAME/Stage17 代码与模型权重。**

### 背景
Stage5 bbox EMA 消融已证明 bbox 平滑本身不损害 FLAME/Gaze/Audio；剩余 head/jaw 抖动主要来自 Stage10 106 landmark 逐帧检测噪声 + FLAME fitting。本实验验证对 Stage10 landmark 施加因果 EMA（α=0.7）能否降低抖动同时保持真实运动。

### 方法
- **Baseline**：Stage10 raw 106 landmark，无平滑。
- **EMA**：`landmark_ema[t] = 0.7·landmark_raw[t] + 0.3·landmark_ema[t-1]`，frame0=raw；仅作用于 [106,2] 坐标，不改 bbox/模型；不使用 Butterworth / filtfilt / 未来帧。
- 两分支共用 raw SCRFD bbox（`docs/stage5_roi_ablation/baseline_results.json`）与 Stage5 crop；仅 landmark 不同 → Stage11 仿射对齐不同 → Stage12-17 全链路经官方 runner 重跑。

### 结果（200 帧，1 clip）
- landmark jitter（全部106）：1.752→1.484px（**−15.3%**）；max 4.60→3.71px；P95 3.41→2.98px；5 区域（contour/eyebrow/eye/nose/mouth）均降 13.5–17.2%。
- motion energy retention **0.847**（≥0.8，无过度平滑）；mouth opening mean retention **0.999**（曲线相关系数 0.982）；jaw amplitude retention **1.000**。
- FLAME motion ratios：expression **0.996** / head **0.942** / jaw **0.995**（全部 ≥0.9 保持）。
- Stage17 layout_ok=True（两分支，visual_feature [200,112]）。
- 结论：**推荐进入生产**。

### 产物
- 脚本：`scripts/stage10_landmark_temporal_ema_ablation.py`、`scripts/run_single_video_dataset_landmark_ema_ablation.sh`
- 两分支 Stage10-17：`outputs/ablation_landmark_ema/{baseline,ema_alpha07}/`（含 `visual_feature.npy`）
- 指标：`outputs/ablation_landmark_ema/comparison/{ablation_metrics.json,jitter_metrics.json,motion_preservation.json}`
- overlay 视频×2：`docs/stage10_landmark_temporal_ema_ablation/videos/{baseline_landmark_overlay.mp4,ema07_landmark_overlay.mp4}`
- 报告：`docs/Stage10_Landmark_Temporal_EMA_Ablation_Report.md`

### 未修改
生产 pipeline 代码 / 模型 / Stage17 layout / config 均未改动。

---

## 2026-09-01 — Stage12 EMICA vs SPECTRE Full Parameter Source Ablation (AISDD)

### 变更范围
为 Stage12 增加 SPECTRE 完整 FLAME 参数提取变体实验（Experiment B: 全 spectre），
生产默认（Experiment A: 全 emica）保持不变。

### 代码变更

| File | Change |
|---|---|
| `face_pipeline_v2/models/wrappers/spectre.py` | 新增 expression 提取：E_flame latent slice `parameters[:,150:200]`（50 维）zero-pad 至 100 维输出 |
| `face_pipeline_v2/stages/stage_12_flame/processor.py` | 新增 `expression_source`/`head_source`/`jaw_source` 真实来源切换 + `validate_sources()`（禁止 mixed source）+ `_pad_expression()` |
| `face_pipeline_v2/stages/stage_12_flame/runner.py` | 同来源切换逻辑（subprocess 路径）+ 元数据真实记录三 source + CLI 三参数 |
| `face_pipeline_v2/stages/stage_12_flame/stage.py` | 从 config 读取三 source 并透传 subprocess/direct 两条路径 |
| `face_pipeline_v2/stages/stage_12_flame/integration.py` | `invoke_stage12_runner` 传递三 source |
| `scripts/stage0_17_e2e_stage12.py` | 新增 `--expression_source`/`--head_source`，`--jaw_source` 语义改为真实切换 |
| `face_pipeline_v2/stages/stage_17_fusion/export.py` | 新增 `SOURCE_ALLOWED` 来源校验（emica/spectre/mediapipe/chinese_hubert） |
| `configs/config.yaml`, `configs/config_dataset_mode.yaml` | stage12 新增 `expression_source`/`head_source`（默认 emica） |
| `configs/config_dataset_mode_spectre.yaml` | 新增 Experiment B 配置（三 source = spectre） |
| `scripts/run_stage12_emica_vs_spectre_ablation.sh` | 新增实验驱动 |
| `scripts/stage12_emica_vs_spectre_export.py` | 新增 ablation export（真实 source 标签 + 校验） |
| `scripts/generate_stage12_emica_vs_spectre_report.py` | 新增报告生成器 |
| `tests/*` | spectre expression / processor 来源选择 / 禁混源 测试 |

### 实验
- 复用 `outputs/stage0_17_e2e/dataset_validation` aligned_faces（同输入）。
- 输出 `outputs/ablation_stage12_emica_vs_spectre/{emica,spectre}/...`。
- 报告 `docs/Stage12_EMICA_vs_SPECTRE_Full_Ablation_Report.md`。

### 重要语义说明
- SPECTRE 无独立 ExpressionEncoder：expression 为 E_flame 输出的 50 维 latent slice，
  后 50 维 zero-padding 至 100。与 EMICA 专用 encoder 语义不同，稳定性对比只能说明
  "sequence temporally smoother"，不能说明 accuracy 更高。

### 未修改
- 生产默认 config（三 source = emica）、`visual_feature_dim=112`、feature layout、gaze/audio 接口。

---



## 2026-08-29 — Stage4 Dynamic Clip Design Update (Documentation Phase)

### 变更范围
本阶段**仅更新设计文档**；未修改 pipeline 代码 / `configs/config.yaml` / stage 实现 / tests。

### 背景
- 审计（`docs/Stage4_Removal_Impact_Audit.md`）确认：Stage4 固定 8s window + 4s stride 并非下游 Stage5-17 的硬性依赖，Stage5-17 均支持动态长度 T。
- 新训练需求：Stage3 输出完整 single-face segment（如 16s）→ 1 个 clip → Stage5-17 → `TrainingSample(T=400)`。
- 设计决策：不删除 Stage4，而是重新定义为「Clip Generation / Clip Adapter Stage」，支持双模式（`sliding_window` / `segment`）。

### 文档变更
- **新增** `docs/Stage4_Dynamic_Clip_Design_Update.md`：Stage4 重新定义、ClipInfo 动态 duration 数据契约、双模式配置设计、数据流更新、设计决策记录（为什么不删除 Stage4）。
- **更新** `docs/01_System_Architecture.md`：Stage 4 从「Sliding Window Cropping / 生成固定长度 Clip」改为「Clip Generation / Clip Adapter（sliding_window | segment）」。
- **更新** `docs/05_Data_Flow_and_Output.md`：Stage 4 数据流（表 5.2 / 5.4.1 / 5.13）改为双模式描述。
- **更新** `docs/03_Pipeline_Stage_Specification.md` §3.7：Stage 4 规格改为双模式（默认 `sliding_window`：8s / 4s；`segment`：整段 1:1 ClipInfo，动态时长）。
- **更新** `PROJECT_CONTEXT.md`：数据流图、Per-Stage Config 表、Design Decisions 中的「8s clips, 4s stride」同步为双模式描述。

### 未修改（按要求保持原样）
- pipeline 代码、`configs/config.yaml`、stage 实现、tests —— 未做任何改动。

---

## 2026-08-24 — Stage12 Pre-Integration Asset Validation (COMPREHENSIVE)

### 验证范围
对 Stage12 所需的所有 third-party 模型、checkpoints、源码、依赖包进行穷举式验证，识别所有缺失项、路径不匹配、语义冲突。

### 1. 模型权重文件状态

| 模型 | 路径 | 大小 | 状态 |
|------|------|------|------|
| SPECTRE checkpoint | `third_party/inferno/spectre/spectre/data/deca_model.tar` | 415MB | ✅ 存在 |
| FLAME geometry | `models/FLAME/geometry/generic_model.pkl` | 51MB | ✅ 存在 |
| EMICA checkpoint | `models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/` | 828MB | ✅ 存在 |
| MICA checkpoint | `models/MICA/model/mica.tar` | — | ✅ 存在 |
| EMOCA checkpoints | `models/EMOCA/` | — | ❌ **目录不存在** |
| DECA checkpoint | `models/DECA/` | — | ❌ **目录不存在** |
| EMOCA/DECA assets | `third_party/inferno/inferno-master/assets/` | — | ❌ **空目录** |
| EMOCA/DECA data | `third_party/inferno/inferno-master/data/` | — | ❌ **空目录** |
| HuBERT | `models/hubert/hubert_base_ls960.pt` | ~1.1GB | ✅ 存在 |

### 2. Third-Party 源码完整性

| 组件 | 路径 | 状态 |
|------|------|------|
| inferno-master (DECA/EMOCA) | `third_party/inferno/inferno-master/` | ✅ 完整 |
| SPECTRE | `third_party/inferno/spectre/` | ✅ 完整 |
| EMICA third_party | `third_party/EMICA/` | ❌ **空目录** |
| EMICA emica_env | `third_party/emica_env/` | ❌ **空目录** |

### 3. 路径不匹配问题

| 问题 | 期望路径 | 实际路径 | 严重度 |
|------|----------|----------|--------|
| SPECTRE FLAME2020 | `spectre/data/FLAME2020/generic_model.pkl` | 不存在 | 🔴 CRITICAL |
| inferno-master FLAME | `assets/FLAME/geometry/generic_model.pkl` | `models/FLAME/geometry/` | 🟡 MEDIUM |
| MICA FLAME | `mica/data/FLAME2020/generic_model.pkl` | 不存在 | 🟡 MEDIUM |

解决方案: 创建 symlink `spectre/data/FLAME2020 → models/FLAME/geometry`，并在 Stage12 wrapper 中通过 config 覆盖所有路径。

### 4. EMOCA & DECA Checkpoints 缺失详情

根据 `download_assets.sh` 分析，需下载以下 assets（未完成）：
```
assets/EMOCA/models/
  ├── EMOCA.zip          → EMOCA v1
  ├── EMOCA_v2_mp.zip    → EMOCA v2 (mediapipe)
  ├── EMOCA_v2_lr_mse_20.zip
  ├── EMOCA_v2_lr_cos_1.5.zip
  └── DECA.zip           → DECA backbone
assets/DECA/             → DECA assets (FLAME_texture, etc.)
assets/FaceRecognition/  → VGG Face recognition
assets/FLAME/            → FLAME geometry, texture, mask
```

### 5. Docker 环境缺失依赖

以下包在 Docker `face_pipeline_lxs` (emica_env) 中**全部缺失**：
`smplx`, `trimesh`, `pytorch-lightning`, `omegaconf`, `hydra-core`, `face_alignment`, `mediapipe`, `yacs`, `pytorch3d`, `torchaudio`, `chumpy`, `torchfile`, `kornia`, `gdl`, `scikit-image`, `librosa`

### 6. 🔴 CRITICAL: EMICA vs EMOCA — Expression Encoder 维度纠错

**用户设计文档曾声称**: EMOCA `expcode[100]`，SPECTRE `exp[50]`，维度不一致不可融合

**实际代码验证 (2026-08-24)**:
- EMICA `cfg.yaml`: `predicts` 包含 `expcode(100)` — EMICA ExpressionEncoder 输出 100 维
- `emoca_conf/model/sizes/default.yaml`: `n_exp: 50`
- `emoca_conf/training/emoca.yaml` (coarse & detail): `n_exp: 50`
- `spectre/config.py`: `n_exp: 50`

**结论**: 
- **EMICA** ExpressionEncoder 输出 `expression_code[100]` — 这是 Stage 12 的真正输出
- **EMOCA** (原始 DECA-based expression encoder) 输出 `expcode[50]` — 不是 Stage 12 输出
- FLAME 模型内部 `n_exp: 100` 是参数容量（encoder 50维 + 50维 zero padding for EMOCA path），不是 EMOCA encoder 维度
- 设计文档中混淆了 EMICA 和 EMOCA 的 expression encoder 维度

### 7. posecode 语义一致性验证 ✅

| 模型 | posecode[0:3] | posecode[3:6] | FLAME decoder 对应 |
|------|--------------|--------------|-------------------|
| DECA/EMOCA | jaw | global | `pose_params[:,:3]`=jaw |
| SPECTRE | jaw | global | 同 DECA |

EMOCA 和 SPECTRE 的 posecode 语义一致，无需 swap。DECA 代码中 `globpose = posecode[..., :3]` 变量命名有误导性（实际 posecode[..., :3] 是 jaw），但不影响使用。

### 8. EMOCA/DECA encode() 输出 key 完整清单 (仅供参考，非 Stage 12 输出)

```
shapecode[100]   # shape
texcode[50]      # texture
expcode[50]      # expression (EMOCA original, NOT Stage 12! Stage 12 uses EMICA ExpressionEncoder[100])
posecode[6]      # [jaw(3), global(3)]
cam[3]           # orthographic camera
lightcode[27]    # SH lighting
detailcode[128]  # detail (EMOCA only)
```

### 8b. EMICA 输出 key 清单 (Stage 12 实际输出)

```
expression_code[100]  # EMICA ExpressionEncoder
jawpose[3]            # jaw pose
globalpose[3]         # global pose
eye_pose[6]           # always zeros(6) — EMICA does NOT predict eye_pose
cam[3]                # camera (optional)
shape[300]            # MICA shape code (optional)
texcode[50]           # texture (optional)
lightcode[27]         # SH lighting (optional)
```

### 9. Stage12 实现就绪评估 (Updated 2026-08-24)

| 条件 | 状态 | 行动 |
|------|------|------|
| 源码 | ✅ | inferno-master + SPECTRE 完整 |
| SPECTRE checkpoint | ✅ | deca_model.tar 存在 |
| FLAME 模型 | ✅ | generic_model.pkl 存在 |
| EMICA checkpoint | ✅ | EMICA-CVT_flame2020_notexture 存在 (828MB) |
| EMOCA/DECA checkpoints | ❌ | 需下载 EMOCA/DECA assets (~2-3GB) — 用于 EMICA 的 DECA backbone |
| SPECTRE FLAME2020 path | ❌ | 需创建 symlink |
| Docker 依赖 | ❌ | 需安装 15+ Python 包 |
| 设计文档 EMICA/EMOCA 混淆 | ✅ | 已修正（2026-08-24）：EMICA ExpressionEncoder[100], EMOCA[50] |
| eye_pose 来源确认 | ✅ | 已确认（2026-08-24）：EMICA 不预测 eye_pose，始终 zeros(6) |

### 10. 推荐行动顺序

1. 安装 Docker 依赖: `pip install smplx trimesh pytorch-lightning omegaconf hydra-core face_alignment mediapipe yacs pytorch3d torchaudio chumpy torchfile kornia gdl scikit-image librosa`
2. 创建路径 symlink: `spectre/data/FLAME2020 → models/FLAME/geometry`
3. 下载 EMOCA/DECA assets 到 `third_party/inferno/inferno-master/assets/`（用于 EMICA DECA backbone）
4. 实现 EMICA wrapper: 支持 module enable/disable (MICA/DECA/ExpressionEncoder/SPECTRE)，输出 `expression_code[100]`, `pose_code[6]`, `eye_pose[6]`, optional `shape_code[300]`, optional `camera_parameters`
5. 开始 Stage12 实现

### 修改文件

| 文件 | 变更 |
|------|------|
| `DEVELOPMENT_LOG.md` | 本条目 |

---

## 2026-08-24 — EMICA eye_pose 验证 & 设计文档修正

### 验证目标

确认 EMICA encoder 是否直接预测 eye_pose。

### 验证方法

1. **配置检查**: `models/EMICA/EMICA-CVT_flame2020_notexture/cfg.yaml` — `predicts` 列表包含 `expcode(100), texcode(50), jawpose(3), globalpose(3), cam(3), lightcode(27), shape(300)`，**无 eye_pose 条目**。
2. **源码检查**: `FaceEncoder.py` `_decompose_code()` 方法不处理任何 eye 参数。
3. **Checkpoint 检查**: 仅包含 `shape_model.flame.eye_pose`（固定 zeros buffer）和 `renderer.uv_face_eye_mask`（渲染 mask），**无 encoder 的 eye_pose 权重**。

### 结论

- **EMICA 不预测 eye_pose**。eye_pose 始终为 `zeros(6)`，由 FLAME 模型内部默认值提供。
- 若需要 eye_pose，必须由 Stage 13 (Gaze / MediaPipe) 提供。

### 设计文档修正 (PROJECT_CONTEXT.md & DEVELOPMENT_LOG.md)

1. **EMICA / EMOCA 关系澄清**:
   - EMICA ExpressionEncoder 产生 `expression_code[100]` — 这是 Stage 12 输出
   - EMOCA 原始 expcode: 50 — 不是 Stage 12 输出
   - 删除 "EMOCA expcode[100]" 错误描述

2. **Stage 12 状态更新**: "EMOCA extraction" → "EMICA FLAME extraction"

3. **Models 部分增加**: EMICA modules: MICA, DECA, ExpressionEncoder, SPECTRE

4. **Data Model 更新**: FLAMEFeature 不固定 112d，改为结构化字段

5. **Known Issues 增加**: 需实现 EMICA wrapper，支持 module enable/disable，输出 `expression_code[100]`, `pose_code[6]`, `eye_pose[6]`, optional `shape_code[300]`, optional `camera_parameters`

### 修改文件

| 文件 | 变更 |
|------|------|
| `PROJECT_CONTEXT.md` | 全局 EMOCA→EMICA 替换；Section 10b 重写（EMICA 架构、eye_pose 确认、EMICA vs EMOCA 关系、模块列表）；Known Issues 增加 EMICA wrapper 需求；Latest Development Status 更新 |
| `DEVELOPMENT_LOG.md` | Section 6 重写（EMICA vs EMOCA 维度纠错）；Section 8b 新增（EMICA 输出 key 清单）；Section 9 更新（eye_pose 来源确认）；Section 2026-08-23 设计冻结/集成进度全面更新；本条目新增 |

---

## 2026-08-23 — Stage12 Design Finalized (FLAME Motion Extraction) → Updated 2026-08-24

### 核心决策 (Updated)

1. **expression**: EMICA ExpressionEncoder produces `expression_code[100]` — 这是 Stage 12 的真正输出。EMOCA 原始 `expcode[50]` 不是 Stage 12 输出。
2. **jaw_pose**: EMICA 初始值 + SPECTRE speech-aware refinement (lambda=0.6)
3. **pose_code**: refined jaw[0:3] + global_pose[3:6]
4. **eye_pose**: EMICA 不预测 eye_pose（已确认 2026-08-24），始终 zeros(6)，待 Stage 13 gaze estimator 替换
5. **shape_code**: optional EMICA MICA module 300维
6. **camera**: optional
7. **FLAMEFeature**: 结构化字段，禁止 Stage 12 flatten，由 Stage 17 负责

### 废弃旧设计

- `expcode[:50] + posecode[:6]` 56维固定向量 → 结构化 FLAMEFeature
- EMOCA-only → EMICA + SPECTRE jaw refinement architecture

### 修改文档

| 文档 | 变更 |
|---|---|
| `docs/02_Data_Model_and_Interface.md` | FLAMEFeature 从固定56维改为结构化字段 |
| `docs/03_Pipeline_Stage_Specification.md` | Stage12 完全重写：EMICA+SPECTRE 架构、jaw refinement formula、module switch |
| `docs/04_Config_and_Model_Management.md` | FLAME config 增加 module switch、jaw_refinement config |
| `docs/01_System_Architecture.md` | Stage12 描述更新 |
| `docs/05_Data_Flow_and_Output.md` | FLAME data flow 描述更新 |
| `docs/06_Engineering_Guideline.md` | Stage12 model 更新 |
| `PROJECT_CONTEXT.md` | 全局引用更新：EMOCA→EMICA；新增 EMICA vs EMOCA 关系；eye_pose 确认 |
| `docs/Stage12_Design_Finalization_Report.md` | 新增：设计冻结报告 |

### Implementation Ready: YES

---

## 2026-08-23 — Pipeline Integration Progress (EMICA/SPECTRE/HuBERT/PPU) → Updated 2026-08-24

### 1. EMICA/SPECTRE 融合设计确认 (Updated)

完成 FLAME motion extraction 从 EMOCA-only 到 **EMICA + SPECTRE** jaw refinement 的架构升级：

- **expression**: EMICA ExpressionEncoder 输出 `expression_code[100]`。EMOCA 原始 `expcode[50]` 不是 Stage 12 输出。
- **global_pose[3]**: EMICA 提供
- **jaw_pose[3]**: EMICA 初始值 + SPECTRE speech-aware refinement
- **jaw refinement formula**: `final_jaw = EMICA_jaw + λ × (SPECTRE_jaw - EMICA_jaw)`，λ 默认 0.6，可配置 0.5~0.7
- **pose_code[6]**: `refined_jaw[0:3] + global_pose[3:6]`
- **eye_pose[6]**: EMICA 不预测 eye_pose（已确认），始终 zeros(6)，预留 Stage 13 gaze estimator 替换接口
- **FLAMEFeature**: 结构化 dataclass，flatten 推迟到 Stage 17
- **Module switch**: 通过 config 控制 MICA/DECA/ExpressionEncoder/SPECTRE 开关，默认 ExpressionEncoder=true, SPECTRE=true, MICA=false, DECA=false

### 2. HuBERT Audio Feature 准备

- 模型权重就绪: `models/hubert/hubert_base_ls960.pt` (~1.1GB)
- 目标输出: audio embedding 768 dims per frame，与 FLAME frame 级时间对齐
- 输入: Stage 7.1 `AVAlignmentInfo.corrected_audio_path`
- Wrapper stub: `face_pipeline_v2/models/wrappers/wav2vec2.py`（空，待实现）

### 3. PPU 环境确认

- **Docker**: `emodia` 容器
- **Python**: 3.12.3
- **PPU torch**: 2.9.0（`torch.cuda.is_available()=True`，device=`PPU-ZW810E`，compute capability `sm_80`）
- **ONNX Runtime**: `onnxruntime_gpu==1.21.1+ppu1.7.0.oe`（阿里官方 PPU 适配版）
- **Provider 验证**: `ort.get_available_providers()` → `['CUDAExecutionProvider', 'CPUExecutionProvider']`
- **SCRFD ONNX 验证**: PPU 上 ~6.03ms/帧（640×640），CPU ~62-182ms/帧，加速 10-30×
- **PPU-ZW810E**: x16 卡，96GB/卡，CUDA 兼容（驱动 alixpu v1.5.1，编译器 HGGC）

### 4. Affine Alignment 问题分析

- **现象**: 部分 aligned face crop 缺少头顶 / 下巴
- **分析**: 纯几何分析验证完成
- **根因**: 当前 crop 扩展和 alignment mapping 导致部分极端姿态情况下 chin 映射超过 256 canvas
- **结论**: 需要调整 alignment 策略

### 5. 后续计划

1. 完成 Stage12 FLAME 接入: `flame_feature.py` dataclass → EMICA inference wrapper (module enable/disable: MICA, DECA, ExpressionEncoder, SPECTRE) → SPECTRE jaw refinement integration → Stage12 processor
2. HuBERT feature extraction wrapper
3. Gaze 模块接入 (Stage13 MediaPipe)
4. FLAME/audio/gaze timestamp fusion (Stage17)
5. 修复 PPU 容器内 PyTorch 模型加载 (sixdrepnet/HyperIQA/SyncNet)

### 修改文件

| 文件 | 变更 |
|---|---|
| `PROJECT_CONTEXT.md` | 更新 date；新增 FLAME/EMICA/SPECTRE 设计章节；新增 HuBERT 设计章节；PPU 环境更新；affine alignment 问题记录；优先级更新 |
| `DEVELOPMENT_LOG.md` | 本条目 |

---

## 2026-08-19 — Stage 5.2/5.3 改为真正 Early Hard Filter

### 问题发现

Stage 5.2 SingleFaceFilter 与 Stage 5.3 TemporalStabilityCheck 实际是 annotation 而非 hard filter：
- 5.2 `process()`：无论 `identity_result.passed` 是 True/False 都 append。
- 5.3 `process()`：无论 `stability_result.passed` 是 True/False 都 append。

导致被拒绝的 sequence 仍进入 Stage 6（6DRepNet）/ Stage 7（SyncNet）/ Stage 8（HyperIQA），消耗 GPU 推理资源；Progressive Gating 设计失效；`passed` 字段成为无效 metadata。

### 原因分析

- 5.2/5.3 只负责「评估 + 附加结果」，没有在 `process()` 内丢弃未通过项。
- 下游 Stage 6/7/8 只读 sequence 本身，不消费 `passed`，因此拒绝不生效。

### 设计文档修改（第二阶段）

| 文档 | 章节 | 变更 |
|---|---|---|
| `docs/01_System_Architecture.md` | 1.1 / 1.2 / 1.4.3 | 改为 Early Rejection Gates（硬过滤门），明确拒绝的 sequence 不进入 Stage 6-9，补充 Progressive Gating 原则 |
| `docs/02_Data_Model_and_Interface.md` | 2.3 + 注释 | Stage I/O 表改硬过滤语义；补 Gating 语义说明 |
| `docs/03_Pipeline_Stage_Specification.md` | 3.9 / 3.9.0 / 3.9.2 / 3.9.3 | 改名为 Hard Filter，补 Behavior/Output/Gating/Independence/Configuration |
| `docs/04_Config_and_Model_Management.md` | 4.4.6 | enabled=true 激活硬过滤，false 为纯透传 |
| `docs/05_Data_Flow_and_Output.md` | 5.2 / 5.4.2 / 5.13 | 数据流表 + 总结明确 rejected 不进入 Stage 6-9 |
| `docs/06_Engineering_Guideline.md` | 6.10.1 | 补 Progressive Gating 原则（cheap filtering 减少下游 GPU 推理） |

### 代码修改（第三阶段）

- `face_pipeline_v2/stages/stage_05_2_single_face_filter/stage.py` — `process()` 硬过滤：只 append `identity_result.passed=True` 的 sequence；新增 `input_count`/`passed_count`/`rejected_count`/`reject_reason_counts` 统计；保留 reject reason 与 debug 日志。
- `face_pipeline_v2/stages/stage_05_3_temporal_stability/stage.py` — 同理硬过滤；保留对 Stage 5.2 的独立性（无 `identity_result` 时 fallback 全 bbox 逻辑）。

未修改：`FaceSequence` dataclass、`identity_result`/`stability_result` 结构、`filter_sequence`/`analyze_stability` 核心逻辑。

### 测试修改与结果（第四/六阶段）

- `tests/test_stage_05_2_single_face_filter.py` / `tests/test_stage_05_3_temporal_stability.py`：
  - `test_mixed_pass_and_fail` 改为断言输出只含通过项（硬过滤）。
  - 新增 `test_all_rejected_returns_empty`（空列表 → 下游 skip）。
  - 新增 `test_disabled_passthrough_keeps_all`（disabled 透传、不丢弃、不附加结果）。
- `tests/stage0_11_integration_test.py` — 5.2/5.3 record 改为 `input: passed/rejected (hard filter, dropped)`。
- 结果：**42 passed**（5.2+5.3 全部单测，含 6 个新用例）；py_compile 通过。

### 后续（未执行，超出本次范围）

- 放宽阈值：5.2 `max_identity_anomaly_ratio` 0.005 → 0.05（talking-head）；5.3 `max_center_motion` 15 → 30-50px。
- 实跑 relaxed Stage 0-11 集成测试，量化 Stage 6/7/8 减少的无效推理量。
- 生产 `pipeline_runner.py` 实现 / Stage 9 导出阈值放宽。

---

## 2026-08-18 — PPU ONNX Runtime 适配与 SCRFD GPU 加速验证

### 今日目标

让 SCRFD ONNX 在 PPU 上真正 GPU 加速（08-16 代码就绪但 PPU 未跑通）。

### 原问题

- PPU-ZW810E 是 CUDA 兼容卡，但 PyPI 的 `onnxruntime-gpu` 没有「CUDA 12 + cuDNN 8.9」组合（≤1.18.x = CUDA 11 + cuDNN 8；≥1.19.x = CUDA 12 + cuDNN 9），SCRFD 只能 CPU 推理。
- 源码编译 onnxruntime 1.18.x 又因服务器无法访问 github/镜像而卡死。

### 排查过程

1. 确认当前 PPU-ZW810E 环境本身正常，问题只在于原版 ONNX Runtime 没有 PPU 支持。
2. 确认**不需要自己源码编译 ONNX Runtime**。
3. 联系阿里 PPU 技术支持，确认官方提供 **PPU 适配版 ONNX Runtime pip 包**。

### 当前 Docker 环境

| 项 | 值 |
|---|---|
| Python | 3.12.3 |
| OS | Ubuntu 24.04 |
| CUDA | 12.9 |
| PPU SDK | 2.0 |
| `CUDA_HOME` | `/usr/local/PPU_SDK/CUDA_SDK` |
| `PPU_SDK` | `/usr/local/PPU_SDK` |

### 安装官方 onnxruntime_gpu

```bash
pip install onnxruntime_gpu==1.21.1+ppu1.7.0.oe
```

### Provider 验证

```python
import onnxruntime as ort
ort.get_available_providers()
# ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

**结论**：PPU 通过 `CUDAExecutionProvider` 兼容方式工作，**不显示单独 `PPUExecutionProvider`**。代码侧 `execution_provider=ppu → CUDAExecutionProvider` 映射无需改动。

### SCRFD Benchmark 结果（2026-08-18）

| 项 | 值 |
|---|---|
| 模型 | `models/SCRFD/scrfd_10g_bnkps.onnx` |
| 输入 | 640×640 |
| PPU（CUDAExecutionProvider） | **约 6.03 ms/帧** |
| CPU（CPUExecutionProvider） | 约 62~182 ms/帧 |
| 加速 | **约 10~30 倍** |

### 最终结论

- **SCRFD 已经成功接入 PPU 加速**，ONNX Runtime PPU 适配完成。
- **不再需要源码编译 ONNX Runtime**。
- Stage 3 / Stage 5 的 SCRFD ONNX 推理均支持 PPU（共享 `create_onnx_session()` 后端 + `execution_provider` 配置）。

### 下一步

1. **继续 Stage 12 EMICA（FLAME）**。
2. 修复 PPU 容器内 PyTorch 模型加载（sixdrepnet / HyperIQA / SyncNet）。
3. Stage 9 阈值用更大 validation set 标定。

---

## 2026-08-16 — SCRFD PPU 后端加速：代码就绪，PPU 未跑通（BLOCKED）

### 今日目标

让 SCRFD ONNX 在 PPU 上推理加速。约束：不换模型、不改 PyTorch、保持 `SCRFD ONNX → onnxruntime wrapper → FaceInfo` 输出格式不变，只替换推理后端。

### 完成内容

#### Runtime / Backend（基础设施，非单个 Stage）

- `face_pipeline_v2/runtime/backend.py`：修复 `AzureExecutionProvider` 被误判为 PPU 的 bug。
- `face_pipeline_v2/runtime/onnx_runtime.py`：`create_onnx_session()` 新增 `execution_provider`（auto/ppu/cuda/cpu）+ `resolve_execution_providers()` + 逐个 fallback + 打印实际 provider。
- `face_pipeline_v2/models/wrappers/scrfd.py`：新增 `execution_provider` 参数 + `active_providers` 属性 + 加载时打印 `session.get_providers()`。**未改动 detect / decode / FaceInfo**。
- `face_pipeline_v2/stages/stage_03_face_screening/stage.py`、`stage_05_face_detection/stage.py`：从 config 读取并透传 `execution_provider`（保持 ModelManager 结构不变）。
- `configs/config.yaml`：新增顶层 `scrfd.execution_provider: auto` 和 `models.scrfd.execution_provider: auto`（auto/ppu/cpu）。
- `scripts/test_scrfd_backend.py`：新增验证脚本（输出 ort 版本 / available providers / SCRFD 实际 provider / 单图耗时）。

#### 环境调查结论

- PPU = **T-Head（平头哥）PPU-ZW810E**（16 卡，96GB/卡，驱动 alixpu v1.5.1，编译器 HGGC），**CUDA 兼容卡，无 `PPUExecutionProvider`**。
- 部署方式判定：**onnxruntime-gpu 的 `CUDAExecutionProvider`**，ONNX 无需转换（故未创建 `tools/convert_scrfd_to_ppu.py`）。
- 宿主机 `/opt/pg1`（CUDA 12.3 + glibc 2.32）无法加载 onnxruntime CUDA EP（`__cudaRegisterFatBinaryEnd` 处 abort）。
- 容器 `face_pipeline_lxs`（emodia 镜像，CUDA 12.9 + cuDNN 8.9.5 + glibc 2.39）：torch 2.9.0 CUDA 可用（device=PPU-ZW810E，compute capability sm_80）。

### Bug 修复

- **问题**：`backend.py` 把 `AzureExecutionProvider`（CPU 提供者）列入 `_PPU_PROVIDER_NAMES`，导致纯 CPU 环境被误判为 `Backend.PPU`。
- **原因**：`AzureExecutionProvider` 是 Azure 的 CPU 优化 EP，不是 GPU/PPU 后端。
- **解决**：从 `_PPU_PROVIDER_NAMES` 与 `_classify_backend` 中移除。

### 测试结果

- `scripts/test_scrfd_backend.py`：宿主机跑通，onnxruntime 1.23.2，providers=[Azure, CPU]，backend=cpu，active=[CPUExecutionProvider]，约 2500ms/张（CPU）。
- pytest：`test_config` / `test_context` / `test_stage_interface` / `test_stage_03_coarse_face_screening` / `test_stage_10_landmark` = **53 passed**。`test_stage_05_face_detection.py` 有 4 个**既有失败**（`TestFaceCropper` ×3 + `test_face_id_includes_video_clip_and_face_index`，仅涉及 crop/face_id，不引用本次改动模块，与本次无关）。
- 容器内实测：`onnxruntime-gpu 1.18.1` CPU fallback 约 113ms/张；CUDA EP 加载失败（见下）。

### 当前问题

- **onnxruntime-gpu 版本与 SDK cuDNN 错配**：SDK = CUDA 12.9 + cuDNN 8.9.5。PyPI 上 ≤1.18.x = CUDA 11 + cuDNN 8（缺 `libcublasLt.so.11`）；≥1.19.x = CUDA 12 + cuDNN 9（缺 `libcudnn.so.9`）。**无「CUDA 12 + cuDNN 8.9」组合**。
- **源码编译 onnxruntime 1.18.x 被网络卡死**：服务器（host 网络）github 及所有镜像（gitclone 403 / gitee 无 onnxruntime 镜像 / ghproxy 等不可达）均拿不到源码；PyPI 无 onnxruntime sdist。
- 需要：上传 onnxruntime 源码 tarball，或 T-Head `art-pub.eng.t-head.cn` token，或 T-Head 提供 cuDNN 9 shim。

### 明日计划

1. 获取 onnxruntime 1.18.x 源码（上传 tarball）或 T-Head token / cuDNN 9 shim，继续 SCRFD PPU 加速。
2. 修复 PPU 容器内 PyTorch 模型加载（sixdrepnet PYTHONPATH / HyperIQA `weights_only=False` / SyncNet）——延续 08-13 未完成项。
3. Stage 9 阈值用更大 validation set 标定。

---

## 2026-08-13 — Stage 10/11 坐标映射修复 + Stage 7.1 三态对齐 + Stage 9 阈值标定

### Completed

- **Stage 10/11 坐标映射修复（核心）**：Stage 5 使用 `expanded_bbox` 裁剪 raw crop，但 Stage 10 `_map_to_frame` 与 Stage 11 `align_faces` 仍用 `tight_bbox` 做坐标映射，导致 landmark 从 crop 坐标回映到 frame 坐标时偏移，aligned face 错位。现改为使用 `expanded_bbox` offset 做 1:1 无缩放映射（`frame = crop + expanded_bbox.xy1`，`crop = frame − expanded_bbox.xy1`），并保留 `tight_bbox` 的 legacy fallback（兼容无 expanded_bbox 的旧 FaceInfo）。
- **Stage 7.1 三态对齐状态机**：从布尔 `correction` 重构为三态 `already_aligned` / `corrected` / `severe_offset_unavailable`；新增 `classify_alignment()`、`extract_clip_audio()`、`should_correct()`（向后兼容）；`AVAlignmentInfo` 新增 `aligned_audio_path`（最终可用 clip 级音频）+ `alignment_status`（三态）；`max_offset_ms` 统一由 `sync.max_offset_frames` 派生。
- **Stage 9.5 改为读取 `aligned_audio_path`**，并处理 `severe_offset_unavailable` 导致的 audio_unavailable（视频仍导出，音频置 None）。
- **Stage 9 阈值标定**：以 RD_Radio1_000（534×534, 75s, 17 clips）作为阈值参考，调整 `quality.quality_threshold=0.32`、`quality.sync.min_lse_c=3.0`（临时）、`stage_09.min_pose_ratio=0.85`。
- **测试脚本**：新增/更新 4 个脚本（见 Testing）。

### Fixed Issues

- 修复「expanded_bbox 裁剪，但使用 tight_bbox 坐标映射导致 aligned face 错位」——Stage 10/11 统一改用 expanded_bbox offset，坐标 round-trip 一致。
- 修复 Stage 7.1 音频严重错位时仍生成错误音频的问题——三态状态机中 `severe_offset_unavailable` 状态明确不生成音频，`aligned_audio_path=None`。
- 修复 Stage 9.5 读取 `corrected_audio_path` 时漏掉「无需校正但需裁剪 clip 音频」的情况——改为读取 `aligned_audio_path`。
- **修复 Stage 11 aligned face「半张脸」的根因（landmark 预处理 scalefactor 错误）**——`insightface_landmark.py._preprocess` 原把输入 `/255.0` 归一化到 [0,1]，但 2d106det.onnx 图内已有 `Sub(127.5)` + `Mul(0.0078)` 归一化层，期望输入 **[0,255] 原始像素**。`/255` 后模型收到接近常数（≈-127.5）的输入，landmark 输出整体错乱。修复：去掉 `/255`，直接传 [0,255] RGB。
- **修复 Stage 11 aligned face「半张脸」的第二个根因（landmark 关键点索引错误）**——`affine_transform.py` 原用 `LEFT_EYE=63-66+71-74`、`RIGHT_EYE=67-70+75-78`、`NOSE=55-62`，实测这些索引指向的是嘴/鼻下部（y≈133-159）。真实索引（2026-08-13 实证）：左眼 `33-42`、右眼 `87-96`、鼻尖 `83-85`。

### Investigation

- **PPU GPU 环境 PyTorch 模型加载失败**（RD_Radio2_000, PPU-ZW810E + PyTorch 2.9.0, cuda available）：
  - Stage 6 6DRepNet：`No module named 'sixdrepnet'` — GPU 容器 PYTHONPATH 未包含 third_party/sixdrepnet。
  - Stage 8 HyperIQA：`Cannot use weights_only=True with files saved in legacy .tar format` — PyTorch 2.6+ 默认 `weights_only=True`，`koniq_pretrained.pkl` 是 legacy .tar，需 `weights_only=False`。
  - Stage 7 SyncNet：模型 fallback → 全零输出（待确认原因）。
  - ONNX 模型（SCRFD Stage 3/5、InsightFace Stage 10）在 CPU 正常，Stage 10 3800 landmarks / Stage 11 3800 aligned faces 全部成功。
- **EMICA**：`third_party/EMICA`、`models/wrappers/emica.py` 仍为空，Stage 12 FLAME 未启动（EMICA checkpoint 已确认存在，设计已冻结，待实现 wrapper）。EMOCA/DECA checkpoints 仍需下载用于 DECA backbone。

### Testing

- `test_scripts/test_stage71_audio.py`：Stage 7.1 三态单元测试（offset=1/5/20 → already_aligned/corrected/severe），3/3 PASS。
- `test_scripts/run_threshold_reference.py`：RD_Radio1_000 全 Stage 0-11，记录逐 clip 参数。结果：LSE-C mean 5.91（4.93–6.54）、offset 统一 1 帧、HyperIQA 46.95、pose 全 frontal（yaw 0.19±1.85°）；默认阈值下 17/17 被拒（quality 0.4696 < 0.5），据此下调阈值。
- `test_scripts/run_stage8_11_reuse.py`：复用 Stage 5 磁盘 crop，跳过 ~35min SCRFD，跑 Stage 6-11+9.5。RD_Radio1_000 17 clips 全接受（fusion ~0.84），manifest.json 导出，audio 全部 already_aligned（offset=1 帧=40ms < 50ms 阈值）。
- `test_scripts/run_full_pipeline.py`：normal_talking.mp4（640×360, 15.12s, 2 clips）全流程 0-11+9.5。Stage 9 1/2 接受；Stage 7.1 {corrected:1, severe_offset_unavailable:1}；Stage 10 198 landmarks / Stage 11 198 aligned；Stage 9.5 导出 1 video + 1 audio。

### Current Problems

- PPU GPU 上三个 PyTorch 模型（6DRepNet/SyncNet/HyperIQA）无法加载，需在 GPU 容器内修复 sixdrepnet 路径、`weights_only=False`、SyncNet 依赖。
- Stage 9 阈值（0.32/3.0/0.85）仍为临时值，需真实 validation set 标定。
- Stage 5 SCRFD 仍 CPU 推理（validation_output_rd 中 47min），未启用 GPU/ONNX provider。
- Stage 12（EMICA/FLAME）及 Stage 11.5、13-17 仍为 stub。

### Next Steps

- 修复 PPU GPU 容器内 PyTorch 模型加载（sixdrepnet PYTHONPATH、HyperIQA weights_only、SyncNet）。
- 接入 EMICA（Stage 12 FLAME）——先确认权重/依赖，再实现 wrapper。
- Stage 9 阈值用更大 HDTF validation set 标定。
- 补全 `pipeline_runner.py` 多 stage 编排。

---

## 2026-08-12 — Pipeline Optimization and Preprocessing Redesign

### 1. Face Crop Pipeline Redesign

**Problem**: Stage 5 used SCRFD tight bbox → crop directly. The tight bbox only covers eyes/nose/mouth, missing forehead, chin, and cheeks. Downstream stages tried to "expand" post-crop — but this only added black padding, not real face pixels.

**Solution**: Redesigned Stage 5 to expand the bbox ON THE ORIGINAL FRAME before cropping:
```
Before: SCRFD tight bbox → crop → post-crop expand (black padding)
After:  SCRFD tight bbox → expand_bbox on frame → crop (real context)
```
Config: `stage_05_face_crop.expand_ratio=2.0`. FaceInfo now stores both `bbox` (original) and `expanded_bbox` (after expansion).

### 2. Per-Model Preprocessing Decentralized

Each downstream stage now performs its own resize. Stage 5 no longer resizes to 256×256 — it saves raw crops at native resolution.

| Stage | Input | Preprocess | Config |
|-------|-------|-----------|--------|
| 6 (6DRepNet) | Stage 5 raw crop | resize 256×256 | `stage_06.preprocess.input_size` |
| 7 (SyncNet) | Stage 5 raw crop | use as-is for video | (expand done at Stage 5 bbox level) |
| 8 (HyperIQA) | Stage 5 raw crop | resize 224×224 | `stage_08.preprocess.input_size` |
| 10 (InsightFace) | Stage 5 raw crop | resize 256×256 | landmark model internal |

### 3. Stage 7 Clip-Level Audio Fix

**Problem**: SyncNet was using the FULL video audio track from time 0, comparing face crops from [4s-12s] with audio from [0s-8s] — time window mismatch.

**Solution**: `sync_estimator._extract_clip_audio()` truncates audio to `[clip.start_time, clip.start_time+clip.duration]` via ffmpeg. SyncNet now compares face crops with the CORRECT time-aligned audio segment.

### 4. Stage 7.1 Per-Clip Corrected Audio

`align_audio()` now accepts optional `clip_start`/`clip_duration` parameters. When present, the output is trimmed to clip boundaries after offset correction. This produces per-clip audio segments ready for final asset export.

### 5. Stage 9.5 Clip Export

New stage that exports Stage-9-accepted clips as final training data assets:
- Video: ffmpeg cut from Stage 1 standardized video
- Audio: Stage 7.1 per-clip corrected audio (offset-aligned)
- Manifest: `manifest.json` with per-clip quality metadata

Only `is_accepted=True` clips are exported.

### 6. Stage 11 Affine Estimation Fix

**Problem**: `cv2.estimateAffinePartial2D` with RANSAC method failed on small face crops (~100px). Only 3 points (eyes+nose) with RANSAC is overkill — consensus threshold of 3px on 100px images caused failures.

**Solution**: Changed to `cv2.LMEDS` (Least-Median) which is more stable with exactly 3 points. Result: 183/198 aligned (15 edge-frame failures at extreme angles).

### 7. Stage 11 Landmark Coordinate Fix

Simplified frame→crop coordinate mapping from `(frame_x - x1) * crop_w / bbox_w` to pure offset `frame_x - x1` (valid for raw crops where `crop_w == bbox_w`). Added scale correction only for boundary-clamped edge cases. Debug logging added for first 3 frames per clip.

### 8. Stage 5 Performance Optimization

**Frame Detection Cache**: `Dict[int, List[FaceInfo]]` cache deduplicates SCRFD inference across overlapping clips.
- Two-phase processing: Phase 1 reads all unique frames sequentially + detects → cache. Phase 2 serves clips from cache.
- Hit ratio: 57% (2 clips) / ~46% expected (14 clips).
- `_clone_for_clip()` copies crop files to new clip directories on cache hit.

**Video Reader Abstraction**: `runtime/video_reader.py` — `VideoFrameReader` ABC with `FFmpegVideoReader` (pipe-based) and `OpenCVVideoReader` (seek-based). Config: `runtime.video_reader: ffmpeg|opencv`. FFmpeg test: 378 frames decoded in 0.3s, pixel-identical to OpenCV.

### 9. Performance Test Results

60s video (170_gzfKeagBy1k_1min, 640×360, 14 clips):
- Stage 5: ~60 min (64% of total) — SCRFD ONNX CPU inference on ~1500 unique frames
- Stage 8: ~10 min — HyperIQA CPU inference
- Stage 6: ~5 min — 6DRepNet CPU inference
- All models CPU-bound on dev machine. Expected 10-50× speedup on PPU GPU.

### 10. Current Next Steps

- Run Stage 0-11 full pipeline on real 60s video with new expand=2.0 crops
- Verify: face crop completeness, LSE-C values, aligned_faces quality, Stage 9 acceptance rate
- GPU deployment: ONNX CUDAExecutionProvider for Stage 3/5/10, PyTorch CUDA for Stage 6/7/8

---

## 2026-08-12 — Input Chain Refactoring: Per-Model Preprocess + Clip-Level Audio

### Problem

Stage 5 was resizing all face crops to 256×256, destroying the original bbox resolution. Different models need different preprocessing (expand ratios, input sizes), and SyncNet was using the full video audio track instead of per-clip segments — causing severe time-window mismatch.

### Changes

#### 1. Stage 5: raw crops at native resolution
- `face_cropper.py`: removed `cv2.resize` — saves raw bbox crop at original dimensions
- Downstream stages are responsible for their own preprocessing

#### 2. New utility: `face_pipeline_v2/utils/face_preprocess.py`
- `expand_face_crop(image, expand_ratio)` — centre-expand with black padding
- `resize_face(image, target_size)` — square resize
- `preprocess_face_crop(path, expand_ratio, target_size)` — full pipeline

#### 3. Stage 6/7/8 each apply own preprocess
| Stage | expand_ratio | input_size | Config |
|-------|-------------|------------|--------|
| 6 (6DRepNet) | 1.4 | 256 | `stage_06.preprocess` |
| 7 (SyncNet) | 1.6 | 256 (for video) | `stage_07.preprocess` |
| 8 (HyperIQA) | 1.6 | 224 | `stage_08.preprocess` |

#### 4. Stage 7: clip-level audio
- `sync_estimator._extract_clip_audio()` — ffmpeg trim to clip boundaries
- SyncNet now compares face crops with the SAME time window's audio
- Previously: face crops from [4s-12s] vs audio from [0s-8s] (WRONG)

#### 5. Stage 7.1: per-clip corrected audio
- `align_audio()` accepts optional `clip_start`/`clip_duration` for trimming
- Output is per-clip audio segment (not full-track)
- Input dict accepts optional `clip_infos` key

#### 6. Validation output: accepted clips only
- GPU validation script now saves only Stage-9-accepted clips
- manifest.json generated with per-clip quality metadata
- Stage 10/11 confirmed: only process accepted clips (aligned_faces count = accepted landmarks count)

### Smoke test (normal_talking.mp4, 15s)
- Stage 9: 1/2 accepted (was 0/2 before fixes!)
- LSE-C similar to before (0.19-0.23) — 640×360 resolution is the limiting factor
- All per-stage preprocess config read correctly
- Stage 10: 198 LandmarkInfo from 1 accepted clip ✅

### Files modified
| File | Change |
|---|---|
| `utils/face_preprocess.py` | NEW — expand + resize utilities |
| `stage_05_face_detection/face_cropper.py` | Removed resize; raw crop save |
| `stage_06_pose/pose_estimator.py` + `stage.py` | Preprocess config + expand→resize |
| `stage_07_sync/sync_estimator.py` + `stage.py` | Clip-audio + expand preprocess |
| `stage_07_1_av_alignment/processor.py` + `stage.py` | Per-clip trimming |
| `stage_08_quality/quality_estimator.py` + `stage.py` | Preprocess config + expand→resize |
| `configs/config.yaml` | Added stage_06/07/08.preprocess sections |
| `test_scripts/gpu_validation/run_gpu_validation.py` | Accepted-only assets + manifest.json |
| `PROJECT_CONTEXT.md` | Updated design decisions |
| `DEVELOPMENT_LOG.md` | This entry |

---

## 2026-08-11 (late) — Stage 9 Sync Hard Filter Redesign

### Change: Sync Hard Filter Uses Raw LSE-C + Offset

**Problem**: Stage 9 hard filter used `compute_sync_score_norm()` which clamped LSE-C to `[0, 1]`. Real LSE-C range is 3-10 (verified in expand_ratio experiment). Clamping destroyed signal discrimination — all LSE-C ≥ 1 became score=1.0.

**Design change**:
- Hard filter now uses raw `AVSyncInfo.offset` and `AVSyncInfo.sync_lse_c` directly
- Config: `quality.sync.max_offset_frames=8`, `quality.sync.min_lse_c=5.0`
- Accept condition: `abs(offset) <= 8 AND sync_lse_c >= 5.0`
- Specific reject reasons: `sync_offset_exceeded`, `sync_lse_c_too_low`
- Normalized `sync_score` retained for fusion weighting + Stage 9.1 ranking only

**Files modified**:
| File | Change |
|---|---|
| `configs/config.yaml` | Added `quality.sync` section; updated `sync` section comment |
| `stage_09_quality_fusion/fusion.py` | FusionThresholds: `sync_threshold`→`min_lse_c`; apply_hard_filter: raw offset+LSE-C checks |
| `stage_09_quality_fusion/stage.py` | _build_thresholds reads from quality.sync; enhanced logging with raw offset/LSE-C |
| `tests/test_stage_09_quality_fusion.py` | Updated FusionThresholds defaults and mock configs |
| `test_scripts/e2e_stage0_11/run_e2e_test.py` | Config overrides for test video (min_lse_c=0.2, max_offset=15) |

**E2E test**: 16/16 stages PASS. Stage 9 logs: `offset=-9 (max=15) | lse_c=0.27 (min=0.2)` ✅

---

## 2026-08-11 — Stage0-Stage11 Pipeline Integration and Asset Validation

### Today's Achievements

#### New Stages Completed

- **Stage 7.1 AV Alignment** — A/V sync correction via ffmpeg. Consumes Stage 7 AVSyncInfo + Stage 1 StandardizedVideoInfo, produces AVAlignmentInfo with corrected_audio_path. Regular pipeline stage (not enhancement). Correction threshold: 50ms; max offset: 15 frames.

- **Stage 9 Quality Fusion** — Pure rule-based logic. Fuses Stage 6 (Pose), Stage 7 (Sync), Stage 8 (Quality) into unified QualityMetrics per clip. Three gates: pose (min_pose_ratio + angle thresholds), sync (LSE-C + offset), quality (HyperIQA score). Weighted fusion score. Produces `accepted_clip_ids` for Stage 10.

- **Stage 9.1 Quality Ranking** — Enhancement module. Weighted multi-metric ranking: pose(0.3), sync(0.3), face_quality(0.4). Assigns quality levels (excellent/good/normal). Does NOT modify Stage 9's is_accepted/reject_reasons.

- **Stage 10 Landmark Extraction** — InsightFace 2d106det ONNX model. Extracts 106-point facial landmarks per face. Only processes clips accepted by Stage 9. Output: LandmarkInfo (212 float values in original frame coordinates).

- **Stage 11 Face Alignment** — OpenCV affine transform. Selects key points (left eye, right eye, nose tip) from landmarks, estimates affine to 256×256 template, warps face crop. Output: AlignedFaceInfo with affine_matrix (6 values), original_image_size, aligned_face_path.

#### Final Asset Export Verified

After Stage 11, each accepted clip has:
- Video clip (Stage 4)
- Face crops (Stage 5)
- Head pose sequence (Stage 6)
- AV sync metrics + corrected audio (Stage 7 + 7.1)
- Face quality scores (Stage 8)
- Quality fusion metrics (Stage 9)
- 106-point landmarks (Stage 10)
- Aligned canonical faces 256×256 + affine matrix (Stage 11)

#### Stage0-Stage11 E2E Integration Test

| Metric | Value |
|---|---|
| Test video | sample.mp4 (640×360, 25fps, 15.12s, h264+aac) |
| Stages executed | 16 (0→1→2→3→4→5→5.1→5.2→5.3→6→7→7.1→8→9→9.1→10→11) |
| Pass rate | 16/16 (100%) |
| Total duration | 573.9s (~9.6 min) |
| Data flow checks | 7/7 passed |
| Face crops saved | 388 |
| Landmarks extracted | 378 |
| Aligned faces saved | 378 |

### Integration Hardening — Problems Found and Fixed

#### Fix 1: Stage 11 Image Save Bug (High 🔴)
- **Problem**: 378 AlignedFaceInfo produced but 0 images saved to disk.
- **Root cause**: `aligned_path.parent.mkdir()` in `else` branch of `if save_images:`. `cv2.imwrite` silently fails when parent dir doesn't exist.
- **Fix**: Move `mkdir` before `if/else`; add `RuntimeError` on `cv2.imwrite` failure.
- **File**: `stage_11_alignment/processor.py`

#### Fix 2: SCRFD Network Download on Startup (Medium 🟡)
- **Problem**: SCRFD wrapper tried insightface buffalo_sc download before falling back to local ONNX.
- **Root cause**: `SCRFDWrapper.load()` had insightface Path 1 first, ONNX Path 2 as fallback.
- **Fix**: Reversed priority — ONNX local-first, insightface only if pre-downloaded. No auto-download.
- **File**: `models/wrappers/scrfd.py`

#### Fix 3: Stage 9 Debug Logging (Medium 🟡)
- **Problem**: Reject reasons lacked per-dimension metric values — couldn't distinguish which threshold caused rejection.
- **Fix**: Added `_format_reject_detail()` method with per-dimension PASS/FAIL + metric value + threshold.
- **File**: `stage_09_quality_fusion/stage.py`

#### Fix 4: ffprobe Fallback Logging (Low 🟢)
- **Problem**: ffprobe unavailable warning not clear enough.
- **Fix**: Added explicit `MissingDependencyError` handling with install instructions.
- **Files**: `stage_00_video_check/stage.py`, `run_e2e_test.py`

#### Fix 5: EnhancementStage Return Type Consistency (Low 🟢)
- **Problem**: Stage 9.1 disabled passthrough returned `dict` instead of `List[QualityRankingInfo]`.
- **Root cause**: EnhancementStage always returned `input_data` when disabled; input/output types differ for Stage 9.1.
- **Fix**: Added `_get_passthrough_output()` hook. Stage 9.1 overrides to return `[]`. Stages 5.1/5.2/5.3 unchanged (types match).
- **Files**: `pipeline/enhancement_stage.py`, `stage_09_1_quality_ranking/stage.py`

#### Fix 6: torch.load FutureWarning (Low 🟢)
- **Problem**: PyTorch 2.4 FutureWarning about `weights_only=False` default.
- **Fix**: Set `weights_only=True` in 3 locations (sixdrepnet, SyncNet, HyperIQA ResNet50). Added comment on HyperIQA checkpoint line 126 explaining why `weights_only=False` is required.
- **Files**: `sixdrepnet/regressor.py`, `SyncNet/SyncNetInstance_calc_scores.py`, `hyperiqa.py`

### Key Findings from E2E Test

1. **Stage 9 default thresholds too strict**: Test video 2/2 clips rejected with defaults (sync_th=0.5, quality_th=0.5). Test used relaxed thresholds (sync=0.2, quality=0.3, min_pose_ratio=0.5) to allow flow-through. Real calibration needed.
2. **SyncNet LSE-C on small video**: LSE-C values 0.27/0.44 on 640×360 video — consistently below 0.5. May be resolution-dependent.
3. **All 5 models loaded successfully**: SCRFD (ONNX), 6DRepNet (PyTorch), SyncNet (PyTorch), HyperIQA (PyTorch), InsightFace 2d106det (ONNX).
4. **Enhancement toggle switches verified**: All 4 enhancement modules correctly passthrough when `enabled=false`.
5. **Stage 7.1→Stage 14 interface ready**: 2 AVAlignmentInfo with corrected_audio_path ready for downstream consumption.

### Files Modified Today

| File | Change |
|---|---|
| `stage_11_alignment/processor.py` | Fix mkdir + cv2.imwrite error handling |
| `models/wrappers/scrfd.py` | ONNX local-first loading, no auto-download |
| `stage_09_quality_fusion/stage.py` | Per-dimension debug logging |
| `stage_00_video_check/stage.py` | ffprobe MissingDependencyError handling |
| `pipeline/enhancement_stage.py` | _get_passthrough_output() hook |
| `stage_09_1_quality_ranking/stage.py` | Override passthrough to return [] |
| `sixdrepnet/regressor.py` | weights_only=True |
| `SyncNet/SyncNetInstance_calc_scores.py` | weights_only=True |
| `hyperiqa.py` | weights_only=True + comment |
| `test_scripts/e2e_stage0_11/run_e2e_test.py` | New E2E integration harness |
| `PROJECT_CONTEXT.md` | Updated for 2026-08-11 state |
| `DEVELOPMENT_LOG.md` | This entry |

### Current Status

```
Stage 0  — FFprobe video check              ✅
Stage 1  — AV standardization               ✅
Stage 2  — Scene detection                  ✅
Stage 3  — SCRFD coarse screening           ✅
Stage 4  — Temporal clip generation         ✅
Stage 5  — Face detection & crop            ✅
Stage 5.1— ByteTrack tracking               ✅ (enhancement)
Stage 5.2— Single face continuity           ✅ (enhancement)
Stage 5.3— Temporal stability               ✅ (enhancement)
Stage 6  — 6DRepNet head pose               ✅
Stage 7  — SyncNet AV sync                  ✅
Stage 7.1— Audio alignment correction       ✅
Stage 8  — HyperIQA quality                 ✅
Stage 9  — Quality fusion + hard filter     ✅
Stage 9.1— Quality ranking                  ✅ (enhancement)
Stage 10 — InsightFace 106 landmark         ✅
Stage 11 — Face alignment                   ✅
─────────────────────────────────────────────────
Stage 11.5— Face validity                    🔴 stub
Stage 12 — FLAME (EMICA)                     🔴 stub
Stage 13 — Gaze (MediaPipe)                  🔴 stub
Stage 14 — Audio feature (wav2vec2)          🔴 stub
Stage 15 — Quality statistics                🔴 stub
Stage 16 — Data annotation                   🔴 stub
Stage 17 — Feature alignment & fusion        🔴 stub
```

**Stage0-Stage11 pipeline is fully closed-loop.** The pipeline can now produce high-quality video data assets (clips, aligned faces, landmarks, affine matrices, quality metrics, corrected audio) ready for downstream FLAME/Gaze/Audio-Feature extraction.

**Next: Stage 12 FLAME (EMICA) → Stage 13 Gaze (MediaPipe) → Stage 14 Audio Feature (wav2vec2)**

---

## 2026-08-07 ~ 2026-08-10 — Stage 6-8 Model Integration

### Stage 6 — 6DRepNet PyTorch Wrapper

完成：
- 6DRepNet PyTorch wrapper 实现，使用官方 SixDRepNet 模型（third_party/sixdrepnet）
- 从 ONNX 方案切换到 PyTorch 方案，直接加载 `6DRepNet_300W_LP_AFLW2000.pth`
- third_party 离线部署，无需网络下载
- 输出 yaw / pitch / roll
- device="auto" 支持（CUDA 自动检测 → CPU fallback）
- PPU GPU backend 支持

### Stage 7 — SyncNet Wrapper

完成：
- SyncNet wrapper 使用 `syncnet_v2.model` 加载
- 通过 third_party/SyncNet 加载真实 SyncNet PyTorch 模型
- AV sync score 输出（offset, score, conf）
- device="auto" 支持

### Stage 8 — HyperIQA Wrapper

完成：
- HyperIQA wrapper 使用 `koniq_pretrained.pkl` 加载
- resnet50 本地权重（`resnet50-19c8e357.pth`），patch model_zoo.load_url 实现离线推理
- 处理 `base_model.` 前缀兼容性（strip → load_state_dict）
- HyperNet + TargetNet 完整推理链路
- 输出 quality_score (0-100)
- device="auto" 支持

### Runtime Backend

完成：
- device="auto" 统一设备选择（CUDA available → cuda, else → cpu）
- CPU fallback 机制
- PPU GPU backend 支持（LD_PRELOAD workaround）
- 模型加载流程：config → PipelineContext → ModelManager → Wrapper

### Problems Solved

1. **6DRepNet ONNX→PyTorch 迁移** — ONNX 方案依赖预导出 .onnx 文件且 rotation_utils 复杂；切换到 third_party/sixdrepnet 官方 PyTorch 实现，直接调用 SixDRepNet_Detector.predict()。

2. **HyperIQA checkpoint 兼容性** — 早期 `.pth` checkpoint (HyperIQA-resnet50-koniq10k-c96c41b1.pth) 与 third_party/HyperIQA/models.py 架构不匹配。解决方案：切换到 `koniq_pretrained.pkl` checkpoint，strip `base_model.` 前缀后成功加载。

3. **HyperIQA 离线部署** — HyperNet 初始化时 resnet50_backbone(pretrained=True) 尝试从 pytorch.org 下载权重。解决方案：patch `torch.utils.model_zoo.load_url` 指向本地 `resnet50-19c8e357.pth`。

4. **SCRFD bbox decode 修复**（前期） — ONNX 输出 4-distance 格式 (dx1,dy1,dx2,dy2)，错误使用 center+exp(width) decode。修复为 grid±distance 直接解码。

5. **PyTorch CUDA/PPU 环境问题** — PPU GPU (非 NVIDIA) 上 PyTorch CUDA 库符号不兼容 (libcusparse.so.12)。LD_PRELOAD workaround 解决。

6. **Docker PPU 设备映射** — Docker 容器内 PPU 设备访问配置。

### Current Status

- Stage 0-8: ✅ completed
- Stage 9: 🔴 stub — Quality Fusion 待实现
- Stages 10-17: 🔴 stub

**Next: Stage 9 Quality Fusion**

---

## 2026-08-06

---

## Completed Today (2026-08-06)

### 1. SCRFD BBox Decode Fix (Critical Bug)
- **Problem**: SCRFD ONNX wrapper used wrong `center+exp(width)` decode, causing bboxes to cover 78-90% of the frame.
- **Root cause**: SCRFD 10G outputs 4-distance format `(dx1, dy1, dx2, dy2)`, not `(cx, cy, w, h)`. The `exp()` call on `dx2` blew up bbox sizes.
- **Fix**: Changed `_decode_detections()` to use correct formula:
  ```python
  x1 = (grid_x - dx1) * stride   # not cx - exp(w)/2
  x2 = (grid_x + dx2) * stride   # not cx + exp(w)/2
  ```
- **Result**: Bboxes restored to 6-8% frame area (normal face size). Stage 3 now correctly passes 2/6 scenes (was 1/6 with corrupted bboxes).
- **Files**: `models/wrappers/scrfd.py`

### 2. Stage 6 — 6DRepNet Real ONNX Integration
- **Problem**: `_load_torch()` raised `NotImplementedError` → NoOp fallback → all yaw/pitch/roll = 0.
- **Environment**: PyTorch CUDA broken on PPU machine. `SixDRepNet.onnx` (151MB) was pre-prepared.
- **Fix**: Rewrote `SixDRepNetWrapper` to use ONNX runtime. Input `(1,3,224,224)` → output `(1,3,3)` rotation matrix → `rotation_utils.compute_euler_angles_from_rotation_matrix()` → yaw/pitch/roll.
- **Config**: Added `models.sixdrepnet` section to config.yaml.
- **ModelManager**: Registered sixdrepnet in `_REGISTRY`.
- **Test 1** (`debug_6drepnet/run_debug.py`): Frontal face ~yaw=16-32°, flipped face roll=23°, rotated face roll=36°. All non-zero.
- **Test 2** (`debug_6drepnet/test_stage6_integration.py`): 200 frames Stage 5→Stage 6, all non-zero. yaw [4.7, 48.9]°, pitch [-39.6, -7.5]°, roll [-13.7, 12.1]°.
- **Files**: `models/wrappers/sixdrepnet.py`, `configs/config.yaml`, `models/model_manager.py`, `test_scripts/debug_6drepnet/*`

### 3. Stage 7 — SyncNet Real PyTorch Integration
- **Problem**: SyncNet wrapper couldn't import `SyncNetInstance_calc_scores` → NoOp fallback → fixed (offset=0, score=0.5).
- **Fix**:
  - Fixed `third_party/SyncNet/` path in wrapper (parents[4]→parents[3]).
  - Rewrote `SyncNetWrapper` for proper `model_path` / `device` / `batch_size` / `vshift` params.
  - Works with `LD_PRELOAD` for PyTorch on PPU machine.
- **Config**: Added `models.syncnet` and `stage_07.device` to config.yaml.
- **ModelManager**: Registered syncnet; `load_model()` now passes `**extra_kwargs` (batch_size, vshift).
- **Test** (`debug_syncnet/run_debug_syncnet.py`): Normal clip offset=0, score=0.877; delayed clip offset=7, score=0.362. Not NoOp.
- **Files**: `models/wrappers/syncnet.py`, `configs/config.yaml`, `models/model_manager.py`, `test_scripts/debug_syncnet/*`

### 4. Stage 7 Interface Audit & Consistency Fixes
- **Audit**: 5/5 checks PASS. Found 4 issues (2 medium, 2 minor).
- **C-1 Fix**: Added `device: cpu` to `stage_07` in config.yaml.
- **C-2 Fix**: `ModelManager.load_model()` now accepts and passes `**extra_kwargs` to wrapper constructor.
- **I-3 Fix**: Removed direct `SyncNetWrapper` instantiation fallback from `Stage7._get_model()`. Stage 7 now uses only `ModelManager.get_model()` / `load_model()`.
- **Wrapper API**: Renamed `SyncNetWrapper.__init__(checkpoint=)` → `(model_path=)` for ModelManager consistency.
- **Files**: `configs/config.yaml`, `models/model_manager.py`, `stages/stage_07_sync/stage.py`, `models/wrappers/syncnet.py`

### 5. Stage 8 — HyperIQA Investigation (Incomplete)
- **Problem**: Wrapper falls back to NoOp → quality_score = 50.0. Real PyTorch inference needed.
- **Analysis**: Checkpoint `HyperIQA-resnet50-koniq10k-c96c41b1.pth` has keys like `base_model.conv1.weight`, `base_model.layer1.*`, `base_model.lda_modules.*`. After stripping `base_model.` prefix, keys still don't match `third_party/HyperIQA/models.py` which uses `res.conv1.weight`, `res.layer1.*`, `res.lda1_pool.*`. These are different HyperIQA implementations.
- **Status**: Blocked — need matching third_party code or ONNX export.
- **Files**: `models/wrappers/hyperiqa.py` (rewritten but can't load checkpoint), `test_scripts/debug_hyperiqa.py`

### 6. Validation & Debug Tools Created
- `test_scripts/stage0_8_validation/` — Full pipeline validation (run_validation.py, visualization.py, validation_report.py, README.md)
- `test_scripts/debug_scrfd_output.py` — ONNX output format analysis
- `test_scripts/debug_bbox_fix.py` — Bbox decode verification
- `test_scripts/debug_6drepnet/` — 6DRepNet debug + Stage 6 integration test
- `test_scripts/debug_syncnet/` — SyncNet debug + interface audit report
- `test_scripts/debug_hyperiqa.py` — HyperIQA test (blocked)

---

## Stage Progress

| Stage | Status | Notes |
|---|---|---|
| **0** | ✅ Complete | Video integrity check (ffprobe/ffmpeg) |
| **1** | ✅ Complete | AV standardization |
| **2** | ✅ Complete | Scene segmentation |
| **3** | ✅ Complete | Coarse face screening — **SCRFD bbox decode fixed today** |
| **4** | ✅ Complete | Sliding window clip generation |
| **5** | ✅ Complete | Full face detection & crop |
| **5.1** | ✅ Complete | Face tracking (ByteTrack) |
| **5.2** | ✅ Complete | Single face identity filter |
| **5.3** | ✅ Complete | Temporal stability check |
| **6** | ✅ Complete | **Real 6DRepNet PyTorch integrated** |
| **7** | ✅ Complete | **Real SyncNet PyTorch integrated; interface audit passed** |
| **8** | ✅ Complete | **HyperIQA integrated (koniq_pretrained.pkl + resnet50 local)** |
| **9** | 🔴 Stub | Quality fusion (not started) |
| **10-17** | 🔴 Stub | Downstream stages (not started) |

---

## Important Changes

### Architecture
- **ModelManager.load_model()** now supports `**extra_kwargs` for wrapper-specific params (batch_size, vshift for SyncNet).
- **Stage 7** is first stage to fully conform to "ModelManager only" pattern (no direct wrapper fallback).
- **Stage 6** 6DRepNet PyTorch wrapper via third_party/sixdrepnet (official implementation).

### Files Modified Today
| File | Change |
|---|---|
| `models/wrappers/scrfd.py` | Fixed bbox decode: center+exp→distance format |
| `models/wrappers/sixdrepnet.py` | Rewrote: PyTorch via third_party/sixdrepnet; device="auto" |
| `models/wrappers/syncnet.py` | Rewrote: PyTorch third_party path, model_path param |
| `models/wrappers/hyperiqa.py` | Rewrote: koniq_pretrained.pkl + resnet50 local; offline deployment |
| `models/model_manager.py` | Rewrote: registry, **extra_kwargs, 4 models |
| `stages/stage_07_sync/stage.py` | Removed direct wrapper fallback |
| `configs/config.yaml` | Added sixdrepnet, syncnet, hyperiqa to models section; device to stage_07 |

### Files Created Today
| File | Purpose |
|---|---|
| `test_scripts/stage0_8_validation/*` | Stage 0-8 integration test |
| `test_scripts/debug_scrfd_output.py` | ONNX output format analysis |
| `test_scripts/debug_bbox_fix.py` | Bbox decode verification |
| `test_scripts/debug_6drepnet/*` | 6DRepNet tests |
| `test_scripts/debug_syncnet/*` | SyncNet tests + audit |
| `test_scripts/debug_hyperiqa.py` | HyperIQA debug (blocked) |
| `PROJECT_CONTEXT.md` | Updated for 2026-08-06 |
| `DEVELOPMENT_LOG.md` | This file |

---

## Problems Encountered

| # | Problem | Root Cause | Status |
|---|---|---|---|
| 1 | SCRFD bbox covering whole screen | Wrong `exp(dx2)` decode (center+size instead of 4-distance) | ✅ Fixed |
| 2 | 6DRepNet all-zero output | PyTorch broken; `_load_torch()` NotImplementedError | ✅ Fixed (PyTorch via third_party/sixdrepnet) |
| 3 | SyncNet NoOp fallback | Import path wrong; PyTorch broken without LD_PRELOAD | ✅ Fixed |
| 4 | PyTorch CUDA library incompatibility | PPU GPU, not NVIDIA; `libcusparse.so.12` symbol mismatch | ⚠️ Workaround: LD_PRELOAD |
| 5 | HyperIQA checkpoint won't load | `.pth` checkpoint arch mismatch with third_party code | ✅ Fixed (koniq_pretrained.pkl + base_model prefix strip) |
| 6 | Stage 3 only 1/6 scenes passing | SCRFD bbox bug caused false multi_face detection | ✅ Fixed (now 2/6) |
| 7 | insightface downloads buffalo_sc.zip twice | Stage 3 + Stage 5 each load SCRFD independently | ⚠️ Minor |

---

## 2026-08-11 — Stage 7 SyncNet Crop Expansion Experiment & Cleanup

### Experiment: Face Crop Expand Ratio Optimization

**Background**: SyncNet LSE-C abnormally low (0.615 on HDTF data, expected 3-10). Root cause identified as input distribution shift — Stage 5's tight bbox crop (mean pixel ~170) mismatched SyncNet's training distribution (full-frame, mean ~110).

**Experiment**: Tested 5 expand_ratio values on RD_Radio1_000 clip_0000.

| expand_ratio | lse_c | lse_d | pixel_mean | face_ratio | boundary_clip |
|---|---|---|---|---|---|
| 1.0 (original) | 0.615 | 11.69 | 169.6 | 100% | no |
| 1.5 | 5.742 | 7.62 | 154.3 | 44% | no |
| 1.55 | 5.891 | 7.53 | 152.7 | 42% | no |
| **1.6** | **6.034** | **7.51** | 151.1 | 39% | no |
| 1.8 | 5.677 | 8.02 | 146.6 | 32% | yes (bottom) |

**Conclusion**: `expand_ratio=1.6` is optimal. LSE-C improved from 0.615→6.034 (9.8x). At 1.8, performance declines due to boundary clipping and diluted lip signal.

### SyncNet Field Renaming

Renamed AVSyncInfo fields to match SyncNet paper terminology:
- `sync_score` → `sync_lse_c` (LSE-C = median(d) - min(d))
- `confidence` → `sync_lse_d` (LSE-D = min(d))
- Added `median_distance` for debug

### third_party Cleanup

- Removed all debug prints from `third_party/SyncNet/SyncNetInstance_calc_scores.py`
- Restored clean `evaluate()` returning `(offset, conf, minval, median_val)` — raw model outputs only
- Metric calculation (lse_c, lse_d) is in pipeline code: `SyncNetWrapper.predict()` and `sync_estimator.estimate_sync()`

### Stage 7: Face-Crop-Based Input

Changed Stage 7 from full-frame video input to face crops (per revised AISDD 3.11):
- Removed `_resolve_video_path()` — no longer depends on standardized full-frame video
- Added `_collect_face_crops()` + `_build_face_crop_video()` — builds SyncNet input from Stage 5 crops
- Updated config: `stage_05.crop.expand_ratio=1.6`, `boundary_check.fallback_ratio=1.5`

### Design Docs Updated

- `docs/03_Pipeline_Stage_Specification.md` — Stage 7 input redefined: FaceSequence + AudioInfo + ClipInfo
- `docs/05_Data_Flow_and_Output.md` — Data flow tables updated

### Files Modified

| File | Change |
|---|---|
| `third_party/SyncNet/SyncNetInstance_calc_scores.py` | Debug removal, clean restore |
| `stages/stage_07_sync/sync_estimator.py` | Face-crop-based video generation |
| `stages/stage_07_sync/stage.py` | Removed _resolve_video_path; added _resolve_temp_base |
| `data/models.py` | AVSyncInfo field rename; added QualityRankingInfo |
| `stages/stage_09_quality_fusion/fusion.py` | Updated to use sync_lse_c |
| `stages/stage_09_1_quality_ranking/*` | New Stage 9.1 (Quality Ranking enhancement) |
| `docs/03_Pipeline_Stage_Specification.md` | Stage 7 redesign |
| `docs/05_Data_Flow_and_Output.md` | Data flow update |
| `configs/config.yaml` | expand_ratio, boundary_check, quality_ranking sections |
| `tests/test_stage_07_sync.py` | Rewritten for face-crop-based Stage 7 |
| `tests/test_stage_09_1_quality_ranking.py` | New tests |

---

## Next Steps (Updated 2026-08-10)

1. **Stage 9 Quality Fusion** — Pure rule-based logic. All input types (PoseSequence, AVSyncInfo, FaceQualitySequence) and output type (QualityMetrics) are defined. No model dependencies.

2. **Pipeline Runner** — Wire Stage 0→9 for end-to-end testing.

3. **Stages 10-17** — Landmark, Alignment, FLAME, Gaze, Audio Feature, Dataset Assembly.

4. **ModelManager cleanup** — Stages 3, 5, 6, 8 still have direct wrapper loading fallbacks (legacy from when ModelManager was a skeleton). Should be cleaned to match Stage 7 pattern.

5. **requirements.txt** — Document all Python dependencies.

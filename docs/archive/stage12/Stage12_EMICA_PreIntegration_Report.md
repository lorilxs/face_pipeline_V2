# Stage12 EMICA Pre-Integration Report

**Date**: 2026-08-23
**Version**: v1.1 (Final Asset Check)
**Status**: ✅ PreIntegration Ready — 所有资产就绪，可进入下一阶段

---

## 目录

1. [设计文档修改情况](#1-设计文档修改情况)
2. [Stage12 新架构](#2-stage12-新架构)
3. [FLAMEFeature 设计](#3-flamefeature-设计)
4. [pose_code 设计说明](#4-pose_code-设计说明)
5. [为什么不能固定112维](#5-为什么不能固定112维)
6. [EMICA 环境检查](#6-emica-环境检查)
7. [PyTorch/CUDA 检查](#7-pytorchcuda-检查)
8. [源码完整性](#8-源码完整性)
9. [权重完整性](#9-权重完整性)
    - [9.4 Unified FLAME Asset Management](#94-unified-flame-asset-management-统一-flame-资产管理)
10. [SPECTRE 专项审计](#10-spectre-专项审计)
11. [缺失资源](#11-缺失资源)
12. [下一阶段实现建议](#12-下一阶段实现建议)

---

## 1. 设计文档修改情况

### 1.1 文档修改总表

| 文件 | 章节 | 是否需要修改 | 修改原因 |
|------|------|-------------|----------|
| **00_Project_Overview.md** | 0.7 Tech Stack (L618-622) | ✅ 需修改 | EMOCA/EMICA → EMICA 为主，含4个子模块 |
| **01_System_Architecture.md** | 1.3.1 (L176) | ✅ 需修改 | Stage 12 描述更新为 EMICA |
| **01_System_Architecture.md** | 1.6 (L489-497) | ✅ 需修改 | 增加 EMICA 模块化架构 |
| **02_Data_Model_and_Interface.md** | 2.8.1 FLAMEFeature (L761-783) | ❗ 需重写 | 固定维度→结构化 Feature Container |
| **02_Data_Model_and_Interface.md** | 2.3 (L163) | ✅ 需修改 | Stage 12 描述更新 |
| **03_Pipeline_Stage_Specification.md** | 3.18 Stage 12 (L1460-1485) | ❗ 需重写 | 旧 EMOCA → 新 EMICA |
| **04_Config_and_Model_Management.md** | 4.4 flame config (L244-248) | ❗ 需重写 | 模块级 enable/disable 配置 |
| **04_Config_and_Model_Management.md** | 4.5 (L638) | ✅ 需修改 | EMICA weights 路径 |
| **05_Data_Flow_and_Output.md** | 5.2 (L71), 5.9 (L478,493) | ✅ 需修改 | 输出根据配置变化 |
| **06_Engineering_Guideline.md** | 6.x | ✅ 需增加 | EMICA 模块化推理架构规范 |

### 1.2 关键修改摘要

**02_Data_Model_and_Interface.md** — 旧 FLAMEFeature → 新 FLAMEFeature:

旧: `expcode(50) + posecode(6) + mouth_parameters + full_parameters`
新: 结构化字段 (expression_code, pose_code, eye_pose, shape_code?, camera_parameters?, spectre_features?)

**03_Pipeline_Stage_Specification.md** — Stage 12 完全重写:

旧: EMOCA, Expression 50, Pose 6, Mouth Related
新: EMICA (MICA+DECA+EMOCA+SPECTRE), 模块化输出

**04_Config_and_Model_Management.md** — flame config:

旧: `model: EMICA, expression_dim: 50, pose_dim: 6`
新: 模块级 `modules.mica.enabled`, `modules.deca.enabled` 等
---

## 2. Stage12 新架构

### 2.1 EMICA 整体架构

```
Stage 12: EMICA FLAME Feature Extraction
│
├── Config: modules.{mica,deca,emoca,spectre}.enabled
│
├── Model Loading (ModelManager)
│   ├── EMICA checkpoint (867MB) — always loaded
│   ├── MICA checkpoint (mica.tar) — only if mica.enabled=true
│   ├── DECA encoder — only if deca.enabled=true
│   ├── EMOCA encoder — only if emoca.enabled=true
│   └── SPECTRE checkpoint — only if spectre.enabled=true
│
├── Inference Pipeline
│   ├── Input: Stage 11 AlignmentInfo → aligned face image [3,224,224]
│   ├── EMICA forward pass → codedict
│   │   ├── MICA (if enabled):  shape_code [300]
│   │   ├── DECA (if enabled):  jawpose[3], globalpose[3], cam[3]
│   │   ├── EMOCA:              expcode[100], eye_pose[6]
│   │   └── SPECTRE (if enabled): spectre features
│   └── Output: FLAMEFeature (structured)
│
└── Output
    └── FLAMEFeature per frame
```

### 2.2 模块职责

| 模块 | 作用 | 输出字段 | 维度 | 默认状态 |
|------|------|----------|------|----------|
| **MICA** | Identity / Face Shape | `shape_code` | 300 | disabled |
| **DECA** | Face Detail / Camera | `camera_parameters` | 3+3+3=9 | disabled |
| **EMOCA** | Expression | `expression_code`, `eye_pose` | 100+6 | **enabled** |
| **SPECTRE** | Speech-related Refinement | `spectre_features` | TBD | **enabled** |

### 2.3 推理流程

```
Aligned Face Image (Stage 11)
    ↓
EMICA FaceEncoder.encode()
    ↓
codedict = {
    'expcode':    Tensor[100],    # EMOCA
    'jawpose':    Tensor[3],      # DECA (if enabled)
    'globalpose': Tensor[3],      # DECA (if enabled)
    'cam':        Tensor[3],      # DECA (if enabled)
    'shape':      Tensor[300],    # MICA (if enabled)
    'eye_pose':   Tensor[6],      # EMOCA/EMICA
}
    ↓
FLAMEFeature:
    expression_code = codedict['expcode']     # 100
    pose_code = concat(jawpose, globalpose)    # 6
    eye_pose = codedict['eye_pose']            # 6
    shape_code = codedict['shape'] (if MICA)   # 300
    camera_parameters = ... (if DECA)          # N
    spectre_features = ... (if SPECTRE)        # M
```

### 2.4 EMICA 源码关键路径

| 组件 | 路径 |
|------|------|
| FaceReconstructionBase | `inferno/models/FaceReconstruction/FaceRecBase.py` |
| FaceEncoder | `inferno/models/FaceReconstruction/FaceEncoder.py` |
| MICA | `inferno/models/mica/mica.py` |
| DECA | `inferno/models/DECA.py` |
| SPECTRE | `inferno/spectre/src/spectre.py` |
| FLAME Model | `inferno/models/FLAME.py` |
| Load Model | `inferno_apps/FaceReconstruction/utils/load.py` |
| Demo (inference) | `inferno_apps/FaceReconstruction/demo/demo_face_rec_on_images.py` |
---

## 3. FLAMEFeature 设计

### 3.1 设计原则

1. **结构化**: 不再是固定长度 vector，而是命名字段
2. **模块化**: 每个字段对应一个模块，模块 disabled 时字段为 None
3. **可扩展**: 增加新模块只需增加 optional 字段
4. **Stage 12 不负责 flatten**: flatten 由 Stage 17 完成

### 3.2 字段定义

| 字段 | 类型 | 维度 | 模块 | 必选 |
|------|------|------|------|------|
| `frame_index` | `int` | 1 | — | ✅ |
| `expression_code` | `Tuple[float, ...]` | 100 | EMOCA | ✅ |
| `pose_code` | `Tuple[float, ...]` | 6 | DECA | ✅ |
| `eye_pose` | `Tuple[float, ...]` | 6 | EMOCA | ✅ |
| `shape_code` | `Optional[Tuple[float, ...]]` | 300 | MICA | ❌ |
| `camera_parameters` | `Optional[Tuple[float, ...]]` | N | DECA | ❌ |
| `spectre_features` | `Optional[Tuple[float, ...]]` | M | SPECTRE | ❌ |

### 3.3 Default Config 输出

默认配置 (MICA=false, DECA=false, EMOCA=true, SPECTRE=true):

```
FLAMEFeature(
    frame_index=0,
    expression_code=(100,),    # 100 dims
    pose_code=(6,),            # 6 dims (jaw+head)
    eye_pose=(6,),             # 6 dims
    shape_code=None,           # MICA disabled
    camera_parameters=None,    # DECA disabled
    spectre_features=(M,)      # SPECTRE enabled
)
```

Flattened: 100 + 6 + 6 + M = 112+M dims

**注意**: 112d 只是 EMOCA+SPECTRE 默认配置下的 flatten 结果，不是固定接口。

---

## 4. pose_code 设计说明

### 4.1 来源

根据 EMICA cfg.yaml:
- `jawpose: 3` — 下颌姿态 (axis-angle)
- `globalpose: 3` — 头部全局姿态 (axis-angle)

### 4.2 合并方式

```
pose_code = concat(jawpose[3], globalpose[3]) = Tensor[6]
```

其中:
- `pose_code[0:3]` = jawpose (下颌)
- `pose_code[3:6]` = globalpose (头部)

### 4.3 设计理由

**禁止分开设计**:
```
❌ jaw_pose: Tensor[3]
❌ head_pose: Tensor[3]
```

**统一设计**:
```
✅ pose_code: Tensor[6]  # 合并
```

原因:
1. 与旧 pipeline 兼容 (原来就是 posecode[:6])
2. 下游 Stage 17 期望统一的 pose 表示
3. 简化接口

---

## 5. 为什么不能固定112维

### 5.1 112d 的来源

当前默认配置下:
```
expression_code: 100 (EMOCA)
pose_code:        6 (DECA jaw+global)
eye_pose:         6 (EMOCA)
────────────────
Total:          112 dims
```

### 5.2 未来扩展

| 场景 | 新增维度 | 总维度 |
|------|----------|--------|
| 默认 | — | 112d |
| +MICA | +300 (shape_code) | 412d |
| +DECA | +9 (camera: 3+3+3) | 421d |
| +MICA+DECA | +300+9 | 421d |
| +Full | +300+9+M(SPECTRE) | 421+M d |

### 5.3 固定112d的危害

1. **不可扩展**: 无法在保持向后兼容的前提下增加 shape_code
2. **语义丢失**: 不知道哪些维度是 expression、哪些是 pose
3. **下游困惑**: Stage 17 需要硬编码维度索引
4. **违反 AISDD 设计原则**: 数据模型应工程化，而非 hardcode

### 5.4 正确做法

- Stage 12 输出**结构化** FLAMEFeature
- 每个字段有明确的语义和模块来源
- Stage 17 负责 flatten，根据 config 确定最终维度
---

## 6. EMICA 环境检查

### 6.1 环境信息

**Docker 容器**: `face_pipeline_lxs`
**EMICA 专用环境**: `source /workspace/emica_env/bin/activate`
**Python 版本**: 3.10.12

### 6.2 依赖包状态

| 包名 | 状态 | 版本 | 备注 |
|------|------|------|------|
| torch | ✅ | 2.4.1+cpu | ⚠️ CPU 版本，无 CUDA |
| torchvision | ✅ | 0.24.0 | |
| numpy | ✅ | 1.26.0 | |
| opencv-python | ✅ | 4.10.0.84 | |
| pytorch-lightning | ✅ | 已安装 | |
| omegaconf | ✅ | 已安装 | |
| chumpy | ✅ | 已安装 | |
| mediapipe | ✅ | 已安装 | |
| einops | ✅ | 0.6.1 | |
| scipy | ✅ | 1.11.3 | |
| transformers | ✅ | 5.2.0 | |
| fsspec | ✅ | 2026.2.0 | |
| scikit-learn | ✅ | 1.3.2 | |
| smplx | ❌ | **缺失** | FLAME 模型依赖 |
| trimesh | ❌ | **缺失** | 3D mesh 处理 |
| pyrender | ❌ | **缺失** | 渲染依赖 |
| face_alignment | ⚠️ | 已安装但崩溃 | `import` 导致 core dump |
| hydra-core | ❌ | **缺失** | 配置管理 |

### 6.3 关键发现

1. **PyTorch CPU 版本**: `torch 2.4.1+cpu`，无 CUDA。867MB 模型 CPU 推理不可行。
2. **缺失 FLAME 依赖**: `smplx` 和 `trimesh` 是 FLAME 核心依赖，必须安装。
3. **face_alignment 崩溃**: 已安装但导入即崩溃。
4. **pyrender 缺失**: 如需渲染可视化需要安装。

---

## 7. PyTorch/CUDA 检查

### 7.1 当前状态

```
PyTorch Version: 2.4.1+cpu
CUDA Available:  False
Device:          CPU
```

### 7.2 影响

- EMICA 推理需要 GPU 加速，CPU 推理极慢
- 867MB checkpoint 在 CPU 上推理不可行
- 需要安装 CUDA 版 PyTorch

### 7.3 建议

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## 8. 源码完整性

### 8.1 EMICA 核心源码

| 组件 | 路径 | 状态 |
|------|------|------|
| **FaceReconstructionBase** | `inferno/models/FaceReconstruction/FaceRecBase.py` | ✅ |
| **FaceEncoder** | `inferno/models/FaceReconstruction/FaceEncoder.py` | ✅ |
| **Model Loader** | `inferno_apps/FaceReconstruction/utils/load.py` | ✅ |
| **Inference Test** | `inferno_apps/FaceReconstruction/utils/test.py` | ✅ |
| **Demo (Images)** | `inferno_apps/FaceReconstruction/demo/demo_face_rec_on_images.py` | ✅ |
| **Demo (Video)** | `inferno_apps/FaceReconstruction/demo/demo_face_rec_on_video.py` | ✅ |

### 8.2 各子模块源码

| 模块 | 关键文件 | 状态 |
|------|----------|------|
| **MICA** | `mica.py`, `config.py`, `arcface.py`, `generator.py` | ✅ |
| **DECA** | `DECA.py`, `DecaEncoder.py`, `DecaDecoder.py`, `DecaFLAME.py` | ✅ |
| **EMOCA** | `EmoDECA.py`, `EmoCnnModule.py`, `EmoNetModule.py`, `EmoMLP.py` | ✅ |
| **SPECTRE** | `spectre.py`, `encoders.py`, `FLAME.py`, `renderer.py` | ✅ |
| **FLAME** | `FLAME.py`, `TemporalFLAME/` | ✅ |
---

## 9. 权重完整性

### 9.1 模型权重总表

| 模块 | 权重文件 | 路径 | 大小 | 状态 |
|------|----------|------|------|------|
| **EMICA** | `loss_total=3.88806534.ckpt` | `models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/` | 828 MB | ✅ |
| **MICA** | `mica.tar` | `models/MICA/model/mica.tar` | 480 MB | ✅ |
| **DECA** | — | (包含在 EMICA checkpoint 中) | — | ✅ |
| **EMOCA** | — | (包含在 EMICA checkpoint 中) | — | ✅ |
| **SPECTRE** | `deca_model.tar` | `third_party/inferno/spectre/spectre/data/deca_model.tar` | 415 MB | ✅ |
| **FLAME (统一)** | `generic_model.pkl` | `models/FLAME/geometry/` | 51 MB | ✅ |
| **FLAME (统一)** | `female_model.pkl` | `models/FLAME/geometry/` | 51 MB | ✅ |
| **FLAME (统一)** | `male_model.pkl` | `models/FLAME/geometry/` | 51 MB | ✅ |
| **FLAME Geometry** | `head_template.obj` | `models/FLAME/geometry/` | 551 KB | ✅ |
| **FLAME Geometry** | `landmark_embedding.npy` | `models/FLAME/geometry/` | 31 KB | ✅ |
| **FLAME Geometry** | `mediapipe_landmark_embedding.npz` | `models/FLAME/geometry/` | 4.5 KB | ✅ |

> ⚠️ 旧路径 `models/FLAME2020/` 已删除。所有模块统一使用 `models/FLAME/geometry/`。

### 9.2 EMICA Checkpoint 分析

`models/EMICA/EMICA-CVT_flame2020_notexture/cfg.yaml` 确认:
- `pl_module_class: FaceReconstructionBase`
- **Expression Encoder**: `expcode: 100` ✅
- **DECA Encoder**: `texcode: 50`, `jawpose: 3`, `globalpose: 3`, `cam: 3` ✅
- **MICA Encoder**: `shape: 300` ✅
- 828MB checkpoint 包含所有上述权重 ✅
- FLAME path: `FLAME/geometry/generic_model.pkl` (相对路径，无 "2020") ✅

### 9.3 SPECTRE 权重详情

详见 [第10节 SPECTRE 专项审计](#10-spectre-专项审计)。

**摘要**:
- 源码: ✅ 完整（来自 `https://github.com/radekd91/SPECTRE.git`）
- 预训练权重 `deca_model.tar`: ✅ **已就绪** (415MB, `spectre/spectre/data/deca_model.tar`)
- 辅助数据 (texture, mask, displacement, head_template, landmark): ✅ 完整
- FLAME 模型: ⚠️ SPECTRE config.py 硬编码 `data/FLAME2020/`，需通过 SpectrePreprocessor 运行时覆盖为 `models/FLAME/geometry/`

---

### 9.4 Unified FLAME Asset Management (统一 FLAME 资产管理)

**原则**: 整个 `Face_Pipeline_V2` 统一维护一个 FLAME geometry 资产，EMICA / MICA / DECA / SPECTRE 共享同一份 FLAME 模型文件。

**统一路径**: `models/FLAME/geometry/`

| 文件 | 用途 | 大小 |
|------|------|------|
| `generic_model.pkl` | FLAME 2020 通用模型 | 51 MB |
| `female_model.pkl` | FLAME 2020 女性模型 | 51 MB |
| `male_model.pkl` | FLAME 2020 男性模型 | 51 MB |
| `head_template.obj` | 头部拓扑模板 | 551 KB |
| `landmark_embedding.npy` | 3D landmark embedding | 31 KB |
| `mediapipe_landmark_embedding.npz` | MediaPipe landmark embedding | 4.5 KB |

**旧路径 `models/FLAME2020/`**: 已删除 ❌

**各模块 FLAME 路径引用方式**:

| 模块 | 路径配置 | 方式 |
|------|----------|------|
| **EMICA** | cfg.yaml: `flame_model_path: FLAME/geometry/generic_model.pkl` | 相对路径，无 "2020" ✅ |
| **MICA** | config.py: `data/FLAME2020/generic_model.pkl` | 硬编码 "FLAME2020" ⚠️ |
| **DECA** (inferno) | DECA.py: `FLAME/geometry/generic_model.pkl` | 相对路径 ✅ |
| **SPECTRE** | config.py: `data/FLAME2020/generic_model.pkl` | 硬编码 "FLAME2020" ⚠️ |
| **SPECTRE** (via SpectrePreprocessor) | 运行时覆盖: `resolve_path(cfg.flame_model_path)` | 可指向 `models/FLAME/geometry/` ✅ |

> **实现阶段注意事项**: MICA 和 SPECTRE 直接调用时需在初始化前覆盖 `flame_model_path` 为 `models/FLAME/geometry/generic_model.pkl`。

---

## 10. SPECTRE 专项审计

### 10.1 审计结论

```
SPECTRE 源码: ✅ 完整可用
SPECTRE 预训练权重: ✅ 已就绪 (deca_model.tar, 415MB)
SPECTRE FLAME 模型: ⚠️ config.py 硬编码 data/FLAME2020/，需运行时覆盖
SPECTRE 辅助数据: ✅ 完整
SPECTRE 子模块: ✅ 完整 (face_alignment, face_detection, av_hubert, VSR)
SPECTRE 整体状态: ✅ 可集成 (路径覆盖后即可使用)
```

### 10.2 源码目录结构

**路径**: `third_party/inferno/spectre/`

**来源**: `https://github.com/radekd91/SPECTRE.git` (via ghproxy)
**Git 分支**: `master`
**子模块**: 4 个

**数据目录**: 实际数据位于 `spectre/spectre/data/`（嵌套结构），包含 `deca_model.tar` (415MB) 等。config.py 的 `project_dir` 误指向外层 `spectre/`，因此 expects `spectre/data/`。需在调用时修正路径。

| 子模块 | 用途 | 状态 |
|--------|------|------|
| `external/face_alignment` | 人脸关键点检测 | ✅ |
| `external/face_detection` | 人脸检测 | ✅ |
| `external/av_hubert` | AV-HuBERT 唇语识别 | ✅ |
| `external/Visual_Speech_Recognition_for_Multiple_Languages` | 唇语识别框架 | ✅ |

**SPECTRE 自有 Python 文件 (29 个)**:

| 组件 | 文件 | 用途 |
|------|------|------|
| 主模型 | `src/spectre.py` | SPECTRE 类定义 |
| 编码器 | `src/models/encoders.py` | PerceptualEncoder + ResnetEncoder |
| FLAME | `src/models/FLAME.py` | FLAME 3D 人脸模型 |
| 损失函数 | `src/models/expression_loss.py` | 表情损失 (ExpressionLossNet) |
| 训练器 | `src/trainer_spectre.py` | 训练循环 |
| 渲染器 | `src/utils/renderer.py` | 3D→2D 渲染 |
| 配置 | `config.py` | 默认配置 (yacs) |
| 入口 | `main.py` | 训练入口 |
| 演示 | `demo.py` | 推理演示 |
| 数据集 | `datasets/` | LRS3, MEAD, TCDTIMIT |
| 工具 | `utils/` | 帧提取、唇语工具、AV-HuBERT 等 |

### 10.3 SPECTRE 模型架构

```
SPECTRE (nn.Module)
├── E_flame (ResnetEncoder)
│   └── ResNet50 → Linear(2048→1024) → Linear(1024→236)
│       └── 输出: n_shape(100)+n_tex(50)+n_exp(50)+n_pose(6)+n_cam(3)+n_light(27)
│
├── E_expression (PerceptualEncoder)
│   └── MobileNetV2/ResNet50 → Conv1d(temporal,kernel=5) → Linear(256→53)
│       └── 输出: exp(50) + jaw_pose(3)
│
├── flame (FLAME decoder)
├── flametex (FLAMETex, optional)
└── render (SRenderY)
```

**关键特性**:
- 时序模型: E_expression 使用 Conv1d(kernel_size=5) 进行时序建模
- 输入: K 帧连续图像 (K=20 默认), [B, T, 3, 224, 224]
- 需要 padding 处理: 首尾 2 帧因 Conv1d padding 无效
### 10.4 SPECTRE encode() 输出

```python
def encode(self, images):
    # images: [T, 3, 224, 224] (视频帧)
    parameters = self.E_flame(images)  # → [B, 236]
    codedict = decompose_code(parameters)  # → dict
    # codedict.keys() = ['shape','tex','exp','pose','cam','light']
    deca_exp = codedict['exp'].clone()  # [B, 50]
    deca_jaw = codedict['pose'][:,3:].clone()  # [B, 3]
    
    codedict['exp'], jaw = self.E_expression(images)  # [B,50], [B,3]
    codedict['pose'][:, 3:] = jaw  # 替换 jaw pose
    codedict['images'] = images
    
    return codedict, deca_exp, deca_jaw
```

**codedict 字段**:

| 字段 | 维度 | 含义 |
|------|------|------|
| `shape` | 100 | 身份/形状参数 |
| `tex` | 50 | 纹理参数 |
| `exp` | 50 | 表情参数 (E_expression 输出) |
| `pose` | 6 | [global_pose(3), jaw_pose(3)] |
| `cam` | 3 | 相机参数 |
| `light` | 27 | 光照参数 (9×3) |
| `images` | — | 原始输入图像 |

**额外返回值**: `deca_exp`(50dim), `deca_jaw`(3dim)

**关于 Audio**: SPECTRE **不需要音频输入**，仅处理视频帧。demo.py 中的 audio 仅用于输出视频时添加音频轨道。

### 10.5 SPECTRE 权重检查

| 文件 | 位置 | 状态 |
|------|------|------|
| **deca_model.tar** (预训练权重) | `spectre/spectre/data/deca_model.tar` | ✅ **已就绪** (415MB) |
| `generic_model.pkl` (FLAME) | `spectre/spectre/data/FLAME2020/` | ❌ **缺失** (需运行时覆盖) |
| `fixed_displacement_256.npy` | `spectre/spectre/data/` | ✅ |
| `head_template.obj` | `spectre/spectre/data/` | ✅ |
| `landmark_embedding.npy` | `spectre/spectre/data/` | ✅ |
| `texture_data_256.npy` | `spectre/spectre/data/` | ✅ |
| `uv_face_mask.png` | `spectre/spectre/data/` | ✅ |
| `uv_face_eye_mask.png` | `spectre/spectre/data/` | ✅ |
| `mean_texture.jpg` | `spectre/spectre/data/` | ✅ |
| `LRS3_lists.pkl` | `spectre/spectre/data/` | ✅ |

> **注意**: SPECTRE 数据目录实际位于 `spectre/spectre/data/`（嵌套结构），而 config.py 的 `project_dir` 指向外层 `spectre/`，因此 expects `spectre/data/`。但 deca_model.tar 等文件已在 `spectre/spectre/data/` 中。

**子模块权重 (已存在)**:
- `face_alignment`: 2dfan2.pth, 2dfan2_alt.pth, 2dfan4.pth ✅
- `face_detection`: mobilenet0.25_Final.pth, Resnet50_Final.pth, s3fd_weights.pth ✅

### 10.6 SPECTRE 依赖分析

**Python 包依赖** (requirements.txt):
| 包 | 用途 | 推理必需 |
|------|------|----------|
| `scikit_image` | 图像处理 | ✅ |
| `kornia` | 计算机视觉工具 | ✅ |
| `chumpy` | FLAME 模型加载 | ✅ |
| `librosa` | 音频处理 | ❌ (仅 demo) |
| `av` | 视频处理 | ❌ (仅 demo) |
| `loguru` | 日志 | ❌ (仅训练) |
| `tensorboard` | 可视化 | ❌ (仅训练) |
| `pytorch_lightning` | 训练框架 | ❌ (仅训练) |
| `opencv-python` | 图像 I/O | ✅ |
| `phonemizer` | 音素转换 | ❌ (仅训练) |
| `jiwer` | WER 评估 | ❌ (仅训练) |

**推理最小依赖**: torch, torchvision, numpy, scipy, scikit_image, opencv-python, kornia, chumpy, yacs, pytorch3d

### 10.7 SPECTRE 与 Inferno 集成关系

**inferno 框架中的 SPECTRE 引用**:

| 文件 | 用途 |
|------|------|
| `inferno/models/temporal/external/SpectrePreprocessor.py` | SPECTRE 预处理器 |
| `inferno/models/temporal/Renderers.py` | 引用 SPECTRE 渲染 |
| `inferno/datasets/LRS3DataModule.py` | SPECTRE 训练数据 |
| `inferno/datasets/FaceVideoDataModule.py` | 通用视频数据 |

**集成方式**:
- `SpectrePreprocessor` 导入: `from spectre.src.spectre import SPECTRE`
- 路径解析: `get_path_to_externals()` → `third_party/inferno/`
- 独立运行: SPECTRE **不依赖** inferno 框架，可独立加载

**EMICA 与 SPECTRE 的关系**:
- EMICA 的 FaceEncoder 不直接引用 SPECTRE
- SPECTRE 是独立模型，与 EMICA 并行使用
- 在 Stage12 中，SPECTRE 作为可选模块，独立推理

### 10.8 Stage12 集成 SPECTRE 所需资源清单

| 项目 | 状态 | 路径/说明 |
|------|------|-----------|
| 源码 | ✅ | `third_party/inferno/spectre/` |
| 配置 | ✅ | `config.py` |
| 预训练权重 (deca_model.tar) | ✅ | `spectre/spectre/data/deca_model.tar` (415MB) |
| FLAME 模型 | ⚠️ | 项目 `models/FLAME/geometry/` 有，需通过 SpectrePreprocessor 运行时覆盖路径 |
| 辅助数据 | ✅ | `spectre/spectre/data/` 完整 |
| 子模块 | ✅ | 含权重 |
| 推理入口 | ✅ | `spectre.encode(images)` |
| 输出接口 | ✅ | codedict + deca_exp + deca_jaw |
| 输入格式 | ✅ | [T, 3, 224, 224] tensor |
| 是否需要 audio | ❌ | 不需要 |
| 是否需要 inferno | ❌ | 可独立运行 |

### 10.9 SPECTRE 权重获取方案

**状态: 已完成 ✅**

`deca_model.tar` 已于 2026-08-23 下载至 `spectre/spectre/data/` (415MB)。

**FLAME 路径覆盖方案** (实现阶段):

SPECTRE 的 `config.py` 硬编码 `data/FLAME2020/generic_model.pkl`，但 inferno 的 `SpectrePreprocessor.py` 已在运行时覆盖:

```python
# SpectrePreprocessor.py line 45
spectre_cfg.model.flame_model_path = resolve_path(cfg.flame_model_path)
```

其中 `cfg.flame_model_path` 由 inferno 配置传入，指向 `models/FLAME/geometry/generic_model.pkl`。

**Stage12 直接调用时**，需在初始化 SPECTRE 前设置:
```python
spectre_cfg.model.flame_model_path = "models/FLAME/geometry/generic_model.pkl"
spectre_cfg.model.flame_lmk_embedding_path = "models/FLAME/geometry/landmark_embedding.npy"
```

> 不再需要 `models/FLAME2020/` 路径（已删除）。
## 11. 缺失资源

### 11.1 必须解决

| 资源 | 优先级 | 影响 | 解决方案 |
|------|--------|------|----------|
| **PyTorch CUDA 版本** | 🔴 高 | CPU 推理不可行 | 安装 CUDA 版 PyTorch |
| **smplx** | 🔴 高 | FLAME 模型无法加载 | `pip install smplx` |
| **trimesh** | 🟡 中 | 3D mesh 处理不可用 | `pip install trimesh` |
| **SPECTRE FLAME 路径覆盖** | 🟡 中 | 需在运行时覆盖 config.py 硬编码路径 | 调用前设置 `flame_model_path` 指向 `models/FLAME/geometry/` |

### 11.2 建议解决

| 资源 | 优先级 | 影响 | 解决方案 |
|------|--------|------|----------|
| **pyrender** | 🟢 低 | 渲染可视化不可用（推理不需要） | 可选 |
| **face_alignment** | 🟢 低 | 已崩溃，但项目用 MediaPipe | 修复或移除 |
| **hydra-core** | 🟢 低 | 配置管理，可替代 | 可选 |

## 12. 下一阶段实现建议

### 12.1 实现顺序

**Phase 1: 环境修复** (必须)
1. 安装 CUDA 版 PyTorch
2. 安装 smplx, trimesh
3. 获取 SPECTRE 模型权重

**Phase 2: 设计文档更新** (当前阶段)
1. 更新 01_System_Architecture.md
2. 重写 02_Data_Model_and_Interface.md 的 FLAMEFeature
3. 重写 03_Pipeline_Stage_Specification.md 的 Stage 12
4. 更新 04_Config_and_Model_Management.md
5. 更新 05_Data_Flow_and_Output.md
6. 更新 06_Engineering_Guideline.md

**Phase 3: 代码实现** (下一阶段)
1. 创建 `data/flame_models.py` — 新 FLAMEFeature dataclass
2. 创建 `models/wrappers/emica.py` — EMICA wrapper
3. 创建 `stages/stage_12_flame/stage.py` — Stage 12 processor
4. 更新 `configs/config.yaml` — EMICA 模块配置
5. 创建 `tests/test_stage12_emica.py` — 集成测试

**Phase 4: 验证**
1. 验证 EMOCA+SPECTRE 默认配置输出
2. 验证 MICA/DECA 可选模块
3. 验证与 Stage 17 的接口兼容性

### 12.2 关键实现注意事项

1. **EMICA 模型加载**: 使用 `FaceReconstructionBase.instantiate(cfg, checkpoint=...)`
2. **推理接口**: `model(batch, training=False, validation=False)` 返回 codedict
3. **输入格式**: `batch['image']` = `[B, 3, 224, 224]` tensor
4. **输出提取**:
   - `codedict['expcode']` → expression_code
   - `codedict['jawpose']` + `codedict['globalpose']` → pose_code
   - `codedict['shape']` → shape_code (MICA)
   - `codedict['cam']` → camera_parameters (DECA)
5. **SPECTRE 集成**: 需要单独调用 SPECTRE.encode() 方法
6. **Module 开关**: disabled 模块不加载、不推理、输出 None
7. **Stage 12 不 flatten**: 输出结构化 FLAMEFeature，flatten 由 Stage 17 负责

### 12.3 风险提示

| 风险 | 等级 | 说明 |
|------|------|------|
| EMICA checkpoint 兼容性 | 🟡 中 | 可能依赖特定 PyTorch/PyTorch-Lightning 版本 |
| SPECTRE FLAME 路径硬编码 | 🟡 中 | config.py 硬编码 `data/FLAME2020/`，需运行时覆盖 |
| CUDA 环境 | 🔴 高 | 当前 EMICA 环境无 CUDA，867MB 模型 CPU 推理不可行 |
| face_alignment 崩溃 | 🟢 低 | 不影响推理，EMICA 使用 MediaPipe |
| 依赖版本冲突 | 🟡 中 | smplx/trimesh 安装后可能与现有包冲突 |

---

## 附录 A: EMICA cfg.yaml 关键输出维度确认

来源: `models/EMICA/EMICA-CVT_flame2020_notexture/cfg.yaml`

```yaml
model:
  face_encoder:
    type: EmicaEncoder
    encoders:
      expression_encoder:       # EMOCA
        predicts:
          expcode: 100           # ✅ expression_code: 100
      mica_deca_encoder:
        encoders:
          deca_encoder:          # DECA
            predicts:
              texcode: 50
              jawpose: 3         # ✅
              globalpose: 3      # ✅
              cam: 3             # ✅
              lightcode: 27
          mica_encoder:          # MICA
            predicts:
              shape: 300         # ✅ shape_code: 300
```

**pose_code = jawpose(3) + globalpose(3) = 6**

---

## 附录 B: SPECTRE 输出维度 (已确认)

**SPECTRE 源码** (`spectre.py`):
- `encode()` 返回 `codedict, deca_exp, deca_jaw`
- `codedict` 包含:

| 字段 | 维度 | 说明 |
|------|------|------|
| `shape` | 100 | 形状参数 |
| `tex` | 50 | 纹理参数 |
| `exp` | 50 | 表情参数 (注意: 50 维，与 EMOCA 的 100 维不同) |
| `pose` | 6 | 姿态参数 (jaw(3) + head(3)) |
| `cam` | 3 | 相机参数 |
| `light` | 27 | 光照参数 |

**额外返回**:
- `deca_exp`: 50 维 (DECA 表情)
- `deca_jaw`: 3 维 (DECA 下颌姿态)

**与 EMICA 输出对比**:

| 参数 | EMICA (EMOCA) | SPECTRE | 兼容性 |
|------|---------------|---------|--------|
| expression_code | 100 | 50 | ❌ 维度不同，需分别存储 |
| pose_code | 6 | 6 | ✅ 一致 |
| shape_code | 300 (MICA) | 100 | 仅 MICA 输出 |
| camera_parameters | 3 (DECA) | 3 | ✅ 一致 |

---

## 附录 C: 需要修改的文件清单

| 文件 | 修改类型 | 优先级 |
|------|----------|--------|
| `docs/01_System_Architecture.md` | 更新 | 🔴 高 |
| `docs/02_Data_Model_and_Interface.md` | 重写 | 🔴 高 |
| `docs/03_Pipeline_Stage_Specification.md` | 重写 | 🔴 高 |
| `docs/04_Config_and_Model_Management.md` | 更新 | 🔴 高 |
| `docs/05_Data_Flow_and_Output.md` | 更新 | 🟡 中 |
| `docs/06_Engineering_Guideline.md` | 新增 | 🟡 中 |
| `docs/00_Project_Overview.md` | 更新 | 🟢 低 |
| `face_pipeline_v2/data/models.py` | 重写 FLAMEFeature | 🔴 高 |
| `configs/config.yaml` | 更新 flame config | 🔴 高 |
| `face_pipeline_v2/stages/stage_12_flame/` | 新建 | 🔴 高 |
| `face_pipeline_v2/models/wrappers/` | 新建 emica.py | 🔴 高 |

---

## 附录 D: 当前 Stage 12 代码状态

路径: `face_pipeline_v2/stages/stage_12_flame/`
内容: 只有空的 `__init__.py`
状态: 尚未实现

---

**报告结束**

*本报告由 Stage12 EMICA 预集成审查生成，只包含审查、检查、分析内容，不包含任何代码实现。*
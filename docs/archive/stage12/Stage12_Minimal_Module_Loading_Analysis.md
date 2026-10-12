# Stage12 最小模块加载策略分析报告

**Date**: 2026-08-24
**Version**: v1.0
**Status**: 只读架构验证，不修改代码

---

## 1. Stage12 最终输出定义

### 1.1 必选输出

| 字段 | 维度 | 来源 | 说明 |
|------|------|------|------|
| `expression_code` | `[B, 100]` | EMICA ExpressionEncoder | FLAME expression 100维，不使用 EMOCA 原始 expcode[50] |
| `pose_code` | `[B, 6]` | EMICA DECA + SPECTRE | `head_pose[3]` + `jaw_pose_final[3]` |
| `eye_pose` | `[B, 6]` | `torch.zeros(6)` | FLAME decoder 需要，当前无真实 gaze 模块 |

### 1.2 可选输出

| 字段 | 维度 | 来源 | 说明 |
|------|------|------|------|
| `shape_code` | `[B, 300]` | MICA | 作为扩展能力保留 |
| `camera_params` | `[B, 3]` | EMICA DECA | 可选 |

### 1.3 pose_code 组成

```
pose_code = head_pose[3] + jaw_pose_final[3]

head_pose:
  来源: EMICA DECA → globalpose [3]

jaw_pose_final:
  lambda * SPECTRE_jaw_pose [3] + (1-lambda) * EMICA_jaw_pose [3]

  其中:
    EMICA_jaw_pose: DECA → jawpose [3]
    SPECTRE_jaw_pose: SPECTRE E_expression → jaw [3]
```

---

## 2. 输出字段 → 模块依赖关系

```
expression_code[100]
    ↓
  EMICA ExpressionEncoder
    ├── ResNet50 backbone (独立)
    └── prediction head → 100-d

head_pose[3]
    ↓
  EMICA DECA
    └── ResNet50 backbone → prediction head → globalpose[3]

EMICA_jaw_pose[3] (jaw_pose 融合的 EMICA 分量)
    ↓
  EMICA DECA
    └── ResNet50 backbone (同上) → prediction head → jawpose[3]

SPECTRE_jaw_pose[3] (jaw_pose 融合的 SPECTRE 分量)
    ↓
  SPECTRE E_expression
    ├── MobileNetV2 backbone (默认) 或 ResNet50
    ├── Conv1d(kernel=5) temporal layer
    └── prediction head → 53-d → jaw[3]

shape_code[300] (可选)
    ↓
  MICA
    └── ResNet50 backbone + MICA decoder → 300-d

camera_params[3] (可选)
    ↓
  EMICA DECA
    └── ResNet50 backbone (同上) → cam[3]
```

---

## 3. 最小模块加载方案

### 3.1 当前训练需求

| 模块 | 是否需要 | 原因 |
|------|---------|------|
| **MICA** | ❌ 不需要 | `shape_code[300]` 是可选输出，非 Stage12 当前训练输入 |
| **EMICA DECA** | ✅ 需要 | 输出 `globalpose[3]` + `jawpose[3]` + `cam[3]` |
| **EMICA ExpressionEncoder** | ✅ 需要 | 输出 `expcode[100]` |
| **SPECTRE E_expression** | ✅ 需要 | 输出 `jaw[3]` (SPECTRE jaw_pose) |
| **SPECTRE E_flame** | ❌ 不需要 | 其 `pose[0:3]` 被 EMICA DECA 替代，`exp[50]` + `jaw[3]` 被 E_expression 覆盖 |
| **SPECTRE FLAME decoder** | ❌ 不需要 | 仅需要 encoder 输出 |
| **SPECTRE FLAMETex** | ❌ 不需要 | 纹理非必需 |
| **SPECTRE renderer** | ❌ 不需要 | 渲染非必需 |

### 3.2 最低加载清单

```
EMICA:
  [✗] MICA
  [✓] DECA (DecaEncoder)
  [✓] ExpressionEncoder

SPECTRE:
  [✓] E_expression (PerceptualEncoder)
  [✗] E_flame (ResnetEncoder)
  [✗] FLAME decoder
  [✗] FLAMETex
  [✗] Renderer
```
---

## 4. EMICA 源码支持情况

### 4.1 源码文件

| 组件 | 文件 | 行号 |
|------|------|------|
| EmicaEncoder | `third_party/inferno/inferno-master/inferno/models/FaceReconstruction/FaceEncoder.py` | L258-283 |
| MicaDecaEncoder | 同上 | L186-219 |
| MicaEncoder | 同上 | L144-185 |
| DecaEncoder | 同上 | L41-87 |
| ExpressionEncoder | 同上 | L222-229 |
| FaceReconstructionBase | `FaceRecBase.py` | L170-764 |

### 4.2 问题 1：EmicaEncoder 初始化时是否无条件加载 MICA / DECA / ExpressionEncoder？

**答案：是的，无条件加载全部三个子模块。**

源码证据：

```python
# FaceEncoder.py L258-264
class EmicaEncoder(FaceEncoderBase):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.mica_deca_encoder = MicaDecaEncoder(cfg=self.cfg.encoders.mica_deca_encoder)
        self.expression_encoder = ExpressionEncoder(cfg=self.cfg.encoders.expression_encoder)
```

`MicaDecaEncoder.__init__` 同样无条件加载 MICA + DECA：

```python
# FaceEncoder.py L186-191
class MicaDecaEncoder(FaceEncoderBase):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.mica_encoder = MicaEncoder(config=self.cfg.encoders.mica_encoder)
        self.deca_encoder = DecaEncoder(config=self.cfg.encoders.deca_encoder)
```

**不存在任何 `enabled` / `load_mica` / `skip` 等配置开关。**

### 4.3 问题 2：encode() forward 过程中是否无条件执行全部三个 encoder？

**答案：是的，无条件执行全部。** 调用链如下：

```
EmicaEncoder.encode(batch)                                    # L269
  ├── self.mica_deca_encoder.encode(batch)                    # L270
  │     ├── self.mica_encoder.encode(batch)                   # L211
  │     │     └── E_mica.encode(image, mica_image)            # L147-177
  │     │         → batch['shapecode'] = [B, 300]
  │     └── self.deca_encoder.encode(batch)                   # L213
  │           └── ResNet50 → prediction head
  │               → batch['texcode']    [B, 50]
  │               → batch['jawpose']    [B, 3]
  │               → batch['globalpose'] [B, 3]
  │               → batch['cam']        [B, 3]
  │               → batch['lightcode']  [B, 27]
  └── self.expression_encoder.encode(batch)                   # L271
        └── ResNet50 → prediction head
            → batch['expcode']    [B, 100]
```

**不存在任何 `if enable_mica:` / `if enable_deca:` 等条件分支。**

### 4.4 问题 3：是否可以通过现有 config 关闭 MICA？

**答案：不能。** 当前 EMICA cfg.yaml 结构：

```yaml
# cfg.yaml L90-130
encoders:
  mica_deca_encoder:
    encoders:
      deca_encoder:
        type: DecaEncoder
        trainable: false
      mica_encoder:
        type: MicaEncoder
        trainable: false
  expression_encoder:
    type: ExpressionEncoder
    trainable: true
```

- `trainable: false` 只控制梯度回传 (`requires_grad_(False)`)，**不控制模块是否加载或执行**。
- 没有 `enabled: false` 或 `load: false` 等配置项。
- `MicaDecaEncoder.__init__` 和 `encode` 均为硬编码，无条件执行。

### 4.5 问题 4：如果必须修改源码，需要修改哪些类？

**需要修改的类：**

| 类 | 文件 | 需要增加的逻辑 |
|----|------|---------------|
| `EmicaEncoder` | `FaceEncoder.py` L258 | `if enable_mica/deca/expression: load submodule` |
| `MicaDecaEncoder` | `FaceEncoder.py` L186 | `if enable_mica: load MicaEncoder`; `if enable_deca: load DecaEncoder` |
| `EmicaEncoder.encode()` | `FaceEncoder.py` L269 | `if self.mica_deca_encoder: encode()`; `if self.expression_encoder: encode()` |
| `MicaDecaEncoder.encode()` | `FaceEncoder.py` L210 | `if self.mica_encoder: encode()`; `if self.deca_encoder: encode()` |

**额外好处：**
- `MicaEncoder.encode()` 需要 `mica_image` 字段（FAN 预处理后的图像），如果跳过 MICA 则不需要提供此字段，简化数据预处理。


---

## 5. SPECTRE 源码分析

### 5.1 源码位置

| 组件 | 文件 | 行号 |
|------|------|------|
| SPECTRE 主类 | `third_party/inferno/spectre/src/spectre.py` | L31-259 |
| PerceptualEncoder (E_expression) | `third_party/inferno/spectre/src/models/encoders.py` | L8-49 |
| ResnetEncoder (E_flame) | 同上 | L52-70 |
| SPECTRE config | `third_party/inferno/spectre/config.py` | L1-182 |

### 5.2 SPECTRE 初始化同样无条件加载全部模块

```python
# spectre.py L59-77
def _create_model(self, model_cfg):
    self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
    self.E_expression = PerceptualEncoder(model_cfg.n_exp, model_cfg).to(self.device)
    self.flame = FLAME(model_cfg).to(self.device)
    if model_cfg.use_tex:
        self.flametex = FLAMETex(model_cfg).to(self.device)
    ...
    self._setup_renderer(model_cfg)  # L43: 无条件加载 renderer + masks + textures
```

`_setup_renderer` 无条件加载：
- `SRenderY` (PyTorch3D 渲染器)
- `uv_face_eye_mask.png`
- `uv_face_mask.png`
- `fixed_displacement_256.npy`
- `mean_texture.jpg`
- `dense_template` (texture_data_256.npy)

→ 即使只需要 E_expression，当前代码也会加载完整的 FLAME decoder + renderer + 所有辅助数据。

### 5.3 SPECTRE E_expression 架构

```python
# encoders.py L8-49
class PerceptualEncoder(nn.Module):
    def __init__(self, outsize, cfg):
        if cfg.backbone == "mobilenetv2":
            self.encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
            feature_size = 1280
        elif cfg.backbone == "resnet50":
            self.encoder = resnet.load_ResNet50Model()
            feature_size = 2048

        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.layers = nn.Sequential(nn.Linear(256, 53))

    def forward(self, inputs):
        # inputs: [T, 3, 224, 224]
        features = self.encoder(inputs).squeeze(-1).squeeze(-1)  # [T, feature_size]
        features = features.permute(1, 0).unsqueeze(0)            # [1, feature_size, T]
        features = self.temporal(features)                        # [1, 256, T]
        features = features.squeeze(0).permute(1, 0)              # [T, 256]
        parameters = self.layers(features)                        # [T, 53]
        parameters[..., 50] = F.relu(parameters[..., 50])         # jaw x >= 0
        return parameters[..., :50], parameters[..., 50:]         # exp[50], jaw[3]
```

**关键约束：** E_expression 的 `Conv1d(kernel_size=5, padding=2)` 要求序列输入。对于单帧推理 (T=1)，padding=2 会进行零填充，需要验证边界行为。

### 5.4 SPECTRE 默认 backbone

```python
# config.py L96
cfg.model.backbone = 'mobilenetv2'  # perceptual encoder backbone
```

SPECTRE 的 E_expression 默认使用 **MobileNetV2**（非 ResNet50），与 EMICA 的 ResNet50 不同。

---

## 6. 重复计算分析

### 6.1 当前架构中的 Backbone 使用情况

| 模块 | Backbone | 输出维度 | 共享？ |
|------|---------|---------|--------|
| EMICA DECA | ResNet50 | 2048 -> 86 (tex+jaw+pose+cam+light) | 独立 |
| EMICA ExpressionEncoder | ResNet50 | 2048 -> 100 (exp) | 独立 |
| SPECTRE E_flame | ResNet50 | 2048 -> 236 (shape+tex+exp+pose+cam+light) | 独立 |
| SPECTRE E_expression | MobileNetV2 (默认) | 1280 -> 256 -> 53 | 独立 |

### 6.2 冗余分析

**冗余 1：EMICA 内部双重 ResNet50**

```
EMICA 推理:
  image -> ResNet50 (DECA) -> [texcode, jawpose, globalpose, cam, lightcode]
  image -> ResNet50 (ExpressionEncoder) -> [expcode 100]
```

同一张图像被两个独立的 ResNet50 各处理一次。这是 EMOCA 架构的已知设计特点（DECA 冻结 + ExpressionEncoder 可训练）。ExpressionEncoder 的 `initialize_from()` 方法暗示它可以从 DECA encoder 初始化权重，说明两者 Backbone 结构兼容，理论上可以共享。

**冗余 2：EMICA DECA vs SPECTRE E_flame**

两者都是 ResNet50 -> FC 预测头，但：
- 预测目标不同（DECA 86维 vs SPECTRE 236维）
- 权重独立（来源不同：DECA 来自 EMICA checkpoint，SPECTRE 来自 `deca_model.tar`）
- SPECTRE E_flame 的输出在 Stage12 中**不需要**（globalpose 用 EMICA 的，exp 被 E_expression 覆盖）

**冗余 3：SPECTRE E_expression 默认 MobileNetV2**

与 EMICA 的 ResNet50 不同，无直接共享机会。

### 6.3 显存估算

| 模块 | 参数量 (约) | 显存 (fp32) |
|------|------------|------------|
| EMICA DECA (ResNet50) | ~25.5M | ~102 MB |
| EMICA ExpressionEncoder (ResNet50) | ~25.5M | ~102 MB |
| SPECTRE E_flame (ResNet50) | ~25.5M | ~102 MB |
| SPECTRE E_expression (MobileNetV2) | ~3.5M | ~14 MB |
| **合计** | **~80M** | **~320 MB** |

如果只加载 DECA + ExpressionEncoder + E_expression（不含 MICA + E_flame + FLAME decoder + renderer），可节省约 **50% 显存**。

### 6.4 可优化方向（仅分析，不实现）

1. **EMICA 内部共享 ResNet50**：DECA 和 ExpressionEncoder 共用 backbone，仅分别接不同预测头。
2. **E_flame 完全移除**：Stage12 不需要 SPECTRE E_flame 的任何输出。
3. **MICA 可选加载**：通过 config 开关控制 MICA 的加载和执行。

---

## 7. 源码修改需求总结

### 7.1 结论：需要源码级优化

当前源码不支持以下 Stage12 所需的"最小加载"能力：

| 需求 | 当前支持？ | 原因 |
|------|-----------|------|
| 加载 EMICA 时不加载 MICA | 否 | `EmicaEncoder` 硬编码加载 `MicaDecaEncoder`，后者硬编码加载 `MicaEncoder` |
| 加载 EMICA 时不执行 MICA encode | 否 | `encode()` 无条件调用 `mica_encoder.encode()` |
| 加载 SPECTRE 时仅加载 E_expression | 否 | `SPECTRE.__init__` 无条件加载 E_flame + FLAME + renderer + 所有辅助数据 |
| 通过 config 控制子模块加载 | 否 | 无 `enabled` 字段，`trainable=False` 仅控制梯度 |

### 7.2 需要修改的类（不实现，仅标注）

```
FaceEncoder.py:
  EmicaEncoder.__init__()      -> 增加 enable_mica / enable_deca / enable_expression 配置读取
  EmicaEncoder.encode()        -> 增加条件跳过
  MicaDecaEncoder.__init__()   -> 增加 enable_mica / enable_deca 配置读取
  MicaDecaEncoder.encode()     -> 增加条件跳过

spectre.py:
  SPECTRE._create_model()      -> 增加 enable_flame / enable_expression 配置读取
  SPECTRE._setup_renderer()    -> 增加条件跳过
```

---

## 8. 附录

### 8.1 EMICA 完整 encoder 输出字段

| 字段 | 来源 | 维度 | 类型 | Stage12 使用 |
|------|------|------|------|-------------|
| `shapecode` | MICA | `[B, 300]` | Tensor | 可选 |
| `texcode` | DECA | `[B, 50]` | Tensor | 不输出 |
| `jawpose` | DECA | `[B, 3]` | Tensor | jaw_pose 融合分量 |
| `globalpose` | DECA | `[B, 3]` | Tensor | head_pose |
| `cam` | DECA | `[B, 3]` | Tensor | 可选 |
| `lightcode` | DECA | `[B, 27]` | Tensor | 不输出 |
| `expcode` | ExpressionEncoder | `[B, 100]` | Tensor | expression_code |

### 8.2 SPECTRE encode() 输出 (codedict)

| 字段 | 维度 | 来源 | Stage12 使用 |
|------|------|------|-------------|
| `pose[:, 0:3]` | `[T, 3]` | E_flame | 不用 (用 EMICA 替代) |
| `pose[:, 3:6]` | `[T, 3]` | E_expression | SPECTRE jaw_pose 分量 |
| `exp` | `[T, 50]` | E_expression | 不用 (用 EMICA 100维替代) |
| `shape` | `[T, 100]` | E_flame | 不用 |
| `tex` | `[T, 50]` | E_flame | 不用 |
| `cam` | `[T, 3]` | E_flame | 不用 |
| `light` | `[T, 9, 3]` | E_flame | 不用 |

---

*报告结束。下一步：等待用户确认分析结论，决定是否修改源码。*

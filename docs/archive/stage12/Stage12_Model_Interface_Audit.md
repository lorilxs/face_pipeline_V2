# Stage12 模型接口审计报告

**Date**: 2026-08-23
**Version**: v1.0
**Status**: 只读审计，不修改代码

---

## 1. EMOCA / EMICA 接口审计

### 1.1 源码位置

| 组件 | 文件 |
|------|------|
| 入口 | `third_party/inferno/inferno-master/inferno/models/FaceReconstruction/FaceRecBase.py` |
| 编码器 | `third_party/inferno/inferno-master/inferno/models/FaceReconstruction/FaceEncoder.py` |
| 配置 | `models/EMICA/EMICA-CVT_flame2020_notexture/cfg.yaml` |
| 形状模型 | `third_party/inferno/inferno-master/inferno/models/temporal/TemporalFLAME.py` |

### 1.2 推理入口

```python
# FaceRecBase.py line 332
def forward(self, batch, training=True, validation=False, render=True, **kwargs):
    batch = self.encode(batch, training=training)   # 编码
    batch = self.decode(batch, training=training)    # 解码 (FLAME)
    batch = self.render(batch, training=training)    # 渲染
    return batch

# FaceRecBase.py line 514
def encode(self, batch, training=True, validation=False, return_features=False):
    batch = self.face_encoder(batch, return_features=return_features)
    return batch
```

### 1.3 EMICA Encoder 数据流

```
EmicaEncoder.encode(batch)
│
├── MicaDecaEncoder.encode(batch)
│   ├── MicaEncoder.encode(batch)        → batch['shapecode'] = mica_shapecode  [300]
│   └── DecaEncoder.encode(batch)        → batch['texcode']    [50]
│                                         → batch['jawpose']    [3]
│                                         → batch['globalpose'] [3]
│                                         → batch['cam']        [3]
│                                         → batch['lightcode']  [27]
│
└── ExpressionEncoder.encode(batch)      → batch['expcode']    [100]
```

### 1.4 编码器输出字段 (cf. cfg.yaml)

#### EMICA 完整 encoder 输出

| 字段 | 来源 | 维度 | 类型 | 说明 |
|------|------|------|------|------|
| `shapecode` | MICA | `[B, 300]` | Tensor | MICA 形状参数 |
| `texcode` | DECA | `[B, 50]` | Tensor | 纹理参数 |
| `jawpose` | DECA | `[B, 3]` | Tensor | 下颌姿态 (axis-angle) |
| `globalpose` | DECA | `[B, 3]` | Tensor | 头部全局姿态 (axis-angle) |
| `cam` | DECA | `[B, 3]` | Tensor | 正交相机参数 (scale, tx, ty) |
| `lightcode` | DECA | `[B, 27]` | Tensor | 球谐光照 (9×3) |
| `expcode` | ExpressionEncoder | `[B, 100]` | Tensor | EMOCA 表情参数 |

#### EMOCA-only encoder 输出 (无 MICA)

| 字段 | 维度 | 来源 |
|------|------|------|
| `texcode` | 50 | DECA |
| `jawpose` | 3 | DECA |
| `globalpose` | 3 | DECA |
| `cam` | 3 | DECA |
| `lightcode` | 27 | DECA |
| `expcode` | 100 | ExpressionEncoder |

### 1.5 eye_pose 审计

```python
# DecaFLAME.py line 79-81
default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
---

## 2. SPECTRE 接口审计

### 2.1 源码位置

| 组件 | 文件 |
|------|------|
| 入口 | `third_party/inferno/spectre/src/spectre.py` |
| 配置 | `third_party/inferno/spectre/config.py` |
| Demo | `third_party/inferno/spectre/demo.py` |

### 2.2 推理入口

```python
# spectre.py line 139
def encode(self, images):
    with torch.no_grad():
        parameters = self.E_flame(images)       # ResnetEncoder → [T, 236]

    codedict = self.decompose_code(parameters, self.param_dict)
    deca_exp = codedict['exp'].clone()          # [T, 50]
    deca_jaw = codedict['pose'][:, 3:].clone()  # [T, 3]

    codedict['exp'], jaw = self.E_expression(images)  # PerceptualEncoder → [T, 50], [T, 3]
    codedict['pose'][:, 3:] = jaw

    return codedict, deca_exp, deca_jaw
```

### 2.3 输入

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `images` | Tensor | `[T, 3, 224, 224]` | 连续 K 帧图像，首尾各 2 帧因 padding 无效 |

### 2.4 输出 codedict 字段

| 字段 | 维度 | 说明 |
|------|------|------|
| `shape` | `[T, 100]` | 形状参数 |
| `tex` | `[T, 50]` | 纹理参数 |
| `exp` | `[T, 50]` | 表情参数 (E_expression 输出) |
| `pose` | `[T, 6]` | 姿态 (`pose[:, 0:3]` = globalpose, `pose[:, 3:6]` = jawpose) |
| `cam` | `[T, 3]` | 相机参数 |
| `light` | `[T, 9, 3]` | 球谐光照 (reshape 后) |
| `images` | `[T, 3, 224, 224]` | 输入图像 (写入 codedict) |

### 2.5 额外返回值

| 字段 | 维度 | 说明 |
|------|------|------|
| `deca_exp` | `[T, 50]` | E_flame 原始表情 (被 E_expression 覆盖前 clone) |
| `deca_jaw` | `[T, 3]` | E_flame 原始下颌 (被 E_expression 覆盖前 clone) |

### 2.6 Demo 中的后处理

```python
# demo.py line 177-179
codedict, initial_deca_exp, initial_deca_jaw = spectre.encode(images_array)
codedict['exp'] = codedict['exp'] + initial_deca_exp
codedict['pose'][..., 3:] = codedict['pose'][..., 3:] + initial_deca_jaw
```

> **注意**: Demo 将 `deca_exp` + `deca_jaw` 加到 codedict 上，这是 DEMO 行为，非 `encode()` 本身。`encode()` 返回的 codedict 中 `exp` 和 `pose[3:]` 已经是 E_expression 的输出。

### 2.7 与 EMICA 输出对比

| 参数 | EMICA (EMOCA) | SPECTRE | 兼容性 |
|------|---------------|---------|--------|
| expression_code | 100 | 50 | ❌ 维度不同 |
| pose_code | 6 (jawpose+globalpose) | 6 (pose[:3]+pose[3:]) | ✅ 语义一致 |
| shape_code | 300 (MICA only) | 100 | ❌ 不同来源 |
| camera_parameters | 3 | 3 | ✅ 一致 |
| eye_pose | 无 (FLAME 常数) | 无 (FLAME 常数) | ✅ 均无 |

---

## 3. MICA 接口审计

### 3.1 源码位置

| 组件 | 文件 |
|------|------|
| 编码器 | `third_party/inferno/inferno-master/inferno/models/FaceReconstruction/FaceEncoder.py` (MicaEncoder) |
| 模型 | `third_party/inferno/inferno-master/inferno/models/mica/mica.py` |
| 配置 | `third_party/inferno/inferno-master/inferno/models/mica/config.py` |

### 3.2 推理入口

```python
# FaceEncoder.py line 147
class MicaEncoder(FaceEncoderBase):
    def encode(self, batch, return_features=False):
        with torch.no_grad():
            mica_encoding = self.E_mica.encode(image, mica_image)
            mica_decoding = self.E_mica.decode(mica_encoding, predict_vertices=False)
            mica_shapecode = mica_decoding['pred_shape_code']
            batch['shapecode'] = mica_shapecode   # ← shape code 输出
            return batch
```

### 3.3 MICA 输出

| 字段 | 维度 | 说明 |
|------|------|------|
| `shapecode` | `[B, 300]` | MICA 形状参数 (config: `n_shape: 300`) |

### 3.4 权重

| 文件 | 路径 | 大小 |
|------|------|------|
| `mica.tar` | `models/MICA/model/mica.tar` | 480 MB |

---

## 4. DECA 接口审计

### 4.1 源码位置

| 组件 | 文件 |
|------|------|
| 编码器 | `third_party/inferno/inferno-master/inferno/models/FaceReconstruction/FaceEncoder.py` (DecaEncoder) |
| 模型 | `third_party/inferno/inferno-master/inferno/models/DECA.py` |

### 4.2 推理入口

```python
# FaceEncoder.py line 67
class DecaEncoder(FaceEncoderBase):
    def encode(self, batch, return_features=False):
        image = batch['image']
        code_vec = self.encoder(image)                # ResNet50 → concatenated code
        batch = self._decompose_code(batch, code_vec) # 按 prediction_code_dict 拆分
        return batch
```

### 4.3 DECA 输出 (来自 EMICA cfg.yaml)

| 字段 | 维度 | 说明 |
|------|------|------|
| `texcode` | `[B, 50]` | 纹理参数 |
| `jawpose` | `[B, 3]` | 下颌姿态 (axis-angle) |
| `globalpose` | `[B, 3]` | 头部全局姿态 (axis-angle) |
| `cam` | `[B, 3]` | 正交相机参数 (scale, tx, ty) |
| `lightcode` | `[B, 27]` | 球谐光照系数 |

### 4.4 camera_parameters 详情

```
cam: [B, 3] = [scale, tx, ty]
```

正交投影相机，用于 3D→2D 投影：
```python
landmarks2d = util.batch_orth_proj(landmarks3d, codedict['cam'])[:, :, :2]
```
self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
```

**关键发现**: `eye_pose` **不是** encoder 的输出字段。它是 FLAME 模型内部的一个**固定常数参数** (全零 [6])，不可训练。

在 FLAME decode 时组合为完整 pose：
```python
# DecaFLAME.py line 242
full_pose = torch.cat([
    pose_params[:, :3],   # globalpose  [3]
    self.neck_pose,       # neck_pose   [3] (固定零)
    pose_params[:, 3:],   # jawpose     [3]
    eye_pose_params       # eye_pose    [6] (固定零)
], dim=1)  # → [15]
```

> **审计结论**: 如果 Stage12 需要输出 `eye_pose`，应使用 `torch.zeros(6)` 作为 placeholder，或从 FLAME 模型中读取 `model.flame.eye_pose`。`eye_pose` 不是 encoder 预测的结果。
---

## 5. 汇总：Stage12 可用输出字段

### 5.1 所有 encoder 输出汇总

| 字段 | 维度 | 来源 | 必选 | 说明 |
|------|------|------|------|------|
| `expcode` | `[B, 100]` | EMOCA (ExpressionEncoder) | ✅ | 表情参数 |
| `jawpose` | `[B, 3]` | DECA | ✅ | 下颌姿态 |
| `globalpose` | `[B, 3]` | DECA | ✅ | 头部全局姿态 |
| `texcode` | `[B, 50]` | DECA | ❌ | 纹理参数 |
| `lightcode` | `[B, 27]` | DECA | ❌ | 球谐光照 |
| `cam` | `[B, 3]` | DECA | ❌ | 相机参数 |
| `shapecode` | `[B, 300]` | MICA | ❌ | 形状参数 |
| `eye_pose` | — | FLAME 常数 | ⚠️ | 不是 encoder 输出 |

### 5.2 SPECTRE 独立输出

| 字段 | 维度 | 来源 | 说明 |
|------|------|------|------|
| `codedict['shape']` | `[T, 100]` | SPECTRE E_flame | 形状参数 |
| `codedict['tex']` | `[T, 50]` | SPECTRE E_flame | 纹理参数 |
| `codedict['exp']` | `[T, 50]` | SPECTRE E_expression | 表情参数 (时序) |
| `codedict['pose']` | `[T, 6]` | SPECTRE E_flame+E_expression | 姿态 (global+jaw) |
| `codedict['cam']` | `[T, 3]` | SPECTRE E_flame | 相机参数 |
| `codedict['light']` | `[T, 9, 3]` | SPECTRE E_flame | 光照参数 |
| `deca_exp` | `[T, 50]` | SPECTRE E_flame | 原始 DECA 表情 |
| `deca_jaw` | `[T, 3]` | SPECTRE E_flame | 原始 DECA 下颌 |

### 5.3 与 FLAMEFeature 设计映射

| FLAMEFeature 字段 | 来源 | 维度 | 状态 |
|------|------|------|------|
| `expression_code` | EMOCA `expcode` | 100 | ✅ 匹配 |
| `pose_code` | DECA `jawpose[3]` + `globalpose[3]` | 6 | ✅ 匹配 |
| `eye_pose` | ⚠️ 无 encoder 输出 | 6 | ⚠️ 需从 FLAME 模型读取或填零 |
| `shape_code` | MICA `shapecode` | 300 | ✅ 匹配 (MICA enabled) |
| `camera_parameters` | DECA `cam` | 3 | ✅ 匹配 (DECA enabled) |
| `spectre_features` | SPECTRE `codedict` | — | ✅ 匹配 (SPECTRE enabled) |

### 5.4 eye_pose 处理建议

`eye_pose` 在所有 FLAME 变体中都是固定常数 `torch.zeros([1, 6])`，不是 encoder 预测值。

**Stage12 实现建议**:
1. 在 FLAMEFeature 中保留 `eye_pose` 字段
2. 值设为 `torch.zeros(6)` 或从 FLAME 模型读取 `model.flame.eye_pose`
3. 如果未来需要动态 eye_pose，可扩展模块

---

## 6. 关键发现与风险

### 6.1 维度差异

| 参数 | EMICA (EMOCA) | SPECTRE | 说明 |
|------|---------------|---------|------|
| expression | 100 | 50 | ❌ 不可互换 |
| shape | 300 (MICA) | 100 (SPECTRE) | ❌ 不同来源 |
| pose | 6 | 6 | ✅ 语义一致 |

### 6.2 路径硬编码

| 模块 | 硬编码路径 | 影响 |
|------|-----------|------|
| SPECTRE config.py | `data/FLAME2020/generic_model.pkl` | 需运行时覆盖 |
| MICA config.py | `data/FLAME2020/generic_model.pkl` | 需运行时覆盖 |
| EMICA cfg.yaml | `FLAME/geometry/generic_model.pkl` | ✅ 相对路径，正确 |

### 6.3 数据目录差异

SPECTRE 数据实际位于 `spectre/spectre/data/`（嵌套结构），但 `config.py` 的 `project_dir` 指向外层 `spectre/`，导致 expects `spectre/data/`。`deca_model.tar` 已在 `spectre/spectre/data/` 中。

---

**报告结束**
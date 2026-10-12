# Stage12 Interface Design Review

**Date**: 2026-08-25
**Status**: 审查完成 — 待讨论

---

## 1. 当前 Stage12 目录结构

```
face_pipeline_v2/stages/stage_12_flame/
├── __init__.py          # 空文件
├── stage.py             # Stage12FLAME BaseStage 实现
└── processor.py         # extract_flame_parameters() 处理器
```

**现状**: 目录已存在，Stage12FLAME 已实现，但仅支持 EMICA 单模型。
- 输入：`List[AlignedFaceInfo]` (来自 Stage11)
- 输出：`List[FLAMEFeature]`
- 调用链：`Stage12FLAME._get_model()` → `EMICAWrapper` → `processor.extract_flame_parameters()`

**问题**: 未集成 SPECTRE，未实现 jaw fusion。

---

## 2. 已有 FLAMEFeature 定义

**位置**: `face_pipeline_v2/data/models.py` (行 605-654)

```python
@dataclass(frozen=True)
class FLAMEFeature:
    frame_index: int
    shapecode: Tuple[float, ...]           # [300] MICA identity
    expcode: Tuple[float, ...]             # [100] expression
    jawpose: Tuple[float, ...]             # [3] EMICA jaw pose
    globalpose: Tuple[float, ...]          # [3] head pose
    cam: Tuple[float, ...]                 # [3] camera
    texcode: Optional[Tuple[float, ...]] = None   # [50]
    lightcode: Optional[Tuple[float, ...]] = None  # [27]

    @property
    def posecode(self) -> Tuple[float, ...]:
        """globalpose(3) + jawpose(3) = 6 dims."""
        return tuple(self.globalpose) + tuple(self.jawpose)
```

### 与目标接口的对比

| 目标字段 | 当前 FLAMEFeature | 差异 |
|----------|-------------------|------|
| `expression_code` [100] | `expcode` [100] | ✅ 一致 |
| `pose_code` [6] = head_pose[3] + fused_jaw[3] | `posecode` property | ❌ jawpose 未融合 SPECTRE |
| `eye_pose` [6] zeros | 无 | ❌ 缺失 |
| `shape_code` [300] optional | `shapecode` [300] | ✅ 一致 |
| `camera_params` [3] optional | `cam` [3] | ✅ 一致 |
| 原始 EMICA jawpose [3] | `jawpose` | ⚠️ 融合后会覆盖 |
| 原始 SPECTRE jawpose [3] | 无 | ❌ 缺失 |

### 建议修改（最小侵入式）

新增字段，不改名现有字段（向后兼容）：

```python
@dataclass(frozen=True)
class FLAMEFeature:
    frame_index: int
    shapecode: Tuple[float, ...]              # [300] — 不变
    expcode: Tuple[float, ...]                # [100] — 不变
    jawpose: Tuple[float, ...]                # [3] — 改为 fused jaw
    globalpose: Tuple[float, ...]             # [3] — 不变 (head_pose)
    cam: Tuple[float, ...]                    # [3] — 不变
    texcode: Optional[Tuple[float, ...]] = None
    lightcode: Optional[Tuple[float, ...]] = None
    eye_pose: Optional[Tuple[float, ...]] = None        # [6] zeros — 新增
    spectre_jawpose: Optional[Tuple[float, ...]] = None  # [3] — 审计用

    @property
    def posecode(self) -> Tuple[float, ...]:
        """globalpose(3) + jawpose(3) = 6 dims (jawpose is now fused)."""
        return tuple(self.globalpose) + tuple(self.jawpose)
```

---

## 3. 模型 Wrapper 风格审查

### 3.1 现有 EMICAWrapper

**位置**: `face_pipeline_v2/models/wrappers/emica.py`

**接口**:
```
class EMICAWrapper:
    __init__(checkpoint_path, flame_path, device)
    load() -> None
    encode(images: Tensor) -> Dict[str, Tensor]    # shapecode, expcode, jawpose, ...
    decode(batch) -> Dict[str, Tensor]
    forward(images) -> Dict[str, Tensor]            # encode + decode
    extract_parameters(batch, frame_indices) -> List[Dict]
    preprocess_image(image, target_size) -> Tensor  # BGR uint8 → [-1,1]
```

**注意**: 当前使用 `FaceReconstructionBase.instantiate()` 完整加载，不需要修改。

### 3.2 需新增 SPECTREWrapper

**推荐接口** (与 EMICAWrapper 保持一致):

```
class SPECTREWrapper:
    __init__(checkpoint_path, flame_model_path, device)
    load() -> None
    encode(images: Tensor) -> Dict[str, Tensor]    # pose (含 head+jaw), exp, cam, ...
    extract_parameters(batch, frame_indices) -> List[Dict]
    preprocess_image(image, target_size) -> Tensor
```

**关键**: SPECTRE 的 `encode()` 返回 `pose` [...,6] = `head_pose[3] + jaw_pose[3]`。分离逻辑在 processor 中处理。

**加载方式**: 使用 `ResnetEncoder(outsize=n_param)` 直接加载 `E_flame` 权重，绕过 `pytorch3d` 渲染器。设置 `cfg.model.backbone = "resnet50"`。

### 3.3 ModelManager 注册

**位置**: `face_pipeline_v2/models/model_manager.py`

需新增 spectre 注册：
```python
"spectre": (
    "face_pipeline_v2.models.wrappers.spectre",
    "SPECTREWrapper",
),
```

---

## 4. config.yaml 审查

### 4.1 当前 flame 配置

```yaml
flame:
  model: EMICA
  device: cuda:0
  expression_dim: 50
  pose_dim: 6
```

**问题**: `expression_dim: 50` 不准确(实际 100)。缺少 SPECTRE 和 jaw fusion lambda。

### 4.2 建议新增 stage12 配置

```yaml
stage12:
  enabled: true

  backend:
    emica:
      enabled: true
      checkpoint_path: models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/...
      flame_path: models/FLAME/geometry/generic_model.pkl
      device: auto

    spectre:
      enabled: true
      checkpoint_path: third_party/inferno/spectre/spectre/data/deca_model.tar
      flame_model_path: models/FLAME/geometry/generic_model.pkl
      device: auto
      backbone: resnet50       # 避免 mobilenetv2 的 torch.hub 依赖

  jaw_fusion_lambda: 0.5       # λ: EMICA weight, (1-λ): SPECTRE weight
  batch_size: 8
  eye_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 固定 zeros
```

**设计说明**:
- `jaw_fusion_lambda`: 0.0 = 纯 SPECTRE jaw, 1.0 = 纯 EMICA jaw
- 保留旧 `flame:` section 作为向后兼容（`stage12` 优先）

---

## 5. 输出接口修改评估

### 5.1 下游消费者

| 消费者 | 消费字段 | 影响 |
|--------|----------|------|
| Stage13 Gaze | `AlignedFaceInfo` | ✅ 无影响 |
| Stage17 Fusion | `FLAMEFeature` → `TrainingSample.flame_feature` | ⚠️ 需确认 |
| 外部训练脚本 | 直接读取 FLAMEFeature 文件 | ⚠️ 新增字段不影响旧消费者 |

### 5.2 已知 Bug

`tests/test_datamodel.py` 行 158:
```python
FLAMEFeature(0, (0.0,) * 50, (0.0,) * 6, (0.0,) * 10, (0.0,) * 100)
```
与实际 FLAMEFeature 构造函数签名不匹配（缺少 `cam` 参数，维度也不对）。需在 Stage12 实现时一并修复。

---

## 6. 总结与行动项

### 需修改文件

| 文件 | 变更类型 | 内容 |
|------|----------|------|
| `data/models.py` | 修改 | FLAMEFeature 新增 `eye_pose`, `spectre_jawpose` |
| `models/wrappers/spectre.py` | **新建** | SPECTREWrapper |
| `models/wrappers/emica.py` | 无需修改 | 现有加载方式可用 |
| `models/model_manager.py` | 修改 | 注册 spectre |
| `stages/stage_12_flame/processor.py` | 修改 | 双模型推理 + jaw fusion |
| `stages/stage_12_flame/stage.py` | 修改 | 加载两个模型，传入 lambda |
| `configs/config.yaml` | 修改 | 新增 stage12 section |
| `tests/test_datamodel.py` | 修复 | FLAMEFeature 构造签名 |

### 关键设计决策待确认

1. **FLAMEFeature 字段命名**：保持 `expcode`/`jawpose`/`globalpose`，还是改为 `expression_code`/`head_pose`/`fused_jawpose`？
2. **jaw_fusion_lambda 默认值**：建议 0.5（等权重），实际值需通过实验确定
3. **SPECTRE 是否可选**：如果 SPECTRE checkpoint 不存在，是否 fallback 到仅 EMICA？
4. **eye_pose 来源**：是否确认固定为 zeros[6]？EMICA/SPECTRE 都不预测 eye_pose

### 不做的事

- ❌ 不重命名 FLAMEFeature 现有字段（避免破坏下游）
- ❌ 不修改 EMICAWrapper 加载逻辑（当前可用）
- ❌ 不实现完整推理代码（本次仅审查）
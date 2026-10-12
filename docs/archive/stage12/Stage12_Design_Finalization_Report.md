# Stage12 Design Finalization Report

Version: v1.0

Date: 2026-08-23

Status: **DESIGN FROZEN** ✅

---

## 1. Final Architecture

```
AlignedFaceInfo (Stage11 output)
    │
    ├──→ EMOCA Encoder (per-frame)
    │       ├── expression_code:  Tensor[100]  ← expcode
    │       ├── global_pose:      Tensor[3]    ← globalpose
    │       └── initial_jaw_pose: Tensor[3]    ← jawpose
    │
    └──→ SPECTRE Encoder (video sequence, temporal)
            └── refined_jaw_pose: Tensor[3]    ← speech-aware refinement
                    │
                    ▼
    final_jaw_pose = EMOCA_jaw + λ × (SPECTRE_jaw − EMOCA_jaw)
                    │
                    ▼
    pose_code = concat(final_jaw_pose, global_pose)  → Tensor[6]
                    │
                    ▼
              FLAMEFeature
```

---

## 2. Tensor Dimensions

| Field | Type | Dimension | Source | Notes |
|---|---|---|---|---|
| `expression_code` | Tensor | [100] | EMOCA expcode | 禁止 SPECTRE exp 融合 |
| `pose_code` | Tensor | [6] | EMOCA + SPECTRE | jaw[0:3] + global[3:6] |
| `eye_pose` | Tensor | [6] | FLAME default | zeros(6), 待 Stage13 替换 |
| `shape_code` | Optional[Tensor] | [300] | MICA | 仅 MICA.enabled=true |
| `camera_parameters` | Optional[Tensor] | [3] | DECA | 仅 DECA.enabled=true |
| `spectre_features` | Optional[dict] | — | SPECTRE | 原始输出，debug用 |

---

## 3. Module Responsibility

| Module | Role | Required | Output |
|---|---|---|---|
| **EMOCA** | Primary encoder | ✅ Yes | expression(100), global_pose(3), jaw_pose(3) |
| **SPECTRE** | Jaw refinement | ✅ Yes | refined_jaw_pose(3), exp(50), cam(3) |
| MICA | Shape | ❌ No (default) | shapecode(300) |
| DECA | Camera | ❌ No (default) | cam(3) |

---

## 4. Jaw Refinement Formula

```
final_jaw_pose = EMOCA_jaw_pose + λ × (SPECTRE_jaw_pose − EMOCA_jaw_pose)
```

| Parameter | Default | Range | Config Key |
|---|---|---|---|
| λ (lambda) | 0.6 | 0.5 ~ 0.7 | `flame.jaw_refinement.lambda` |

When `flame.jaw_refinement.enabled = false`:
```
final_jaw_pose = EMOCA_jaw_pose  (no refinement)
```

---

## 5. Config Design

```yaml
flame:
  model: EMOCA
  device: cuda:0
  expression_dim: 100
  pose_dim: 6

  modules:
    mica:
      enabled: false
    deca:
      enabled: false
    emoca:
      enabled: true
    spectre:
      enabled: true

  jaw_refinement:
    enabled: true
    lambda: 0.6
```

---

## 6. Key Design Decisions

### 6.1 expression_code 禁止融合 SPECTRE exp

| Source | Dimension | Latent Space |
|---|---|---|
| EMOCA | [100] | EMOCA expression latent space |
| SPECTRE | [50] | DECA expression latent space |

两个 latent space 不一致，禁止 concat 或替换。

### 6.2 eye_pose 全部为零

EMOCA/EMICA/SPECTRE 均不预测 eye_pose。FLAME 模型中注册为 `nn.Parameter(torch.zeros(1,6))`。

**当前**: `eye_pose = torch.zeros(6)`
**未来**: Stage13 gaze estimator 替换

### 6.3 flatten 不在此阶段执行

旧设计 `expcode[:50] + posecode[:6]` 在 Stage12 完成 flatten。
新设计要求 Stage12 保持结构化 `FLAMEFeature`，flatten 由 Stage17 负责。

### 6.4 shape_code 可选

MICA 默认关闭。使用泥脸（zero shape）替代。如需 identity shape，开启 `mica.enabled=true`。

---

## 7. Remaining Implementation Tasks

| # | Task | Status |
|---|---|---|
| 1 | 实现 `flame_feature.py` 定义 FLAMEFeature dataclass | 🔴 pending |
| 2 | 编写 EMOCA 推理 wrapper | 🔴 pending |
| 3 | 编写 SPECTRE 推理 wrapper | 🔴 pending |
| 4 | 实现 Stage12 processor (EMOCA + SPECTRE + jaw fusion) | 🔴 pending |
| 5 | 实现 config 解析与 module switch | 🔴 pending |
| 6 | 集成到 ModelManager | 🔴 pending |
| 7 | 单元测试与集成测试 | 🔴 pending |

---

## 8. Implementation Ready

**YES** ✅

设计冻结完成，所有接口、维度、融合公式、配置项已明确，可进入代码实现阶段。
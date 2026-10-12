# Stage12 Pose 根因审计报告

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


> 只读审计（本报告**不修改任何代码**，只新增审计脚本与文档）。目的：定位
> FLAME decoder 重建 mesh 时的三类异常——「表情正常」「表情+head_pose 只见
> 头顶」「表情+jaw_pose 下巴严重扭曲」——的根因，并给出优先级排序与修复
> 建议，等待确认后再动手改代码。
>
> 审计根目录：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2`
> 审计日期：2026-08-27

---

## 1. 当前问题描述

| 现象 | 表现 | 与数值的对应 |
|---|---|---|
| 仅 expression | 人脸正常 | `expcode[100]` 量级正常（std≈0.4–0.7），与 pose 无关 |
| expression + head_pose | 只剩头顶 | Stage12 存下的 `head_pose` ≈ **1.25–1.36 rad（71–78°）**，作为 FLAME global 旋转整体转头 ~75° |
| expression + jaw_pose | 下巴严重扭曲 | `jaw_pose`（融合）≈ **6.7–7.0 rad（382–400°）**，`spectre_jawpose` ≈ **11.1–11.7 rad（638–669°）**，远超物理可能 |
| expression + gaze | 正常 | 排除 gaze 参数问题 |

**排除项**：不是 FLAME decoder 的 pose 顺序问题（代码链路 `[global,jaw]`
正确，见 §2），不是 pose 顺序 swap（那是文档的错），不是 rotation
representation / slice 错误（SPECTRE `[200:206]` 切片正确，见
`docs/Stage12_SPECTRE_Pose_Format_Audit.md`）。

**根因（已实测证实）**：Stage12 给 EMICA / SPECTRE 的输入图统一归一化到
**`[-1,1]`**，而这两个模型（DECA/EMICA/SPECTRE 家族）都是在 **float
`[0,1]`** 图像上训练的。输入分布不匹配 → 两个 encoder 的输出整体爆炸 ~40×，
pose 值全部非物理。见 §6 与证据表。

---

## 2. 数据链路

```
EMICA / SPECTRE  encoder 输出
        ↓
face_pipeline_v2/models/wrappers/emica.py / spectre.py   （提取 pose）
        ↓
face_pipeline_v2/stages/stage_12_flame/processor.py       （jaw 融合）
        ↓  FLAMEFeature（data/models.py）
face_pipeline_v2/stages/stage_12_flame/runner.py:save_outputs
        ↓
outputs/<stage…>/<clip>/flame_features.npz
        ↓  integration.py:read_stage12_outputs
FLAMEFeature.posecode = globalpose[3] + jawpose[3]   （[global, jaw]）
        ↓
FLAME decoder（third_party/…/FLAME.py:191）           （期望 [global, jaw]）
```

**posecode 顺序全程 `[global(3), jaw(3)]`**，与 FLAME decoder 一致
（黄金依据见 `docs/Stage12_Jaw_Head_Pose_Audit.md` §2、§3，此处不再展开）。

Stage12 实际产出 npz 的路径是 **wrapper 路径**
（`scripts/stage0_17_e2e_stage12.py` → `ModelManager` → wrapper →
`processor.extract_flame_parameters()`）；standalone `runner.py` 的完整
SPECTRE 路径当前不可用（`torch.hub` 需联网），且 `spectre_encode()` 存在
传参与返回类型 bug（详见 SPECTRE 审计文档 §6）。

---

## 3. EMICA pose 统计

### 3.1 现状（输入 `[-1,1]`，即当前 Stage12 行为）

在 `face_pipeline_lxs` 容器 + emica_env 内用 `scripts/stage12_model_probe.py`
对真实对齐人脸实测（8 帧，与 npz 量级一致）：

| 字段 | shape | mean | std | min | max | norm 均值 |
|---|---|---|---|---|---|---|
| `jawpose` | [8,3] | -0.073 | 0.330 | -0.424 | +0.385 | **0.586 rad（33.6°）** |
| `globalpose` | [8,3] | -0.083 | 0.765 | -0.802 | +0.997 | **1.333 rad（76.4°）** |
| `expcode` | [8,100] | -0.111 | 0.684 | -1.942 | +2.690 | 6.92 |

逐帧 per-axis（`globalpose`）：`mean=[0.98, -0.79, -0.45]`，帧间
`std=[0.009, 0.007, 0.004]` —— **几乎恒定**（正是“只看头顶”的直接原因）。

### 3.2 修正后（输入 `[0,1]`）

| 字段 | norm 均值 | per-axis mean | 帧间 std |
|---|---|---|---|
| `jawpose` | **0.164 rad（9.4°）** | [0.152, 0.035, -0.048] | [0.054, 0.009, 0.014] |
| `globalpose` | **0.160 rad（9.2°）** | [0.110, -0.113, 0.028] | [0.010, 0.005, 0.005] |
| `expcode` | — | std 0.43 | — |

- `globalpose` 从 76° 掉到 **~9°**：对正面对齐脸，EMICA 输出接近恒定的
  **canonical 朝向**（≈[0.11,-0.11,0.03]），符合预期。
- `jawpose` 从 33.6° 掉到 **9.4°**，回到物理合理区。
- **结论**：EMICA 的 pose 本身没问题，是输入归一化把它放大到非物理。

### 3.3 npz 反推 EMICA jaw（现状，间接验证）

`fused = 0.4·emica + 0.6·spectre` → `emica ≈ (fused − 0.6·spectre)/0.4`，
8 个 clip 反推 `norm_mean ≈ 0.54–0.60 rad（31–34°）`，与上表 3.1 一致。

---

## 4. SPECTRE pose 统计

详见 `docs/Stage12_SPECTRE_Pose_Format_Audit.md`（格式/切片/单位专项审计）。
要点：

### 4.1 现状（输入 `[-1,1]`）

| E_flame 参数块 | 现状 | 说明 |
|---|---|---|
| pose [200:206] | norm **13.9 rad** | `global[0:3]` ≈ [-6.58, 0.09, -1.48]，`jaw[3:6]` ≈ [-7.55, -7.26, 6.12]（norm 12.1）|
| npz `spectre_jawpose` | norm **11.1–11.7 rad（638–669°）** | 8 clip 实测，per-axis ≈ [-6.5, -6.8, 6.4] |
| shape/tex/exp/cam/light | 全部爆炸（std 46–330） | 说明不是“只有 jaw 坏”，而是整个 encoder 输出错乱 |

### 4.2 修正后（输入 `[0,1]`）

| E_flame 参数块 | 修正后 | 说明 |
|---|---|---|
| pose [200:206] | norm **0.334 rad** | `global[0:3]` ≈ [0.31, -0.03, 0.05]，`jaw[3:6]` ≈ [-0.09, 0.02, -0.03]（norm **0.10 rad = 5.7°**）|
| shape/tex/exp/cam/light | 全部合理 | shape norm 3.5、cam mean 3.2、light norm 6.7 |

- 对照参考 `test_data/npy`：head ≈ 0.3 rad、jaw ≈ 0.08 rad —— **量级一致**。
- **结论**：SPECTRE 的 `[200:206]` 切片、`[global, jaw]` 顺序、axis-angle 弧度
  全部正确；唯一错误是**输入归一化**把 E_flame 推离训练分布。

---

## 5. Fusion 逻辑分析

`processor.py:93-96`：

```python
fused_jaw = (1.0 - jaw_fusion_lambda) * emica_jaw + jaw_fusion_lambda * spectre_jaw
# λ = jaw_fusion_lambda = 0.6（SPECTRE 权重）
```

| 检查项 | 结果 |
|---|---|
| 权重取值 | λ=0.6 → SPECTRE 主导（可配置，见 runner/meta `jaw_fusion_lambda`） |
| 维度 / broadcast | ✅ 两端都是 [3]，无广播错 |
| slice 错误 | ✅ 无 |
| **单位混合（degree+rad / 6D+axis-angle）** | ❌ **没有** degree+rad 混用；真正的错是**两个输入都被同一个归一化错误放大**（同因子，因此彼此“一致”，但绝对值错） |
| 融合后（现状） | 0.4×0.59 + 0.6×12.1 ≈ **7.4 rad** ≈ 实测 6.9 rad ✓（完全自洽） |
| 融合后（修正 [0,1]） | 0.4×0.164 + 0.6×0.099 ≈ **0.125 rad（7.2°）** ✓ 物理合理 |

**结论**：融合公式本身正确；融合产物非物理纯粹来自两个输入的归一化错误。

---

## 6. 根因判断（排序）

### 根因 #1（决定性，已实测证实）：输入归一化 `[-1,1]` vs 模型训练分布 `[0,1]`

- `wrappers/spectre.py:200-208` 与 `wrappers/emica.py:240-243` 都用
  `tensor / 127.5 - 1.0`（→ [-1,1]）。
- 参考训练管线（`spectre_backup/datasets/datasets.py`、
  `inferno-master/.../DecaDataModule.py`）经 skimage `warp()` 得到 **float
  [0,1]** 输入。
- 单帧对照实验：同一张图，`[-1,1]` 下 E_flame jaw[3:6] norm=12.1 rad、
  `[0,1]` 下 = **0.10 rad**；EMICA global 1.36 → **0.18 rad**。
- 影响面：**两个模型的全部输出**（不只是 pose），解释了所有症状。

### 根因 #2（次要，残余问题）：EMICA `globalpose` 是 canonical 朝向常量，动画需基线化

- 修正归一化后 EMICA `globalpose` ≈ [0.11,-0.11,0.03]（norm 0.18 rad ≈ 10°）
  对**正面对齐脸几乎恒定**。
- 直接用绝对 raw 值做动画会引入恒定 ~10° 偏移；建议动画用途**减中性帧
  （或均值）用增量**，或配合 EMICA 的 `cam` 一起做正交投影。

### 根因 #3（文档放大项）：三处文档把 pose 顺序写成 `[jaw, global]`

- `docs/02_Data_Model_and_Interface.md:774,834`、
  `docs/03_Pipeline_Stage_Specification.md:1555`、`DEVELOPMENT_LOG.md:79-94`
  一致写成 `pose_code = jaw_pose[0:3] + head_pose[3:6]`。
- 代码与 `docs/Stage12_14_DataFlow_Compatibility_Report.md` 都是 `[global, jaw]`。
- 数值修复后若仍按文档写下游，会再次把 pose 装反。

> 补充：standalone `runner.py` 的完整 SPECTRE 路径另有独立 bug
> （`spectre_encode` 传 dict / 返回 tuple），但不影响已产出的 e2e npz；
> 属于待修项，不列为本次 mesh 异常根因。

---

## 7. 修复建议（不在本次只读范围内执行，等确认）

### P0 —— 修正两个 wrapper 的输入归一化（根因 #1）

- `face_pipeline_v2/models/wrappers/spectre.py` `preprocess_image()`：
  `tensor / 127.5 - 1.0` → `tensor / 255.0`（float [0,1]）。
- `face_pipeline_v2/models/wrappers/emica.py` `preprocess_image()`：同上。
- 同步检查 `runner.py:preprocess_image`（同一处 `[-1,1]` 逻辑）。
- 修正后重跑 Stage12，目标数值：
  `head_pose` norm < 0.6 rad、`jaw_pose` norm < 0.8 rad、
  `spectre_jawpose` norm < 0.8 rad（参考：修正后 ≈ 0.18 / 0.13 / 0.10 rad）。

### P1 —— EMICA global 基线化（根因 #2）

- 动画/重建用途建议保存 **`globalpose - 中性帧(或均值)` 的增量**，或保留 raw
  `globalpose` + `cam` 并在下游用正交投影消费；
- 决策点：npz 里存 raw 还是增量？建议存 raw + 额外存 baseline（向后兼容）。

### P2 —— 统一文档 pose 顺序为 `[global, jaw]`（根因 #3）

- 修订 `docs/02:774,834`、`docs/03:1555`、`DEVELOPMENT_LOG.md:79-94`。

### P3 —— 收尾/回归

- 修复 standalone `runner.py` 的 `spectre_encode()`（传 tensor、解包 tuple），
  或明确下线该路径只留 wrapper 路径。
- 加网格级回归测试：用 `flame_features.npz` 解码顶点，校验 global 只绕根关节
  刚体旋转、jaw 只动下颌顶点、数值在物理区间。
- 用 `scripts/stage12_pose_audit.py` 做每次输出的自动异常检测（head>2 rad /
  jaw>1 rad 告警）。

---

## 附录 A. 交付物索引

| 文件 | 类型 | 说明 |
|---|---|---|
| `docs/Stage12_Pose_Root_Cause_Audit.md` | 本文档 | 根因审计（7 章） |
| `docs/Stage12_SPECTRE_Pose_Format_Audit.md` | 文档 | SPECTRE pose 格式专项审计（切片/顺序/单位/归一化证据） |
| `docs/Stage12_Jaw_Head_Pose_Audit.md` | 文档（此前会话） | pose 顺序链路审计 + 实测数值 + mesh 装配 |
| `scripts/stage12_pose_audit.py` | **新增审计脚本** | 离线扫描 npz，统计 + 异常帧告警（host Python） |
| `scripts/stage12_model_probe.py` | **新增审计脚本** | Docker/emica_env 内加载模型实测两种归一化输出 |

## 附录 B. 复现

```bash
# 1) 离线异常检测（host，只读 npz）
cd /mnt/home/lxs/face_pipeline/Face_Pipeline_V2
python3 scripts/stage12_pose_audit.py --json /tmp/pose_audit.json

# 2) 模型级归一化对照（Docker/emica_env，只读）
docker exec face_pipeline_lxs bash -c \
  'cd /workspace/Face_Pipeline_V2 && PYTHONPATH=/workspace/Face_Pipeline_V2 \
   /workspace/emica_env/bin/python scripts/stage12_model_probe.py \
   --face_dir outputs/stage0_11_batch/.../aligned_faces/..._scene_0000_clip_0001 \
   --emica_ckpt models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt \
   --spectre_tar third_party/inferno/spectre/spectre/data/deca_model.tar \
   --n_frames 8 --norm 0_1'
```


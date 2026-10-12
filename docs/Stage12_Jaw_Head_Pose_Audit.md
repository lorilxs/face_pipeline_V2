# Stage12 Jaw / Head Pose 参数提取审计报告

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


> **2026-08-27 更新（决定性根因已确认）**：本文档 §6 的候选根因已由实测定论——
> 真正根因是 **Stage12 对 EMICA/SPECTRE 的输入归一化用了 `[-1,1]`，而两个模型
> 都是在 float `[0,1]` 图像上训练的**（`/255` 即可修正）。修复后：
> EMICA global ≈ 0.16 rad、jaw ≈ 0.16 rad；SPECTRE jaw ≈ 0.10 rad（原 11.5 rad）。
> 详见 `docs/Stage12_SPECTRE_Pose_Format_Audit.md` 与
> `docs/Stage12_Pose_Root_Cause_Audit.md`。
>
> 只读审计（本报告不修改任何代码）。目的：诊断 FLAME 网格渲染问题
> （表情正常 / 表情+头部姿态只见头顶 / 仅下巴严重扭曲），并给出正确的
> pose 语义与装配方式。
>
> 审计对象根目录：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2`
> 审计日期：2026-08-27

---

## 0. 结论摘要

1. **Stage12 自身代码链路不存在 pose 顺序 swap** —— `flame_features.npz`
   分开保存 `head_pose[3]`/`jaw_pose[3]`，`FLAMEFeature.posecode` 为
   `[globalpose(3), jawpose(3)]`，与 FLAME decoder 期望的
   `[global, jaw]` 一致。**代码是对的。**
2. **文档是错的（而且是「危险地错」）** —— `docs/02_Data_Model_and_Interface.md`、
   `docs/03_Pipeline_Stage_Specification.md`、`DEVELOPMENT_LOG.md` 三处一致写成
   `pose_code = jaw_pose[0:3] + head_pose[3:6]`（jaw 在前）。任何按文档写的
   下游 mesh 脚本都会把 6 维 pose 装反，喂给 FLAME 后头部/下巴错位。
   （`docs/Stage12_14_DataFlow_Compatibility_Report.md` 反而是对的。）
3. **真正的「元凶」是提取出的 pose 数值本身非物理** —— 8 个真实 clip 实测：
   - `head_pose`（EMICA global）：≈ **1.25–1.32 rad（71–75°）**，且逐帧几乎恒定
     （帧间 std 仅 0.01–0.04 rad ≈ 0.6–2.3°）。
   - `spectre_jawpose`：≈ **11.1–11.7 rad（638–669°）**，物理上完全不可能。
   - `jaw_pose`（融合后）：≈ **6.7–7.0 rad（382–400°）**，主要由 SPECTRE 贡献
     （`fused ≈ 0.6 × spectre_jaw`）。
   - 对照参考（`test_data/npy/*.npy`，62 维 FLAME 参数）：head ≈ 0.3 rad（17°），
     jaw ≈ 0.08 rad（4.5°）。Stage12 的 head 大约 **4×**、jaw 大约 **50–90×** 偏大。
4. **症状与数值直接对应**：
   - 「表情 + 头部姿态 → 只见头顶」：把 ≈75° 的 `head_pose` 原样作为 FLAME
     global rotation（绕颈根关节）应用，整个头被转到约 75° 俯仰/侧倾，
     自然只能看到头顶。
   - 「仅下巴 → 严重扭曲」：把 ≈395°（≈49° 有效值，经 Rodrigues 2π 周期）
     的 `jaw_pose` 应用到下巴关节 + pose-corrective 形变，下颌区域被巨大常量
     偏移拉变形。
   - 「表情单独正常」：`expcode[100]` 数值量级正常（参考 std≈1.4，属合理范围），
     与 pose 无关。

**一句话**：先修 pose 数值的量纲/标定（重点：SPECTRE jaw ≈ 11 rad 的来历），
再统一文档与代码的 pose 顺序语义（[global, jaw]）；不要只调换 pose 顺序——
那样依旧会得到 75° 的头顶和 400° 的下巴。

---

## 1. 审计范围与方法

只读检查以下代码与数据：

| 类别 | 路径 |
|---|---|
| Stage12 处理器 | `face_pipeline_v2/stages/stage_12_flame/processor.py` |
| Stage12 runner（Docker 子进程，含 save_outputs） | `face_pipeline_v2/stages/stage_12_flame/runner.py` |
| Stage12 集成（npz 读取） | `face_pipeline_v2/stages/stage_12_flame/integration.py` |
| 数据模型 | `face_pipeline_v2/data/models.py`（`FLAMEFeature.posecode`） |
| 模型 wrapper | `face_pipeline_v2/models/wrappers/emica.py`、`spectre.py` |
| Stage17 消费端 | `face_pipeline_v2/stages/stage_17_fusion/processor.py`、`runner.py` |
| FLAME decoder（vendored 黄金依据） | `third_party/inferno/spectre/src/models/FLAME.py`、`third_party/inferno/inferno-master/inferno/models/FLAME.py`、`inferno/models/DECA.py` |
| SPECTRE 主模块 | `third_party/inferno/spectre/src/spectre.py` |
| 文档 | `docs/02_Data_Model_and_Interface.md`、`docs/03_Pipeline_Stage_Specification.md`、`docs/Stage12_14_DataFlow_Compatibility_Report.md`、`DEVELOPMENT_LOG.md` |
| 实测数据 | `outputs/**/flame_features.npz`（8 个 clip）+ 参考 `test_data/npy/*.npy` |

实测脚本（host Python，只读 npz）对所有可用 `flame_features.npz` 统计了
`head_pose` / `jaw_pose` / `spectre_jawpose` 的逐帧均值幅值、帧间 std，并反推
EMICA jaw 分量。

---

## 2. 黄金依据：FLAME decoder 的 pose 布局 = `[global(3), neck(3), jaw(3), eye(6)]`

三个独立来源一致，`pose_params` 前 3 维是 **global（头部全局旋转）**、后 3 维是
**jaw（下巴旋转）**，neck 固定为 0：

- `third_party/inferno/spectre/src/models/FLAME.py:191`
  ```python
  full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1),
                         pose_params[:, 3:], eye_pose_params], dim=1)
  ```
  → `pose_params[:, :3]` 进 **global** 槽，`pose_params[:, 3:]` 进 **jaw** 槽。
- `third_party/inferno/inferno-master/inferno/models/FLAME.py:220` —— 完全相同。
- `third_party/inferno/inferno-master/inferno/models/DECA.py:638`
  ```python
  posecode = torch.cat([global_pose, jaw_pose], dim=1)
  ```
  → DECA/EMICA 的 `posecode` 即 **`[global(3), jaw(3)]`**。

**结论**：任何 6 维 pose 向量喂给 FLAME 时，标准顺序是
`[global, jaw]`（neck 0、eye 0 由 FLAME 内部补齐）。

---

## 3. Stage12 代码链路（Face_Pipeline_V2）—— 内部一致、无 swap

### 3.1 提取（`stage_12_flame/processor.py:84-110`）

```python
emica_exp    = emica_output["expcode"][i]     # [100]  表情
emica_jaw    = emica_output["jawpose"][i]     # [3]    EMICA jaw
emica_global = emica_output["globalpose"][i]  # [3]    EMICA global
spectre_jaw  = spectre_output["jaw_pose"][i]  # [3]    SPECTRE jaw
fused_jaw = (1.0 - λ) * emica_jaw + λ * spectre_jaw   # λ=0.6
# FLAMEFeature(expcode=emica_exp, jawpose=fused_jaw,
#              globalpose=emica_global, spectre_jawpose=spectre_jaw, ...)
```

- EMICA wrapper（`models/wrappers/emica.py:192-196`）返回独立 key：
  `expcode` / `jawpose` / `globalpose` —— 顺序由 key 名确定，天然无歧义。
- SPECTRE wrapper（`models/wrappers/spectre.py:152-155`）：
  ```python
  pose = parameters[:, 200:206]      # n_shape=100 + n_tex=50 + n_exp=50 → pose 起点 200
  head_pose = pose[:, :3]            # [global]
  jaw_pose  = pose[:, 3:]            # [jaw]
  ```
  与 SPECTRE 源码一致（`spectre/src/spectre.py:143-150`：`pose` 6 维，
  `codedict['pose'][:, 3:]` 为 jaw）。
- runner 中取 `spectre_jaw = pose[3:6] if pose.shape[-1] >= 6 else pose[0:3]`
  （`runner.py` 的 `run_stage12`）——取的是**后半段**，正确。

### 3.2 落盘（`runner.py` 的 `save_outputs`，npz key）

```
expression_code [T,100]   ← EMICA expcode
jaw_pose        [T,3]     ← fused_jaw
head_pose       [T,3]     ← EMICA globalpose
eye_pose        [T,6]     ← zeros（占位）
spectre_jawpose [T,3]     ← SPECTRE raw jaw（审计轨迹）
shapecode       [T,300]   ← zeros（MICA 未加载）
frame_index / timestamp
```
**npz 里不存在拼接后的 `pose`/`posecode` 字段**，head 与 jaw 分开存储。

### 3.3 读回（`integration.py:175-186`）

`jawpose   = data["jaw_pose"]`，`globalpose = data["head_pose"]` —— 正确映射。

### 3.4 `FLAMEFeature.posecode`（`data/models.py:676-678`）

```python
@property
def posecode(self):
    return tuple(self.globalpose) + tuple(self.jawpose)   # [global, jaw] ✅
```

### 3.5 Stage17 消费（`stage_17_fusion/processor.py`）

`VISUAL_LAYOUT = {"expression": (0,100), "pose": (100,106), "gaze": (106,112)}`，
`pose = f.posecode`（[global, jaw] 6 维），注释亦写 `globalpose[3] + jawpose[3]`。

**小结**：从 wrapper → processor → runner → npz → integration → posecode → Stage17，
全程为 `[global, jaw]`，与 FLAME decoder 期望一致。**Stage12 代码里没有 swap。**

---

## 4. 文档与代码矛盾 —— 文档把 pose 顺序写反了（高危陷阱）

| 来源 | 写法 | 是否正确 |
|---|---|---|
| `face_pipeline_v2/data/models.py:676-678` | `posecode = globalpose[3] + jawpose[3]` | ✅ 正确（= FLAME） |
| `docs/Stage12_14_DataFlow_Compatibility_Report.md:63,84` | `posecode = globalpose[3] + jawpose[3]` | ✅ 正确 |
| `docs/02_Data_Model_and_Interface.md:774,834` | `pose_code = concat(jaw_pose, head_pose)` | ❌ 反了（jaw 在前） |
| `docs/03_Pipeline_Stage_Specification.md:1555` | `pose_code = jaw_pose[0:3] + head_pose[3:6]` | ❌ 反了（jaw 在前） |
| `DEVELOPMENT_LOG.md:79-94` | 声称 `posecode[0:3]=jaw`、`[3:6]=global`，且“DECA 的 `globpose=posecode[..., :3]` 命名有误导性（实际是 jaw）” | ❌ 反了 |

**DEVELOPMENT_LOG 的论断可直接被源码证伪**：`DECA.py:638` 明确
`posecode = torch.cat([global_pose, jaw_pose])` —— DECA 里 `posecode[..., :3]`
**就是** global，命名没有误导；SPECTRE FLAME `forward`（`FLAME.py:191`）也是
`pose_params[:, :3]` → global。DEVELOPMENT_LOG 结论“无需 swap”碰巧与代码一致，
但它的**语义前提是错的**，且这个错误语义被 docs/02、docs/03 原样复制。

**风险**：任何下游 mesh 解码脚本如果照文档把 pose 拼成
`[jaw_pose, head_pose]`（jaw 在前），再喂给 FLAME（期望 `[global, jaw]`），
就会把 head（~75°）塞进 jaw 槽、把 jaw（~395°）塞进 global 槽 ——
头部与下颌同时错位、进一步放大“只见头顶 / 下巴扭曲”现象。
**建议统一为 `[global, jaw]` 并修订三处文档。**

---

## 5. 实测：提取出的 pose 数值非物理（真正的根因）

### 5.1 Stage12 输出（8 个 clip 实测，`outputs/**/flame_features.npz`）

| clip | `head_pose` 幅值 | `jaw_pose`(融合) 幅值 | `spectre_jawpose` 幅值 |
|---|---|---|---|
| ..._scene_0000_clip_0001 | 1.32 rad (75.4°) | 6.99 rad (400.2°) | 11.68 rad (669.1°) |
| ..._scene_0002_clip_0000 | 1.28 rad (73.3°) | 6.86 rad (392.8°) | 11.48 rad (657.5°) |
| ..._scene_0002_clip_0001 | 1.24 rad (71.2°) | 6.66 rad (381.6°) | 11.14 rad (638.5°) |
| ..._scene_0002_clip_0002 | 1.25 rad (71.4°) | 6.82 rad (390.6°) | 11.41 rad (653.5°) |
| ..._scene_0002_clip_0003 | 1.26 rad (72.3°) | 6.85 rad (392.7°) | 11.47 rad (657.3°) |
| ..._scene_0002_clip_0004 | 1.27 rad (72.7°) | 6.84 rad (391.9°) | 11.45 rad (656.0°) |
| ..._scene_0002_clip_0005 | 1.24 rad (71.2°) | 6.67 rad (382.4°) | 11.17 rad (639.8°) |
| ..._scene_0000_clip_0001（另一批） | 1.31 rad (75.2°) | 6.89 rad (394.9°) | 11.53 rad (660.6°) |

补充统计（全部 clip）：
- `head_pose` **帧间 std ≈ 0.01–0.04 rad（0.6–2.3°）** —— 几乎恒定，是常量偏移而非逐帧头部运动；
- 由 `fused = 0.4·emica_jaw + 0.6·spectre_jaw` 反推：`emica_jaw ≈ 0.5–0.6 rad（31–33°）`，
  **`fused_jaw ≈ 0.6 × spectre_jaw`** —— 融合后的大下巴偏移基本全部来自 SPECTRE。

### 5.2 对照参考（`test_data/npy/100248_0_4.npy`、`100248_8_12.npy`，62 维 FLAME 参数）

| 参考 | head（50:53） | jaw（53:56） |
|---|---|---|
| 100248_0_4 | 0.29 rad (16.8°) | 0.08 rad (4.3°) |
| 100248_8_12 | 0.31 rad (17.5°) | 0.08 rad (4.5°) |

**Stage12 的 head 约为参考的 4×、jaw 约为参考的 50–90×。**

### 5.3 数值解读

- 头部绕颈根关节旋转 **~75°** 后，观察方向落在头顶一侧 → **“只见头顶”**；
- 下巴关节旋转 **~395°**（Rodrigues 公式按 2π 周期等效 ≈49°，但叠加 pose-corrective
  形变与近关节蒙皮）→ 下颌被巨大常量偏移拉变形 → **“仅下巴严重扭曲”**；
- `expcode[100]` 量级正常（每分量 std≈1.4，属合理范围）→ **“表情单独正常”**。

**即使把 pose 顺序改对，只要 `head_pose≈75°`、`jaw_pose≈395°` 仍原样写入，
三个症状依旧存在。因此顺序问题是次要、数值问题是主要的。**

---

## 6. 根因候选（按可能性排序，需在 emica_env 内进一步验证）

### 6.1 SPECTRE jaw 数值异常（最可疑，主导 fused_jaw）

- `spectre_jawpose ≈ 11.1–11.7 rad`，且 `fused_jaw ≈ 0.6 × spectre_jaw`。
- 若该值本意是 **11.5°（≈0.2 rad）**（SPECTRE/DECA 训练数据的 jaw 典型量级），
  则说明 SPECTRE 输出的 jaw 存在 **量纲/刻度问题**（度数未转弧度，或该 checkpoint
  的 pose 头输出并非弧度）。
- 也可能 `models/wrappers/spectre.py:152-153` 的 pose 切片
  `parameters[:, 200:206]` 与**实际 checkpoint**（`deca_model.tar`）的参数布局不符
  （n_shape/n_tex/n_exp 与 DECA 默认 100/50/50 不一致），导致 200:206 取到的不是
  pose，或 pose 内部顺序不是 `[global, jaw]`。
- **验证**：在 emica_env 内打印 SPECTRE `parameters` 的统计
  （`mean/std/maxabs`，特别看 200:206 是否像 axis-angle 弧度），并与官方 SPECTRE
  输出 `codedict['pose']` 对比；检查 `deca_model.tar` 的 key 与 wrapper 的
  `_POSE_START/_POSE_END` 是否匹配。

### 6.2 EMICA globalpose 是「规范朝向」常量而非逐帧头部运动

- `head_pose ≈ 1.25–1.32 rad` 且帧间 std 仅 0.6–2.3°——这是**常量偏移**。
- DECA/EMICA 的 `globalpose` 是与相机参数 `cam` 联合预测的；Stage12 把
  `cam` 置零（DECA 未开启）。脱离相机单独把 raw global 应用到 FLAME 世界坐标，
  会得到 ~75° 的旋转头。
- **验证**：对同一批 crop，用 EMICA 官方 decode（含 cam 投影）渲染对照，
  确认 raw global 是否应配合相机使用；对动画用途建议
  **减去中性帧（或均值）用增量**，而不是直接用 raw 绝对值。

### 6.3 文档顺序错误放大问题

- 见 §4。即使数值修复后，也必须把文档改为 `[global, jaw]`，否则下游脚本仍可能装反。

---

## 7. 正确的 FLAME mesh 装配（供下游 mesh 脚本对照）

以 `flame_features.npz` 解码一帧 mesh：

```python
import numpy as np
# 假设已加载 flame = FLAME(...)  (vendored: spectre/src/models/FLAME.py)

d = np.load(".../flame_features.npz", allow_pickle=True)
i = 0
expression = d["expression_code"][i]     # [100]
head       = d["head_pose"][i]           # [3]   ← EMICA global（先修量纲/基线！）
jaw        = d["jaw_pose"][i]            # [3]   ← 融合 jaw（先修量纲！）
eye        = d["eye_pose"][i][:6]        # [6]   ← Stage13 占位 zeros

# FLAME forward 期望 full_pose = [global(3), neck(3)=0, jaw(3), eye(6)=0]
full_pose = np.concatenate([head, np.zeros(3), jaw, np.zeros(6)])  # 顺序 [global, jaw]

vertices, landmarks2d, landmarks3d = flame.forward(
    shape_params=torch.zeros(1, 300),        # shapecode 目前为 zeros（MICA 未加载）
    expression_params=torch.from_numpy(expression).float().unsqueeze(0),
    pose_params=torch.from_numpy(full_pose).float().unsqueeze(0),
)
```

**注意**：在 pose 数值问题（§6.1/6.2）修复前，先用
`head - mean(head)`、`jaw - mean(jaw)` 的**增量**做临时验证，观察相对运动是否
正常；绝对姿势若需对齐正面，应减去中性帧或配合 EMICA 的 `cam`。

---

## 8. 建议的修复顺序（不在本次只读审计范围内）

1. **SPECTRE jaw**：核实 `deca_model.tar` 参数布局与 200:206 切片；确认 SPECTRE
   jaw 单位（弧度？），必要时在 wrapper/processor 转弧度后再融合。
2. **EMICA global 基线**：确认 `globalpose` 应配合 `cam` 使用；对动画输出建议保存
   **中性帧减除后**的相对 pose，或在下游统一做基线化。
3. **统一 pose 顺序文档**：修订 `docs/02:774,834`、`docs/03:1555`、
   `DEVELOPMENT_LOG.md:79-94` 为 `[global, jaw]`（与代码及
   `Stage12_14_DataFlow_Compatibility_Report.md` 一致）。
4. **加一个网格级回归测试**：用 `flame_features.npz` 解码顶点，
   校验 global 只绕根关节刚体旋转、jaw 只动下颌顶点、数值范围在物理合理区间
   （head<0.6 rad、jaw<0.8 rad）。

---

## 附录 A. 关键代码索引

| 位置 | 内容 |
|---|---|
| `spectre/src/models/FLAME.py:191` | FLAME `full_pose = [pose[:,:3], neck, pose[:,3:], eye]` → pose=`[global, jaw]` |
| `inferno/models/FLAME.py:220` | 同上（EMICA 侧 FLAME） |
| `inferno/models/DECA.py:638` | `posecode = cat([global_pose, jaw_pose])` → `[global, jaw]` |
| `spectre/src/spectre.py:143-150` | `pose` 6 维；`pose[:,3:]` 为 jaw |
| `wrappers/spectre.py:152-155` | `pose = parameters[:,200:206]`；`head=pose[:,:3]`, `jaw=pose[:,3:]` |
| `wrappers/emica.py:192-196` | 返回 `expcode/jawpose/globalpose` 独立 key |
| `processor.py:86-110` | EMICA 提取 + `fused_jaw = 0.4·emica + 0.6·spectre` |
| `runner.py:save_outputs` | npz keys：`expression_code/jaw_pose/head_pose/eye_pose/spectre_jawpose/shapecode` |
| `integration.py:175-186` | npz → FLAMEFeature（`jaw_pose→jawpose`, `head_pose→globalpose`） |
| `data/models.py:676-678` | `posecode = globalpose + jawpose`（`[global, jaw]`） |
| `stage_17_fusion/processor.py:25-33,231-249` | `pose = f.posecode`；visual_feature `exp[0:100]+pose[100:106]+gaze[106:112]` |
| `docs/02:774,834` / `docs/03:1555` / `DEVELOPMENT_LOG:79-94` | ❌ 错误写成 `[jaw, global]`，需修订 |

## 附录 B. 实测命令（可复现）

```bash
cd /mnt/home/lxs/face_pipeline/Face_Pipeline_V2
python3 - <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob('outputs/**/flame_features.npz', recursive=True)):
    d = np.load(f)
    for k in ('head_pose','jaw_pose','spectre_jawpose'):
        a = d[k]; mag = np.linalg.norm(a, axis=1)
        print(os.path.basename(os.path.dirname(f)), k,
              'mean_mag=%.2frad(%.1fdeg)' % (mag.mean(), np.degrees(mag.mean())),
              'frame_std=', a.std(0).round(3).tolist())
PY
```



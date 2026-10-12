# Stage12 SPECTRE Pose 格式审计报告

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


> 只读审计（不修改任何模型 / 推理逻辑 / FLAME decoder）。目的：回答
> 「SPECTRE 输出里的 global/jaw pose 到底是什么、在哪、单位是什么、
> `parameters[200:206]` 这个 slice 是否正确」。
>
> 审计根目录：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2`
> 审计日期：2026-08-27

---

## 1. 结论（TL;DR）

| # | 问题 | 结论 |
|---|---|---|
| 1 | `parameters[200:206]` 是不是 pose？ | ✅ 是。`236 = 100shape + 50tex + 50exp + 6pose + 3cam + 27light`，pose 恰好在 `[200:206]` |
| 2 | pose 内部顺序？ | ✅ `[global(3), jaw(3)]`（global 在前，jaw 在后）——与 FLAME decoder 一致 |
| 3 | 表示与单位？ | ✅ axis-angle（Rodrigues 向量），**弧度 rad** |
| 4 | `jaw = pose[:, 3:]`（`[203:206]`）是否正确？ | ✅ 正确 |
| 5 | 那为什么 jaw ≈ 11.5 rad（660°）？ | ❌ **不是 slice/顺序/表示错误，是「输入归一化不匹配」** —— 见 §5 |
| 6 | SPECTRE 参考预处理？ | ✅ 图像为 **float [0,1]**（`warp()` 输出），非 `[-1,1]` |

**一句话**：`[200:206]` slice 和 `[global, jaw]` 顺序都**没有错**；错的是
Stage12 把输入图归一化到了 `[-1,1]`，而 SPECTRE 的 `E_flame` 是在 `[0,1]`
图像上训练的。把输入改成 `[0,1]` 后，jaw 从 11.5 rad 掉到 **0.10 rad
（5.7°）**，所有参数块恢复物理合理。

---

## 2. 审计对象与依据

| 依据 | 路径 | 作用 |
|---|---|---|
| SPECTRE 官方 encoder | `third_party/inferno/spectre/src/models/encoders.py` | `ResnetEncoder(outsize=236)` 结构 |
| SPECTRE 官方 config | `third_party/inferno/spectre/config.py` | `n_shape=100,n_tex=50,n_exp=50,n_pose=6,n_cam=3,n_light=27` |
| SPECTRE 主模块 | `third_party/inferno/spectre/src/spectre.py` | `decompose_code()`、`pose[:,3:]` 为 jaw |
| inferno-master 副本 | `third_party/inferno/inferno-master/external/spectre/` | runner 实际解析到的 SPECTRE，布局一致 |
| checkpoint | `third_party/inferno/spectre/spectre/data/deca_model.tar` | `E_flame`（322 keys，ResNet50+regressor） |
| FLAME decoder | `third_party/inferno/spectre/src/models/FLAME.py:191` | `full_pose=[pose[:,:3], neck, pose[:,3:], eye]` |
| 实测探针 | `scripts/stage12_model_probe.py`（Docker/emica_env） | 两种归一化下的 E_flame 输出 |

---

## 3. checkpoint / E_flame 结构审计

`deca_model.tar` 顶层是一个 dict：

```
{'E_flame': <dict 322 keys>, 'E_detail': ..., 'D_detail': ...,
 'opt': ..., 'epoch': 0, 'iter': 39000, 'all_iter': ..., 'batch_size': ...}
```

- `E_flame` 就是 Stage12 使用的 `ResnetEncoder` 权值（322 个 key：
  ResNet50 骨干 + `layers.0`(2048→1024) + `layers.2`(1024→236)）。
- `epoch=0, iter=39000`：SPECTRE 用其自己的 temporal 流程训练的模型。
- 实测 `layers.2.bias` pose 段 `[200:206] = [0.026, -0.035, 0.037, 0.139,
  -0.019, -0.010]` —— 幅值小，属于训练好的模型（不是随机初始化）。

E_flame 输出维度推导（官方 config，`spectre/config.py:34-40`）：

```
param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
n_shape=100  n_tex=50  n_exp=50  n_pose=6  n_cam=3  n_light=27
n_param = 100+50+50+6+3+27 = 236
offset:  shape[0:100] tex[100:150] exp[150:200] pose[200:206]
        cam[206:209] light[209:236]
```

`decompose_code()`（`spectre.py:129-139`）按上表顺序切分 → 与
`SPECTREWrapper._POSE_START=200 / _POSE_END=206`（`wrappers/spectre.py:41-43`）
**完全一致**。

---

## 4. pose 内部布局 = `[global(3), jaw(3)]`

三个独立来源一致：

1. `spectre.py:143-150` `encode()`：`deca_jaw = codedict['pose'][:,3:]`，
   然后 `codedict['pose'][:, 3:] = jaw` —— 前 3 维 global、后 3 维 jaw。
2. inferno-master 副本 `external/spectre/src/spectre.py:139-150`：相同。
3. FLAME decoder `FLAME.py:191`：`full_pose = cat([pose_params[:,:3], neck,
   pose_params[:,3:], eye])` —— 前 3 维进 global 槽、后 3 维进 jaw 槽。

**表示**：axis-angle（旋转向量），单位 **rad**。`jaw_type='aa'`
（`config.py:41`）。E_flame 直接回归 axis-angle 向量，无 6D/quaternion/
euler/旋转矩阵，**不需要任何 rotation conversion**。

> ⚠️ 一个被忽略的细节：官方 `SPECTRE.encode()` 会用 **temporal
> `E_expression`** 的输出**覆盖** `pose[:, 3:]`（jaw）。即官方“最终 jaw”是
> `E_expression` 的时序结果，不是裸 `E_flame[203:206]`。Stage12 的
> `SPECTREWrapper` 只加载了 `E_flame`、直接取 `[203:206]`，这是**简化**
> （未用时序 jaw），但不是本次 11.5 rad 问题的原因——裸 E_flame 在正确的
> `[0,1]` 输入下 jaw ≈ 0.10 rad，依然合理。

## 5. 根因：输入归一化 `[-1,1]` vs 参考 `[0,1]`

### 5.1 参考预处理（SPECTRE 训练 / 官方数据管线）

`spectre_backup/datasets/datasets.py:81-110`（训练管线）与
`inferno-master/inferno/datasets/DecaDataModule.py:415`：

```python
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))
# skimage.transform.warp 对 uint8 输入返回 float，值域 [0,1]
images_list.append(cropped_image.transpose(2, 0, 1))
images_array = torch.from_numpy(np.array(images_list)).float()  # [0,1]
```

**SPECTRE（以及 DECA/EMICA 家族的 encoder）吃的是 float `[0,1]` 图像。**

### 5.2 Stage12 实际预处理（错误）

`face_pipeline_v2/models/wrappers/spectre.py:200-208`（`emica.py:240-243` 同）：

```python
tensor = tensor / 127.5 - 1.0      # [0,255] -> [-1, 1]   ← 与 SPECTRE 训练分布不符
```

### 5.3 实测对照（`scripts/stage12_model_probe.py --norm 0_1 / neg1_1`，8 帧）

同一批真实对齐人脸：

| E_flame 参数块 | 输入 `[-1,1]`（现状） | 输入 `[0,1]`（参考） |
|---|---|---|
| shape [0:100] | std **48.3**（norm 483） | std 0.35（norm 3.5） |
| tex [100:150] | std 48.8（norm 346） | std 0.30（norm 2.1） |
| exp [150:200] | std 46.6（norm 330） | std 0.34（norm 2.4） |
| **pose [200:206]** | **norm 13.9**（jaw[3:6] norm 12.1） | **norm 0.33**（jaw[3:6] norm 0.10） |
| cam [206:209] | mean +231（max +710） | mean 3.2（max 9.8） |
| light [209:236] | norm 477 | norm 6.7 |

`[0,1]` 下 **所有块都物理合理**；`[-1,1]` 下所有块爆炸 ~40×。

### 5.4 与参考数据对照

`test_data/npy/*.npy`（62 维参考 FLAME 参数）中 head ≈ 0.3 rad（17°）、
jaw ≈ 0.08 rad（4.5°）。`[0,1]` 归一化下的 SPECTRE：
global ≈ 0.32 rad（18°）、jaw ≈ 0.10 rad（5.7°）——**量级一致**。

---

## 6. Stage12 里 SPECTRE 的两条路径（都要记着）

| 路径 | 调用方 | 结果 |
|---|---|---|
| **wrapper**（`models/wrappers/spectre.py`） | e2e 官方驱动 `scripts/stage0_17_e2e_stage12.py`（ModelManager → wrapper）→ `processor.extract_flame_parameters()` | **e2e npz 的实际来源**。裸 E_flame `[203:206]` 当 jaw |
| **完整 SPECTRE**（runner CLI） | `face_pipeline_v2/stages/stage_12_flame/runner.py:_load_spectre` | 当前**不可用**：`from spectre.src.spectre import SPECTRE` 会解析到 `inferno-master/external/spectre`，其 `E_expression` 需要 `torch.hub` 联网下载（本环境无网络 → `URLError`）|

runner 独立路径还有两个**潜在 bug**（不影响已产出的 e2e npz，但属于审计发现）：
1. `runner.py:180-185 spectre_encode()` 把 `{"image": images}` 传成 `images`
   —— SPECTRE.`encode()` 期望 tensor；即便传入 tensor，`encode()` 返回的是
   **元组** `(codedict, deca_exp, deca_jaw)`，`spectre_out["pose"]` 会失败。
2. 因此 standalone runner 的 SPECTRE 分支实际走不到 `pose[3:6]` 正确取值。


## 7. 修复建议（不在本次只读范围内执行）

1. **改 wrapper 预处理为 `[0,1]`**：`spectre.py:200-208` 的
   `tensor / 127.5 - 1.0` 改为 `tensor / 255.0`（`emica.py:240-243` 同理）。
2. **统一 wrapper 与参考预处理语义**，并在 wrapper docstring 里写明
   “SPECTRE/DECA/EMICA 均以 float [0,1] 训练”。
3. **SPECTRE jaw 建议改用官方 `E_expression` 时序 jaw**（或至少在文档标注
   当前 `E_flame[203:206]` 是简化、逐帧、非时序）。
4. **修复/下线 standalone runner 的 spectre 路径**：修正
   `spectre_encode()`（传 tensor、解包元组），或统一走 wrapper。
5. 重新生成 `flame_features.npz`，用 `scripts/stage12_pose_audit.py`
   验证 jaw < 1 rad、head < 2 rad（理想 head<0.6、jaw<0.8 rad）。

---

## 附录 A. 复现命令

```bash
# 在 face_pipeline_lxs 容器 + emica_env 内：
cd /workspace/Face_Pipeline_V2
PYTHONPATH=/workspace/Face_Pipeline_V2 /workspace/emica_env/bin/python \
  scripts/stage12_model_probe.py \
  --face_dir outputs/stage0_11_batch/.../aligned_faces/..._scene_0000_clip_0001 \
  --emica_ckpt models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/\
      model-epoch=38-val/loss_total=3.88806534.ckpt \
  --spectre_tar third_party/inferno/spectre/spectre/data/deca_model.tar \
  --n_frames 8 --norm 0_1
```

（`--norm` 可选 `0_1` / `neg1_1`，默认 `neg1_1` 即当前 Stage12 行为。）


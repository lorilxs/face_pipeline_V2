# Stage12 Pose Fix Status Report

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


- 生成时间: 2026-08-28
- 检查模式: **只读 (read-only)** — 未修改任何 Python / 模型 / 文档代码
- 检查范围:
  - `face_pipeline_v2/models/wrappers/emica.py`
  - `face_pipeline_v2/models/wrappers/spectre.py`
  - `face_pipeline_v2/stages/stage_12_flame/runner.py`
  - `face_pipeline_v2/stages/stage_12_flame/processor.py`
  - `face_pipeline_v2/data/models.py`
  - `docs/03_Pipeline_Stage_Specification.md`
  - 既有验证报告 `docs/Stage0_17_End_to_End_Validation_Report.md` / `docs/Stage17_Real_Data_Validation_Report.md`

> 本报告对照既有审计结论 (根因 = 输入归一化 `/127.5-1.0` 应为 `/255.0`)，
> 逐项核查 P0/P1/P2/P3 修复在当前代码中的落地状态。

---

## 1. 当前修改状态总结

列定义 (schema):

| Field Name | 中文名称 | Type | Field Description |
|------------|----------|------|-------------------|
| task_item | 任务项 | string | 修复项目名称 |
| status | 状态 | string | 已完成 / 未完成 / 部分完成 |
| file | 文件 | string | 涉及文件 |
| detail | 详细说明 | string | 当前实现情况 |

修复项状态数据:

| task_item | status | file | detail |
|-----------|--------|------|--------|
| P0 输入归一化 (EMICA) | 已完成 | `emica.py:244` | `preprocess_image` 末尾 `tensor / 255.0` → [0,1]; docstring 明确 "matches DECA/EMICA/SPECTRE training distribution, NOT [-1,1]" |
| P0 输入归一化 (SPECTRE) | 已完成 | `spectre.py:209` | `preprocess_image` 末尾 `tensor / 255.0` → [0,1]; 同 EMICA 分布说明 |
| P0 输入归一化 (Runner) | 已完成 | `runner.py:159-166` | 模块级 `preprocess_image`: BGR→RGB → resize → permute → `/ 255.0` → unsqueeze; EMICA/SPECTRE 共用同一 batch_tensor |
| P1 EMICA global pose baseline 修正 | 未完成 | `runner.py:173-177,236`; `processor.py:88,104` | `emica_encode` 直接返回 `model.encode(images)` 原始输出; `run_stage12` 取 `emica_out["globalpose"]` 原值写入 `head_pose`, 全链路无任何 baseline / mean 偏移 |
| P2 文档 pose 顺序描述修正 | 未完成 | `docs/03_Pipeline_Stage_Specification.md:1555` | 文档仍写 `pose_code = jaw_pose[0:3] + head_pose[3:6]` ([jaw,head]), 与代码 `models.py:676-678` 的 [global,jaw] 及 FLAME 要求相反 |
| P3 SPECTRE standalone runner bug 修复 | 部分完成 / 已回归 | `runner.py:120-153` | 模块级 `return spectre` SyntaxError 已删除 (脚本可解析); 但 `_load_spectre` 函数体内**缺少 `return spectre`** → 隐式返回 `None` → `spectre_encode(None,...)` 触发 `AttributeError` 崩溃 |

---

## 2. P0 输入归一化检查

列定义:

| Field Name | 中文名称 | Type | Field Description |
|------------|----------|------|-------------------|
| model | 模型 | string | EMICA/SPECTRE |
| normalization | 输入归一化 | string | 当前实现 |
| expected | 期望值 | string | 正确训练分布 |
| result | 结果 | string | PASS/FAIL |

检查数据:

| model | normalization | expected | result |
|-------|---------------|----------|--------|
| EMICA (`emica.py:243-245`) | `tensor / 255.0` → [0,1] float | [0,1] (DECA/EMICA/SPECTRE `skimage.warp` 训练分布) | PASS |
| SPECTRE (`spectre.py:208-209`) | `tensor / 255.0` → [0,1] float | [0,1] | PASS |
| Runner preprocess (`runner.py:162-165`) | BGR→RGB → resize → permute → `/ 255.0` → [0,1] | [0,1] | PASS |
| Processor (`processor.py:66`) | 复用 `emica_model.preprocess_image` (/255.0) → [0,1]; EMICA/SPECTRE 共用同一 batch | [0,1] | PASS |

关键证据 (`emica.py:243-245`):
```python
tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
tensor = tensor / 255.0           # [0, 255] -> [0, 1] float (matches DECA/EMICA/SPECTRE training distribution)
tensor = tensor.unsqueeze(0)        # [C, H, W] -> [1, C, H, W]
```

关键证据 (`runner.py:159-166`):
```python
def preprocess_image(img_bgr, target_size=(224,224)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size[::-1])
    tensor = torch.from_numpy(img_resized.astype(np.float32)).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)
```

> 全仓库范围内未检出残留的 `/ 127.5 - 1.0` 归一化路径 (3 处 preprocess_image 均为 `/ 255.0`)。P0 归一化修复已全面落地。

## 3. Head Pose / Jaw Pose 来源检查

列定义:

| Field Name | 中文名称 | Type | Field Description |
|------------|----------|------|-------------------|
| output_field | 输出字段 | string | head_pose/jaw_pose |
| source | 来源 | string | EMICA/SPECTRE/Fusion |
| dimension | 维度 | string | shape |
| status | 状态 | string | 当前状态 |

检查数据 (依据 `runner.py:226-259` 与 `processor.py:78-111`):

| output_field | source | dimension | status |
|--------------|--------|-----------|--------|
| `head_pose` | EMICA `globalpose` (`runner.py:236,255`; `processor.py:88,104`) | [3] axis-angle (`aa`) | 正确: 来自 EMICA DecaEncoder globalpose, 原值直写 |
| `jaw_pose` | Fusion: `(1-λ)·EMICA_jaw + λ·SPECTRE_jaw`, λ=0.6 → 0.4·EMICA + 0.6·SPECTRE (`runner.py:246-247,254`; `processor.py:95-96,103`) | [3] axis-angle | 正确(逻辑): 方案A 融合; **但当前 runner 因 P3 在 `spectre_encode`(L227) 即崩溃, 无法实际产出** |
| `eye_pose` | zeros(6) (`runner.py:256`; `processor.py:106`) | [6] | 占位: EMICA/SPECTRE 均不预测; Stage13 gaze 不在此处注入 |
| `shapecode` | zeros(300) (`runner.py:258`) | [300] | MICA 未加载 |

SPECTRE jaw 抽取约定 (`runner.py:238-240`): 若 `spectre_out` 含 6 维 `pose`, 取 `pose[3:6]` 为 jaw (即 SPECTRE/DECA 约定 pose=[global(3),jaw(3)]), 与 wrapper `spectre.py:153-155` (`head_pose=pose[:,:3]`, `jaw_pose=pose[:,3:]`) 一致。

> **注意**: 由于 P3 (`_load_spectre` 返回 `None`), 当前 `runner.py` 在 `run_stage12` 的 `spectre_encode(spectre_model, batch_tensor)`(`runner.py:227`) 处即 `AttributeError` 崩溃,
> 上述 `head_pose`/`jaw_pose` 无法用**当前代码**重新生成。`outputs/` 下既有 npz 为 P3 回归前的旧产物 (见 §6)。

---

## 4. Jaw 策略检查

列定义:

| Field Name | 中文名称 | Type | Field Description |
|------------|----------|------|-------------------|
| strategy | 当前策略 | string | jaw 融合方式 |
| location | 代码位置 | string | 文件和函数 |
| recommendation | 建议 | string | 是否需要修改 |

检查数据:

| strategy | location | recommendation |
|----------|----------|----------------|
| 方案A: `jaw = 0.4·EMICA + 0.6·SPECTRE` (即 `(1-λ)·emica + λ·spectre`, λ=0.6) | `runner.py:246-247` (`run_stage12`); `processor.py:95-96` (`extract_flame_parameters`) | 代码本身无需修改; 但 `docs/03:1589` 公式方向与代码相反, 需同步文档 |

代码证据 (`runner.py:246-247`):
```python
fused_jaw = ((1.0 - jaw_fusion_lambda) * emica_jaw
             + jaw_fusion_lambda * spectre_jaw)
```
默认 `jaw_fusion_lambda=0.6` (`runner.py:362`) → `0.4·EMICA + 0.6·SPECTRE` = **方案A**。

文档证据 (`docs/03_Pipeline_Stage_Specification.md:1588-1589`):
```
jaw_final = λ · jaw_emica + (1 − λ) · jaw_spectre
```
该公式 λ 乘 EMICA → `0.6·EMICA + 0.4·SPECTRE`, 与代码 `(1-λ)·emica + λ·spectre` **方向相反** (文档/代码不一致)。

> 结论: 代码实现为方案A (与既有审计建议一致); 仅文档公式方向需修正以与代码对齐。

## 5. Pose 顺序检查

列定义:

| Field Name | 中文名称 | Type | Field Description |
|------------|----------|------|-------------------|
| component | 组件 | string | 检查对象 |
| order | 当前顺序 | string | 实际顺序 |
| correct | 正确顺序 | string | FLAME 要求 |
| mismatch | 是否错误 | bool | 是否需要修改 |

检查数据:

| component | order | correct | mismatch |
|-----------|-------|---------|---------|
| `FLAMEFeature.posecode` (`models.py:676-678`) | `globalpose(3) + jawpose(3)` = [global, jaw] | [global, jaw] | false ✓ |
| FLAME decoder `full_pose` (`DecaFLAME.py` / `FLAME.py:190-191`) | `pose_params[:, :3]`=global, `pose_params[:, 3:]`=jaw → [global, jaw] | [global, jaw] | false ✓ |
| Stage12 npz 存储 (`runner.py:307-316`) | `head_pose` / `jaw_pose` 分离存储 (非合并 pose_code) | — | false ✓ |
| `docs/03:1555` pose_code 描述 | `jaw_pose[0:3] + head_pose[3:6]` = [jaw, head] | [global, jaw] | **true ✗** |

代码证据 (`models.py:675-678`):
```python
@property
def posecode(self) -> Tuple[float, ...]:
    """拼接后的姿态参数 — globalpose(3) + jawpose(3) = 6 dims."""
    return tuple(self.globalpose) + tuple(self.jawpose)
```

FLAME decoder 证据 (`third_party/inferno/spectre_backup/src/models/FLAME.py:190-191`):
```python
betas = torch.cat([shape_params, expression_params], dim=1)
full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1),
                        pose_params[:, 3:], eye_pose_params], dim=1)
```
→ FLAME 取 `pose_params[:3]` 为 **global**, `pose_params[3:]` 为 **jaw**。故正确拼接顺序为 **[global, jaw]**。

> 结论: 代码侧 (models.py / npz 存储 / FLAME decoder) 全部为 [global, jaw], 正确;
> **仅 `docs/03:1555` 文档描述为 [jaw, head], 与代码及 FLAME 相反, 需修正** (P2 未完成)。

---

## 6. 当前 mesh 异常风险判断

症状: 只有头顶 + 下巴扭曲 + exp 正常 + pose 异常。

依据: 既有 `Stage0_17` 报告 (2026-08-27) 显示 `exp std ≈ 0.68` (正常), `pose std ≈ 2.75` (异常偏大 — axis-angle 正常量级应 < 1.0 rad)。该量级与"归一化 bug 期 `head_pose/jaw_pose` 异常偏大"特征吻合。

最大概率原因排序:

1. **P3 回归阻断 + 旧 npz 残留归一化 bug (最高概率)**
   当前 `runner.py` 因 `_load_spectre` 返回 `None` 在 `spectre_encode`(`L227`) 崩溃, **无法用修复后的 `/255.0` 重新生成 npz**。
   外部 decoder 渲染所用 npz 为 P3 回归前 (归一化 bug 期) 产物, `pose std ≈ 2.75` 异常 → `global_pose` 幅值过大 → 头部大幅 pitch → mesh 只剩头顶 + 下颌扭曲; 同时 `exp std ≈ 0.68` 正常, 与"exp 正常 / pose 异常"症状完全吻合。

2. **P1 EMICA global pose 无 baseline 修正 (次高, 待 P3 修复后验证)**
   即便 P3 修复后用 `/255.0` 重跑, EMICA `globalpose` 若存在训练 baseline 偏移且未做 `globalpose -= baseline` 修正, 头部仍可能带常量 pitch 偏置 → 偏头顶。当前代码全链路无 baseline 处理。

3. **P2 外部 decoder 按 [jaw,head] 错序拼接 (契约风险)**
   若外部 decoder 误信 `docs/03:1555` 的 [jaw,head] 描述, 以 `jaw_pose` 拼在前喂入 FLAME, 则 `global` 落位 jaw 槽、`jaw` 落位 global 槽 → 头部不旋转 + 下颌异常张开 (与"只剩头顶"不完全吻合, 但属 pose 契约风险)。

> 备注: expression 维度 (100 vs decoder n_exp) 在本任务范围外; 既有审计已确认 EMICA 训练 `n_exp=100` (`cfg.yaml:137`), Stage12 输出 `expression[100]`, 二者一致, 且报告显示 `exp std` 正常, 故 expression 非本次"头顶"症状主因。

## 7. 后续执行建议

> 本任务严格只读, 以下仅为建议, 未执行任何修改。

### 已完成
- **P0 输入归一化**: `emica.py` / `spectre.py` / `runner.py` 三处 `preprocess_image` 全部改为 `/ 255.0` → [0,1], 与 DECA/EMICA/SPECTRE 训练分布一致。

### 未完成
- **P3**: `runner.py` `_load_spectre` 函数体内**缺少 `return spectre`** (回归, 直接阻断 Stage12 重跑)。
- **P2**: `docs/03:1555` `pose_code` 顺序描述仍为 [jaw,head] (与代码/FLAME 的 [global,jaw] 相反)。
- **P2 (附加)**: `docs/03:1589` jaw 融合公式方向 (λ·emica+(1-λ)·spectre) 与代码 ((1-λ)·emica+λ·spectre) 相反。
- **P1**: EMICA `globalpose` 无 baseline 修正。

### 需重新验证
- 修复 P3 后用 `/255.0` 重跑 Stage12, 校验新 npz `pose std` 是否从 ~2.75 回落到 < 1.0 量级 (确认归一化修复生效、排除旧 npz 干扰)。
- 用新 npz 在外部 FLAME decoder 重渲染, 确认 mesh 恢复正常面部结构 (排除 P1/P2 残余影响)。

### 下一步执行顺序

**Step 1 (阻断修复, 最高优先级)**: 在 `runner.py` `_load_spectre` 函数末尾 (当前 `runner.py:153` 之后、`L155` 模块级注释之前) 补回
```python
    return spectre
```
使 `_load_spectre` 正确返回 SPECTRE 模型, 解除 `spectre_encode` 崩溃。此为当前 Stage12 重新可运行的硬前置。

**Step 2 (文档对齐)**:
- 将 `docs/03:1555` `pose_code` 描述改为 `head_pose[0:3] + jaw_pose[3:6]` ([global, jaw]), 与 `models.py:676-678` 及 FLAME decoder 对齐。
- 将 `docs/03:1589` jaw 融合公式改为 `jaw_final = (1−λ)·jaw_emica + λ·jaw_spectre`, 与 `runner.py:246-247` 代码方向一致。

**Step 3 (重跑验证 + 评估 P1)**: 修复 Step 1 后, 在 Docker `emica_env` 内重跑 Stage12 (用既有 EMICA/SPECTRE checkpoint + `/255.0` 预处理), 生成新 npz;
- 若新 npz `pose std < 1.0` 且 mesh 恢复正常 → P0 即为根因, P1 baseline 非必需, 任务收口。
- 若新 npz `pose std` 仍异常或 mesh 仍偏头顶 → 评估 P1: 是否需对 EMICA `globalpose` 引入 baseline 修正 (需先实测 EMICA globalpose 在正脸对齐帧上的统计均值以判定是否存在系统偏置)。

---

## 附录: 关键证据行号速查

| 项 | 位置 | 内容 |
|----|------|------|
| P0 EMICA 归一化 | `emica.py:244` | `tensor / 255.0` |
| P0 SPECTRE 归一化 | `spectre.py:209` | `tensor / 255.0` |
| P0 Runner 归一化 | `runner.py:165` | `... / 255.0` |
| P1 无 baseline | `runner.py:173-177,236` | `emica_encode` 直返原值; `emica_global` 原值写 `head_pose` |
| P2 doc pose 错序 | `docs/03:1555` | `jaw_pose[0:3] + head_pose[3:6]` |
| P2 doc jaw 公式反向 | `docs/03:1589` | `λ·jaw_emica + (1−λ)·jaw_spectre` |
| P3 缺 return | `runner.py:120-153` | `_load_spectre` 无 `return spectre` |
| P3 崩溃点 | `runner.py:227` | `spectre_encode(spectre_model, batch_tensor)` (model=None) |
| jaw 融合 (方案A) | `runner.py:246-247` | `(1-λ)·emica + λ·spectre` |
| posecode 正确序 | `models.py:676-678` | `globalpose + jawpose` |
| FLAME 正确序 | `FLAME.py:191` | `pose[:3]=global, pose[3:]=jaw` |
| 旧 npz pose 异常 | `Stage0_17 报告` | `pose std ≈ 2.75` (异常), `exp std ≈ 0.68` (正常) |

---

*End of Stage12_Pose_Fix_Status_Report.md — read-only status check, no code modified.*



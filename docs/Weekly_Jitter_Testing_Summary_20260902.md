# 抖动问题测试周报（2026-08-31 ~ 09-02）

> 目的：定位并消除 FLAME mesh 时间抖动（重点：嘴部 / jaw）。
> 本文件汇总本周针对抖动的全部验证测试、方法、定量结果与结论，供周会汇报使用。
> 测试样本统一为 `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`（200 帧 / 25fps，1920×1080），
> 部分端到端消融复用同视频的 11 个 clip（2188 帧）扩展验证。

## 0. 一句话结论链

1. **上游（Stage5 bbox / Stage10 landmark）的时序平滑能显著降检测框抖动，但对最终 FLAME 参数几乎无影响**（motion energy ratio ≈ 1.0）→ 抖动根源不在检测层。
2. **SPECTRE jaw 时序更平滑但 mesh 物理不正确（嘴部扭曲）→ 生产正式采用 EMICA jaw**。
3. **本周最新发现 Stage12 输入归一化存在 [-1,1] bug（正确为 [0,1]）**，污染了 36 个历史 npz，已全部清理，需修复后重跑 Stage12。

---

## 1. 抖动来源定位审计（Stage5 vs Stage10）— 8/31

**目的**：判断 mesh 抖动更多来自哪一级。
**方法**：逐帧重跑 SCRFD bbox + 2d106det landmark，统计相邻帧位移（px，对脸对角线归一化）。
**结果**：

| 指标 | 值 (px) | 占脸对角线 346.7px |
|---|---|---|
| bbox 中心抖动 (mean) | 2.265 | 0.653% |
| landmark 平均点位移 (mean) | 2.032 | 0.586% |
| bbox 最大跳变 | 14.031 | - |
| 异常帧 (mean+3σ) | 9 | - |

**结论**：两者贡献相当，无法单一定位；且 landmark 坐标经 `expanded_bbox` 映射，部分位移来自 bbox。
➡ 由此启动 Stage5 / Stage10 两级平滑实验。

---

## 2. Stage5 人脸框 Temporal ROI 约束 — 8/31

**方法**：以上一帧 bbox 为 temporal prior，在扩展 ROI 内执行 SCRFD（扫描 expand_ratio 1.0/1.2/1.5/2.0）。
**结果**：

| 指标 | Baseline | ROI 1.2 | 变化 |
|---|---|---|---|
| bbox 平均抖动 | 2.265 | 2.760 | +21.8% ✗ |
| bbox 峰值跳变 | 14.031 | 8.647 | -38.4% ✓ |
| 孤立异常帧 (>8px) | 8 | 3 | -5 ✓ |
| landmark 平均抖动 | 1.752 | 1.797 | +2.6% ✗ |

expand_ratio 扫描：ratio=1.0 时 bbox 平均抖动 1.749px 优于基线 2.265px，但平均抖动整体未降。
**结论**：能压大跳变、但平均抖动不降反升 → **默认 1.2 不进入生产**，生产保持全图检测。

---

## 3. Stage5 bbox 因果 EMA 平滑（α=0.5/0.7/0.9）— 8/31~9/1

**方法**：SCRFD 输出后对 `[x1,y1,x2,y2]` 四坐标做因果一阶 EMA（仅历史+当前帧，无离线滤波）。
**bbox 层结果**：

| 方法 | center jitter (px) | max jump (px) | 孤立异常帧 | landmark jitter (px) |
|---|---|---|---|---|
| Baseline | 2.265 | 14.031 | 8 | 1.752 |
| EMA0.5 | **1.607 (-29.0%)** | **6.029 (-57.0%)** | **0** | 1.707 (-2.6%) |
| EMA0.7 | 1.916 (-15.4%) | 9.502 (-32.3%) | 1 | 1.711 (-2.3%) |
| EMA0.9 | 2.155 (-4.9%) | 12.683 (-9.6%) | 8 | 1.723 (-1.7%) |

**端到端验证（EMA α=0.5，全链路 Stage5→10→11→12→13→14→17）——FLAME 层结果**：

| FLAME 参数 | motion energy ratio (EMA/base) |
|---|---|
| expression | **0.992** |
| head_pose | **0.999** |
| jaw_pose | **0.980** |

**关键结论（用户关注的"bbox EMA 但 flame 没变"）**：
> bbox EMA 把人脸框抖动降了 29%（峰值跳变 -57%、孤立异常 8→0），**但 FLAME 的 expression/head/jaw 运动能量几乎原样（ratio≈0.98~0.999）** ——
> 上游人脸框平滑不会传导到 FLAME 参数，说明抖动主要不是由检测框抖动引起。

---

## 4. Stage10 landmark 因果 EMA（α=0.7）— 9/1

**方法**：对 106 点 2D 坐标逐点因果 EMA，端到端跑通 Stage12/13/14/17。
**landmark 层结果**：

| 指标 | baseline | EMA0.7 | 变化 |
|---|---|---|---|
| 全点平均抖动 | 1.752 | 1.484 | **-15.3%** |
| mouth 区域 | 2.129 | 1.841 | -13.5% |
| landmark motion energy retention | — | 0.847 | OK (≥0.8) |

**FLAME 层结果**：

| FLAME 参数 | motion energy ratio (EMA/base) |
|---|---|
| expression | **0.996** |
| head_pose | **0.942** |
| jaw_pose | **0.995** |

**结论**：landmark 平滑同样**几乎不传导到 FLAME**（head 略降 0.942 属保留量级内）。
➡ 两次独立实验共同证实：**抖动根源在 Stage12 FLAME 参数提取层，而非 Stage5/10 检测噪声**。

---

## 5. Stage12 jaw 来源消融（EMICA / SPECTRE / 融合）— 8/31~9/1

**背景**：生产原为融合 `jaw = 0.4·EMICA + 0.6·SPECTRE`（λ=0.6）。
**方案对比（同 50 帧，jaw 角度指标）**：

| 指标 | A 融合 | B SPECTRE | C EMICA | D (EMICA head+SPECTRE jaw) |
|---|---|---|---|---|
| jaw_angle_mean (°) | 3.86 | 1.62 | 8.25 | **1.62** |
| jaw_angle_std (°) | 1.34 | 0.44 | 2.91 | **0.44** |
| jaw_rest_bias (°) | 3.70 | 1.48 | 7.92 | **1.48** |
| jaw_velocity (°/frame) | 0.86 | 0.56 | 1.72 | **0.56** |

**全量消融（11 clip / 2188 帧，EMICA vs SPECTRE 全参数）**：

| 参数 | 帧间 mean_delta | 更稳定 |
|---|---|---|
| expression | EMICA 0.934 vs SPECTRE 0.551 | SPECTRE |
| jaw_pose (°/frame) | EMICA 0.975 vs SPECTRE 0.602 | **SPECTRE** |
| head_pose (°/frame) | EMICA 0.784 vs SPECTRE 0.738 | SPECTRE |

**决定性验证 —— FLAME mesh 实际重建**：
- SPECTRE jaw 版本：**嘴部区域存在扭曲**（mouth-region distortion）
- EMICA jaw 版本：mesh 重建效果更好

**结论**：SPECTRE jaw 时序更平滑、无静止偏置，但 **mesh 物理不正确**；
**正式采用 EMICA jaw**（`jaw_pose = EMICA_jaw_pose`，jaw_fusion_lambda 废弃，SPECTRE 原始 jaw 保留为 `spectre_jawpose` debug 字段）。

---

## 6. SPECTRE E_expression 接入与输入归一化排查 — 9/2

**背景**：怀疑 jaw/exp 抖动另一根源在 SPECTRE 的 E_expression（PerceptualEncoder）接入方式与输入。
**测试内容**：
- 接入 E_expression（mobilenetv2 PerceptualEncoder），验证残差语义 `exp50 = E_flame_exp + E_expression_exp`、`jaw = E_flame_jaw + E_expression_jaw`
- **输入归一化值域三组对照：[-1,1] vs [0,1] vs [0,255]**
- 边界策略：pad 2 / drop 2
- 诊断测试：噪声图 / 黑图 / 白图 / 真实人脸、crop 尺度 0.6~1.0、BN 状态、mobilenet 饱和检查

**结果**：
- **确认 [-1,1] 为归一化 bug**：正确应为 **[0,1]**（与 DECA/EMICA/SPECTRE 训练分布一致，skimage `warp()` 输出），wrapper 的 `preprocess_image` 已按 `/255.0` 修复
- 被污染的存量产物中 jaw 出现异常大数值：`jaw_pose`（EMICA 均值 -4.2）、旧 `spectre_jawpose`（约 -6.7），正常量级应 < 1 rad
- 污染范围：`stage12_flame/flame_features.npz` 18 个 + `stage17_fusion/training_sample.npz` 18 个（各含 7 主 + 11 validation）= **36 个，已全部删除**
- 保留：`stage13_gaze` / `stage14_audio`（18+18 个，不受该 bug 影响）

**后续待办**：
- 用修复后的 [0,1] 归一化重跑 Stage12（生成正确 `flame_features.npz`）→ Stage13/14/17
- Stage12 仍需处理 `batch_size=8` 的 temporal 时间连续性问题（改为按 clip 整段前向），否则 E_expression 在 batch 边界引入额外抖动

---

## 7. 各报告原始出处

| 测试 | 报告 |
|---|---|
| 来源定位审计 | `docs/Stage5_Stage10_Temporal_Stability_Report.md` |
| ROI 约束 | `docs/Stage5_ROI_Temporal_Ablation_Report.md`、`docs/Stage5_BBox_Center_Trajectory_Report.md` |
| bbox EMA | `docs/Stage5_BBox_EMA_Ablation_Report.md`、`docs/Stage5_EMA_Motion_Preservation_Report.md` |
| landmark EMA | `docs/Stage10_Landmark_Temporal_EMA_Ablation_Report.md` |
| jaw 来源消融 | `docs/Stage12_SPECTRE_Jaw_Final_Adoption_Report.md`（已取代）、`docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`、`docs/Stage12_EMICA_vs_SPECTRE_Full_Ablation_Report.md` |
| 归一化排查/清理 | 本周会话记录（无独立报告，输出已清理） |

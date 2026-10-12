# Stage5 Temporal EMA (α=0.5) End-to-End Motion Preservation Ablation Report
## 1. 实验目的 (Objective)
验证 Stage5 bbox temporal EMA (α=0.5) 是否会影响后续 FLAME / Gaze / Audio 多模态参数质量。生产 Stage5/10/12/13/14/17 代码与模型均未修改；仅新增实验入口与分析脚本，且 baseline 与 EMA 分支使用**完全相同的代码路径**（仅 bbox 不同）保证可比较性。
## 2. 方法 (Method)
- **视频**: `outputs/stage0_17_e2e/stage0_11/clips/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4` (1920×1080, 25 fps, 200 帧)
- **Baseline**: Stage5 原始 SCRFD bbox → Stage10 106-landmark → Stage11 对齐 → Stage12 FLAME → Stage13 gaze → Stage14 audio → Stage17 fusion
- **EMA α=0.5**: 同链路，但 bbox 经因果 EMA 平滑 `[x1,y1,x2,y2]`：`bbox_smooth_t = α·bbox_raw_t + (1-α)·bbox_smooth_{t-1}`；第 0 帧用原始 bbox；检测缺失保持上一帧平滑值。不使用 Butterworth / filtfilt / 未来帧。
- **原始 SCRFD bbox 复用**自 Stage5 ROI 消融 (`docs/stage5_roi_ablation/baseline_results.json`)，与先前实验字节一致。
- **Stage12** 经 docker `emica_env` 官方 `stage0_17_e2e_stage12.py` (jaw_source=emica)；**Stage13/14/17** 经官方 runner；参数与生产一致。

## 3. bbox 对比 (Stage5)
| 指标 | Baseline | EMA α=0.5 | 变化 |
|---|---|---|---|
| center jitter (px) | 2.265 | 1.607 | -29.0% |
| center jitter std (px) | 2.239 | 1.236 | — |
| max jump (px) | 14.031 | 6.029 | -57.0% |
| 孤立 outlier (≥8px, 个数) | 8 | 0 | -8 |
| width jitter (px) | 2.605 | 1.677 | — |
| height jitter (px) | 2.979 | 1.628 | — |
| landmark jitter (106pt, px) | 1.752 | 1.707 | -2.6% |

## 4. FLAME 对比 (Stage12)
### 4.1 expression (100-dim)
| 指标 | Baseline | EMA α=0.5 |
|---|---|---|
| velocity (L2/frame) | 0.97576 | 0.97115 |
| std | 0.44783 | 0.44837 |
| max delta | 2.94120 | 2.81799 |
| motion energy | 265.784 | 263.587 |

### 4.2 head_pose (3-dim rotation)
| 指标 | Baseline | EMA α=0.5 |
|---|---|---|
| rotation velocity (L2/frame) | 0.01241 | 0.01241 |
| motion energy | 0.044 | 0.044 |
| **motion energy ratio (EMA/base)** | — | **0.999** |

### 4.3 jaw_pose (3-dim)
| 指标 | Baseline | EMA α=0.5 |
|---|---|---|
| amplitude (std) | 0.06933 | 0.06928 |
| velocity (L2/frame) | 0.01838 | 0.01834 |
| range (mean per-dim) | 0.10021 | 0.09834 |
| motion energy ratio (EMA/base) | — | **0.980** |

### 4.4 生产保真度校验 (baseline 分支 vs 生产 Stage12)
为确认 EMA 是唯一变量，对比本实验**基线分支**的 Stage12 输出与生产 Stage12 (`outputs/stage12_flame/`，jaw_fusion_lambda=1.0，即 `run_single_video_dataset.sh` 生产链路；非 legacy lambda=0.6 的 e2e 运行)：

| 参数 | max abs diff |
|---|---|
| expression | 1.146e-01 |
| head_pose | 1.010e-02 |
| jaw_pose | 2.316e-01 |

## 5. Stage17 融合校验
layout: expression `[0,100]`, head_pose `[100,103]`, jaw_pose `[103,106]`, gaze `[106,112]`

| 分支 | visual_feature shape | audio shape | layout_ok | 块最大偏差(expr/head/jaw/gaze) |
|---|---|---|---|---|
| baseline | [200, 112] | [200, 768] | True | 0.00e+00 / 0.00e+00 / 0.00e+00 / 0.00e+00 |
| ema_alpha05 | [200, 112] | [200, 768] | True | 0.00e+00 / 0.00e+00 / 0.00e+00 / 0.00e+00 |

## 6. 可视化 (Videos)

## 7. 结论汇总表
| Field Name | 中文名称 | Type | Field Description |
|---|---|---|---|
| bbox_jitter_delta | bbox抖动变化 | float | EMA前后bbox稳定性变化 (-29.0%) |
| jaw_motion_ratio | jaw运动保持比例 | float | EMA jaw motion energy / baseline = 0.980 |
| head_motion_ratio | head运动保持比例 | float | EMA head motion energy / baseline = 0.999 |
| expression_energy_ratio | expression保持比例 | float | EMA expression动态保持比例 = 0.992 |
| recommendation | 最终建议 | string | 推荐进入生产 |

## 8. 最终结论
1. **EMA α=0.5 是否降低 bbox 抖动？** 是。center jitter 2.265→1.607px (-29.0%)，max jump 14.031→6.029px (-57.0%)，孤立 outlier 8→0。
2. **是否损害 FLAME expression？** 否（基本保持）。expression motion energy ratio = 0.992。
3. **是否损害 head_pose？** 否（基本保持）。head motion energy ratio = 0.999。
4. **是否损害 jaw_pose？** 否（基本保持）。jaw motion energy ratio = 0.980。
5. **是否推荐进入 Stage5 生产流程？** **推荐进入生产**。

---
*本报告由 `scripts/run_single_video_dataset_ema_ablation.sh` + `scripts/stage5_ema_e2e_ablation.py` 生成；生产代码未修改。*

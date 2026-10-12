# Stage10 Landmark Temporal EMA (α=0.7) End-to-End Motion Preservation Ablation Report
## 1. 实验目的
当前 mesh 抖动问题已排除主要来自 bbox EMA（Stage5 消融），head/jaw FLAME 参数抖动主要来源于 Stage10 106 landmark 逐帧检测噪声以及后续 FLAME fitting。本实验验证：对 Stage10 输出的 106 landmark 2D 坐标施加因果 temporal EMA（当前帧权重 0.7）后，是否能够降低 landmark 高频抖动，同时保持真实人脸运动与表情（motion / mouth / jaw）不受损。

## 2. 方法
- **输入**: 与 Stage5 bbox EMA 消融相同的视频与 200 帧 clip (`outputs/stage0_17_e2e/stage0_11/clips/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4`)
- **Baseline**: Stage10 raw 106 landmark，不做任何平滑
- **EMA**: 逐点因果 EMA（作用于 [106,2] 坐标）:
  `landmark_ema[t] = 0.7·landmark_raw[t] + (1-0.7)·landmark_ema[t-1]`; 第 0 帧 = raw; 当前帧权重 0.7 / 历史权重 0.3; 不使用 Butterworth / filtfilt / 未来帧。
- **SCRFD bbox 两分支共用**（raw bbox，复用自 Stage5 ROI 消融 `baseline_results.json`，与先前实验字节一致）；仅 landmark 不同 → Stage11 对齐不同。
- **Stage12** 经 docker `emica_env` 官方 `stage0_17_e2e_stage12.py`（jaw_source=emica）；Stage13/14/17 经官方 runner；参数与生产一致。

## 3. Landmark 稳定性结果
| metric | baseline | EMA0.7 | change |
|-|-|-|-|
| all mean jitter (px) | 1.752 | 1.484 | +15.3% |
| all std jitter (px) | 0.816 | 0.703 | — |
| all max jitter (px) | 4.601 | 3.713 | — |
| all P95 jitter (px) | 3.409 | 2.982 | — |
| contour mean jitter (px) | 1.761 | 1.492 | +15.3% |
| contour std jitter (px) | 1.029 | 0.867 | — |
| contour max jitter (px) | 7.375 | 5.393 | — |
| contour P95 jitter (px) | 3.497 | 3.196 | — |
| eyebrow mean jitter (px) | 1.462 | 1.211 | +17.2% |
| eyebrow std jitter (px) | 0.854 | 0.763 | — |
| eyebrow max jitter (px) | 4.803 | 4.288 | — |
| eyebrow P95 jitter (px) | 3.195 | 2.841 | — |
| eye mean jitter (px) | 1.810 | 1.511 | +16.5% |
| eye std jitter (px) | 1.260 | 1.017 | — |
| eye max jitter (px) | 7.750 | 5.926 | — |
| eye P95 jitter (px) | 4.050 | 3.379 | — |
| nose mean jitter (px) | 1.505 | 1.284 | +14.7% |
| nose std jitter (px) | 0.828 | 0.763 | — |
| nose max jitter (px) | 4.380 | 3.919 | — |
| nose P95 jitter (px) | 3.094 | 2.725 | — |
| mouth mean jitter (px) | 2.129 | 1.841 | +13.5% |
| mouth std jitter (px) | 1.162 | 0.935 | — |
| mouth max jitter (px) | 6.938 | 4.759 | — |
| mouth P95 jitter (px) | 4.320 | 3.706 | — |

## 4. Motion Preservation
| 指标 | Baseline | EMA α=0.7 | Retention |
|---|---|---|---|
| landmark motion energy (px/frame) | 1.7524 | 1.4842 | **0.847** |
| mouth opening mean (px) | 41.006 | 40.984 | 0.999 |
| mouth opening max (px) | 55.752 | 53.910 | — |
| mouth opening std (px) | 6.494 | 6.075 | — |
| mouth opening 曲线相关系数 | — | — | 0.982 |
| jaw amplitude (std) | 0.06933 | 0.06934 | 1.000 |
| jaw max abs | 0.23838 | 0.23767 | — |
| jaw frame delta mean (deg) | 1.0528 | 1.0542 | — |
| jaw frame delta std (deg) | 1.1438 | 1.1372 | — |
| jaw frame delta max (deg) | 7.3784 | 7.3932 | — |
- **motion energy retention 0.847**: OK（≥0.8）

## 5. FLAME 参数变化 (Stage12)
### 5.1 expression (100-dim)
| 指标 | Baseline | EMA α=0.7 | Ratio |
|---|---|---|---|
| frame delta mean (L2) | 0.97576 | 0.97477 | — |
| frame delta std (L2) | 0.61927 | 0.61668 | — |
| frame delta max (L2) | 2.94120 | 3.01935 | — |
| motion energy | 265.7837 | 264.7631 | **0.996** |
### 5.2 head_pose (rotvec → degree)
| 指标 | Baseline | EMA α=0.7 | Ratio |
|---|---|---|---|
| rotation delta mean (deg) | 0.7107 | 0.6800 | — |
| rotation delta std (deg) | 0.4714 | 0.4718 | — |
| rotation delta max (deg) | 3.1486 | 2.9001 | — |
| motion energy (deg²) | 144.743 | 136.330 | **0.942** |
### 5.3 jaw_pose (rotvec → degree)
| 指标 | Baseline | EMA α=0.7 | Ratio |
|---|---|---|---|
| rotation delta mean (deg) | 1.0528 | 1.0542 | — |
| rotation delta std (deg) | 1.1438 | 1.1372 | — |
| rotation delta max (deg) | 7.3784 | 7.3932 | — |
| motion energy (deg²) | 480.949 | 478.511 | **0.995** |

## 6. 视频结果
- `docs/stage10_landmark_temporal_ema_ablation/videos/baseline_landmark_overlay.mp4` ✓
- `docs/stage10_landmark_temporal_ema_ablation/videos/ema07_landmark_overlay.mp4` ✓
人工观察重点：眼睛是否漂移、嘴部运动是否被压制、轮廓是否更稳定。

## 7. 最终建议
- jitter 降低: 是
- motion energy retention: 0.847 (OK)
- mouth opening retention: 0.999 (OK)
- jaw amplitude retention: 1.000 (OK)
- **是否推荐进入生产 pipeline: 推荐进入生产**

---
*本报告由 `scripts/run_single_video_dataset_landmark_ema_ablation.sh` + `scripts/stage10_landmark_temporal_ema_ablation.py` 生成；生产代码未修改。*

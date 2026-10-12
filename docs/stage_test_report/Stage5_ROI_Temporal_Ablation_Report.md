# Stage5 SCRFD Temporal ROI Constraint Ablation Report

## 11.1 实验目的

生产管线中 Stage5 对每一帧执行 **全图 SCRFD 独立检测**。人物说话时，嘴部运动会引起 SCRFD 检测框轻微变化，进而导致 5 点 landmark 漂移、Stage10 106 点 landmark 输入变化、FLAME 参数时间抖动、mesh 稳定性下降。

本实验只做**算法验证**（不改生产代码、不改模型、不加滤波），检验：以上一帧 bbox 为 temporal prior、在扩展后的 ROI 区域内执行 SCRFD，是否能够限制检测搜索范围、降低 bbox 与 landmark 的时间抖动。

## 11.2 实验设置

| 项目 | 值 |
|---|---|
| 视频 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_17_e2e/stage0_11/clips/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4` |
| 分辨率 / 帧率 | 1920x1080 / 25 fps |
| 帧数量 | 200（时长 8.0s） |
| SCRFD 模型 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/models/SCRFD/scrfd_10g_bnkps.onnx` |
| confidence_threshold | 0.6（生产 `face.confidence_threshold`） |
| 生产 expand_ratio（Stage10 crop） | 2.0 |
| **Temporal ROI expand_ratio（实验）** | **1.2**（默认） |
| 孤立跳变阈值 | 8 px（第 8.3 节规则） |
| 扫描范围（推荐比例） | 1.0, 1.2, 1.5, 2.0 |
| 主脸选择 | 最大面积 + conf≥0.6（生产 `largest_area`） |

## 11.3 定量结果

| Metric | Baseline | ROI SCRFD | Difference |
|---|---|---|---|
| bbox jitter mean (px) | 2.265 | 2.760 | +0.494 (+21.8%) |
| bbox jitter std (px) | 2.239 | 1.742 | -0.497 (-22.2%) |
| bbox max jump (px) | 14.031 | 8.647 | -5.384 (-38.4%) |
| width change mean (px) | 2.605 | 0.979 | -1.626 (-62.4%) |
| height change mean (px) | 2.978 | 5.682 | +2.704 (+90.8%) |
| outlier count (>8px 孤立跳变) | 8 | 3 | -5 |
| landmark jitter mean (px) | 1.752 | 1.797 | +0.045 (+2.6%) |
| landmark max jump (px) | 16.823 | 16.807 | -0.016 (-0.1%) |
| landmark outlier count | 2 | 2 | +0 |

异常帧（孤立跳变 > 8 px）：
- Baseline bbox: [7, 9, 23, 28, 35, 40, 42, 129]
- ROI bbox: [7, 22, 170]
- Baseline landmark: [40, 128]
- ROI landmark: [154, 170]

### expand_ratio 扫描

> 各比例共用同一 temporal prior 链（主比例 1.2 的上一帧 bbox），唯一变量为 ROI 大小。

| expand_ratio | bbox jitter mean | landmark jitter mean | bbox max jump | full-image 次数 |
|---|---|---|---|---|
| 1.0 | 1.749 | 1.736 | 5.239 | 1 |
| 1.2 | 2.760 | 1.797 | 8.647 | 1 |
| 1.5 | 1.936 | 1.744 | 7.029 | 1 |
| 2.0 | 1.851 | 1.737 | 6.502 | 1 |

## 11.4 可视化结果

| 视频 | 路径 | 显示内容 |
|---|---|---|
| Baseline overlay | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/baseline_bbox_overlay.mp4` | 绿色 bbox、frame_id、confidence |
| Temporal ROI overlay | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/roi_temporal_bbox_overlay.mp4` | 黄色 ROI、绿色 bbox、frame_id、confidence、红色 fallback |

## 11.5 最终结论

| Field | 值 |
|---|---|
| roi_effectiveness / roi_improve | False |
| recommended_ratio | 1.0 |
| fallback_rate | 0.0000（0 次 / 199 个转换） |
| production_recommendation | 有条件推荐：默认 1.2 未改善平均抖动；但扫描中较小比例 1.0 将 bbox 抖动降至 1.749（低于基线 2.265），需更多数据验证 landmark 稳定性后再决定 |

### 结论说明

默认比例 1.2 下 ROI **未改善**平均抖动（bbox 平均抖动反而 +21.8%，landmark 平均抖动 +2.6%），因此**不直接采用默认 1.2 进入生产**。
但 expand_ratio 扫描显示：比例为 1.0 时 bbox 平均抖动 (1.749 px) 低于基线 (2.265 px)，同时孤立跳变从 8 降至 3。建议在更多 clip 上复验 landmark 稳定性后再考虑以较小 ROI 比例实现；当前生产管线保持全图检测不变（本实验为只读验证，未修改任何生产代码）。

**数据文件**：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/baseline_results.json`、`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/roi_results.json`、`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/ablation_metrics.json`

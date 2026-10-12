# Stage5 SCRFD BBox Temporal EMA Stabilization Ablation Report

## 11.1 实验目的

Stage5 使用 SCRFD 逐帧检测人脸，单帧检测结果存在短时间抖动：bbox center 孤立跳变 → crop 变化 → 106 landmark 不稳定 → FLAME mesh 时间抖动。上一轮 ROI temporal constraint 实验表明 ROI 可降低极端跳变，但整体稳定性提升有限。

本实验验证：在 SCRFD 输出之后引入轻量 temporal tracking —— 对 bbox 四坐标 [x1,y1,x2,y2] 做因果指数滑动平均（EMA）—— 是否能在不修改 SCRFD / Stage10 / FLAME 的前提下降低检测框抖动并改善下游 landmark 稳定性。

## 11.2 方法

### 对比方法

- **Baseline**：`bbox_t = SCRFD(frame_t)`（生产全图检测，未修改）。
- **Temporal EMA**：`bbox_smooth_t = α·bbox_raw_t + (1-α)·bbox_smooth_{t-1}`
- 帧 0 直接使用 SCRFD 结果；EMA 作用于四个坐标 [x1,y1,x2,y2]，非仅中心点。
- 平滑强度：**α = 0.5 / 0.7 / 0.9**。
- 因果策略：仅使用当前帧与历史帧，**不使用** Butterworth / filtfilt / 离线未来帧信息。

### 评测

- 视频：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_17_e2e/stage0_11/clips/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4`，200 帧，25 fps
- bbox 稳定性：相邻帧中心距离（mean/std/max）；孤立跳变阈值 8 px；宽高变化。
- landmark 影响：对 Baseline 与各 EMA bbox 分别做 Stage10 式 crop（expand_ratio=2）并调用 2d106det，比较相邻帧 106 点平均位移。

- baseline 检测复用自上一轮实验：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/baseline_results.json`（同视频 / 同模型 / 同主脸选择，保证与 ROI 实验可比）。

## 11.3 定量结果

| Method | center jitter (px) | max jump (px) | outlier | landmark jitter (px) |
|---|---|---|---|---|
| Baseline | 2.265 | 14.031 | 8 | 1.752 |
| EMA0.5 | 1.607 | 6.029 | 0 | 1.707 |
| EMA0.7 | 1.916 | 9.502 | 1 | 1.711 |
| EMA0.9 | 2.155 | 12.683 | 8 | 1.723 |

### 与 Baseline 对比（% 变化）

| Method | center jitter | max jump | outlier | landmark jitter |
|---|---|---|---|---|
| EMA0.5 | 1.607 (-29.0%) | 6.029 (-57.0%) | 0 (Δ-8) | 1.707 (-2.6%) |
| EMA0.7 | 1.916 (-15.4%) | 9.502 (-32.3%) | 1 (Δ-7) | 1.711 (-2.3%) |
| EMA0.9 | 2.155 (-4.9%) | 12.683 (-9.6%) | 8 (Δ+0) | 1.723 (-1.7%) |

### 孤立跳变帧（>8px）

- Baseline bbox 异常帧: [7, 9, 23, 28, 35, 40, 42, 129]
- EMA0.5 bbox 异常帧: 无
- EMA0.7 bbox 异常帧: [9]
- EMA0.9 bbox 异常帧: [7, 9, 23, 28, 35, 40, 42, 129]

## 11.4 视频结果

| 视频 | 路径 | 显示内容 |
|---|---|---|
| Baseline | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/videos/baseline_bbox.mp4` | 绿色 bbox、frame_id、confidence |
| EMA α=0.5 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/videos/ema_alpha05_bbox.mp4` | 橙色 EMA bbox、蓝色原始 bbox、frame_id、alpha=0.5 |
| EMA α=0.7 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/videos/ema_alpha07_bbox.mp4` | 橙色 EMA bbox、蓝色原始 bbox、frame_id、alpha=0.7 |
| EMA α=0.9 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/videos/ema_alpha09_bbox.mp4` | 橙色 EMA bbox、蓝色原始 bbox、frame_id、alpha=0.9 |

## 11.5 最终建议

| Field | 值 |
|---|---|
| best_alpha | 0.5 |
| jitter_reduce | 29.0%（相比 baseline 中心抖动均值） |
| landmark_improve | 是（Δ-0.046 px） |
| production_candidate | 推荐：EMA 显著降低 bbox 抖动且 landmark 同步改善，可进入 Stage5 候选验证 |

### 结论说明

在 α ∈ {0.5, 0.7, 0.9} 中，**α = 0.5** 将 bbox 中心平均抖动从 2.265 px 降至 1.607 px（降低 29.0%），孤立跳变从 8 降至 0 次。landmark 平均抖动 1.752 → 1.707 px（改善）。

EMA 为因果一阶递归，计算开销可忽略（无额外模型推理，仅 4 坐标线性加权），若结论为推荐/有条件推荐，可作为 Stage5 输出的轻量后处理候选；本实验为只读验证，未修改任何生产代码。

**数据文件**：
- `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/baseline.json`、`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/ema_05.json`、`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/ema_07.json`、`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/ema_09.json`
- `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_bbox_ema_ablation/ablation_metrics.json`

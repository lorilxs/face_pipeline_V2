# Stage5 BBox Center Trajectory Report

## 1. 数据来源

| 项目 | 值 |
|---|---|
| 视频 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_17_e2e/stage0_11/clips/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4` |
| 分辨率 / 帧率 | 1920x1080 / 25 fps |
| 帧数量 | 200（时长 8.0s） |
| 输入 JSON | `docs/stage5_roi_ablation/baseline_results.json`（baseline）、`docs/stage5_roi_ablation/roi_results.json`（temporal ROI 1.2） |
| 异常阈值 | 8 px（来自 `ablation_metrics.json` 的 `outlier_px_threshold`，沿用此前实验第 8.3 节规则） |

本任务为**纯分析/可视化**：未修改 stage_05 生产代码、stage_10 landmark 代码、pipeline 流程或 config.yaml；未添加滤波 / EMA。

## 2. cx / cy 定义

对每一帧 bbox `[x1, y1, x2, y2]` 计算人脸框中心：

| 字段 | 含义 | 公式 |
|---|---|---|
| `cx` | 人脸框中心 X 坐标（pixel） | `(x1 + x2) / 2` |
| `cy` | 人脸框中心 Y 坐标（pixel） | `(y1 + y2) / 2` |
| `frame_id` | 视频帧索引（0 起始、连续） | 与 JSON 中 `frame_id` 一致 |

中心轨迹由 `200` 个连续帧组成（baseline 与 ROI 帧数一致、frame index 连续 0..199）。

## 3. 曲线说明

| 图片 | 路径 | 内容 |
|---|---|---|
| Center X 轨迹 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/visualization/bbox_center_x.png` | 横轴 frame index，纵轴 `cx`（pixel）；蓝色 = baseline，红色 = ROI 1.2；异常帧已标记 |
| Center Y 轨迹 | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/visualization/bbox_center_y.png` | 横轴 frame index，纵轴 `cy`（pixel）；蓝色 = baseline，红色 = ROI 1.2；异常帧已标记 |
| Center 位移（velocity） | `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/visualization/bbox_center_velocity.png` | 相邻帧中心移动距离 `sqrt(dx² + dy²)`；绿色虚线 = 阈值 8 px |

Overlay 视频（可逐帧人工观察检测框是否跳动）：
- Baseline：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/visualization/baseline_bbox_center_overlay.mp4`
- Temporal ROI 1.2：`/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_roi_ablation/visualization/roi_bbox_center_overlay.mp4`

视频内容：绿色 bbox 矩形、红色中心点 `(cx, cy)`、左上角 frame index 与 center 坐标；保持原分辨率 / 原 fps / H.264 编码。

## 4. 异常帧分析

判定规则（沿用此前实验第 8.3 节）：帧 `t-1 -> t` 的中心位移 `> 8 px` 且 `t -> t+1` 的位移恢复正常（`<= 8 px`），标记 `t` 为孤立跳变异常帧。

| 方法 | 异常帧（frame : distance px） | 数量 |
|---|---|---|
| baseline | 7:9.46, 9:14.03, 23:9.76, 28:9.63, 35:8.88, 40:10.95, 42:11.36, 129:10.02 | 8 |
| roi | 7:8.31, 22:8.65, 170:8.36 | 3 |

异常帧列表亦写入 `docs/stage5_roi_ablation/visualization/outlier_frames.json`。

## 5. baseline vs ROI 结论

| 统计量 | baseline | ROI 1.2 | 变化 |
|---|---|---|---|
| cx_mean（中心 X 均值 px） | 933.5338 | 935.5182 | 1.9844 |
| cy_mean（中心 Y 均值 px） | 389.5388 | 379.9083 | -9.6305 |
| cx_std（横向稳定性 px） | 6.0883 | 4.6498 | -1.4385（-23.6%） |
| cy_std（纵向稳定性 px） | 5.7995 | 4.3636 | -1.4359（-24.8%） |
| velocity_mean（平均位移 px） | 2.2653 | 2.7596 | 0.4943（+21.8%） |
| velocity_max（最大位移 px） | 14.0307 | 8.6466 | -5.3841（-38.4%） |
| 孤立跳变异常帧数（>8 px） | 8 | 3 | -5 |

**结论**：

- 平均位移（velocity_mean）：ROI 1.2 为 2.7596 px，baseline 为 2.2653 px（+21.8%）——ROI 约束未降低平均抖动。
- 最大瞬时跳变（velocity_max）：ROI 1.2 为 8.6466 px，baseline 为 14.0307 px（-38.4%）——ROI 显著抑制了最大跳变。
- 孤立跳变异常帧：baseline 8 个（7, 9, 23, 28, 35, 40, 42, 129），ROI 1.2 3 个（7, 22, 170），减少 5 个。
- 稳定性：ROI 1.2 的 `cy_std` 更大（纵向漂移更明显），`cx_std` 变化相对较小；横向/纵向中心均值接近，说明 ROI 未改变人脸框整体位置。

整体而言，temporal ROI 1.2 在该 clip 上**降低了大跳变与孤立异常帧的数量**（峰值跳变从 14.0307 降至 8.6466 px，异常帧从 8 降至 3），但**平均抖动并未下降**（平均位移反而略升），与 `Stage5_ROI_Temporal_Ablation_Report.md` 的定量结论一致。因此默认 ROI 1.2 不直接进入生产；本任务仅用于理解 ROI 方法对 bbox 稳定性的影响，生产管线保持全图检测不变。

**统计文件**：`docs/stage5_roi_ablation/visualization/bbox_center_statistics.json`、`docs/stage5_roi_ablation/visualization/outlier_frames.json`

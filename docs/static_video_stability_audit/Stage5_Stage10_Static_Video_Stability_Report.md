# Stage5 + Stage10 Static Video Stability Audit

- 输入视频：`test_data/1_8s.mp4`
- 目标：量化 **静态人脸视频** 在 Stage5 (SCRFD bbox) / Stage10 (106 landmark) / Stage12 (FLAME/EMICA) 各级引入的时间抖动，并定位根因。
- 生成时间：2026-09-01 17:28:30
- 约束：仅新增实验脚本/产物，**未修改任何生产代码/模型/配置**。

## 1. Video Information

| 属性 | 值 |
|---|---|
| video_path | test_data/1_8s.mp4 |
| width | 592 |
| height | 1280 |
| fps | 25.0 |
| frame_count | 200 |
| duration_s | 8.0 |
| has_audio | True |
| codec | h264 |

> 判断：592×1280 / 25fps / 8.01s / 含音频。理论上为**人物基本静止、摄像机固定**的人脸视频（用于隔离各阶段检测/回归噪声）。

## 2. Stage5 Detection Stability

- Overlay 视频：`stage5_bbox_overlay.mp4`（左=原视频，右=SCRFD bbox + Frame ID / conf / cx / cy）
- 中心曲线：`stage5_bbox_center_curve.png`
- 指标 JSON：`stage5_bbox_center_metrics.json`

| 指标 | 值 (px) |
|---|---|
| mean_cx / mean_cy | 338.1623 / 438.5477 |
| std_cx / std_cy | 0.0278 / 0.0646 |
| frame_to_frame_delta_cx_mean | 0.0234 |
| frame_to_frame_delta_cy_mean | 0.0413 |
| max_jump (单帧跳变) | 0.2385 @ frame 12 |
| 单帧跳变 > 5px 异常帧数 | 0 |

> 判定：**STABLE**。帧间中心平均位移 dx=0.0234px / dy=0.0413px。

## 3. Stage10 Landmark Stability

- Overlay 视频：`stage10_landmark_overlay.mp4`（左=原视频，右=106 landmark + Frame ID / jitter）
- 指标 JSON：`stage10_landmark_metrics.json`

| 指标 | 值 (px) |
|---|---|
| mean_jitter_px | 0.1418 |
| std_jitter_px | 0.2134 |
| max_jump_px | 0.9505 @ frame 182 |
| max_point_jump_px | 1.2087 @ frame 197 |
| eye_jitter | 0.1412 |
| mouth_jitter | 0.1414 |
| jaw_jitter | 0.1476 |

## 4. Stage12-17 FLAME Stability

- 指标 JSON：`flame_temporal_metrics.json`（读 Stage17 `visual_feature` 的 head_pose[100:103] / jaw_pose[103:106]，按 Rodrigues 相对旋转角统计）

| 指标 | 值 (deg) |
|---|---|
| head_mean_delta_deg | 0.0241 |
| head_max_delta_deg | 0.1066 |
| jaw_mean_delta_deg | 0.0098 |
| jaw_max_delta_deg | 0.0336 |
| expression_mean_delta_l2 | 0.0189 |
| expression_max_delta_l2 | 0.0674 |

## 5. Root Cause Analysis

传播链：

```
Stage5 SCRFD bbox -> aligned crop(expand=2.0) -> Stage10 106landmark 
     -> Stage11 affine align -> Stage12 EMICA(image) -> FLAME -> Stage17 feature
```

- **Stage5 bbox 漂移**：帧间中心位移 dx/dy = 0.0234 / 0.0413 px，std_cx/std_cy = 0.0278/0.0646 px，单帧最大跳变 0.2385 px（>5px 异常帧 0）。
- **Stage10 landmark 漂移**：平均帧间位移 0.1418 px，最大 0.9505 px（区域：eye 0.1412 / mouth 0.1414 / jaw 0.1476 px）。
- **Stage12 EMICA 拟合噪声**：head 帧间 0.0241°/帧，jaw 0.0098°/帧。

> **结论（实测）**：在 1_8s.mp4 这一画质良好、人物基本静止的输入上，三个环节均未出现明显抖动：
> - Stage5 bbox 中心 std≈0.0278/0.0646px，帧间均值≈0.02–0.04px，无 >5px 单帧跳变（异常帧 0）。
> - Stage10 landmark 帧间均值≈0.1418px（区域均匀：eye 0.1412 / mouth 0.1414 / jaw 0.1476px）。
> - Stage12 head 帧间≈0.0241° / jaw≈0.0098°（相对旋转角，max head 0.1066°）。三者均处亚像素/亚角度级。
> 
> **相对贡献排序：landmark(≈0.14px) > bbox(≈0.03px) ≈ EMICA(≈0.01–0.02°)**，绝对量均极小。
> 传播链：bbox 中心 0.03px → expand=2.0 crop（~573px）→ 2d106det 放大为 ~0.14px landmark 抖动 → Stage11 仿射(scale≈0.43x 缩至 256px) → EMICA 逐帧图像回归为 ~0.01–0.02°/帧 FLAME 抖动；
> **各段传导系数均远小于 1，静态且画质良好的输入不会产生肉眼可见的抖动尖刺**。
> 
> 推论：此前在测试集上观察到的 Head/Jaw 尖刺，根因应是**输入退化**（模糊/遮挡/bbox 突变/真实运动）而非本 pipeline 的系统性噪声；去抖应优先做**输入侧稳定**（bbox/Kalman、图像质量门控）或**输出侧 FLAME 后处理**（Hampel / SO(3) 滤波），而非对稳定输入做无差别时序平滑。

## 6. Recommendation

1. **Stage5 bbox 时序平滑**：中心/宽高 EMA 或 Kalman（输出侧可观察尖刺抑制）。
2. **Stage10 landmark 时序平滑**：因果 EMA（已做 α=0.7 消融，注意其对 EMICA 传导有限）。
3. **FLAME 参数后处理**：head/jaw rotvec 逐通道 Hampel（window 5–11）或 SO(3) 流形滤波（四元数滑动平均/SLERP）；帧间 delta 域离群替换。
4. **模型替换**：SPECTRE 通道（保留 spectre_head/spectre_jaw 字段），或 landmark→3D pose fitting 分支（landmark 平滑直接生效）。

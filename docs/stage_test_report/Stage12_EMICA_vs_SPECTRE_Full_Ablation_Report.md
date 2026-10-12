# Stage12 EMICA vs SPECTRE Full Ablation Report

> 生成时间: 2026-09-01T06:30:37
> 视频: `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4`
> 输入: 复用 `outputs/stage0_17_e2e/dataset_validation/stage_inputs/*_aligned_faces.json`
> Clip 数: 11

## 1. 实验概览

| 字段 | Experiment A (EMICA) | Experiment B (SPECTRE) |
|---|---|---|
| expression_source | emica | spectre |
| head_source | emica | spectre |
| jaw_source | emica | spectre |
| gaze_source | mediapipe (Stage13, 两版本一致) | mediapipe (Stage13, 两版本一致) |
| audio_source | chinese_hubert (Stage14, 两版本一致) | chinese_hubert (Stage14, 两版本一致) |
| expression_dim | 100 | 100 (50-dim E_flame latent slice + 后 50 zero-padding) |

输出目录:

```
outputs/ablation_stage12_emica_vs_spectre/
    emica/    stage12_flame / stage13_gaze / stage14_audio / stage17_fusion / stage17_multimodal_export
    spectre/  stage12_flame / stage13_gaze / stage14_audio / stage17_fusion / stage17_multimodal_export
```

## 2. Shape Comparison

| clip | spectre expression | emica expression | head_pose | jaw_pose | gaze | audio | 一致 |
|---|---|---|---|---|---|---|---|
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0000 | (191, 100) | (191, 100) | (191, 3) | (191, 3) | (188, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0002 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0003 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0004 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0001 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0002 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0003 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0004 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |
| CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0005 | (200, 100) | (200, 100) | (200, 3) | (200, 3) | (200, 6) | (200, 768) | ✓ |

## 3. Temporal Stability Analysis

相邻帧差 (L2 幅值) 统计; pose 以 degree/frame 计.

| 参数 | 指标 | EMICA | SPECTRE | 较稳定 |
|---|---|---|---|---|
| expression | mean_delta | 0.93422 | 0.55144 | SPECTRE |
| expression | std_delta | 0.63777 | 0.70070 | EMICA |
| expression | max_delta | 3.26426 | 3.64731 | EMICA |
| head_pose (deg/frame) | mean_delta | 0.78445 | 0.73793 | SPECTRE |
| head_pose (deg/frame) | std_delta | 0.59643 | 0.62711 | EMICA |
| head_pose (deg/frame) | max_delta | 3.95959 | 4.80755 | EMICA |
| jaw_pose (deg/frame) | mean_delta | 0.97466 | 0.60177 | SPECTRE |
| jaw_pose (deg/frame) | std_delta | 1.02530 | 0.56448 | SPECTRE |
| jaw_pose (deg/frame) | max_delta | 6.34690 | 3.09935 | SPECTRE |
| gaze | mean_delta | 0.02402 | 0.02402 | = |
| gaze | std_delta | 0.02021 | 0.02021 | = |
| gaze | max_delta | 0.10212 | 0.10212 | = |
| audio | mean_delta | 6.38845 | 6.38845 | = |
| audio | std_delta | 1.27504 | 1.27504 | = |
| audio | max_delta | 9.58145 | 9.58145 | = |

## 4. Output Package Verification

| 检查项 | EMICA | SPECTRE | 结论 |
|---|---|---|---|
| npy 无 NaN/Inf (失败 clip 数) | 0 | 0 | ✓ |
| timestamp 单调递增 (失败 clip 数) | 0 | 0 | ✓ |
| frame_count 两版本一致 | 11 clip | 11 clip | ✓ |
| Stage17 layout (visual_dim=112) | 0 失败 | 0 失败 | ✓ |
| 来源标签 | emica/emica/emica | spectre/spectre/spectre | ✓ |

## 5. 语义说明 (Important)

- **EMICA expression**: 独立 ExpressionEncoder 输出 (100 维, 全部维度有活动)。
- **SPECTRE expression**: `E_flame` 输出的 50 维 latent slice (`parameters[:,150:200]`),
  后 50 维为 pipeline 对齐所需 zero-padding (`expression_dim=100` 保持不变)。
  **两者语义不同, 不能直接比较幅值或视为同一参数空间。**

因此, 如果 SPECTRE temporal stability 更好, 只能说明
"SPECTRE FLAME parameter sequence is temporally smoother",
**不能** 直接说明 "SPECTRE expression accuracy is higher"。

## 6. 结论与 Handoff

- 两套参数包位置:
  - EMICA:   `outputs/ablation_stage12_emica_vs_spectre/emica/`
  - SPECTRE: `outputs/ablation_stage12_emica_vs_spectre/spectre/`
- Stage17 布局固定 `visual_dim=112`: expression[0:100] head_pose[100:103]
  jaw_pose[103:106] gaze[106:112]。
- 最终 mesh 生成由外部 mesh decoder 同事基于上述两套参数包进行, 用于判断
  抖动/嘴部稳定性是否改善; 在 mesh 验证完成前, **生产默认仍为全 EMICA**。

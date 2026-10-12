# Stage17 Multimodal Export Report

- 生成时间: 2026-08-29T11:33:42
- 生成脚本: `scripts/stage17_multimodal_export_verify.py`

## Dataset

- clip_id: `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`
- 帧数 (visual T): 200
- 时长: 8.00s @ 25 FPS
- audio 匹配帧: 200 (100.0%)

## 交付格式 (Data Contract)

| 文件 | Shape | dtype | 内容 |
|------|-------|-------|------|
| `flame/{clip_id}.npy` | `[T,112]` | float32 | 视觉运动特征 (FLAME expression[0:100] + head/jaw pose[100:106] + Gaze[106:112]) |
| `audio/{clip_id}.npy` | `[T,768]` | float32 | HuBERT 音频 embedding |
| `timestamp/{clip_id}.npz` | `[T]` × 4 | int32/float32/bool | frame_index / timestamps / audio_match_idx / dropout_mask |
| `metadata/{clip_id}.json` | - | JSON | 溯源元数据 (含 visual_layout / *_source) |
| `manifest.csv` | - | CSV | 数据集级索引 |

## Summary

| Field Name | 中文名称 | Type | Field Description |
|------------|----------|------|--------------------|
| export_structure | 输出结构 | string | PASS — 目录结构检查 |
| flame_shape | visual维度 | string | PASS — [T,112] |
| audio_shape | audio维度 | string | PASS — [T,768] |
| timestamp_alignment | 时间同步 | string | PASS — frame对齐检查 |
| metadata_check | 元数据检查 | bool | PASS — 字段完整性 |
| final_result | 最终结果 | string | PASS — PASS/FAIL |

## Validation Checks

| Check | Status | Detail |
|-------|--------|--------|
| export_structure | PASS | flame/audio/timestamp/metadata/manifest 齐全 |
| flame_shape | PASS | flame=[200, 112] float32 (期望 [T,112]) |
| audio_shape | PASS | audio=[200, 768] float32 (期望 [T,768]) |
| timestamp_alignment | PASS | T=200, 4 数组长度一致=True, timestamp 单调=True, frame_index 一致=True, dropout 一致=True, audio_match 一致=True |
| metadata_check | PASS | 必含字段 10/10, layout={'expression': [0, 100], 'head_pose': [100, 103], 'jaw_pose': [103, 106], 'gaze': [106, 112]}, head=emica, jaw=emica, gaze=mediapipe, audio=chinese_hubert |
| round_trip | PASS | flame/audio/timestamp 与 fusion 样本逐元素一致 |

## Result

**PASS** — Stage17 多模态导出层完整闭环可用, 交付格式符合数据契约。

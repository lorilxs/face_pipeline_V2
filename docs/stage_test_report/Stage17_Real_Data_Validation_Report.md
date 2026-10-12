# Stage17 Real Data Validation Report

- 生成时间: 2026-08-29T04:14:19
- 生成脚本: `scripts/stage17_real_data_verify.py`

## Dataset

- clip_id: `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`
- frames (FLAME): 200
- duration: 8.00s @ 25 FPS

## Input

| 阶段 | 文件 | Shape | 校验 |
|------|------|-------|------|
| Stage12 FLAME | `flame_features.npz` | exp[200,100] + pose[200,6] | PASS |
| Stage13 Gaze  | `gaze_features.npz`  | gaze_vector[200,6]          | PASS |
| Stage14 Audio | `audio_features.npz` | feature[200,768]           | PASS |

## Output

TrainingSample (`training_sample.npz`):

- visual_feature: `[200, 112]` float32 (expression[0:100] + pose[100:106] + gaze[106:112])
- audio_feature: `[200, 768]` float32 (Chinese HuBERT)
- frame_index: `[200]`
- timestamps: `[200]`
- audio_match_idx: `[200]`
- dropout_mask: `[200]`
- valid_length: 200
- visual_layout: `{'expression': [0, 100], 'pose': [100, 106], 'gaze': [106, 112]}`

## Alignment

Frame count:

- FLAME: T_flame = 200
- Gaze:  T_gaze  = 200
- Audio: T_audio = 200

Alignment result:

- visual T: T_visual = 200
- audio matched: T_audio_match = 200 (100.0%)
- Gaze missing: 0 frames
- Audio unmatched: 0 frames
- dropout ratio: 0.00%

## Validation Checks

| Check | Status | Detail |
|-------|--------|--------|
| FLAME 帧数 > 0 | PASS | 200 frames |
| FLAME expression [T,100] | PASS | expcode dims=[100] |
| FLAME pose [T,6] | PASS | posecode dims=[6] |
| FLAME timestamp 单调 | PASS | strict_increasing=True weak_monotonic=True ts[4.000..11.960] |
| FLAME timestamp 无 NaN/Inf | PASS | nan=0 inf=0 |
| GAZE 帧数 > 0 | PASS | 200 frames |
| GAZE gaze_vector [T,6] | PASS | dims=[6] |
| GAZE timestamp 单调 | PASS | strict_increasing=True weak_monotonic=True ts[4.000..11.960] |
| GAZE timestamp 无 NaN/Inf | PASS | nan=0 inf=0 |
| GAZE vector 无 NaN/Inf | PASS | nan=0 inf=0 |
| GAZE 视线向量 norm≈1 | PASS | left=[1.0000,1.0000] right=[1.0000,1.0000] |
| AUDIO 帧数 > 0 | PASS | 200 frames |
| AUDIO feature [T,768] | PASS | dims=[768] |
| AUDIO timestamp 单调 | PASS | strict_increasing=True weak_monotonic=True ts[4.000..11.960] |
| AUDIO timestamp 无 NaN/Inf | PASS | nan=0 inf=0 |
| AUDIO feature 无 NaN/Inf | PASS | nan=0 inf=0 |
| AUDIO feature 非零比例 > 0 | PASS | nonzero_ratio=1.0000 std=0.2468 |
| 输出样本包 (npz/meta/manifest) | PASS | training_sample.npz/metadata.json/manifest.csv |
| visual_feature 无 NaN/Inf | PASS | nan=0 inf=0 |
| expression std > 0 | PASS | std=0.4488 mean=-0.0337 |
| pose std 合理 (>1e-4) | PASS | std=0.0730 mean=0.0164 |
| visual gaze norm≈1 (有效帧) | PASS | n=200 left=[1.0000,1.0000] right=[1.0000,1.0000] |
| audio_feature 无 NaN/Inf | PASS | nan=0 inf=0 |
| audio_feature 非零比例 (matched) | PASS | nonzero_ratio=1.0000 std=0.2468 |
| audio_feature std > 0 | PASS | std=0.2468 |

## Result

**PASS** — 真实数据链路 Stage12→Stage17 完整闭环可用。

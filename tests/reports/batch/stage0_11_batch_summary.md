# Face Pipeline V2 — Stage 0-11 Batch Test Summary

- **Generated at (UTC)**: 2026-08-20T01:39:56Z
- **Python**: 3.12.3
- **Videos**: 1
- **Command**: `python tests/stage0_11_batch_test.py --input-dir test_data/chinese_talking_videos --config configs/config.yaml --test-base outputs/stage0_11_batch --results-dir tests/reports/batch --limit 1`

## 1. Per-video Final Status

| Video | Final | Total elapsed (s) | Stages PASS / FAIL / SKIP |
| --- | --- | ---: | --- |
| CNFACE_00001_BV18Qg96YE1W_000000.mp4 | PASS | 416.49 | 18 / 0 / 0 |

## 2. Per-stage Aggregate (across videos)

| Stage | PASS | FAIL | SKIPPED | PASS rate | Avg elapsed (s) | Max elapsed (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0 (FFprobe integrity check) | 1 | 0 | 0 | 100.0% | 1.25 | 1.25 |
| Stage 0.5 (Resolution filter) | 1 | 0 | 0 | 100.0% | 0.00 | 0.00 |
| Stage 1 (FFmpeg standardization) | 1 | 0 | 0 | 100.0% | 14.53 | 14.53 |
| Stage 2 (PySceneDetect) | 1 | 0 | 0 | 100.0% | 5.89 | 5.89 |
| Stage 3 (SCRFD coarse screening) | 1 | 0 | 0 | 100.0% | 13.59 | 13.59 |
| Stage 4 (Sliding-window clips) | 1 | 0 | 0 | 100.0% | 0.00 | 0.00 |
| Stage 5 (SCRFD detect/crop) | 1 | 0 | 0 | 100.0% | 27.90 | 27.90 |
| Stage 5.1 (ByteTrack face tracking) | 1 | 0 | 0 | 100.0% | 0.00 | 0.00 |
| Stage 5.2 (Single-face filter) | 1 | 0 | 0 | 100.0% | 0.00 | 0.00 |
| Stage 5.3 (Temporal stability) | 1 | 0 | 0 | 100.0% | 0.00 | 0.00 |
| Stage 6 (6DRepNet pose) | 1 | 0 | 0 | 100.0% | 41.75 | 41.75 |
| Stage 7 (SyncNet A/V sync) | 1 | 0 | 0 | 100.0% | 20.88 | 20.88 |
| Stage 7.1 (A/V alignment) | 1 | 0 | 0 | 100.0% | 1.05 | 1.05 |
| Stage 8 (HyperIQA quality) | 1 | 0 | 0 | 100.0% | 224.41 | 224.41 |
| Stage 9 (Quality fusion/filter) | 1 | 0 | 0 | 100.0% | 0.01 | 0.01 |
| Stage 9.5 (Clip export) | 1 | 0 | 0 | 100.0% | 31.96 | 31.96 |
| Stage 10 (InsightFace landmark) | 1 | 0 | 0 | 100.0% | 6.19 | 6.19 |
| Stage 11 (Affine alignment) | 1 | 0 | 0 | 100.0% | 21.78 | 21.78 |

## 3. Stage × Video Status Matrix

| Video | Stage 0 (FFprobe integrity check) | Stage 0.5 (Resolution filter) | Stage 1 (FFmpeg standardization) | Stage 2 (PySceneDetect) | Stage 3 (SCRFD coarse screening) | Stage 4 (Sliding-window clips) | Stage 5 (SCRFD detect/crop) | Stage 5.1 (ByteTrack face tracking) | Stage 5.2 (Single-face filter) | Stage 5.3 (Temporal stability) | Stage 6 (6DRepNet pose) | Stage 7 (SyncNet A/V sync) | Stage 7.1 (A/V alignment) | Stage 8 (HyperIQA quality) | Stage 9 (Quality fusion/filter) | Stage 9.5 (Clip export) | Stage 10 (InsightFace landmark) | Stage 11 (Affine alignment) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNFACE_00001_BV18Qg96YE1W_000000.mp4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## 4. Errors

No stage raised an exception in any video.


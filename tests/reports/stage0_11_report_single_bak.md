# Face Pipeline V2 — Stage 0-11 Integration Test Report

**Final Result**: `PARTIAL`

## 1. Environment

| Item | Value |
| --- | --- |
| Container | `face_pipeline_lxs` (host `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2` → `/workspace/Face_Pipeline_V2`) |
| Python | 3.12.3 |
| ONNX Runtime | 1.21.1+ppu1.7.0.oe (`CUDAExecutionProvider`, `CPUExecutionProvider`) |
| PyTorch | 2.9.0 (`torch.cuda.is_available()=True`, device `PPU-ZW810E`) |
| OpenCV / PySceneDetect | 4.13.0 / 0.7.1 |
| face_pipeline_v2 | 0.1.0 (editable) |
| Config used | `configs/config.yaml` (loaded via `ConfigLoader`; file unchanged) |
| Test output base | `outputs/stage0_11_hardfilter_suzy_m003` (driver-only override of `storage.output_dir`/`stage_01.output_dir`) |
| Test generated at (UTC) | 2026-08-19T15:39:07Z |

## 2. Input Video

- **Path**: `test_data/BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_seg_m003.mp4`
- **Resolution**: 1920×1080
- **Stage 0.5 resolution check**: 1920×1080 vs min 534×534 → accepted
- **Frame rate / frames / duration**: 25.00 fps / 923 frames / 36.9 s
- **Audio**: yes

## 3. Stage Results

| Stage | Status | Elapsed (s) | Output summary |
| --- | --- | ---: | --- |
| Stage 0 (FFprobe integrity check) | PASS | 0.95 | id=BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_seg_m003 | 1920x1080 | 25.00fps | 36.9s | 923 frames | audio=True |
| Stage 0.5 (Resolution filter) | PASS | 0.0 | 1920x1080 vs min 534x534 | is_valid=True | reject=None |
| Stage 1 (FFmpeg standardization) | PASS | 6.85 | std_video=outputs/stage0_11_hardfilter_suzy_m003/videos/BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_seg_m003/BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_seg_m003_std.mp4 | audio=outputs/stage0_11_hardfilter_suzy_m003/videos/BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_seg_m003/BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_BV1LE411L729_Suzy_攻气单眼皮_衣冠楚楚_斯文败类_seg_m003_audio.wav |
| Stage 2 (PySceneDetect) | PASS | 3.13 | 2 scene(s) |
| Stage 3 (SCRFD coarse screening) | PASS | 5.64 | 2 scene(s): 1 passed, 1 rejected |
| Stage 4 (Sliding-window clips) | PASS | 0.0 | 7 clip(s) |
| Stage 5 (SCRFD detect/crop) | PASS | 14.09 | 7 sequence(s), 495 face detections |
| Stage 5.1 (ByteTrack face tracking) | PASS | 2.82 | 7 sequence(s), 3 unique track(s) |
| Stage 5.2 (Single-face filter) | PASS | 0.0 | 7 input: 0 passed, 7 rejected (hard filter, dropped) |
| Stage 5.3 (Temporal stability) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 6 (6DRepNet pose) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 7 (SyncNet A/V sync) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 7.1 (A/V alignment) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 8 (HyperIQA quality) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 9 (Quality fusion/filter) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 9.5 (Clip export) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 10 (InsightFace landmark) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |
| Stage 11 (Affine alignment) | SKIPPED | 0.0 |  | _skipped:_ Stage 5.2 produced no sequences |

## 4. Errors / Diagnostics

No stage raised an exception.

## 5. Remaining Issues / Notes

- `python_speech_features` (Stage 7 dependency) was installed from PyPI directly (`--index-url https://pypi.org/simple/`) because the private pip mirror returns 401.
- The test harness lives at `tests/stage0_11_integration_test.py`; it drives the existing stage classes via their standard `run()` lifecycle without modifying pipeline code.
- All output artifacts are under `outputs/stage0_11_hardfilter_suzy_m003/`; machine-readable results in `tests/reports/stage0_11_results.json`.

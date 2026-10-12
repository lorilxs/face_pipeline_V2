# Face Pipeline V2 — Stage 0-11 Integration Test Report

**Final Result**: `PASS`

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
| Test output base | `outputs/stage0_11_test_verify` (driver-only override of `storage.output_dir`/`stage_01.output_dir`) |
| Test generated at (UTC) | 2026-08-21T09:14:17Z |

## 2. Input Video

- **Path**: `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4`
- **Resolution**: 1920×1080
- **Stage 0.5 resolution check**: 1920×1080 vs min 534×534 → accepted
- **Frame rate / frames / duration**: 25.00 fps / 1504 frames / 60.2 s
- **Audio**: yes

## 3. Stage Results

| Stage | Status | Elapsed (s) | Output summary |
| --- | --- | ---: | --- |
| Stage 0 (FFprobe integrity check) | PASS | 2.65 | id=CNFACE_00001_BV18Qg96YE1W_000000 | 1920x1080 | 25.00fps | 60.2s | 1504 frames | audio=True |
| Stage 0.5 (Resolution filter) | PASS | 0.0 | 1920x1080 vs min 534x534 | is_valid=True | reject=None |
| Stage 1 (FFmpeg standardization) | PASS | 7.13 | std_video=outputs/stage0_11_test_verify/videos/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_std.mp4 | audio=outputs/stage0_11_test_verify/videos/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_audio.wav |
| Stage 2 (PySceneDetect) | PASS | 4.34 | 3 scene(s) |
| Stage 3 (SCRFD coarse screening) | PASS | 10.56 | 3 scene(s): 2 passed, 1 rejected |
| Stage 4 (Sliding-window clips) | PASS | 0.0 | 11 clip(s) |
| Stage 5 (SCRFD detect/crop) | PASS | 33.56 | 11 sequence(s), 2191 face detections |
| Stage 5.1 (ByteTrack face tracking) | PASS | 0.0 | 11 sequence(s), 0 unique track(s) |
| Stage 5.2 (Single-face filter) | PASS | 0.0 | 11 input: 11 passed, 0 rejected (hard filter, dropped) |
| Stage 5.3 (Temporal stability) | PASS | 0.0 | 11 input: 11 passed, 0 rejected (hard filter, dropped) |
| Stage 6 (6DRepNet pose) | PASS | 111.8 | 11 sequence(s), 2200 pose(s) |
| Stage 7 (SyncNet A/V sync) | PASS | 43.3 | 11 clip(s) |
| Stage 7.1 (A/V alignment) | PASS | 1.43 | 11 clip(s) |
| Stage 8 (HyperIQA quality) | PASS | 115.56 | 11 sequence(s), 2200 score(s) |
| Stage 9 (Quality fusion/filter) | PASS | 0.0 | 11 clip(s), 7 accepted |
| Stage 9.5 (Clip export) | PASS | 13.81 | accepted=7 exported_videos=7 exported_audio=7 manifest=7 failures=0 |
| Stage 10 (InsightFace landmark) | PASS | 5.18 | 1400 landmark(s) |
| Stage 11 (Affine alignment) | PASS | 3.94 | 1400 aligned face(s) |

## 4. Errors / Diagnostics

No stage raised an exception.

## 5. Remaining Issues / Notes

- `python_speech_features` (Stage 7 dependency) was installed from PyPI directly (`--index-url https://pypi.org/simple/`) because the private pip mirror returns 401.
- The test harness lives at `tests/stage0_11_integration_test.py`; it drives the existing stage classes via their standard `run()` lifecycle without modifying pipeline code.
- All output artifacts are under `outputs/stage0_11_test_verify/`; machine-readable results in `tests/reports/stage0_11_results.json`.

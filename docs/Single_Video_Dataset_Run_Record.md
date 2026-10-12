# Single-Video Dataset-Mode Run Record (100008.mp4)

Date: 2026-08-31 · Video: `/mnt/home/lxs/videos/vivihead_chinese_videos/batch_283_videos_flame_audio/videos/100008.mp4`
Mode: `pipeline_mode: dataset` (Stage 6 / 7 / 7.1 / 8 / 9 / 9.5 skipped) · Jaw: **EMICA production scheme** (`jaw_source=emica`).

## Driver

New reusable single-video driver: `scripts/run_single_video_dataset.sh`

```
bash scripts/run_single_video_dataset.sh <video_abs_path> [--smoke]
```

Pipeline (all in one driver):
1. Copies the video into `test_data/video_<id>/` so the docker container
   (mount `Face_Pipeline_V2 -> /workspace/Face_Pipeline_V2`) can read it.
2. **Stage 0–11** — docker `face_pipeline_lxs`, system `python3` (3.12),
   `scripts/stage0_17_e2e_validate.py --config configs/config_dataset_mode.yaml
   --test-base outputs/video_<id>/stage0_11`.
   - Stage 0 uses the ffmpeg `-i` fallback (container ffmpeg is the
     imageio_ffmpeg static build; ffprobe is absent).
3. **Stage 12** (EMICA, docker `emica_env`, CPU) → **Stage 13** (gaze, host) →
   **Stage 14** (Chinese HuBERT audio, host), per accepted clip.
4. **Stage 17** fusion (`face_pipeline_v2/stages/stage_17_fusion/runner.py`).
5. **Production multimodal export + validation** via
   `scripts/export_production_multimodal.py` (now parametrized with
   `--base` / `--export-dir`; defaults keep the legacy
   `outputs/stage0_17_e2e/dataset_validation` behaviour).

Outputs go under `outputs/video_<id>/` — never touches
`outputs/stage0_17_e2e/dataset_validation/`.

## 100008.mp4 run results

- Video: 512×512, 25 fps, 9.0 s, AAC audio → `video_id=100008`.
- 1 scene detected, 1 accepted clip: `100008_scene_0000_clip_0000` (T=200 frames, 8 s @25 fps).
- Stage 0–11: `PASS` (Stage 6/7/7.1/8/9/9.5 all `SKIPPED` — dataset mode).
- Stage 12 `jaw_source=emica` · 200 frames in 20.8 s (CPU).

### Output validation (all PASS)

| Check | Result |
|---|---|
| expression (S12) | `(200, 100)`, no NaN/Inf |
| jaw_pose (S12) | `(200, 3)`, no NaN/Inf |
| head_pose (S12) | `(200, 3)`, no NaN/Inf |
| eye_pose (S12) | `(200, 6)`, no NaN/Inf |
| gaze (S13) | `(200, 6)`, no NaN/Inf |
| audio (S14) | `(200, 768)`, no NaN/Inf |
| fusion visual_feature (S17) | `(200, 112)`, no NaN/Inf |
| fusion audio_feature (S17) | `(200, 768)`, no NaN/Inf |
| metadata `jaw_source` | `emica` (also head/expression `emica`, gaze `mediapipe`, audio `chinese_hubert`) |
| `visual_dim` | `112` (layout: expression [0,100], head [100,103], jaw [103,106], gaze [106,112]) |
| **jaw == emica_jawpose** | `max|jaw − emica_jawpose| = 0.0` (exactly equal) |
| jaw vs SPECTRE | `max|jaw − spectre_jawpose| = 0.162 rad ≈ 9.29°` (audit-only) |
| export `ALL OK` | `flame/audio/timestamp/metadata/manifest.csv` written |

## Artifacts

- Stage0–11 base: `outputs/video_100008/stage0_11/`
  (`stage12_flame/`, `stage13_gaze/`, `stage14_audio/`, `stage17_fusion/`)
- Export: `outputs/video_100008/stage17_multimodal_export/`
- Logs: `logs/video_100008_driver.log`, `logs/stage0_11_100008.log`,
  `logs/stage12|13|14|17_100008_scene_0000_clip_0000.log`, `logs/export_100008.log`
- Last-run Stage0–11 JSON: `outputs/stage0_17_e2e/stage0_11_results.json`
  (shared "last run" record; does not affect `dataset_validation/` outputs).

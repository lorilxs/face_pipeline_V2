# Final Stage0-17 End-to-End Validation — Mesh Export

> **⚠️ 历史记录 (SUPERSEDED)** — 2026-08-31 正式采用 **EMICA jaw** 为最终生产方案（FLAME mesh 重建验证）。本报告为历史审计/验证记录，保留作为历史；最终方案见 `docs/Stage12_EMICA_Jaw_Final_Adoption_Report.md`。


**Pipeline mode:** `dataset` — Stage0→5→10→11→12→13→14→17→17-export.  
**Overall:** `PASS`  
**Clips:** 11 (Stage12/13/14/17 & export all OK = 11/11/11/11/11)

> Generated from `final_mesh_export_checks.json` (validation) + `stage0_11_results.json` (e2e, generated_at=2026-08-31T03:18:12Z).

## 1. Environment

- **Host**: `lxs@304a02301` (bare metal) — validation & Stage13/14/17 driver
- **Docker**: container `face_pipeline_lxs` (python 3.12) — Stage0-11 e2e driver
- **emica_env** (inside container, python 3.10, torch 2.4.1+cpu) — Stage12 EMICA+SPECTRE
- **Compute**: CPU-only for Stage12 (0.21 s/frame; no CUDA in emica_env). Stage13/14 host CPU.
- **Video probe**: ffprobe unavailable in this ffmpeg build → Stage0 uses ffmpeg-based fallback probe
- **Models**: SCRFD (Stage3/5), InsightFace (Stage10), MediaPipe face_landmarker (Stage13), Chinese-HuBERT (Stage14), EMICA + SPECTRE (Stage12)

## 2. Input Video

| Property | Value |
|---|---|
| File | `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4` |
| video_id | `CNFACE_00001_BV18Qg96YE1W_000000` |
| Resolution | 1920×1080 |
| FPS | 25.0 |
| Duration | 60.2 s |
| Total frames | 1504 |
| Has audio | True |
| Accepted clips | 11 |

## 3. Stage Execution Table

| Stage | Status | Detail |
|---|---|---|
| Stage 0 (FFprobe integrity check) | PASS | 1.13s — id=CNFACE_00001_BV18Qg96YE1W_000000 | 1920x1080 | 25.00fps | 60.2s | 1504 frames | audio=True |
| Stage 0.5 (Resolution filter) | PASS | 0.0s — 1920x1080 vs min 534x534 | is_valid=True | reject=None |
| Stage 1 (FFmpeg standardization) | PASS | 10.3s — std_video=outputs/stage0_17_e2e/dataset_validation/videos/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_std.mp4 | audio=outputs/stage0_17_e2e/dataset_validation/videos/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_audio.wav |
| Stage 2 (PySceneDetect) | PASS | 4.6s — 3 scene(s) |
| Stage 3 (SCRFD coarse screening) | PASS | 12.86s — 3 scene(s): 2 passed, 1 rejected |
| Stage 4 (Sliding-window clips) | PASS | 0.0s — 11 clip(s) |
| Stage 5 (SCRFD detect/crop) | PASS | 62.18s — 11 sequence(s), 2191 face detections |
| Stage 5.1 (ByteTrack face tracking) | PASS | 0.0s — 11 sequence(s), 0 unique track(s) |
| Stage 5.2 (Single-face filter) | PASS | 0.0s — 11 input: 11 passed, 0 rejected (hard filter, dropped) |
| Stage 5.3 (Temporal stability) | PASS | 0.0s — 11 input: 11 passed, 0 rejected (hard filter, dropped) |
| Stage 6 (6DRepNet pose) | SKIPPED | — — pipeline_mode=dataset |
| Stage 7 (SyncNet A/V sync) | SKIPPED | — — pipeline_mode=dataset |
| Stage 7.1 (A/V alignment) | SKIPPED | — — pipeline_mode=dataset |
| Stage 8 (HyperIQA quality) | SKIPPED | — — pipeline_mode=dataset |
| Stage 9 (Quality fusion/filter) | SKIPPED | — — pipeline_mode=dataset |
| Stage 9.5 (Clip export) | SKIPPED | — — pipeline_mode=dataset: 11 accepted clip(s), 11 AV alignment(s) from original clip wav |
| Stage 10 (InsightFace landmark) | PASS | 17.13s — 2191 landmark(s) |
| Stage 11 (Affine alignment) | PASS | 6.08s — 2191 aligned face(s) | exported 11 clip metadata |

> Stage6/7/7.1/8/9/9.5 are **SKIPPED** in dataset mode (no quality filtering / A-V sync / clip re-export; Stage9.5 is replaced by the dataset adapter's `build_dataset_av_alignments`). Stage12→17 run in this validation as the mesh-export production chain.

## 4. FLAME Statistics (Stage12)

Per-clip `flame_features.npz`: `expression_code [T,100]`, `head_pose [T,3]`, `jaw_pose [T,3]`, `spectre_jawpose [T,3]`, `emica_jawpose [T,3]`. Gaze `[T,6]` is produced by Stage13 (Section 5) and fused into Stage17's `visual_feature [T,112]`.

| Check | Requirement | Result |
|---|---|---|
| expression shape | [T,100] | PASS (11/11) |
| head_pose shape | [T,3] | PASS (11/11) |
| jaw_pose shape | [T,3] | PASS (11/11) |
| No NaN/Inf | all finite | PASS (11/11) |
| Head rotation | max \|θ\| < 0.5 rad | max=0.329504 PASS |
| Jaw rotation | max \|θ\| < 0.2 rad | max=0.086058 PASS |
| jaw source = SPECTRE | `jaw_pose == spectre_jawpose` | max=0.000000 PASS |

Expression stats across all clips: per-clip mean: avg=-0.0357 (global min -2.5672 .. global max 2.3744), avg std=0.4601

Per-clip FLAME checks:

| clip | T | expr | head | jaw | head_max_rad | jaw_max_rad | jaw==spectre |
|---|---|---|---|---|---|---|---|
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0000` | 191 | 100 | 3 | 3 | 0.2140 | 0.0726 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001` | 200 | 100 | 3 | 3 | 0.2245 | 0.0729 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0002` | 200 | 100 | 3 | 3 | 0.2440 | 0.0729 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0003` | 200 | 100 | 3 | 3 | 0.2440 | 0.0861 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0004` | 200 | 100 | 3 | 3 | 0.2336 | 0.0861 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000` | 200 | 100 | 3 | 3 | 0.3295 | 0.0641 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0001` | 200 | 100 | 3 | 3 | 0.2336 | 0.0641 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0002` | 200 | 100 | 3 | 3 | 0.2375 | 0.0733 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0003` | 200 | 100 | 3 | 3 | 0.2511 | 0.0733 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0004` | 200 | 100 | 3 | 3 | 0.2912 | 0.0696 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0005` | 200 | 100 | 3 | 3 | 0.2912 | 0.0689 | PASS |
## 5. Gaze Statistics (Stage13)

Per-clip `gaze_features.npz`: `gaze_vector [T,6]` = left-gaze `[T,3]` + right-gaze `[T,3]`. Each 3-vector is a unit direction (norm ≈ 1).

| Check | Requirement | Result |
|---|---|---|
| shape | [T,6] | PASS (11/11) |
| No NaN/Inf | all finite | PASS (11/11) |
| norm ≈ 1 (left) | \||left\|| ≈ 1 | max=1.000000 PASS |
| norm ≈ 1 (right) | \||right\|| ≈ 1 | max=1.000000 PASS |

Per-clip gaze checks (max |norm−1| over all frames):

| clip | T | shape | left_norm_mean | left_max_dev | right_norm_mean | right_max_dev |
|---|---|---|---|---|---|---|
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0000` | 188 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0002` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0003` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0004` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0001` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0002` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0003` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0004` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0005` | 200 | 6 | 1.00000 | 5.96e-08 | 1.00000 | 5.96e-08 |

## 6. Audio Statistics (Stage14)

Per-clip `audio_features.npz`: `feature [T,768]` = Chinese-HuBERT hidden-state embedding (768-dim), no NaN/Inf.

| Check | Requirement | Result |
|---|---|---|
| shape | [T,768] | PASS (11/11) |
| No NaN/Inf | all finite | PASS (11/11) |

Aggregate feature stats across all clips: per-clip mean: avg=-0.0012 (global min -4.7930 .. global max 3.3580), avg std=0.2475

Per-clip audio checks:

| clip | T | shape | finite |
|---|---|---|---|
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0000` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0002` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0003` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0004` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0001` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0002` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0003` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0004` | 200 | 768 | PASS |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0005` | 200 | 768 | PASS |

## 7. Export Structure (Stage17 export)

Final delivery directory `outputs/stage17_multimodal_export/` (generated by the official `export_multimodal_sample` per clip):

```
outputs/stage17_multimodal_export/
  flame/{clip_id}.npy      [T,112] float32  visual = expr[100]+head[3]+jaw[3]+gaze[6]
  audio/{clip_id}.npy      [T,768] float32  Chinese-HuBERT embedding
  timestamp/{clip_id}.npz  frame_index / timestamps / audio_match_idx / dropout_mask
  metadata/{clip_id}.json  clip_id / fps / visual_layout / head/jaw/expr/gaze/audio sources
  manifest.csv              dataset-level index (11 clips)
```

| Check | Requirement | Result |
|---|---|---|
| flame/*.npy | [T,112] | PASS (11/11) |
| audio/*.npy | [T,768] | PASS (11/11) |
| timestamp/*.npz | monotonic ts, [T] arrays | PASS (11/11) |
| metadata/*.json | required fields + layout | PASS (11/11) |
| clip ids consistent (stage12/13/14/17) | all == 11 accepted | PASS |
| clip ids consistent (export dirs) | flame/audio/ts/meta | PASS |
| manifest.csv consistent | 11 clip ids | PASS |

Per-clip export checks:

| clip | flame_shape | audio_shape | ts_T | ts_monotonic | dropout | audio_matched | metadata_sources |
|---|---|---|---|---|---|---|---|
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0000` | [191, 112] | [191, 768] | 191 | PASS | 3 | 191 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0002` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0003` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0004` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0001` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0002` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0003` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0004` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
| `CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0005` | [200, 112] | [200, 768] | 200 | PASS | 0 | 200 | `emica/spectre/emica/mediapipe/chinese_hubert` |
## 8. Final PASS/FAIL

| Check | Clips OK |
|---|---|
| Stage12 FLAME | 11/11 |
| Stage13 gaze | 11/11 |
| Stage14 audio | 11/11 |
| Stage17 fusion | 11/11 |
| Stage17 export | 11/11 |
| clip-id consistency (stage12/13/14/17) | PASS |
| clip-id consistency (export dirs) | PASS |
| manifest.csv consistency | PASS |

**Overall: `PASS`** — all multimodal outputs (`flame/*.npy`, `audio/*.npy`, `timestamp/*.npz`) are produced for all 11 accepted clips with consistent clip ids and are ready for downstream FLAME mesh reconstruction validation.

# Stage5 / Stage10 Temporal Stability Audit Report

- **Generated**: 2026-08-31 16:28:38
- **Clip**: `CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001`
- **Video**: `CNFACE_00001_BV18Qg96YE1W_000000`  |  fps `25`
- **Source clip**: `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_17_e2e/stage0_11/clips/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4`
- **Frames**: bbox `200`, landmarks `200` (transitions: 199 bbox / 199 landmark)
- **Crops**: `200` files in `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_17_e2e/stage0_11/faces/CNFACE_00001_BV18Qg96YE1W_000000/CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001/crops`
- **Run dir**: `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/stage0_17_e2e/stage0_11`

## 1. Purpose

Visualise and quantify the temporal stability of two stages that feed the FLAME mesh pipeline and determine which one is more likely responsible for the observed mesh jitter:

- **Stage 5** — SCRFD face detection (`scrfd_10g_bnkps.onnx`): per-frame tight bbox + expanded crop bbox.
- **Stage 10** — InsightFace `2d106det` landmarks mapped to frame coordinates via the Stage 5 `expanded_bbox`.

> Note: the pipeline does **not** persist FaceSequence / LandmarkInfo. This audit therefore regenerates them read-only by re-running the same model wrappers on the clip video and the existing Stage 5 crops. No model or pipeline code was modified.

## 2. Methodology

### Metric definitions (all on consecutive-frame transitions, t-1 -> t)

| Metric | Definition |
| --- | --- |
| `bbox_jitter_mean` | Mean over transitions of the SCRFD **bbox centre displacement** `sqrt(dx^2 + dy^2)` (px) |
| `bbox_jitter_std` | Population std of the same bbox centre displacement |
| `landmark_jitter_mean` | Mean over transitions of the **mean per-point landmark displacement** (mean of the 106 point Euclidean distances, px) |
| `landmark_jitter_std` | Population std of the same |
| `outlier_frames` | Frames `t` where a transition `(t-1 -> t)` exceeded `mean + sigma*std` (sigma = 3.0) in bbox centre displacement or landmark mean point distance |

Outlier threshold: `mean + 3.0 * std` (min abs: bbox 0.0 px, landmark 0.0 px).

## 3. Summary

| Metric | Value (px) | Relative to face diagonal |
| --- | --- | --- |
| Mean face diagonal | 346.7 | 100% |
| **bbox_jitter_mean** (bbox centre displacement) | 2.265 | 0.653% |
| **bbox_jitter_std** | 2.239 | - |
| **landmark_jitter_mean** (mean point distance) | 2.032 | 0.586% |
| **landmark_jitter_std** | 0.886 | - |
| bbox max centre displacement | 14.031 | - |
| landmark max mean-point distance | 2.032 (mean) / 15.367 (max) | - |

## 4. Bbox detail (Stage 5)

| Metric | Mean | Std |
| --- | --- | --- |
| centre displacement (px) | 2.265 | 2.239 |
| dx (px) | -0.063 | 1.530 |
| dy (px) | 0.073 | 2.792 |
| dw (px) | -0.059 | 4.388 |
| dh (px) | 0.041 | 4.173 |

## 5. Landmark detail (Stage 10)

| Metric | Value |
| --- | --- |
| mean point distance (px) | 2.032 ± 0.886 |
| max point distance (px) | 4.451 (mean of max) / 15.367 (worst) |

Per-region mean landmark displacement (px):

| Region | Mean displacement (px) |
| --- | --- |
| eyes | 2.083 |
| nose_tip | 1.810 |
| mouth | 2.369 |
| chin | 2.538 |

## 6. Outlier frames

9 outlier frame(s) flagged (threshold `mean + 3.0*std`):

| frame | bbox centre dist (px) | landmark mean pt dist (px) | flags |
| --- | --- | --- | --- |
| 7 | 9.462 | 0.000 | bbox |
| 9 | 14.031 | 0.000 | bbox |
| 23 | 9.763 | 0.000 | bbox |
| 28 | 9.629 | 0.000 | bbox |
| 40 | 10.951 | 0.000 | bbox |
| 42 | 11.363 | 0.000 | bbox |
| 68 | 0.000 | 4.727 | landmark |
| 129 | 10.019 | 0.000 | bbox |
| 168 | 0.000 | 5.621 | landmark |

## 7. Worst transitions

Top-8 bbox centre displacements:

| t0 | t1 | dx | dy | centre dist | dw | dh |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 9 | 5.989 | -12.688 | **14.031** | 20.691 | 1.099 |
| 41 | 42 | 7.506 | -8.531 | **11.363** | 10.780 | 9.172 |
| 39 | 40 | -4.596 | 9.940 | **10.951** | -14.338 | -0.634 |
| 128 | 129 | 4.556 | -8.924 | **10.019** | 13.361 | 4.011 |
| 22 | 23 | -2.668 | 9.391 | **9.763** | -14.709 | -16.018 |
| 27 | 28 | 5.893 | -7.615 | **9.629** | 13.604 | 10.434 |
| 6 | 7 | -5.750 | 7.514 | **9.462** | -14.757 | -5.293 |
| 34 | 35 | 6.188 | -6.365 | **8.878** | 14.030 | 9.229 |

Top-8 landmark mean point distances:

| t0 | t1 | mean pt dist | max pt dist | max pt idx | eyes | nose | mouth | chin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 167 | 168 | **5.621** | 6.892 | 104 | 4.956 | 6.375 | 6.078 | 5.632 |
| 67 | 68 | **4.727** | 15.133 | 94 | 8.581 | 4.508 | 4.595 | 3.049 |
| 22 | 23 | **4.429** | 13.803 | 40 | 7.604 | 3.106 | 4.011 | 4.295 |
| 165 | 166 | **4.223** | 5.876 | 51 | 3.838 | 5.225 | 4.393 | 2.603 |
| 127 | 128 | **4.165** | 10.273 | 53 | 3.136 | 2.485 | 4.991 | 7.569 |
| 30 | 31 | **4.143** | 13.267 | 40 | 6.325 | 2.982 | 4.537 | 4.508 |
| 171 | 172 | **3.961** | 4.728 | 60 | 3.367 | 4.471 | 4.464 | 4.286 |
| 128 | 129 | **3.953** | 12.217 | 53 | 2.415 | 3.221 | 5.126 | 7.778 |

## 8. Which stage is more likely causing the FLAME mesh jitter?

- **Absolute**: bbox centre jitter `2.265 px` vs landmark mean-point jitter `2.032 px`.
- **Face-size normalised** (mean face diagonal = 346.7 px): bbox `0.653%` vs landmarks `0.586%` of face diagonal per frame.
**Assessment: both stages contribute comparably.** Bbox and landmark jitter are of the same order of magnitude relative to face size. Neither can be singled out from these aggregate figures alone; see the outlier-frame table and overlay videos for the worst frames.
**Caveat**: landmark coordinates already absorb bbox motion (they are mapped through `expanded_bbox`), so part of the landmark displacement is a *consequence* of Stage 5 box jitter. The more reliable discriminator is the *residual* landmark jitter after removing the common translation of the bbox (see `bbox_center_distance` vs `landmark_mean_point_distance` in the per-transition tables).

## 9. Artifacts

- Stage5 overlay video: `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_stage10_audit/overlay_stage5_CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4`
- Stage10 overlay video: `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/docs/stage5_stage10_audit/overlay_stage10_CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4`
- Data: `stage5_frames.json`, `stage10_landmarks.json`, `temporal_stability.json`
- Tool: `scripts/stage5_stage10_audit/` (see its `README.md`)

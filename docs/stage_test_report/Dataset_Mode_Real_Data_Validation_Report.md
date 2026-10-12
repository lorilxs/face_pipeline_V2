# Dataset Pipeline Mode — Real-Data Validation Report

- **Video:** `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4`
- **Config:** `configs/config_dataset_mode.yaml` (`pipeline_mode: dataset`)
- **Validation date:** 2026-08-31
- **Overall verdict:** ✅ **PASS** — `clips_total=11`, `clips_all_stages_ok=11`, `clips_layout_match=11`

This report documents an end-to-end run of the pipeline in **dataset mode** on a real
talking-face video (a Chinese news/interview clip, 1920×1080 @ 25 fps, ~60 s), covering
Stage 0 → Stage 17 including the dataset-marker feature stages (FLAME / gaze / audio
embedding / fusion). All checks are machine-generated and captured in
`docs/dataset_validation_checks.json`.

---

## 1. Environment

| Component | Version / Value |
|---|---|
| Host OS | Linux (Python 3.10.12) |
| ffmpeg | 7.0.2 — **ffprobe disabled in this build** → Stage 0 uses ffmpeg-based probe fallback (`probe_with_ffmpeg`) |
| Host numpy / opencv | 2.1.3 / 5.0.0 |
| Host mediapipe | 0.10.8 (Stage 13, `--delegate cpu`) |
| Host transformers / soundfile | 4.22.2 / 0.14.0 (Stage 14, `chinese-hubert-base`) |
| Host torch | 2.4.1+cpu |
| Docker image `face_pipeline_lxs` | Ubuntu 24.04.2 LTS, emica_env Python 3.10.12, torch 2.4.1+cpu (CUDA **unavailable** → Stage 12 on CPU), numpy 1.23.5, opencv 4.10.0, omegaconf 2.0.6, pytorch_lightning 2.4.0 |
| Project mount | `/workspace/Face_Pipeline_V2` inside container (paths shared with host) |

**Model assets** (all present): SCRFD `models/SCRFD/scrfd_10g_bnkps.onnx`,
insightface 2d106 landmark `models/insightface/2d106det.onnx`,
MediaPipe `models/mediapipe/face_landmarker.task` (3.6 MB), HuBERT
`models/hubert/chinese-hubert-base`, EMICA checkpoint
`models/EMICA/EMICA-CVT_flame2020_notexture/checkpoints/model-epoch=38-val/loss_total=3.88806534.ckpt`,
SPECTRE `third_party/inferno/spectre/spectre/data/deca_model.tar`.

---

## 2. Input video (Stage 0)

| Property | Value |
|---|---|
| id | `CNFACE_00001_BV18Qg96YE1W_000000` |
| resolution | 1920 × 1080 |
| fps | 25.00 |
| duration | 60.2 s |
| total frames | 1504 |
| audio | present |

Stage 0 ffprobe-integrity check passed via the ffmpeg probe fallback (this build has
ffprobe disabled); `video_info` recorded in `outputs/stage0_17_e2e/stage0_11_results.json`.

---

## 3. Stage execution table

**Stage 0 – 11** (run by the e2e driver, docker, dataset mode):

| Stage | Status | Result summary |
|---|---|---|
| Stage 0 – FFprobe integrity | ✅ PASS | 1920×1080 \| 25.00 fps \| 60.2 s \| 1504 frames \| audio=True (ffprobe fallback) |
| Stage 1 – FFmpeg standardization | ✅ PASS | std mp4 + 48 kHz audio wav exported |
| Stage 2 – PySceneDetect | ✅ PASS | 3 scenes detected |
| Stage 3 – SCRFD coarse screening | ✅ PASS | 3 scenes: 2 passed, 1 rejected |
| Stage 4 – Sliding-window clips | ✅ PASS | 11 clips (window 8 s, stride 4 s) |
| Stage 5 – SCRFD detect/crop | ✅ PASS | 11 sequences, 2192 face detections (crops in `faces/`, 3292 jpg files) |
| Stage 5.1 – ByteTrack tracking | ✅ PASS | 11 sequences, single-face filter 11/11 |
| Stage 5.2 / 5.3 – Filters | ✅ PASS | 11 in → 11 passed, 0 rejected |
| Stage 6 – Head pose (6DRepNet) | ⏭️ SKIPPED | dataset mode (not a required feature) |
| Stage 6.5 / 7 / 8 / 9 | ⏭️ SKIPPED | dataset mode |
| Stage 9.5 – Clip export | ✅ PASS | `pipeline_mode=dataset`: **11 accepted clips**, 11 AV alignments (fallback to original clip wav, `already_aligned`, `offset_frames=0`) |
| Stage 10 – Landmark detection | ✅ PASS | 2192 landmarks |
| Stage 11 – Aligned faces | ✅ PASS | 2192 aligned faces, 256×256 (`aligned_faces/`, 2192 jpg files) |

**Stage 12 / 13 / 14 / 17** (run per-clip via `scripts/run_dataset_validation_121314.sh`
and `scripts/run_dataset_validation_17.sh`; Stage 12 in docker, Stages 13/14 on host,
Stage 17 on host):

| Stage | Status | Output per clip | Rows (T) |
|---|---|---|---|
| Stage 12 – FLAME (SPECTRE) | ✅ 11/11 rc=0 | `stage12_flame/<clip>/flame_features.npz` (`expression_code [T,100]`, `head_pose [T,3]`, `jaw_pose [T,3]`) | 192–200 |
| Stage 13 – Gaze (MediaPipe) | ✅ 11/11 rc=0 | `stage13_gaze/<clip>/gaze_features.npz` (`gaze_vector [T,6]`, `timestamp`) | 190–200 |
| Stage 14 – Audio (HuBERT) | ✅ 11/11 rc=0 | `stage14_audio/<clip>/audio_features.npz` (`feature [T,768]`, `dimension`, `source_audio`) | 200 (all) |
| Stage 17 – Fusion | ✅ 11/11 rc=0 | `stage17_fusion/<clip>/training_sample.npz` (`visual_feature [T,112]`, `audio_feature [T,768]`, `timestamps`, `valid_length`, `audio_match_idx`, `dropout_mask`) + `manifest.csv` | 192–200 |

Total feature-marker wall time for the 11-clip Stage 12/13/14 batch: **~10.2 min**
(first Stage 12 at 10:13:00 → batch done 10:23:12); Stage 12 dominates (~0.17 s/frame
on CPU since CUDA is unavailable). Stage 17 fusion for all 11 clips: ~5 s.

---

## 4. Feature statistics (aggregated across 11 clips)

| Feature | Metric | Value (mean / min / max) |
|---|---|---|
| FLAME expression code [100] | std | 0.457 / 0.422 / 0.496 |
| FLAME head pose [3] (rad) | max \|θ\| | 0.255 / 0.212 / 0.333 |
| FLAME jaw pose [3] (rad) | max \|θ\| | 0.072 / 0.064 / 0.083 |
| Gaze vector [6] | row norm | 1.4142 (=√2) / all rows — two unit eye-direction vectors ✓ |
| Gaze rows | T | 199.1 / 190 / 200 (clip_0000: 190, 10 clips: 200) |
| Audio feature [768] | \|x\| mean | 0.137 / 0.136 / 0.138 |
| Audio feature [768] | std | 0.247 / 0.245 / 0.250 |
| Audio rows | T | 200.0 / 200 / 200 (all clips exactly 200 @ 25 fps × 8 s) |
| Fusion visual T | T | 199.3 / 192 / 200 |
| Fusion valid_length | T | 199.1 / 190 / 200 |
| Fusion audio_match_idx ≥ 0 | fraction | 1.0000 (all 11 clips, every visual frame matched) |
| Fusion dropout rows | count | clip_0000: 2 (the 2 gaze-less frames); all others 0 |

All head/jaw pose magnitudes stay well within ±1 rad; gaze rows are exactly unit-length
pairs; audio rows align 1:1 with the 25 fps time grid.

---

## 5. Cross-check: Stage 17 `visual_feature` layout vs Stage 12 / 13

For every clip the fused `visual_feature [T,112]` was compared against the raw
`expression_code [T,100]`, `head_pose [T,3]`, `jaw_pose [T,3]` (Stage 12) and
`gaze_vector [T,6]` (Stage 13). Expected layout: `expr(0:100) + head(100:103) +
jaw(103:106) + gaze(106:112)`.

| Check | Method | Result |
|---|---|---|
| expr / head / jaw columns | element-wise max-abs diff | **maxdiff = 0.0** on 11/11 clips |
| gaze columns | nearest-timestamp match (≤0.06 s), **excluding Stage 17 dropout rows** | **maxdiff = 0.0** on 11/11 clips |
| overall `layout_match` | all of the above | **11 / 11 ✅** |

> Why nearest-timestamp matching for gaze: Stage 13 dropped 2 frames at the start of
> clip_0000 (ts = 0.36 / 0.40 s, early landmarker failures), so Stage 13 and Stage 17
> row indices drift. Stage 17 marks those 2 rows `dropout=True`
> (`valid_length = 190 < T = 192`) and fills their gaze columns with the nearest
> available gaze; the rows are excluded from the training sample. The cross-check
> therefore compares gaze on timestamp (not index) and skips dropout rows. All 2190
> usable gaze rows match exactly.

---

## 6. Dataset-mode specific validations

1. **Accepted-clip count is consistent** across stages: Stage 4 created 11 clips →
   Stage 5/9.5 accepted 11 → Stage 10 produced 2192 landmarks → Stage 11 wrote 2192
   aligned faces → Stages 12/13/14/17 all produced exactly 11 outputs.
2. **AV alignment / audio fallback is correct**: in dataset mode Stage 9.5 reports
   `alignment_status=already_aligned`, `offset_frames=0`, and Stage 14 consumed the
   dataset fallback wav `stage_outputs/av_alignment/<clip>/<clip>.wav` (recorded in
   `source_audio`), not a syncnet-shifted path.
3. **Training sample is coherent**: 11/11 clips have `audio_match_idx ≥ 0` for every
   visual frame (100%), so the fused audio embedding is fully paired with the visual
   grid; `manifest.csv` lists all 11 clips.
4. **Filters behave as expected in dataset mode**: Stage 6–9 (pose/quality/sync
   screening) are intentionally skipped; detection/coarse-screening/filters Stage 0–5.3
   all ran and 11/11 clips passed.

---

## 7. PASS / FAIL summary (from `docs/dataset_validation_checks.json`)

```json
{
  "clips_total": 11,
  "clips_all_stages_ok": 11,
  "clips_layout_match": 11,
  "overall": "PASS"
}
```

✅ **PASS** — the dataset-mode pipeline runs end-to-end on a real talking-face video and
produces a valid, internally-consistent training dataset (FLAME + gaze + HuBERT audio +
fused samples) for all 11 accepted clips.

---

## 8. Known limitations & notes

1. **ffprobe disabled in this ffmpeg build** — Stage 0 falls back to ffmpeg-based
   probing (`probe_with_ffmpeg`). Detection was verified; on a build with ffprobe the
   primary path is used. This is an environment quirk, not a pipeline defect.
2. **CUDA unavailable** — Stage 12 (SPECTRE/FLAME) runs on CPU at ~0.17 s/frame;
   whole 11-clip Stage 12/13/14 batch ~10.2 min. Correctness unaffected.
3. **Stage 13 gaze frame drops** — 2 early frames of clip_0000 produced no gaze;
   Stage 17 flags them as dropout (`valid_length=190`) and excludes them from training.
   Handled and verified (see §5).
4. **Gaze cross-check is timestamp-aligned, not index-aligned** — necessary because of
   the above frame drop; documented in the validation script and §5.
5. **Stage 14 `source_audio`** is stored as a 1-element ndarray in the npz — handled by
   the validation script (`np.asarray(...)[0]`); confirms the dataset-fallback wav was
   used.

---

## 9. Reproducibility

- Stage 0–11 e2e: docker, `scripts/stage0_17_e2e_stage12.py` driver with
  `configs/config_dataset_mode.yaml` → results in `outputs/stage0_17_e2e/stage0_11_results.json`.
- Stage 12/13/14: `scripts/run_dataset_validation_121314.sh`
  (Stage 12 in docker via `/workspace/emica_env/bin/python`; Stage 13 host MediaPipe
  `--delegate cpu`; Stage 14 host HuBERT `--fps 25`) — 33/33 sub-steps `rc=0`,
  log `logs/dataset_validation_121314.log`.
- Stage 17: `scripts/run_dataset_validation_17.sh` — 11/11 `rc=0`.
- Validation: `python3 scripts/validate_dataset_mode_real_run.py` → overwrites
  `docs/dataset_validation_checks.json` and prints the summary in §7.

**Generated artifacts:** `outputs/stage0_17_e2e/dataset_validation/` —
`stage_inputs/` (per-clip metadata: scenes, clips, AV alignment, landmarks, faces),
`videos/`, `scenes/`, `faces/` (3292 crops), `aligned_faces/` (2192 × 256×256),
`stage_outputs/av_alignment/` (11 clip wavs), `stage12_flame/`, `stage13_gaze/`,
`stage14_audio/`, `stage17_fusion/` (11 `training_sample.npz` + `manifest.csv`).

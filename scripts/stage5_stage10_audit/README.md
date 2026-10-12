# Stage5 / Stage10 Temporal Stability Audit

Read-only audit tool that visualises the Stage5 (SCRFD bbox) and Stage10
(InsightFace 106-landmark) outputs of the face pipeline, quantifies temporal
stability (jitter) on consecutive-frame transitions, and produces an
evidence-based report on which stage is more likely causing the observed
FLAME mesh temporal jitter.

The pipeline does **not** persist `FaceSequence` / `LandmarkInfo` to disk, so
this tool regenerates them read-only:

- **Stage5**: re-runs `SCRFDWrapper.detect()` on the clip video
  (same model / pre- & post-processing the pipeline uses).
- **Stage10**: runs `InsightFaceLandmarkWrapper.predict()` on the Stage5 crops
  already on disk (`faces/{video_id}/{clip_id}/crops/frame_XXXXXX_face_000.jpg`)
  and maps crop-space landmarks back to frame coordinates. Because the
  original Stage 5 expanded bbox is not persisted (and re-running SCRFD only
  reproduces it to ~5 px), the **exact** crop origin is recovered by
  normalised-cross-correlation template matching of each crop against its
  source video frame, mirroring `stage_10_landmark/processor.py` to within
  <1 px. If matching fails, the reproduced `expanded_bbox` origin is used as
  a fallback.

No model or pipeline code is modified.

## Layout

| File | Purpose |
| --- | --- |
| `run_audit.py` | CLI orchestrator (extract -> analyze -> render -> report) |
| `extractor.py` | Read-only Stage5/Stage10 data export |
| `data_io.py` | Normalized JSON loaders (canonical + flat variants) |
| `analysis.py` | Bbox / landmark jitter metrics + outlier detection |
| `visualize.py` | Shared OpenCV drawing helpers |
| `visualize_stage5.py` | Stage5 overlay video (bbox + centre/size delta plots) |
| `visualize_stage10.py` | Stage10 overlay video (106 pts + indices + keypoints) |
| `report.py` | Markdown report generator |

## Usage

From the `Face_Pipeline_V2` project root:

```bash
PYTHONPATH=. python -m scripts.stage5_stage10_audit.run_audit \
    --clip-id CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001
```

Fast smoke test (no videos, limited frames):

```bash
PYTHONPATH=. python -m scripts.stage5_stage10_audit.run_audit \
    --max-frames 40 --skip-videos
```

Re-run analysis/report on already-extracted data:

```bash
PYTHONPATH=. python -m scripts.stage5_stage10_audit.run_audit \
    --skip-extract --skip-videos
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--clip-id` | `..._scene_0000_clip_0001` | Target clip |
| `--run-dir` | `outputs/stage0_17_e2e/stage0_11` | Run dir with manifest/clips/faces |
| `--out-dir` | `docs/stage5_stage10_audit` | Data JSONs + overlay videos |
| `--report` | `docs/Stage5_Stage10_Temporal_Stability_Report.md` | Report path |
| `--scrfd-model-dir` | `models/SCRFD` | SCRFD ONNX dir |
| `--landmark-model` | `models/insightface/2d106det.onnx` | 2d106det ONNX |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--expand-ratio` | `2.0` | Must match `config.yaml` Stage5 value |
| `--max-frames` | `None` | Limit frames (smoke test) |
| `--skip-extract` | off | Reuse existing JSONs |
| `--skip-videos` | off | Skip video rendering |
| `--sigma` | `3.0` | Outlier threshold (std devs) |

## Metrics

All jitter metrics are computed on consecutive-frame transitions `(t-1 -> t)`:

- **bbox centre displacement** = `sqrt(dx^2 + dy^2)` of the SCRFD box centre;
  also `dx`, `dy`, `dw`, `dh`.
- **landmark mean point distance** = mean over the 106 points of the
  Euclidean distance between frames; also the max point distance + which
  point, and per-region means (eyes / nose tip / mouth / chin).
- Top-level: `bbox_jitter_mean/std`, `landmark_jitter_mean/std`,
  `outlier_frames` (transitions above `mean + sigma*std`).

## Keypoint semantics (2d106det, verified empirically)

- 0-32 face contour (chin = lowest-y contour point)
- 33-42 left eye, 87-96 right eye (eye corners = min/max-x eye points)
- 43-51 left brow, 97-105 right brow
- 52-71 mouth (corners = min/max-x mouth points)
- 72-86 nose (83-85 nose tip)

## Coordinate note

Stage10 landmarks are in **full-frame coordinates** after the `expanded_bbox`
offset mapping; the overlays draw them directly on the clip video frames
(1:1 with the frame coordinate space). Crop filenames carry *global* frame
numbers (`frame_000100`...), so the tool derives the global->local offset per
clip from the crop filenames.

#!/usr/bin/env python3
"""Stage10 106-landmark Temporal EMA (α=0.7) End-to-End Motion Preservation Ablation.

Experiment (read-only w.r.t. production — no stage_05 / stage_10 / stage_12 code and
no model weights are modified; only experiment scripts + outputs are added):

    Baseline : Stage5 SCRFD raw bbox → Stage10 106-landmark (raw) → Stage11 affine align
               → Stage12 FLAME → Stage13 gaze → Stage14 audio → Stage17 fusion
    EMA α=0.7: the same chain, but the Stage10 106 landmarks are temporally smoothed
               with a causal EMA over the [T, 106, 2] coordinates:
                   landmark_ema[t] = 0.7·landmark_raw[t] + 0.3·landmark_ema[t-1]
               frame 0 = raw landmark.  The SCRFD bbox is NOT touched.

Both branches use the IDENTICAL raw SCRFD bbox (reused from the prior Stage5 ROI
ablation — docs/stage5_roi_ablation/baseline_results.json, byte-identical numbers)
and the IDENTICAL Stage5 crop; they differ ONLY in the Stage10 landmark set that
Stage11 uses for the affine alignment.  Stage12/13/14/17 run through the official
production runners unchanged, which guarantees the baseline and EMA versions are
fully comparable.

Metrics produced (analyze):
  §1 landmark frame-delta jitter (mean/std/max/P95) over all 106 points + the 5
     task regions (contour / eyebrow / eye / nose / mouth) with jitter reduction;
  §2 motion preservation: motion-energy retention, mouth-opening curve (max/mean/
     std + Pearson correlation), jaw amplitude retention (Stage12 jaw_pose);
  §3 FLAME stability: expression frame-delta and head/jaw rotvec→degree rotation
     distance (mean/std/max), baseline vs EMA;
  §4 Stage17 layout verification ([T,112] visual_feature = expr|head|jaw|gaze);
  §5 overlay videos ×2 (raw-landmark vs EMA-landmark on the original clip);
  §6 report docs/Stage10_Landmark_Temporal_EMA_Ablation_Report.md.

Hard constraints honoured (task spec §8): no production Stage10/12 modification,
no EMICA/FLAME/Stage17-layout change, no Butterworth / filtfilt, no future frames.

Usage (from the Face_Pipeline_V2 project root):

    # 1) generate Stage5/10/11 for both branches (crop shared, landmark EMA branch):
    PYTHONPATH=. python3 scripts/stage10_landmark_temporal_ema_ablation.py prep \\
        --clip CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001

    # 2) (run_single_video_dataset_landmark_ema_ablation.sh runs Stage12/13/14/17)

    # 3) comparison metrics + overlay videos + report:
    PYTHONPATH=. python3 scripts/stage10_landmark_temporal_ema_ablation.py analyze
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the pure helpers from the prior Stage5 bbox-EMA ablation (identical
# extraction/metric/video code paths -> the raw-landmark numbers chain to the
# earlier experiments).  Importing the module has no side effects (guarded main).
import stage5_ema_e2e_ablation as s5  # noqa: E402
from face_pipeline_v2.data.models import (  # noqa: E402
    AlignedFaceInfo, FaceInfo, FaceSequence, FrameInfo, LandmarkInfo,
)
from face_pipeline_v2.models.wrappers.insightface_landmark import (  # noqa: E402
    InsightFaceLandmarkWrapper,
)
from face_pipeline_v2.stages.stage_11_alignment.processor import (  # noqa: E402
    align_faces, resolve_alignment_template,
)

# ---------------------------------------------------------------------------
# Experiment configuration (production values + ablation parameters)
# ---------------------------------------------------------------------------

DEFAULT_CLIP = "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"
# Production Stage0-11 run — source of clip video / av-alignment / aligned meta.
SRC_BASE = _PROJECT_ROOT / "outputs/stage0_17_e2e/stage0_11"
# Raw SCRFD bbox reused from the prior Stage5 ROI ablation (identical numbers).
BASELINE_SOURCE = _PROJECT_ROOT / "docs/stage5_roi_ablation/baseline_results.json"
# Experiment output root (self-contained; production untouched).
OUT_BASE = _PROJECT_ROOT / "outputs/ablation_landmark_ema"
COMPARISON_DIR = OUT_BASE / "comparison"
# Overlay videos land in the task-specified doc dir.
VIDEO_DIR = _PROJECT_ROOT / "docs/stage10_landmark_temporal_ema_ablation" / "videos"
REPORT_PATH = _PROJECT_ROOT / "docs/Stage10_Landmark_Temporal_EMA_Ablation_Report.md"

LANDMARK_MODEL = _PROJECT_ROOT / "models/insightface/2d106det.onnx"
PRODUCTION_EXPAND = 2.0   # stage_05 face_crop expand_ratio (Stage10/11 crop)
ALPHA = 0.7               # landmark EMA current-frame weight (this experiment)
FPS = 25.0
ALIGN_RES = 256           # Stage11 aligned face resolution (config output_size)
FFMPEG = Path("/usr/local/bin/ffmpeg")

# visual_feature layout for Stage17 [T, 112]  (task spec §3 / §6)
LAYOUT = {"expression": (0, 100), "head_pose": (100, 103), "jaw_pose": (103, 106),
          "gaze": (106, 112)}

BRANCHES = ("baseline", "ema_alpha07")

# InsightFace 2d106det region indices — verified in production Stage11
# (face_pipeline_v2/stages/stage_11_alignment/affine_transform.py):
#   contour 0-32 | left eye 33-42 | left brow 43-51 | mouth 52-71 |
#   nose 72-86   | right eye 87-96 | right brow 97-105
FACE_CONTOUR = list(range(0, 33))
LEFT_EYE = list(range(33, 43))
LEFT_BROW = list(range(43, 52))
MOUTH = list(range(52, 72))
NOSE = list(range(72, 87))
RIGHT_EYE = list(range(87, 97))
RIGHT_BROW = list(range(97, 106))
# Task regions (§2.1): contour / eyebrow / eye / nose / mouth (+ all 106)
REGIONS: Dict[str, List[int]] = {
    "all": list(range(106)),
    "contour": FACE_CONTOUR,
    "eyebrow": LEFT_BROW + RIGHT_BROW,
    "eye": LEFT_EYE + RIGHT_EYE,
    "nose": NOSE,
    "mouth": MOUTH,
}

# BGR colours for overlay videos.
GREEN = (60, 200, 80)
ORANGE = (0, 140, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_log = logging.getLogger("stage10_landmark_ema")


# ---------------------------------------------------------------------------
# EMA landmark smoothing (causal, no future frames; α=0.7 current-frame weight)
# ---------------------------------------------------------------------------


def landmark_ema_smooth(
    seq: List[Optional[np.ndarray]],
    alpha: float = ALPHA,
) -> List[Optional[np.ndarray]]:
    """Recursive EMA over the [106, 2] landmark coordinates.

    landmark_ema[t] = α·landmark_raw[t] + (1-α)·landmark_ema[t-1]
    Frame 0 (and any frame with no previous smoothed value) keeps the raw
    landmark.  A missing detection (None) is kept None in the EMA branch too
    (no synthesis) and simply breaks the recursion; smoothing restarts from the
    next valid frame.  Smoothing is per-coordinate (no cross-point mixing).
    """
    out: List[Optional[np.ndarray]] = []
    prev: Optional[np.ndarray] = None
    for lms in seq:
        if lms is None:
            out.append(None)
            prev = None
            continue
        if prev is None:
            out.append(np.asarray(lms, dtype=np.float64).copy())
        else:
            out.append(
                alpha * np.asarray(lms, dtype=np.float64)
                + (1.0 - alpha) * prev)
        prev = out[-1]
    return out

# ---------------------------------------------------------------------------
# IO helpers (re-exported from the Stage5 ablation for local clarity)
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON with numpy scalar/array coercion (metrics contain np values)."""
    def _default(o: Any):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default),
                    encoding="utf-8")
    print(f"[json] wrote {path}")


def _to_rel(p: Path) -> str:
    return s5._to_rel(p)


# ---------------------------------------------------------------------------
# Stage5/10/11 branch generation (bbox identical; landmark differs)
# ---------------------------------------------------------------------------


def prep_clip(
    clip_id: str,
    alpha: float = ALPHA,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate both branches' Stage5/10/11 artifacts for one clip.

    The Stage5 SCRFD bbox (raw, reused from the Stage5 ROI ablation) and the
    Stage5 crop are IDENTICAL for both branches.  The raw Stage10 106 landmarks
    are detected once; the ``ema_alpha07`` branch stores the causal EMA-smoothed
    landmarks.  Stage11 affine alignment is then re-run per branch (the eye-line
    / nose anchors change -> different warp -> different aligned faces -> the
    only downstream variable).
    """
    video_path = SRC_BASE / "clips" / f"{clip_id}.mp4"
    if not video_path.exists():
        raise RuntimeError(f"clip video not found: {video_path}")
    src_recs = s5._load_production_aligned_meta(clip_id)
    base = s5._load_baseline(BASELINE_SOURCE)
    if max_frames:
        base = base[:max_frames]
        src_recs = src_recs[:max_frames]
    boxes: List[Optional[Tuple[float, float, float, float]]] = [
        tuple(f["bbox"]) if f.get("bbox") else None for f in base]
    confs: List[float] = [float(f.get("confidence", 0.0)) for f in base]
    assert len(boxes) == len(src_recs), (len(boxes), len(src_recs))
    n = len(src_recs)

    # ---- shared Stage5 assets (identical for both branches) ----
    shared5 = OUT_BASE / "stage5"
    crop_dir = shared5 / "crops"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[prep] clip={clip_id} n={n} video={W}x{H}@{fps:.2f}", flush=True)

    stage5_frames: List[Dict[str, Any]] = []
    lm_records_base: List[Dict[str, Any]] = []   # raw landmark records (baseline)
    raw_seqs: List[Optional[np.ndarray]] = []    # raw (106,2) arrays
    t0 = time.time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            print(f"[prep] WARNING: only {i} frames available in clip", flush=True)
            n = i
            break
        src = src_recs[i]
        b = boxes[i]
        conf = confs[i]

        # ---- Stage5: expand bbox + save native-resolution crop (shared) ----
        from face_pipeline_v2.stages.stage_05_face_detection.face_cropper import (
            expand_bbox)
        exp = expand_bbox(list(b), W, H, PRODUCTION_EXPAND)
        x1, y1, x2, y2 = int(exp[0]), int(exp[1]), int(exp[2]), int(exp[3])
        crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None
        crop_path = crop_dir / f"frame_{i:06d}_face_000.jpg"
        if crop is not None and crop.size:
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop)

        # ---- Stage10: 2d106det on the crop -> full-frame coordinates ----
        lms = s5.extract_106_landmarks(frame, b, W, H)
        rec = {
            "frame_id": i,
            "frame_index": int(src["frame_index"]),
            "timestamp": float(src["timestamp"]),
            "clip_id": clip_id,
            "face_id": src["face_id"],
            "bbox": list(b) if b else None,
            "expanded_bbox": list(exp),
            "confidence": conf,
            "landmarks_106": ([float(v) for v in lms.flatten()]
                               if lms is not None else None),
        }
        lm_records_base.append(rec)
        raw_seqs.append(lms)

        stage5_frames.append({
            "frame_id": i,
            "frame_index": int(src["frame_index"]),
            "timestamp": float(src["timestamp"]),
            "bbox": list(b) if b else None,
            "expanded_bbox": list(exp),
            "confidence": conf,
        })
        if (i + 1) % 50 == 0:
            print(f"  [prep] frame {i + 1} elapsed {time.time() - t0:.0f}s", flush=True)
    cap.release()
    print(f"[prep] {n} frames processed in {time.time() - t0:.1f}s", flush=True)

    # ---- shared Stage5 bbox asset ----
    _write_json(shared5 / "bboxes.json", {
        "experiment": {
            "name": "stage10_landmark_temporal_ema",
            "clip": clip_id,
            "alpha": alpha,
            "fps": FPS,
            "expand_ratio": PRODUCTION_EXPAND,
            "smoothing_target": "stage10_106_landmark",
            "ema_formula": "landmark_ema[t] = 0.7*landmark_raw[t] + 0.3*landmark_ema[t-1]",
            "bbox_source": _to_rel(BASELINE_SOURCE),
            "video": _to_rel(video_path),
        },
        "frames": stage5_frames,
    })

    # ---- Stage10 landmark assets: baseline(raw) vs ema_alpha07(smoothed) ----
    ema_seqs = landmark_ema_smooth(raw_seqs, alpha=alpha)
    branch_lm: Dict[str, List[Optional[np.ndarray]]] = {
        "baseline": raw_seqs, "ema_alpha07": ema_seqs}

    results: Dict[str, Any] = {"clip_id": clip_id, "alpha": alpha,
                               "n_frames": n, "video": _to_rel(video_path)}
    for branch in BRANCHES:
        lm_records = []
        for i, rec in enumerate(lm_records_base):
            lms = branch_lm[branch][i]
            r = dict(rec)
            r["landmarks_106"] = ([float(v) for v in lms.flatten()]
                                   if lms is not None else None)
            lm_records.append(r)
        _write_json(OUT_BASE / branch / "stage10" / "landmarks.json", {
            "clip_id": clip_id, "branch": branch, "alpha": alpha,
            "ema_formula": ("raw" if branch == "baseline"
                             else "0.7*raw[t] + 0.3*ema[t-1]"),
            "frames": lm_records,
        })

        # ---- Stage11: production align_faces on this branch's landmarks ----
        r_branch = _align_branch(branch, clip_id, src_recs[:n], boxes, confs[:n],
                                 branch_lm[branch],
                                 [f["expanded_bbox"] for f in stage5_frames],
                                 W, H)
        results[branch] = {
            "n": n,
            "n_aligned": r_branch["n_aligned"],
            "n_landmarks": r_branch["n_landmarks"],
            "landmark_jitter_mean": r_branch["landmark_jitter_mean"],
            "aligned_metadata": _to_rel(r_branch["aligned_metadata"]),
        }
        print(f"[prep:{branch}] n_aligned={r_branch['n_aligned']} "
              f"n_landmarks={r_branch['n_landmarks']} "
              f"lm_jitter={r_branch['landmark_jitter_mean']:.3f}px", flush=True)

    _write_json(COMPARISON_DIR / "prep_summary.json", results)
    return results

# ---------------------------------------------------------------------------
# Stage11 alignment (per branch; identical to production code path)
# ---------------------------------------------------------------------------


def _align_branch(
    branch: str,
    clip_id: str,
    src_recs: List[Dict[str, Any]],
    boxes: List[Optional[Tuple[float, float, float, float]]],
    confs: List[float],
    landmarks: List[Optional[np.ndarray]],
    expanded_boxes: List[Optional[Tuple[float, float, float, float]]],
    W: int,
    H: int,
) -> Dict[str, Any]:
    """Run production ``align_faces`` for one branch and write its aligned meta."""
    branch_dir = OUT_BASE / branch
    stage5_dir = OUT_BASE / "stage5"
    stage11_dir = branch_dir / "stage11"
    aligned_dir = stage11_dir / "aligned_faces"
    n = len(src_recs)

    face_infos: List[FrameInfo] = []
    for i in range(n):
        src = src_recs[i]
        fi = FaceInfo(
            frame_index=int(src["frame_index"]),
            face_id=src["face_id"],
            bbox=tuple(boxes[i]) if boxes[i] is not None else (0.0, 0.0, 0.0, 0.0),
            landmark_5=(),
            confidence=confs[i],
            face_crop_path=stage5_dir / "crops" / f"frame_{i:06d}_face_000.jpg",
            track_id=None,
            expanded_bbox=(tuple(expanded_boxes[i])
                           if expanded_boxes[i] is not None else ()),
        )
        face_infos.append(FrameInfo(
            frame_index=int(src["frame_index"]),
            clip_id=clip_id,
            timestamp=float(src["timestamp"]),
            image_path=Path(""),
            faces=(fi,),
        ))
    sequence = FaceSequence(
        clip_id=clip_id,
        video_id=clip_id.split("_scene", 1)[0],
        frames=tuple(face_infos),
        identity_result=None,
        stability_result=None,
    )

    landmark_infos: List[LandmarkInfo] = []
    for i in range(n):
        src = src_recs[i]
        lms = landmarks[i]
        if lms is None:
            continue
        landmark_infos.append(LandmarkInfo(
            clip_id=clip_id,
            frame_index=int(src["frame_index"]),
            face_id=src["face_id"],
            timestamp=float(src["timestamp"]),
            landmarks_106=tuple(float(v) for v in lms.flatten()),
            image_width=W,
            image_height=H,
        ))

    import yaml  # noqa: PLC0415
    cfg = yaml.safe_load(
        (_PROJECT_ROOT / "configs/config_dataset_mode.yaml").read_text(encoding="utf-8"))
    template_params = resolve_alignment_template(cfg.get("alignment", {}))
    eyes_template = template_params["eyes"]

    aligned = align_faces(
        landmarks=landmark_infos,
        face_sequences=[sequence],
        output_dir=aligned_dir,
        output_resolution=(ALIGN_RES, ALIGN_RES),
        save_images=True,
        logger=_log,
        eye_template=eyes_template,
    )
    aligned_sorted = sorted(aligned, key=lambda a: int(a.frame_index))
    recs = [{
        "clip_id": a.clip_id,
        "frame_index": int(a.frame_index),
        "face_id": a.face_id,
        "timestamp": float(a.timestamp),
        "aligned_face_path": _to_rel(a.aligned_face_path),
        "affine_matrix": list(a.affine_matrix),
        "original_image_size": list(a.original_image_size),
        "aligned_resolution": list(a.aligned_resolution),
    } for a in aligned_sorted]
    meta = stage11_dir / f"{clip_id}_aligned_faces.json"
    _write_json(meta, recs)
    return {
        "n_aligned": len(recs),
        "n_landmarks": len(landmark_infos),
        "landmark_jitter_mean": (
            s5.landmark_jitter_metrics(landmarks)["landmark_jitter_mean"]),
        "aligned_metadata": meta,
    }


# ---------------------------------------------------------------------------
# Analysis: loaders (new experiment output paths)
# ---------------------------------------------------------------------------


def _clip() -> str:
    p = COMPARISON_DIR / "prep_summary.json"
    if p.exists():
        return str(json.loads(p.read_text(encoding="utf-8")).get("clip_id", DEFAULT_CLIP))
    return DEFAULT_CLIP


def _branch_landmarks(branch: str) -> List[Dict[str, Any]]:
    return json.loads(
        (OUT_BASE / branch / "stage10" / "landmarks.json").read_text(encoding="utf-8")
    )["frames"]


def _landmark_array(recs: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
    out = []
    for f in recs:
        v = f.get("landmarks_106")
        if v is None:
            out.append(None)
            continue
        if isinstance(v, str):
            v = [float(x) for x in v.strip("[]").split(",")]
        out.append(np.asarray(v, dtype=np.float64).reshape(106, 2))
    return out


def _branch_flame(branch: str, clip_id: str) -> Dict[str, np.ndarray]:
    z = np.load(OUT_BASE / branch / "stage12_flame" / clip_id / "flame_features.npz")
    return {"expression_code": z["expression_code"], "head_pose": z["head_pose"],
            "jaw_pose": z["jaw_pose"], "frame_index": z["frame_index"]}


def _branch_gaze(branch: str, clip_id: str) -> np.ndarray:
    return np.load(OUT_BASE / branch / "stage13_gaze" / clip_id
                   / "gaze_features.npz")["gaze_vector"]


# ---------------------------------------------------------------------------
# §1 landmark frame-delta jitter (all 106 + task regions)
# ---------------------------------------------------------------------------


def _transitions(seq: List[Optional[np.ndarray]]):
    """Yield (prev, cur) (106,2) pairs on consecutive valid frames."""
    prev = None
    for lms in seq:
        if lms is None:
            prev = None
            continue
        if prev is not None and prev.shape == lms.shape:
            yield prev, lms
        prev = lms


def region_jitter(seq: List[Optional[np.ndarray]], region: List[int]) -> Dict[str, float]:
    """Per-transition mean point distance within ``region`` -> mean/std/max/P95."""
    jumps = [float(np.mean(np.linalg.norm(cur[region] - prev[region], axis=1)))
             for prev, cur in _transitions(seq)]
    if not jumps:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "p95": 0.0, "count": 0}
    a = np.asarray(jumps, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "max": float(a.max()), "p95": float(np.percentile(a, 95)),
            "count": len(a)}


def jitter_comparison() -> Dict[str, Any]:
    """§1 jitter table for all 106 points + 5 regions, both branches + reduction."""
    lms = {b: _landmark_array(_branch_landmarks(b)) for b in BRANCHES}
    out: Dict[str, Any] = {}
    for region_name, region in REGIONS.items():
        row = {b: region_jitter(lms[b], region) for b in BRANCHES}
        base_mean = row["baseline"]["mean"]
        out[region_name] = {
            "baseline": row["baseline"],
            "ema_alpha07": row["ema_alpha07"],
            "jitter_reduction_pct": (
                (base_mean - row["ema_alpha07"]["mean"]) / base_mean * 100.0
                if base_mean else None),
        }
    return out


# ---------------------------------------------------------------------------
# §2 motion preservation: motion energy / mouth opening / jaw amplitude
# ---------------------------------------------------------------------------


def landmark_motion_energy(seq: List[Optional[np.ndarray]]) -> float:
    """mean(||lm[t]-lm[t-1]||) over all 106 points and valid transitions."""
    jumps = [float(np.mean(np.linalg.norm(cur - prev, axis=1)))
             for prev, cur in _transitions(seq)]
    return float(np.mean(jumps)) if jumps else 0.0


def mouth_open_curve(seq: List[Optional[np.ndarray]]) -> List[float]:
    """Per-frame vertical mouth opening = max_y - min_y over the mouth region.

    The 2d106det mouth points (52-71) do not follow a fixed upper/lower-lip
    ordering in this project's layout, so the opening is defined by the y-extent
    of the mouth landmark set (upper lip centre = min-y point, lower lip centre
    = max-y point).  Robust to point-order noise and gives a clean per-frame
    opening curve in pixels.
    """
    curve = []
    for lms in seq:
        if lms is None:
            curve.append(float("nan"))
            continue
        m = lms[MOUTH]
        curve.append(float(m[:, 1].max() - m[:, 1].min()))
    return curve


def _stats(vals: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "max": float(a.max()), "min": float(a.min())}


def mouth_comparison() -> Dict[str, Any]:
    """§2.2 mouth opening preservation: curve stats + Pearson correlation."""
    lms = {b: _landmark_array(_branch_landmarks(b)) for b in BRANCHES}
    curves = {b: mouth_open_curve(lms[b]) for b in BRANCHES}
    a = np.asarray(curves["baseline"], dtype=np.float64)
    c = np.asarray(curves["ema_alpha07"], dtype=np.float64)
    valid = ~(np.isnan(a) | np.isnan(c))
    corr = float(np.corrcoef(a[valid], c[valid])[0, 1]) if valid.sum() > 2 else None
    return {
        "baseline": _stats(curves["baseline"]),
        "ema_alpha07": _stats(curves["ema_alpha07"]),
        "pearson_correlation": corr,
        "mean_opening_retention": (
            np.nanmean(c) / np.nanmean(a) if np.nanmean(a) else None),
    }


def jaw_metrics(jaw: np.ndarray) -> Dict[str, Any]:
    """§2.3 / §6 jaw_pose: amplitude + rotvec→degree frame deltas."""
    j = np.asarray(jaw, dtype=np.float64)
    degs = rotvec_delta_degrees(j)
    return {
        "amplitude_std": float(np.std(j)),
        "mean_abs": float(np.mean(np.abs(j))),
        "max_abs": float(np.max(np.abs(j))),
        "range_per_dim": float(np.mean(np.ptp(j, axis=0))),
        "frame_delta_deg_mean": _mean(degs),
        "frame_delta_deg_std": _std(degs),
        "frame_delta_deg_max": _max(degs),
        "motion_energy_deg": sum(d * d for d in degs) if degs else 0.0,
    }


def _mean(v): return float(np.mean(v)) if len(v) else 0.0


def _std(v): return float(np.std(v)) if len(v) else 0.0


def _max(v): return float(np.max(v)) if len(v) else 0.0


def rotvec_delta_degrees(seq: np.ndarray) -> List[float]:
    """Per-transition rotation angle (deg) between consecutive rotvecs.

    R_rel = R_t · R_{t-1}^T  →  angle = arccos((trace(R_rel)-1)/2) in degrees.
    """
    out: List[float] = []
    s = np.asarray(seq, dtype=np.float64)
    for t in range(1, len(s)):
        R0, _ = cv2.Rodrigues(s[t - 1])
        R1, _ = cv2.Rodrigues(s[t])
        Rel = R1 @ R0.T
        c = float((np.trace(Rel) - 1.0) / 2.0)
        out.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    return out


def expression_metrics(expr: np.ndarray) -> Dict[str, Any]:
    """§6 expression: per-frame L2 delta of the 100-dim code."""
    e = np.asarray(expr, dtype=np.float64)
    d = np.linalg.norm(np.diff(e, axis=0), axis=1)
    return {"frame_delta_mean": float(d.mean()) if len(d) else 0.0,
            "frame_delta_std": float(d.std()) if len(d) else 0.0,
            "frame_delta_max": float(d.max()) if len(d) else 0.0,
            "std": float(np.std(e)), "motion_energy": float(np.sum(d ** 2))}


def head_metrics(head: np.ndarray) -> Dict[str, Any]:
    degs = rotvec_delta_degrees(head)
    return {"frame_delta_deg_mean": _mean(degs),
            "frame_delta_deg_std": _std(degs),
            "frame_delta_deg_max": _max(degs),
            "motion_energy_deg": sum(d * d for d in degs) if degs else 0.0,
            "std": float(np.std(np.asarray(head, dtype=np.float64)))}

# ---------------------------------------------------------------------------
# Motion preservation summary (§2) and FLAME comparison (§3/§6)
# ---------------------------------------------------------------------------


def motion_preservation() -> Dict[str, Any]:
    """§2: landmark motion-energy retention + mouth opening + jaw amplitude."""
    lms = {b: _landmark_array(_branch_landmarks(b)) for b in BRANCHES}
    me = {b: landmark_motion_energy(lms[b]) for b in BRANCHES}
    mouth = mouth_comparison()

    flame = {b: _branch_flame(b, _clip()) for b in BRANCHES}
    jaw = {b: jaw_metrics(flame[b]["jaw_pose"]) for b in BRANCHES}
    return {
        "landmark_motion_energy": me,
        "motion_energy_retention": (me["ema_alpha07"] / me["baseline"]
                                     if me["baseline"] else None),
        "over_smoothed": (me["ema_alpha07"] / me["baseline"] < 0.8
                           if me["baseline"] else None),
        "mouth_opening": mouth,
        "jaw": {"baseline": jaw["baseline"], "ema_alpha07": jaw["ema_alpha07"],
                "amplitude_retention": (jaw["ema_alpha07"]["amplitude_std"]
                                         / jaw["baseline"]["amplitude_std"]
                                         if jaw["baseline"]["amplitude_std"] else None)},
    }


def flame_comparison(clip_id: str) -> Dict[str, Any]:
    """§6 FLAME stability: expression / head_pose / jaw_pose baseline vs EMA."""
    flame = {b: _branch_flame(b, clip_id) for b in BRANCHES}
    out: Dict[str, Any] = {}
    for b in BRANCHES:
        f = flame[b]
        out[b] = {
            "expression": expression_metrics(f["expression_code"]),
            "head_pose": head_metrics(f["head_pose"]),
            "jaw_pose": head_metrics(f["jaw_pose"]),
        }
    ratios = {
        "expression_energy_ratio": (
            out["ema_alpha07"]["expression"]["motion_energy"]
            / out["baseline"]["expression"]["motion_energy"]
            if out["baseline"]["expression"]["motion_energy"] else None),
        "head_motion_ratio": (
            out["ema_alpha07"]["head_pose"]["motion_energy_deg"]
            / out["baseline"]["head_pose"]["motion_energy_deg"]
            if out["baseline"]["head_pose"]["motion_energy_deg"] else None),
        "jaw_motion_ratio": (
            out["ema_alpha07"]["jaw_pose"]["motion_energy_deg"]
            / out["baseline"]["jaw_pose"]["motion_energy_deg"]
            if out["baseline"]["jaw_pose"]["motion_energy_deg"] else None),
    }
    return {"metrics": out, "motion_energy_ratios": ratios}


# ---------------------------------------------------------------------------
# Stage17 fusion layout verification
# ---------------------------------------------------------------------------


def verify_stage17(clip_id: str) -> Dict[str, Any]:
    """§4 Stage17: visual_feature [T,112] layout vs source stages."""
    out: Dict[str, Any] = {"layout": {k: list(v) for k, v in LAYOUT.items()}}
    for branch in BRANCHES:
        ts = np.load(OUT_BASE / branch / "stage17_fusion" / clip_id
                     / "training_sample.npz")
        vf = ts["visual_feature"]
        af = ts["audio_feature"]
        T = vf.shape[0]
        blocks = {k: vf[:, v[0]:v[1]] for k, v in LAYOUT.items()}
        flame = _branch_flame(branch, clip_id)
        gaze = _branch_gaze(branch, clip_id)
        maxdiff = {
            "expression": float(np.max(np.abs(
                blocks["expression"] - flame["expression_code"][:T]))),
            "head_pose": float(np.max(np.abs(
                blocks["head_pose"] - flame["head_pose"][:T]))),
            "jaw_pose": float(np.max(np.abs(
                blocks["jaw_pose"] - flame["jaw_pose"][:T]))),
            "gaze": float(np.max(np.abs(blocks["gaze"] - gaze[:T]))),
        }
        out[branch] = {
            "visual_feature_shape": list(vf.shape),
            "audio_feature_shape": list(af.shape),
            "valid_length": int(ts["valid_length"]),
            "layout_ok": bool(
                list(vf.shape) == [T, 112] and list(af.shape) == [T, 768]
                and all(maxdiff[k] < 1e-4 for k in maxdiff)),
            "block_max_abs_diff_vs_source": maxdiff,
        }
    return out


# ---------------------------------------------------------------------------
# Overlay videos: original clip + 106 landmarks (raw vs EMA)
# ---------------------------------------------------------------------------


def _label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    s5._label(img, text, x, y, color)


def render_overlays(clip_id: str, skip: bool = False) -> List[str]:
    """§5 overlay videos: original video + 106 landmark dots + frame index.

    baseline_landmark_overlay.mp4 (raw landmarks) and
    ema07_landmark_overlay.mp4 (EMA-smoothed landmarks) into
    docs/stage10_landmark_temporal_ema_ablation/videos/.  Same fps / resolution
    as the source clip, H.264 / yuv420p / faststart (directly playable).
    """
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_path = SRC_BASE / "clips" / f"{clip_id}.mp4"
    specs = [
        ("baseline_landmark_overlay.mp4", "baseline",
         "Baseline  Stage10 raw 106-lmk", GREEN),
        ("ema07_landmark_overlay.mp4", "ema_alpha07",
         "EMA alpha=0.7  Stage10 106-lmk", ORANGE),
    ]
    produced: List[str] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS

    for out_name, branch, title, color in specs:
        if skip:
            print(f"[video] skipped {out_name} (--skip-videos)")
            continue
        recs = _branch_landmarks(branch)
        n = len(recs)
        raw_tmp = VIDEO_DIR / ("_raw_" + out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_tmp), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"cannot create video writer: {raw_tmp}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            rec = recs[i]
            lmf = rec.get("landmarks_106")
            if lmf:
                if isinstance(lmf, str):
                    lmf = [float(x) for x in lmf.strip("[]").split(",")]
                pts = np.asarray(lmf).reshape(106, 2)
                for (px, py) in pts:
                    cv2.circle(frame, (int(round(px)), int(round(py))), 2, CYAN, -1)
            tag = f"{title}  |  frame {int(rec['frame_id']):03d}"
            if rec.get("confidence") is not None:
                tag += f"  |  conf {rec['confidence']:.3f}"
            _label(frame, tag, 12, 30, WHITE)
            writer.write(frame)
        writer.release()
        final = VIDEO_DIR / out_name
        print(f"[video] rendering {out_name} ...", flush=True)
        s5._h264_ffmpeg(raw_tmp, final)
        produced.append(str(final))
        print(f"[video] wrote {final}")
    cap.release()
    return produced

# ---------------------------------------------------------------------------
# Decision helper
# ---------------------------------------------------------------------------


def recommend(jit: Dict[str, Any], motion: Dict[str, Any],
              flame: Dict[str, Any]) -> Dict[str, Any]:
    """Does landmark EMA stabilise landmarks while preserving motion?"""
    row = jit["all"]
    base_j = row["baseline"]["mean"]
    ema_j = row["ema_alpha07"]["mean"]
    q1 = ema_j < base_j                                   # jitter reduced
    q2 = (motion["motion_energy_retention"] is not None
          and motion["motion_energy_retention"] >= 0.8)  # no over-smoothing
    q3 = (motion["mouth_opening"]["mean_opening_retention"] is not None
          and motion["mouth_opening"]["mean_opening_retention"] >= 0.8)
    q4 = (motion["jaw"]["amplitude_retention"] is not None
          and motion["jaw"]["amplitude_retention"] >= 0.8)
    rec = "推荐进入生产" if (q1 and q2 and q3 and q4) else "暂不推荐"
    return {"q1_jitter_reduced": q1, "q2_motion_ok": q2,
            "q3_mouth_ok": q3, "q4_jaw_ok": q4, "recommendation": rec}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(base: float, val: float) -> str:
    if not base:
        return "—"
    return f"{val:.3f} ({100.0 * (val - base) / base:+.1f}%)"


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return f"{x:.{nd}f}" if x is not None else "—"


def build_report(clip_id: str, jit: Dict[str, Any], motion: Dict[str, Any],
                 flame: Dict[str, Any], s17: Dict[str, Any],
                 videos: List[str], rec: Dict[str, Any]) -> str:
    """docs/Stage10_Landmark_Temporal_EMA_Ablation_Report.md (§7)."""
    fb = flame["metrics"]["baseline"]
    fe = flame["metrics"]["ema_alpha07"]
    ratios = flame["motion_energy_ratios"]
    mouth = motion["mouth_opening"]
    jaw = motion["jaw"]
    me = motion["landmark_motion_energy"]
    ret = motion["motion_energy_retention"]

    L: List[str] = []
    A = L.append
    A("# Stage10 Landmark Temporal EMA (α=0.7) End-to-End Motion Preservation Ablation Report\n")
    A("## 1. 实验目的\n")
    A("当前 mesh 抖动问题已排除主要来自 bbox EMA（Stage5 消融），head/jaw FLAME 参数抖动主要"
      "来源于 Stage10 106 landmark 逐帧检测噪声以及后续 FLAME fitting。本实验验证：对 Stage10 "
      "输出的 106 landmark 2D 坐标施加因果 temporal EMA（当前帧权重 0.7）后，是否能够降低 "
      "landmark 高频抖动，同时保持真实人脸运动与表情（motion / mouth / jaw）不受损。\n")
    A("\n## 2. 方法\n")
    A(f"- **输入**: 与 Stage5 bbox EMA 消融相同的视频与 200 帧 clip (`{_to_rel(SRC_BASE / 'clips' / (clip_id + '.mp4'))}`)\n")
    A("- **Baseline**: Stage10 raw 106 landmark，不做任何平滑\n")
    A("- **EMA**: 逐点因果 EMA（作用于 [106,2] 坐标）:\n")
    A(f"  `landmark_ema[t] = {ALPHA}·landmark_raw[t] + (1-{ALPHA})·landmark_ema[t-1]`; "
      f"第 0 帧 = raw; 当前帧权重 {ALPHA} / 历史权重 {1.0 - ALPHA:.1f}; 不使用 Butterworth / "
      "filtfilt / 未来帧。\n")
    A("- **SCRFD bbox 两分支共用**（raw bbox，复用自 Stage5 ROI 消融 `baseline_results.json`，"
      "与先前实验字节一致）；仅 landmark 不同 → Stage11 对齐不同。\n")
    A("- **Stage12** 经 docker `emica_env` 官方 `stage0_17_e2e_stage12.py`（jaw_source=emica）；"
      "Stage13/14/17 经官方 runner；参数与生产一致。\n")

    A("\n## 3. Landmark 稳定性结果\n")
    A("| metric | baseline | EMA0.7 | change |\n|-|-|-|-|\n")
    for region in REGIONS:
        row = jit[region]
        b, e = row["baseline"], row["ema_alpha07"]
        red = row["jitter_reduction_pct"]
        reds = f"{red:+.1f}%" if red is not None else "—"
        A(f"| {region} mean jitter (px) | {b['mean']:.3f} | {e['mean']:.3f} | {reds} |\n")
        A(f"| {region} std jitter (px) | {b['std']:.3f} | {e['std']:.3f} | — |\n")
        A(f"| {region} max jitter (px) | {b['max']:.3f} | {e['max']:.3f} | — |\n")
        A(f"| {region} P95 jitter (px) | {b['p95']:.3f} | {e['p95']:.3f} | — |\n")

    A("\n## 4. Motion Preservation\n")
    A("| 指标 | Baseline | EMA α=0.7 | Retention |\n|---|---|---|---|\n")
    A(f"| landmark motion energy (px/frame) | {me['baseline']:.4f} | "
      f"{me['ema_alpha07']:.4f} | **{_fmt(ret)}** |\n")
    A(f"| mouth opening mean (px) | {mouth['baseline']['mean']:.3f} | "
      f"{mouth['ema_alpha07']['mean']:.3f} | {_fmt(mouth['mean_opening_retention'])} |\n")
    A(f"| mouth opening max (px) | {mouth['baseline']['max']:.3f} | "
      f"{mouth['ema_alpha07']['max']:.3f} | — |\n")
    A(f"| mouth opening std (px) | {mouth['baseline']['std']:.3f} | "
      f"{mouth['ema_alpha07']['std']:.3f} | — |\n")
    A(f"| mouth opening 曲线相关系数 | — | — | {_fmt(mouth['pearson_correlation'])} |\n")
    A(f"| jaw amplitude (std) | {jaw['baseline']['amplitude_std']:.5f} | "
      f"{jaw['ema_alpha07']['amplitude_std']:.5f} | {_fmt(jaw['amplitude_retention'])} |\n")
    A(f"| jaw max abs | {jaw['baseline']['max_abs']:.5f} | "
      f"{jaw['ema_alpha07']['max_abs']:.5f} | — |\n")
    A(f"| jaw frame delta mean (deg) | {jaw['baseline']['frame_delta_deg_mean']:.4f} | "
      f"{jaw['ema_alpha07']['frame_delta_deg_mean']:.4f} | — |\n")
    A(f"| jaw frame delta std (deg) | {jaw['baseline']['frame_delta_deg_std']:.4f} | "
      f"{jaw['ema_alpha07']['frame_delta_deg_std']:.4f} | — |\n")
    A(f"| jaw frame delta max (deg) | {jaw['baseline']['frame_delta_deg_max']:.4f} | "
      f"{jaw['ema_alpha07']['frame_delta_deg_max']:.4f} | — |\n")
    A(f"- **motion energy retention {_fmt(ret)}**: "
      f"{'⚠️ 可能过度平滑（<0.8）' if motion['over_smoothed'] else 'OK（≥0.8）'}\n")

    A("\n## 5. FLAME 参数变化 (Stage12)\n")
    A("### 5.1 expression (100-dim)\n")
    A("| 指标 | Baseline | EMA α=0.7 | Ratio |\n|---|---|---|---|\n")
    A(f"| frame delta mean (L2) | {fb['expression']['frame_delta_mean']:.5f} | "
      f"{fe['expression']['frame_delta_mean']:.5f} | — |\n")
    A(f"| frame delta std (L2) | {fb['expression']['frame_delta_std']:.5f} | "
      f"{fe['expression']['frame_delta_std']:.5f} | — |\n")
    A(f"| frame delta max (L2) | {fb['expression']['frame_delta_max']:.5f} | "
      f"{fe['expression']['frame_delta_max']:.5f} | — |\n")
    A(f"| motion energy | {fb['expression']['motion_energy']:.4f} | "
      f"{fe['expression']['motion_energy']:.4f} | **{_fmt(ratios['expression_energy_ratio'])}** |\n")
    A("### 5.2 head_pose (rotvec → degree)\n")
    A("| 指标 | Baseline | EMA α=0.7 | Ratio |\n|---|---|---|---|\n")
    A(f"| rotation delta mean (deg) | {fb['head_pose']['frame_delta_deg_mean']:.4f} | "
      f"{fe['head_pose']['frame_delta_deg_mean']:.4f} | — |\n")
    A(f"| rotation delta std (deg) | {fb['head_pose']['frame_delta_deg_std']:.4f} | "
      f"{fe['head_pose']['frame_delta_deg_std']:.4f} | — |\n")
    A(f"| rotation delta max (deg) | {fb['head_pose']['frame_delta_deg_max']:.4f} | "
      f"{fe['head_pose']['frame_delta_deg_max']:.4f} | — |\n")
    A(f"| motion energy (deg²) | {fb['head_pose']['motion_energy_deg']:.3f} | "
      f"{fe['head_pose']['motion_energy_deg']:.3f} | **{_fmt(ratios['head_motion_ratio'])}** |\n")
    A("### 5.3 jaw_pose (rotvec → degree)\n")
    A("| 指标 | Baseline | EMA α=0.7 | Ratio |\n|---|---|---|---|\n")
    A(f"| rotation delta mean (deg) | {fb['jaw_pose']['frame_delta_deg_mean']:.4f} | "
      f"{fe['jaw_pose']['frame_delta_deg_mean']:.4f} | — |\n")
    A(f"| rotation delta std (deg) | {fb['jaw_pose']['frame_delta_deg_std']:.4f} | "
      f"{fe['jaw_pose']['frame_delta_deg_std']:.4f} | — |\n")
    A(f"| rotation delta max (deg) | {fb['jaw_pose']['frame_delta_deg_max']:.4f} | "
      f"{fe['jaw_pose']['frame_delta_deg_max']:.4f} | — |\n")
    A(f"| motion energy (deg²) | {fb['jaw_pose']['motion_energy_deg']:.3f} | "
      f"{fe['jaw_pose']['motion_energy_deg']:.3f} | **{_fmt(ratios['jaw_motion_ratio'])}** |\n")

    A("\n## 6. 视频结果\n")
    for v in (VIDEO_DIR / "baseline_landmark_overlay.mp4",
              VIDEO_DIR / "ema07_landmark_overlay.mp4"):
        status = "✓" if v.exists() else "(未生成)"
        A(f"- `{_to_rel(v)}` {status}\n")
    A("人工观察重点：眼睛是否漂移、嘴部运动是否被压制、轮廓是否更稳定。\n")

    A("\n## 7. 最终建议\n")
    A(f"- jitter 降低: {'是' if rec['q1_jitter_reduced'] else '否'}\n")
    A(f"- motion energy retention: {_fmt(ret)} "
      f"({'OK' if rec['q2_motion_ok'] else '⚠️'})\n")
    A(f"- mouth opening retention: {_fmt(mouth['mean_opening_retention'])} "
      f"({'OK' if rec['q3_mouth_ok'] else '⚠️'})\n")
    A(f"- jaw amplitude retention: {_fmt(jaw['amplitude_retention'])} "
      f"({'OK' if rec['q4_jaw_ok'] else '⚠️'})\n")
    A(f"- **是否推荐进入生产 pipeline: {rec['recommendation']}**\n")
    A("\n---\n*本报告由 `scripts/run_single_video_dataset_landmark_ema_ablation.sh` + "
      "`scripts/stage10_landmark_temporal_ema_ablation.py` 生成；生产代码未修改。*\n")
    return "".join(L)

# ---------------------------------------------------------------------------
# Analyze orchestration
# ---------------------------------------------------------------------------


def analyze(skip_videos: bool = False) -> Dict[str, Any]:
    """Full comparison: jitter + motion preservation + FLAME + Stage17 + videos + report."""
    clip_id = _clip()
    print(f"[analyze] clip={clip_id}", flush=True)

    jit = jitter_comparison()
    all_row = jit["all"]
    print(f"[analyze] landmark jitter all106: "
          f"{all_row['baseline']['mean']:.3f} -> "
          f"{all_row['ema_alpha07']['mean']:.3f} px "
          f"({all_row['jitter_reduction_pct']:+.1f}%)", flush=True)

    motion = motion_preservation()
    print(f"[analyze] motion energy retention={motion['motion_energy_retention']:.3f} "
          f"mouth corr={motion['mouth_opening']['pearson_correlation']:.3f} "
          f"jaw amp retention={motion['jaw']['amplitude_retention']:.3f}", flush=True)

    flame = flame_comparison(clip_id)
    ratios = flame["motion_energy_ratios"]
    print(f"[analyze] FLAME ratios: expr={ratios['expression_energy_ratio']:.3f} "
          f"head={ratios['head_motion_ratio']:.3f} jaw={ratios['jaw_motion_ratio']:.3f}",
          flush=True)

    s17 = verify_stage17(clip_id)
    for branch in BRANCHES:
        print(f"[analyze] stage17 {branch}: "
              f"visual={s17[branch]['visual_feature_shape']} "
              f"layout_ok={s17[branch]['layout_ok']}", flush=True)

    videos = render_overlays(clip_id, skip=skip_videos)
    rec = recommend(jit, motion, flame)

    report = build_report(clip_id, jit, motion, flame, s17, videos, rec)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[report] wrote {REPORT_PATH}")

    metrics = {
        "clip_id": clip_id,
        "alpha": ALPHA,
        "jitter": jit,
        "motion_preservation": motion,
        "flame": flame,
        "stage17": s17,
        "recommendation": rec,
        "report": _to_rel(REPORT_PATH),
    }
    _write_json(COMPARISON_DIR / "ablation_metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep", help="generate Stage5/10/11 for both branches")
    p_prep.add_argument("--clip", default=DEFAULT_CLIP)
    p_prep.add_argument("--alpha", type=float, default=ALPHA)
    p_prep.add_argument("--max-frames", type=int, default=None,
                        help="limit frames (smoke test)")
    p_prep.set_defaults(func=cmd_prep)

    p_ana = sub.add_parser("analyze", help="comparison + videos + report")
    p_ana.add_argument("--skip-videos", action="store_true")
    p_ana.set_defaults(func=cmd_analyze)

    return ap.parse_args(argv)


def cmd_prep(args: argparse.Namespace) -> int:
    print(f"[prep] clip={args.clip} alpha={args.alpha} "
          f"max_frames={args.max_frames or 'all'}", flush=True)
    r = prep_clip(args.clip, args.alpha, args.max_frames)
    for branch in BRANCHES:
        print(f"[prep] {branch}: n={r[branch]['n']} "
              f"aligned={r[branch]['n_aligned']} "
              f"landmarks={r[branch]['n_landmarks']} "
              f"lm_jitter={r[branch]['landmark_jitter_mean']:.3f}px", flush=True)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    analyze(skip_videos=args.skip_videos)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

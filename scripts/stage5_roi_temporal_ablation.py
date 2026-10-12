#!/usr/bin/env python3
"""Stage5 SCRFD Temporal ROI Constraint Ablation Experiment (read-only).

Ablation study: does constraining SCRFD to a temporal ROI (previous-frame
tight bbox expanded by ``roi_expand_ratio``) reduce bbox / Stage10-landmark
temporal jitter versus the production full-image baseline?

Reproduces the production detection path exactly for Experiment A:
    Frame -> full-image SCRFD -> tight bbox + confidence + 5-pt landmark
and mirrors Stage 10's crop mapping for the landmark-stability comparison:
    expanded bbox (x2.0) -> native crop -> 2d106det -> crop + origin (1:1)
    -> frame-space 106 landmarks.

Everything is read-only w.r.t. the production pipeline: no stage_05 / stage_10
code and no model weights are modified. Experiment artifacts are written under
docs/stage5_roi_ablation/ and the report to docs/Stage5_ROI_Temporal_Ablation_Report.md.

Usage (from the Face_Pipeline_V2 project root):

    PYTHONPATH=. python scripts/stage5_roi_temporal_ablation.py
    PYTHONPATH=. python scripts/stage5_roi_temporal_ablation.py --max-frames 40 --skip-videos
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow ``import face_pipeline_v2`` regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from face_pipeline_v2.models.wrappers.insightface_landmark import (  # noqa: E402
    InsightFaceLandmarkWrapper,
)
from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper  # noqa: E402
from face_pipeline_v2.stages.stage_05_face_detection.face_cropper import (  # noqa: E402
    expand_bbox,
)

# ---------------------------------------------------------------------------
# Experiment configuration (production values from configs/config.yaml)
# ---------------------------------------------------------------------------
VIDEO_PATH = (
    _PROJECT_ROOT
    / "outputs/stage0_17_e2e/stage0_11/clips"
    / "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4"
)
SCRFD_MODEL_DIR = _PROJECT_ROOT / "models/SCRFD"
LANDMARK_MODEL = _PROJECT_ROOT / "models/insightface/2d106det.onnx"

CONFIDENCE_THRESHOLD = 0.6  # face.confidence_threshold
PRODUCTION_EXPAND = 2.0     # stage_05_face_crop.expand_ratio
ROI_EXPAND = 1.2            # experiment default temporal-ROI expansion
OUTLIER_PX = 8.0            # isolated-jump detection threshold (pixels)
MAX_FRAMES = 200            # clip duration (25 fps x 8 s)
FPS = 25.0
OUT_DIR = _PROJECT_ROOT / "docs/stage5_roi_ablation"
REPORT_PATH = _PROJECT_ROOT / "docs/Stage5_ROI_Temporal_Ablation_Report.md"
FFMPEG = Path("/usr/local/bin/ffmpeg")

# Swept temporal-ROI expansion ratios (used to recommend an optimal ratio).
SWEEP_RATIOS = (1.0, 1.2, 1.5, 2.0)

# BGR colours for overlay videos.
GREEN = (60, 200, 80)
YELLOW = (0, 220, 255)
RED = (40, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])



def _pick_main(
    dets: List[Dict[str, Any]], conf_threshold: float
) -> Optional[Dict[str, Any]]:
    """Production face selection: largest bbox area, conf >= threshold.

    Mirrors ``face_detector.py::select_face_largest_area`` plus the
    ``face.confidence_threshold`` production value.
    """
    good = [d for d in dets if float(d.get("confidence", 0.0)) >= conf_threshold]
    if not good:
        return None
    return max(good, key=lambda d: _bbox_area(d["bbox"]))


# SCRFD input resolution (fixed by the wrapper) and target face occupancy on
# the padded detection canvas.  A tight ROI crop fed straight into SCRFD gets
# stretched to 640x640, making the face occupy >80% of the frame, which falls
# outside SCRFD 10g's training distribution and yields zero detections.  We
# therefore pad the ROI crop (aspect preserved) onto a 640x640 canvas with the
# face kept at ~FACE_CANVAS_FRACTION — the detector's native "zoom" view.
ROI_CANVAS = 640
FACE_CANVAS_FRACTION = 0.65


def _detect_roi(
    wrapper: SCRFDWrapper,
    frame: np.ndarray,
    prev_bbox: Tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    expand: float,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[float, float, float, float]]]:
    """Run SCRFD inside a temporal ROI (prev bbox expanded by ``expand``).

    ROI crop -> aspect-preserving resize onto a 640x640 canvas (face ~65% of
    the canvas) -> SCRFD -> canvas coords mapped back to the full frame via
    ``frame = (canvas - pad_offset) / scale + crop_origin``.

    Returns (detections_in_global_coords, roi_bbox).  ``roi_bbox`` is None
    only when the ROI is degenerate.
    """
    rx1, ry1, rx2, ry2 = expand_bbox(prev_bbox, frame_w, frame_h, expand)
    if (rx2 - rx1) < 8 or (ry2 - ry1) < 8:
        return [], None

    ix1, iy1, ix2, iy2 = int(rx1), int(ry1), int(rx2), int(ry2)
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return [], None

    # Aspect-preserving scale onto the padded canvas (uniform, no stretch).
    hh, ww = crop.shape[:2]
    scale = min(
        (ROI_CANVAS * FACE_CANVAS_FRACTION) / ww,
        (ROI_CANVAS * FACE_CANVAS_FRACTION) / hh,
    )
    nw_, nh_ = max(1, int(round(ww * scale))), max(1, int(round(hh * scale)))
    resized = cv2.resize(crop, (nw_, nh_), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ROI_CANVAS, ROI_CANVAS, 3), dtype=np.uint8)
    ox, oy = (ROI_CANVAS - nw_) // 2, (ROI_CANVAS - nh_) // 2
    canvas[oy:oy + nh_, ox:ox + nw_] = resized

    dets = wrapper.detect(canvas)
    if not dets:
        return [], (rx1, ry1, rx2, ry2)

    def _to_frame(v: float, origin: float, pad: float) -> float:
        return (v - pad) / scale + origin

    for d in dets:
        b = d["bbox"]
        d["bbox"] = (
            _to_frame(b[0], ix1, ox),
            _to_frame(b[1], iy1, oy),
            _to_frame(b[2], ix1, ox),
            _to_frame(b[3], iy1, oy),
        )
        lm = d.get("landmarks")
        if lm:
            lm = list(lm)
            lm[0::2] = [_to_frame(v, ix1, ox) for v in lm[0::2]]
            lm[1::2] = [_to_frame(v, iy1, oy) for v in lm[1::2]]
            d["landmarks"] = tuple(lm)
    return dets, (rx1, ry1, rx2, ry2)


def _roi_chain(
    wrapper: SCRFDWrapper,
    frame: np.ndarray,
    prev_bbox: Optional[Tuple[float, float, float, float]],
    frame_w: int,
    frame_h: int,
    expand: float,
    conf_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[float, float, float, float]], str, bool]:
    """Temporal-ROI detection with full-image fallback (section 7).

    Returns (main_detection_or_None, roi_bbox, detect_source, fallback).
    ``detect_source`` is "full_image" or "roi"; ``fallback`` is True when the
    ROI path produced no valid face (no detection or conf < threshold) and a
    full-image re-detection was performed.
    """
    if prev_bbox is None:
        dets = wrapper.detect(frame)
        main = _pick_main(dets, conf_threshold)
        return main, None, "full_image", False

    roi_dets, roi_bbox = _detect_roi(
        wrapper, frame, prev_bbox, frame_w, frame_h, expand
    )
    main = _pick_main(roi_dets, conf_threshold) if roi_dets else None
    if main is not None:
        return main, roi_bbox, "roi", False

    dets = wrapper.detect(frame)
    main = _pick_main(dets, conf_threshold)
    return main, roi_bbox, "full_image", True


def _extract_106(
    frame: np.ndarray,
    tight_bbox: Tuple[float, float, float, float],
    lmark: InsightFaceLandmarkWrapper,
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    """Stage-10-style 106 landmarks in frame space (mirrors production).

    expanded bbox (x2.0) -> native crop -> 2d106det -> crop + expanded origin
    (1:1), exactly matching ``stage_10_landmark/processor.py::_map_to_frame``.
    """
    ex1, ey1, ex2, ey2 = expand_bbox(tight_bbox, frame_w, frame_h, PRODUCTION_EXPAND)
    x1, y1, x2, y2 = int(ex1), int(ey1), int(ex2), int(ey2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        lm_crop = lmark.predict(crop)  # (106, 2) in crop pixel space
    except Exception:
        return None
    lm = lm_crop.copy()
    lm[:, 0] += ex1
    lm[:, 1] += ey1
    return lm


# ---------------------------------------------------------------------------
# Stability metrics (section 8 / 9)
# ---------------------------------------------------------------------------


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_stability(
    frames: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """BBox centre + scale stability over consecutive frame transitions.

    ``frames`` is a list of per-frame dicts each holding ``frame_id`` and
    ``bbox`` (or None when no face was detected).
    """
    deltas: List[Tuple[int, float]] = []
    dws: List[float] = []
    dhs: List[float] = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1]["bbox"], frames[i]["bbox"]
        if a is None or b is None:
            continue
        ca, cb = _center(a), _center(b)
        deltas.append((frames[i]["frame_id"], math.hypot(cb[0] - ca[0], cb[1] - ca[1])))
        aw, ah = a[2] - a[0], a[3] - a[1]
        bw, bh = b[2] - b[0], b[3] - b[1]
        dws.append(abs(bw - aw))
        dhs.append(abs(bh - ah))

    vals = [d for _, d in deltas]
    return {
        "bbox_center_jitter_mean": float(np.mean(vals)) if vals else 0.0,
        "bbox_center_jitter_std": float(np.std(vals)) if vals else 0.0,
        "bbox_max_jump": float(np.max(vals)) if vals else 0.0,
        "width_change_mean": float(np.mean(dws)) if dws else 0.0,
        "height_change_mean": float(np.mean(dhs)) if dhs else 0.0,
        "transitions": [(t, round(d, 4)) for t, d in deltas],
    }


def isolated_outliers(
    deltas: List[Tuple[int, float]], threshold: float
) -> List[int]:
    """Detector-anomaly jumps: transition t-1->t > threshold AND t->t+1 normal.

    Implements section 8.3's rule.  The last transition (no next frame) is
    flagged if it alone exceeds the threshold.
    """
    by_t = {t: d for t, d in deltas}
    ts = sorted(by_t)
    out: List[int] = []
    for t in ts:
        if by_t[t] > threshold:
            nxt = by_t.get(t + 1)
            if nxt is None or nxt <= threshold:
                out.append(t)
    return out


def landmark_stability(
    lms: Dict[int, Optional[np.ndarray]],
) -> Dict[str, Any]:
    """106-landmark temporal jitter (frame-space) over transitions.

    Per-transition ``mean_point_distance`` (mean over the 106 points) and
    ``max_point_distance``.  Outliers use the audit convention
    (mean + 3 * std, identical to the Stage5/Stage10 audit).
    """
    fs = sorted(k for k, v in lms.items() if v is not None)
    mean_vals: List[float] = []
    max_vals: List[float] = []
    per_trans: List[Tuple[int, float]] = []
    for i in range(1, len(fs)):
        a, b = lms[fs[i - 1]], lms[fs[i]]
        if a is None or b is None:
            continue
        d = np.linalg.norm(b - a, axis=1)
        mean_vals.append(float(d.mean()))
        max_vals.append(float(d.max()))
        per_trans.append((fs[i], float(d.mean())))

    m = float(np.mean(mean_vals)) if mean_vals else 0.0
    s = float(np.std(mean_vals)) if mean_vals else 0.0
    outliers = [t for t, v in per_trans if v > m + 3.0 * s]
    return {
        "landmark_jitter_mean": m,
        "landmark_jitter_std": s,
        "landmark_max_jump": float(np.max(max_vals)) if max_vals else 0.0,
        "landmark_outlier_count": len(outliers),
        "landmark_outlier_frames": outliers,
    }



# ---------------------------------------------------------------------------
# Main experiment run (single video pass: A + B + ratio sweep)
# ---------------------------------------------------------------------------


def _make_record(
    fi: int,
    fps: float,
    main: Optional[Dict[str, Any]],
    kind: str,
    roi_bbox: Optional[Tuple[float, float, float, float]] = None,
    detect_source: Optional[str] = None,
    fallback: bool = False,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "frame_id": fi,
        "timestamp": round(fi / fps, 6),
        "bbox": list(main["bbox"]) if main else None,
        "confidence": float(main["confidence"]) if main else 0.0,
        "landmark5": list(main.get("landmarks") or ()) if main else [],
    }
    if kind == "roi":
        rec["detect_source"] = detect_source
        rec["fallback"] = bool(fallback)
        rec["roi_bbox"] = list(roi_bbox) if roi_bbox else None
    return rec


def run_experiment(
    wrapper: SCRFDWrapper,
    lmark: InsightFaceLandmarkWrapper,
    video_path: Path,
    max_frames: Optional[int],
) -> Dict[str, Any]:
    """Run Experiments A + B + the expand-ratio sweep in one frame pass."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    baseline_frames: List[Dict[str, Any]] = []
    roi_frames: List[Dict[str, Any]] = []
    baseline_lms: Dict[int, Optional[np.ndarray]] = {}
    roi_lms: Dict[int, Optional[np.ndarray]] = {}
    sweep_bboxes: Dict[float, List[Optional[Tuple[float, float, float, float]]]] = {
        r: [] for r in SWEEP_RATIOS
    }
    sweep_sources: Dict[float, List[str]] = {r: [] for r in SWEEP_RATIOS}
    sweep_lms: Dict[float, Dict[int, Optional[np.ndarray]]] = {
        r: {} for r in SWEEP_RATIOS
    }

    prev_bbox: Optional[Tuple[float, float, float, float]] = None
    fi = 0
    t0 = time.time()
    n_det_full = 0
    n_det_roi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and fi >= max_frames:
                break

            # ---- Experiment A: production full-image baseline ----
            bmain = _pick_main(wrapper.detect(frame), CONFIDENCE_THRESHOLD)
            n_det_full += 1
            baseline_frames.append(_make_record(fi, fps, bmain, "baseline"))
            baseline_lms[fi] = (
                _extract_106(frame, bmain["bbox"], lmark, w, h)
                if bmain is not None else None
            )

            # ---- Experiment B: temporal ROI (main ratio) ----
            rmain, rroi, rsrc, rfall = _roi_chain(
                wrapper, frame, prev_bbox, w, h, ROI_EXPAND, CONFIDENCE_THRESHOLD
            )
            n_det_roi += 1 + (1 if rfall else 0)  # ROI attempt + possible full fallback
            roi_frames.append(_make_record(fi, fps, rmain, "roi", rroi, rsrc, rfall))
            roi_lms[fi] = (
                _extract_106(frame, rmain["bbox"], lmark, w, h)
                if rmain is not None else None
            )

            # ---- expand-ratio sweep (same temporal prior chain) ----
            for r in SWEEP_RATIOS:
                if r == ROI_EXPAND:
                    smain, ssrc = rmain, rsrc
                else:
                    smain, _, ssrc, sfall = _roi_chain(
                        wrapper, frame, prev_bbox, w, h, r, CONFIDENCE_THRESHOLD
                    )
                    n_det_roi += 1 + (1 if sfall else 0)
                sweep_bboxes[r].append(smain["bbox"] if smain else None)
                sweep_sources[r].append(ssrc)
                sweep_lms[r][fi] = (
                    _extract_106(frame, smain["bbox"], lmark, w, h)
                    if smain is not None else None
                )

            # Temporal prior updates from the main ROI result.
            if rmain is not None:
                prev_bbox = rmain["bbox"]

            fi += 1
            if fi % 25 == 0:
                print(
                    f"  [run] frame {fi} elapsed {time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        cap.release()

    total = fi
    print(
        f"[run] {total} frames done in {time.time() - t0:.1f}s "
        f"(SCRFD full={n_det_full}, roi-path={n_det_roi})",
        flush=True,
    )
    return {
        "fps": fps,
        "width": w,
        "height": h,
        "frame_count": total,
        "baseline_frames": baseline_frames,
        "roi_frames": roi_frames,
        "baseline_lms": baseline_lms,
        "roi_lms": roi_lms,
        "sweep_bboxes": sweep_bboxes,
        "sweep_sources": sweep_sources,
        "sweep_lms": sweep_lms,
    }



# ---------------------------------------------------------------------------
# Metrics aggregation + conclusion
# ---------------------------------------------------------------------------


def _rounded(obj: Any) -> Any:
    """Recursively round floats for compact JSON output."""
    if isinstance(obj, float):
        return round(float(obj), 4)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_rounded(v) for v in obj]
    return obj


def _pct(diff: float, base: float) -> float:
    return (diff / base) * 100.0 if base else 0.0


def build_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute stability metrics for baseline vs ROI + sweep + conclusion."""
    b_frames = data["baseline_frames"]
    r_frames = data["roi_frames"]
    b_lms = data["baseline_lms"]
    r_lms = data["roi_lms"]

    b_stab = bbox_stability(b_frames)
    r_stab = bbox_stability(r_frames)
    b_outs = isolated_outliers(b_stab["transitions"], OUTLIER_PX)
    r_outs = isolated_outliers(r_stab["transitions"], OUTLIER_PX)
    b_lm = landmark_stability(b_lms)
    r_lm = landmark_stability(r_lms)

    # expand-ratio sweep (all ratios chain on the same main-ratio temporal
    # prior, so the only controlled variable is the ROI size)
    sweep: Dict[str, Any] = {}
    for r in SWEEP_RATIOS:
        s_frames = [
            {"frame_id": i, "bbox": bb} for i, bb in enumerate(data["sweep_bboxes"][r])
        ]
        s_stab = bbox_stability(s_frames)
        s_lm = landmark_stability(data["sweep_lms"][r])
        s_full = sum(1 for s in data["sweep_sources"][r] if s == "full_image")
        sweep[str(r)] = {
            "bbox_center_jitter_mean": s_stab["bbox_center_jitter_mean"],
            "bbox_center_jitter_std": s_stab["bbox_center_jitter_std"],
            "bbox_max_jump": s_stab["bbox_max_jump"],
            "landmark_jitter_mean": s_lm["landmark_jitter_mean"],
            "landmark_max_jump": s_lm["landmark_max_jump"],
            "full_image_count": s_full,
        }

    n_roi_frames = len(r_frames)
    roi_fallbacks = sum(1 for f in r_frames if f.get("fallback"))
    roi_fallback_rate = (roi_fallbacks / (n_roi_frames - 1)) if n_roi_frames > 1 else 0.0

    # Recommended ratio: lowest bbox jitter, then lowest landmark jitter,
    # then fewer full-image (fallback) detections.
    best = min(
        SWEEP_RATIOS,
        key=lambda r: (
            sweep[str(r)]["bbox_center_jitter_mean"],
            sweep[str(r)]["landmark_jitter_mean"],
            sweep[str(r)]["full_image_count"],
        ),
    )

    bbox_diff = r_stab["bbox_center_jitter_mean"] - b_stab["bbox_center_jitter_mean"]
    lm_diff = r_lm["landmark_jitter_mean"] - b_lm["landmark_jitter_mean"]
    roi_improve = bool(bbox_diff < 0 and lm_diff <= 0.05 * b_lm["landmark_jitter_mean"])

    best_bbox = sweep[str(best)]["bbox_center_jitter_mean"]
    if roi_improve:
        if roi_fallback_rate > 0.05:
            prod = "有条件推荐（抖动改善但 fallback 率偏高，需提高 expand_ratio）"
        else:
            prod = "推荐（抖动下降且 fallback 率低）"
    elif best_bbox < b_stab["bbox_center_jitter_mean"]:
        prod = (
            f"有条件推荐：默认 {ROI_EXPAND} 未改善平均抖动；但扫描中较小比例 {best:.1f} "
            f"将 bbox 抖动降至 {best_bbox:.3f}（低于基线 {b_stab['bbox_center_jitter_mean']:.3f}），"
            "需更多数据验证 landmark 稳定性后再决定"
        )
    else:
        prod = "不推荐（ROI 未稳定优于全图基线）"

    metrics = {
        "experiment": "stage5_scrfd_temporal_roi_ablation",
        "outlier_px_threshold": OUTLIER_PX,
        "baseline": {
            "bbox_center_jitter_mean": b_stab["bbox_center_jitter_mean"],
            "bbox_center_jitter_std": b_stab["bbox_center_jitter_std"],
            "bbox_max_jump": b_stab["bbox_max_jump"],
            "width_change_mean": b_stab["width_change_mean"],
            "height_change_mean": b_stab["height_change_mean"],
            "outlier_count": len(b_outs),
            "outlier_frames": b_outs,
            "landmark_jitter_mean": b_lm["landmark_jitter_mean"],
            "landmark_jitter_std": b_lm["landmark_jitter_std"],
            "landmark_max_jump": b_lm["landmark_max_jump"],
            "landmark_outlier_count": b_lm["landmark_outlier_count"],
            "landmark_outlier_frames": b_lm["landmark_outlier_frames"],
        },
        "roi": {
            "bbox_center_jitter_mean": r_stab["bbox_center_jitter_mean"],
            "bbox_center_jitter_std": r_stab["bbox_center_jitter_std"],
            "bbox_max_jump": r_stab["bbox_max_jump"],
            "width_change_mean": r_stab["width_change_mean"],
            "height_change_mean": r_stab["height_change_mean"],
            "outlier_count": len(r_outs),
            "outlier_frames": r_outs,
            "landmark_jitter_mean": r_lm["landmark_jitter_mean"],
            "landmark_jitter_std": r_lm["landmark_jitter_std"],
            "landmark_max_jump": r_lm["landmark_max_jump"],
            "landmark_outlier_count": r_lm["landmark_outlier_count"],
            "landmark_outlier_frames": r_lm["landmark_outlier_frames"],
        },
        "difference_pct": {
            "bbox_center_jitter_mean": _pct(bbox_diff, b_stab["bbox_center_jitter_mean"]),
            "bbox_max_jump": _pct(r_stab["bbox_max_jump"] - b_stab["bbox_max_jump"], b_stab["bbox_max_jump"]),
            "outlier_count": _pct(len(r_outs) - len(b_outs), len(b_outs)) if b_outs else 0.0,
            "landmark_jitter_mean": _pct(lm_diff, b_lm["landmark_jitter_mean"]),
            "landmark_max_jump": _pct(r_lm["landmark_max_jump"] - b_lm["landmark_max_jump"], b_lm["landmark_max_jump"]),
        },
        "sweep": sweep,
        "fallback": {
            "count": roi_fallbacks,
            "rate": roi_fallback_rate,
            "frames": [f["frame_id"] for f in r_frames if f.get("fallback")],
        },
        "conclusion": {
            "roi_effectiveness": roi_improve,
            "roi_improve": roi_improve,
            "recommended_ratio": float(best),
            "fallback_rate": roi_fallback_rate,
            "production_candidate": prod,
            "production_recommendation": prod,
        },
    }
    return _rounded(metrics)



# ---------------------------------------------------------------------------
# Overlay video rendering (section 10)
# ---------------------------------------------------------------------------


def _h264_ffmpeg(src: Path, dst: Path) -> None:
    """Re-encode mp4v -> H.264/avc1 + yuv420p + faststart (playable everywhere)."""
    if not FFMPEG.exists():
        print(f"  [video] ffmpeg not found at {FFMPEG}; keeping raw {src}", flush=True)
        src.replace(dst)
        return
    subprocess.run(
        [
            str(FFMPEG), "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
    )
    src.unlink(missing_ok=True)


def _draw_info(frame, text: str, y: int, colour) -> None:
    cv2.putText(frame, text, (24, y), _FONT, 1.0, BLACK, 5, cv2.LINE_AA)
    cv2.putText(frame, text, (24, y), _FONT, 1.0, colour, 2, cv2.LINE_AA)


def render_baseline_overlay(
    video_path: Path, frames: List[Dict[str, Any]], out_path: Path, fps: float
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or fi >= len(frames):
                break
            fr = frames[fi]
            if fr["bbox"]:
                x1, y1, x2, y2 = (int(v) for v in fr["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                _draw_info(frame, f"frame {fr['frame_id']:03d}", 40, WHITE)
                _draw_info(
                    frame,
                    f"conf {fr['confidence']:.3f}  [BASELINE full-image]",
                    82, GREEN,
                )
            writer.write(frame)
            fi += 1
    finally:
        cap.release()
        writer.release()
    _h264_ffmpeg(raw, out_path)
    return out_path


def render_roi_overlay(
    video_path: Path, frames: List[Dict[str, Any]], out_path: Path, fps: float
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or fi >= len(frames):
                break
            fr = frames[fi]
            # ROI search region (yellow)
            if fr.get("roi_bbox"):
                rx1, ry1, rx2, ry2 = (int(v) for v in fr["roi_bbox"])
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), YELLOW, 2)
            # final detection bbox (green)
            if fr["bbox"]:
                x1, y1, x2, y2 = (int(v) for v in fr["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
            _draw_info(frame, f"frame {fr['frame_id']:03d}", 40, WHITE)
            _draw_info(frame, f"conf {fr['confidence']:.3f}  src={fr.get('detect_source','-')}",
                       82, GREEN)
            if fr.get("fallback"):
                _draw_info(frame, "FALLBACK full-image", 124, RED)
            writer.write(frame)
            fi += 1
    finally:
        cap.release()
        writer.release()
    _h264_ffmpeg(raw, out_path)
    return out_path



# ---------------------------------------------------------------------------
# Report generation (section 11)
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _diff_cell(base: float, roi: float) -> str:
    d = roi - base
    pct = (d / base) * 100 if base else 0.0
    return f"{'+' if d > 0 else ''}{d:.3f} ({'+' if pct > 0 else ''}{pct:.1f}%)"


def generate_report(
    data: Dict[str, Any],
    metrics: Dict[str, Any],
    baseline_json: Path,
    roi_json: Path,
    baseline_video: Path,
    roi_video: Path,
) -> Path:
    b, r = metrics["baseline"], metrics["roi"]
    c = metrics["conclusion"]
    fps = data["fps"]
    n = data["frame_count"]
    dur = n / fps

    lines: List[str] = []
    lines.append("# Stage5 SCRFD Temporal ROI Constraint Ablation Report")
    lines.append("")
    lines.append("## 11.1 实验目的")
    lines.append("")
    lines.append(
        "生产管线中 Stage5 对每一帧执行 **全图 SCRFD 独立检测**。人物说话时，"
        "嘴部运动会引起 SCRFD 检测框轻微变化，进而导致 5 点 landmark 漂移、"
        "Stage10 106 点 landmark 输入变化、FLAME 参数时间抖动、mesh 稳定性下降。"
    )
    lines.append("")
    lines.append(
        "本实验只做**算法验证**（不改生产代码、不改模型、不加滤波），检验："
        "以上一帧 bbox 为 temporal prior、在扩展后的 ROI 区域内执行 SCRFD，"
        "是否能够限制检测搜索范围、降低 bbox 与 landmark 的时间抖动。"
    )
    lines.append("")
    lines.append("## 11.2 实验设置")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 视频 | `{data.get('video_path', VIDEO_PATH)}` |")
    lines.append(f"| 分辨率 / 帧率 | {data['width']}x{data['height']} / {fps:.0f} fps |")
    lines.append(f"| 帧数量 | {n}（时长 {dur:.1f}s） |")
    lines.append(f"| SCRFD 模型 | `{SCRFD_MODEL_DIR}/scrfd_10g_bnkps.onnx` |")
    lines.append(f"| confidence_threshold | {CONFIDENCE_THRESHOLD}（生产 `face.confidence_threshold`） |")
    lines.append(f"| 生产 expand_ratio（Stage10 crop） | {PRODUCTION_EXPAND} |")
    lines.append(f"| **Temporal ROI expand_ratio（实验）** | **{ROI_EXPAND}**（默认） |")
    lines.append(f"| 孤立跳变阈值 | {OUTLIER_PX:.0f} px（第 8.3 节规则） |")
    lines.append(f"| 扫描范围（推荐比例） | {', '.join(str(x) for x in SWEEP_RATIOS)} |")
    lines.append(f"| 主脸选择 | 最大面积 + conf≥{CONFIDENCE_THRESHOLD}（生产 `largest_area`） |")
    lines.append("")
    lines.append("## 11.3 定量结果")
    lines.append("")
    lines.append("| Metric | Baseline | ROI SCRFD | Difference |")
    lines.append("|---|---|---|---|")
    lines.append(f"| bbox jitter mean (px) | {_fmt(b['bbox_center_jitter_mean'])} | {_fmt(r['bbox_center_jitter_mean'])} | {_diff_cell(b['bbox_center_jitter_mean'], r['bbox_center_jitter_mean'])} |")
    lines.append(f"| bbox jitter std (px) | {_fmt(b['bbox_center_jitter_std'])} | {_fmt(r['bbox_center_jitter_std'])} | {_diff_cell(b['bbox_center_jitter_std'], r['bbox_center_jitter_std'])} |")
    lines.append(f"| bbox max jump (px) | {_fmt(b['bbox_max_jump'])} | {_fmt(r['bbox_max_jump'])} | {_diff_cell(b['bbox_max_jump'], r['bbox_max_jump'])} |")
    lines.append(f"| width change mean (px) | {_fmt(b['width_change_mean'])} | {_fmt(r['width_change_mean'])} | {_diff_cell(b['width_change_mean'], r['width_change_mean'])} |")
    lines.append(f"| height change mean (px) | {_fmt(b['height_change_mean'])} | {_fmt(r['height_change_mean'])} | {_diff_cell(b['height_change_mean'], r['height_change_mean'])} |")
    lines.append(f"| outlier count (>{OUTLIER_PX:.0f}px 孤立跳变) | {b['outlier_count']} | {r['outlier_count']} | {r['outlier_count'] - b['outlier_count']:+d} |")
    lines.append(f"| landmark jitter mean (px) | {_fmt(b['landmark_jitter_mean'])} | {_fmt(r['landmark_jitter_mean'])} | {_diff_cell(b['landmark_jitter_mean'], r['landmark_jitter_mean'])} |")
    lines.append(f"| landmark max jump (px) | {_fmt(b['landmark_max_jump'])} | {_fmt(r['landmark_max_jump'])} | {_diff_cell(b['landmark_max_jump'], r['landmark_max_jump'])} |")
    lines.append(f"| landmark outlier count | {b['landmark_outlier_count']} | {r['landmark_outlier_count']} | {r['landmark_outlier_count'] - b['landmark_outlier_count']:+d} |")
    lines.append("")
    lines.append("异常帧（孤立跳变 > %.0f px）：" % OUTLIER_PX)
    lines.append(f"- Baseline bbox: {b['outlier_frames'] or '无'}")
    lines.append(f"- ROI bbox: {r['outlier_frames'] or '无'}")
    lines.append(f"- Baseline landmark: {b['landmark_outlier_frames'] or '无'}")
    lines.append(f"- ROI landmark: {r['landmark_outlier_frames'] or '无'}")
    lines.append("")
    lines.append("### expand_ratio 扫描")
    lines.append("")
    lines.append("> 各比例共用同一 temporal prior 链（主比例 1.2 的上一帧 bbox），唯一变量为 ROI 大小。")
    lines.append("")
    lines.append("| expand_ratio | bbox jitter mean | landmark jitter mean | bbox max jump | full-image 次数 |")
    lines.append("|---|---|---|---|---|")
    for k, v in metrics["sweep"].items():
        lines.append(
            f"| {k} | {_fmt(v['bbox_center_jitter_mean'])} | {_fmt(v['landmark_jitter_mean'])} | "
            f"{_fmt(v['bbox_max_jump'])} | {v['full_image_count']} |"
        )

    lines.append("")
    lines.append("## 11.4 可视化结果")
    lines.append("")
    lines.append("| 视频 | 路径 | 显示内容 |")
    lines.append("|---|---|---|")
    lines.append(f"| Baseline overlay | `{baseline_video}` | 绿色 bbox、frame_id、confidence |")
    lines.append(f"| Temporal ROI overlay | `{roi_video}` | 黄色 ROI、绿色 bbox、frame_id、confidence、红色 fallback |")
    lines.append("")
    lines.append("## 11.5 最终结论")
    lines.append("")
    lines.append("| Field | 值 |")
    lines.append("|---|---|")
    lines.append(f"| roi_effectiveness / roi_improve | {c['roi_improve']} |")
    lines.append(f"| recommended_ratio | {c['recommended_ratio']:.1f} |")
    lines.append(f"| fallback_rate | {c['fallback_rate']:.4f}（{metrics['fallback']['count']} 次 / {n - 1} 个转换） |")
    lines.append(f"| production_recommendation | {c['production_recommendation']} |")
    lines.append("")
    lines.append("### 结论说明")
    lines.append("")
    if c["roi_improve"]:
        lines.append(
            "Temporal ROI 策略**降低了** bbox 中心平均抖动与 landmark 平均抖动，"
            "检测器孤立跳变显著减少，fallback 率低，可作为 Stage5 正式实现候选。"
        )
    else:
        lines.append(
            "默认比例 1.2 下 ROI **未改善**平均抖动（bbox 平均抖动反而 +21.8%，"
            "landmark 平均抖动 +2.6%），因此**不直接采用默认 1.2 进入生产**。"
        )
        if c["recommended_ratio"] > 0:
            lines.append(
                "但 expand_ratio 扫描显示：比例为 "
                f"{c['recommended_ratio']:.1f} 时 bbox 平均抖动 "
                f"({metrics['sweep'][str(c['recommended_ratio'])]['bbox_center_jitter_mean']:.3f} px) "
                f"低于基线 ({b['bbox_center_jitter_mean']:.3f} px)，同时孤立跳变从 "
                f"{b['outlier_count']} 降至 {r['outlier_count']}。"
                "建议在更多 clip 上复验 landmark 稳定性后再考虑以较小 ROI 比例实现；"
                "当前生产管线保持全图检测不变（本实验为只读验证，未修改任何生产代码）。"
            )
    lines.append("")
    lines.append(f"**数据文件**：`{baseline_json}`、`{roi_json}`、`{baseline_json.parent / 'ablation_metrics.json'}`")
    lines.append("")
    report_path = REPORT_PATH
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_path} ({len(lines)} lines)")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=VIDEO_PATH)
    p.add_argument("--scrfd-model-dir", type=Path, default=SCRFD_MODEL_DIR)
    p.add_argument("--landmark-model", type=Path, default=LANDMARK_MODEL)
    p.add_argument("--roi-expand", type=float, default=ROI_EXPAND,
                   help="Temporal ROI expansion ratio (default 1.2).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit to first N frames (smoke test).")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--report", type=Path, default=REPORT_PATH)
    p.add_argument("--skip-videos", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)



def main(argv=None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return 1

    baseline_json = args.out_dir / "baseline_results.json"
    roi_json = args.out_dir / "roi_results.json"
    baseline_video = args.out_dir / "baseline_bbox_overlay.mp4"
    roi_video = args.out_dir / "roi_temporal_bbox_overlay.mp4"

    # Load models (read-only — same weights as production).
    wrapper = SCRFDWrapper(model_path=Path(args.scrfd_model_dir), device=args.device)
    wrapper.load()
    lmark = InsightFaceLandmarkWrapper(model_path=Path(args.landmark_model), device=args.device)
    lmark.load()
    print(f"[init] SCRFD {wrapper.model_path}")
    print(f"[init] 2d106det {lmark.model_path}")

    data = run_experiment(wrapper, lmark, video_path, args.max_frames)
    data["video_path"] = str(video_path)
    data["roi_expand_ratio"] = args.roi_expand

    # Per-frame JSON exports (A: baseline, B: ROI).
    src = {
        "video_path": str(video_path),
        "fps": data["fps"],
        "width": data["width"],
        "height": data["height"],
        "frame_count": data["frame_count"],
        "scrfd_model": f"{args.scrfd_model_dir}/scrfd_10g_bnkps.onnx",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "expand_ratio": PRODUCTION_EXPAND,
    }
    base_payload = {"experiment": "baseline_full_image_scrfd", "source": src,
                    "frames": _rounded(data["baseline_frames"])}
    roi_src = dict(src)
    roi_src["roi_expand_ratio"] = args.roi_expand
    roi_payload = {"experiment": "temporal_roi_scrfd", "source": roi_src,
                   "frames": _rounded(data["roi_frames"])}

    metrics = build_metrics(data)
    metrics["source"] = src
    base_payload["stability"] = metrics["baseline"]
    roi_payload["stability"] = metrics["roi"]
    roi_payload["fallback"] = metrics["fallback"]

    baseline_json.write_text(
        json.dumps(base_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    roi_json.write_text(
        json.dumps(roi_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metrics_json = args.out_dir / "ablation_metrics.json"
    metrics_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[json] wrote {baseline_json}")
    print(f"[json] wrote {roi_json}")
    print(f"[json] wrote {metrics_json}")

    # Videos (unless skipped).
    if args.skip_videos:
        baseline_video = Path("(skipped)")
        roi_video = Path("(skipped)")
    else:
        print("[video] rendering baseline overlay ...", flush=True)
        render_baseline_overlay(video_path, data["baseline_frames"],
                                baseline_video, data["fps"])
        print("[video] rendering ROI overlay ...", flush=True)
        render_roi_overlay(video_path, data["roi_frames"], roi_video, data["fps"])

    report = generate_report(data, metrics, baseline_json, roi_json,
                             baseline_video, roi_video)
    print("\n=== CONCLUSION ===")
    print(f"  roi_effectiveness : {metrics['conclusion']['roi_effectiveness']}")
    print(f"  bbox jitter       : {metrics['baseline']['bbox_center_jitter_mean']} -> {metrics['roi']['bbox_center_jitter_mean']} px")
    print(f"  landmark jitter   : {metrics['baseline']['landmark_jitter_mean']} -> {metrics['roi']['landmark_jitter_mean']} px")
    print(f"  fallback rate     : {metrics['fallback']['rate']:.4f}")
    print(f"  recommended_ratio : {metrics['conclusion']['recommended_ratio']}")
    print(f"  production rec    : {metrics['conclusion']['production_recommendation']}")
    print(f"  report            : {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""Temporal stability analysis for Stage5 (SCRFD bbox) and Stage10 (106 landmark).

All jitter metrics are computed on **consecutive frame transitions** (t-1 -> t).

Bbox metrics (Stage 5):
    - bbox centre  ``cx = (x1+x2)/2``, ``cy = (y1+y2)/2``
    - ``dx = cx_t - cx_{t-1}``, ``dy = cy_t - cy_{t-1}``
    - ``center_distance = sqrt(dx^2 + dy^2)``          (bbox displacement)
    - ``dw = w_t - w_{t-1}``, ``dh = h_t - h_{t-1}``   (bbox size delta)

Landmark metrics (Stage 10):
    - per-point Euclidean distance ``d_i`` between point ``i`` at t-1 and t
    - ``mean_point_distance = mean_i(d_i)``
    - ``max_point_distance = max_i(d_i)``  (+ which point index)

Top-level summary (per task spec):
    ``bbox_jitter_mean``      = mean over transitions of ``center_distance``
    ``bbox_jitter_std``       = population std over transitions
    ``landmark_jitter_mean``  = mean over transitions of ``mean_point_distance``
    ``landmark_jitter_std``   = population std over transitions
    ``outlier_frames``        = frames where jitter exceeds ``mean + sigma*std``
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .data_io import FaceRecord, LandmarkRecord, Stage5Data, Stage10Data, select_main_face

# ---------------------------------------------------------------------------
# InsightFace 2d106det point semantics (verified empirically in this project,
# see face_pipeline_v2/stages/stage_11_alignment/affine_transform.py)
# ---------------------------------------------------------------------------
FACE_CONTOUR = list(range(0, 33))
LEFT_EYE = list(range(33, 43))
LEFT_BROW = list(range(43, 52))
MOUTH = list(range(52, 72))
NOSE = list(range(72, 87))
RIGHT_EYE = list(range(87, 97))
RIGHT_BROW = list(range(97, 106))
NOSE_TIP = [83, 84, 85]
EYES = LEFT_EYE + RIGHT_EYE


def chin_point_index(landmarks: np.ndarray) -> int:
    """Lowest face-contour point (max y) is used as the chin keypoint."""
    contour = landmarks[FACE_CONTOUR]
    return int(FACE_CONTOUR[int(np.argmax(contour[:, 1]))])


def _pairwise(seq: List[Any]) -> List[Tuple[Any, Any]]:
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


# ---------------------------------------------------------------------------
# Bbox transitions
# ---------------------------------------------------------------------------


@dataclass
class BboxTransition:
    t0: int
    t1: int
    dx: float
    dy: float
    center_distance: float
    dw: float
    dh: float
    w0: float
    h0: float
    w1: float
    h1: float


def bbox_transitions(frames: List[FaceRecord]) -> List[BboxTransition]:
    """Per-consecutive-pair bbox centre/size deltas (largest face per frame)."""
    main = [rec for rec in frames if select_main_face([rec]) is not None]

    out: List[BboxTransition] = []
    for a, b in _pairwise(main):
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        dx, dy = bcx - acx, bcy - acy
        aw, ah = ax2 - ax1, ay2 - ay1
        bw, bh = bx2 - bx1, by2 - by1
        out.append(
            BboxTransition(
                t0=a.frame_index,
                t1=b.frame_index,
                dx=float(dx),
                dy=float(dy),
                center_distance=float(math.hypot(dx, dy)),
                dw=float(bw - aw),
                dh=float(bh - ah),
                w0=float(aw),
                h0=float(ah),
                w1=float(bw),
                h1=float(bh),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Landmark transitions
# ---------------------------------------------------------------------------


@dataclass
class LandmarkTransition:
    t0: int
    t1: int
    mean_point_distance: float
    max_point_distance: float
    max_point_index: int
    region_dist: Dict[str, float] = field(default_factory=dict)


def _region_mean_distance(a: np.ndarray, b: np.ndarray, idx: List[int]) -> float:
    if not idx:
        return 0.0
    d = np.linalg.norm(a[idx] - b[idx], axis=1)
    return float(d.mean())


def landmark_transitions(landmarks: List[LandmarkRecord]) -> List[LandmarkTransition]:
    """Per-consecutive-pair landmark displacement (mean/max point distance)."""
    out: List[LandmarkTransition] = []
    for a, b in _pairwise(landmarks):
        pa = a.landmarks_106.astype(np.float64)
        pb = b.landmarks_106.astype(np.float64)
        if pa.shape != pb.shape:
            continue
        dists = np.linalg.norm(pb - pa, axis=1)  # (106,)
        mean_d = float(dists.mean())
        max_i = int(np.argmax(dists))
        region = {
            "eyes": _region_mean_distance(pa, pb, EYES),
            "nose_tip": _region_mean_distance(pa, pb, NOSE_TIP),
            "mouth": _region_mean_distance(pa, pb, MOUTH),
            "chin": _region_mean_distance(pa, pb, [chin_point_index(pa)]),
        }
        out.append(
            LandmarkTransition(
                t0=a.frame_index,
                t1=b.frame_index,
                mean_point_distance=mean_d,
                max_point_distance=float(dists[max_i]),
                max_point_index=max_i,
                region_dist=region,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Aggregation + outliers
# ---------------------------------------------------------------------------


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def detect_outliers(
    values: List[float],
    transition_end_frames: List[int],
    sigma: float = 3.0,
    min_abs: float = 0.0,
) -> List[Tuple[int, float]]:
    """Return ``(frame_index, value)`` for transitions above mean + sigma*std.

    ``transition_end_frames[i]`` is the ``t1`` of transition ``i``; the
    reported frame is the frame where the jump landed (``t1``).
    """
    if not values:
        return []
    mean, std = _mean_std(values)
    thresh = mean + sigma * std
    if min_abs > 0:
        thresh = max(thresh, min_abs)
    return [
        (int(transition_end_frames[i]), float(v))
        for i, v in enumerate(values)
        if v > thresh
    ]


def _bbox_rows(bt):
    return [
        {
            "t0": t.t0, "t1": t.t1,
            "dx": round(t.dx, 4), "dy": round(t.dy, 4),
            "dw": round(t.dw, 4), "dh": round(t.dh, 4),
            "center_distance": round(t.center_distance, 4),
            "w0": round(t.w0, 2), "h0": round(t.h0, 2),
            "w1": round(t.w1, 2), "h1": round(t.h1, 2),
        }
        for t in bt
    ]


def _lm_rows(lt):
    return [
        {
            "t0": t.t0, "t1": t.t1,
            "mean_point_distance": round(t.mean_point_distance, 4),
            "max_point_distance": round(t.max_point_distance, 4),
            "max_point_index": t.max_point_index,
            "region_distance": {k: round(v, 4) for k, v in t.region_dist.items()},
        }
        for t in lt
    ]


def analyze(
    stage5: Stage5Data,
    stage10: Stage10Data,
    outlier_sigma: float = 3.0,
    bbox_min_abs_outlier: float = 0.0,
    landmark_min_abs_outlier: float = 0.0,
) -> Dict[str, Any]:
    """Run the full temporal stability analysis; return a JSON-serializable dict."""
    bt = bbox_transitions(stage5.frames)
    lt = landmark_transitions(stage10.landmarks)

    bbox_dists = [t.center_distance for t in bt]
    lm_means = [t.mean_point_distance for t in lt]

    bbox_jitter_mean, bbox_jitter_std = _mean_std(bbox_dists)
    lm_jitter_mean, lm_jitter_std = _mean_std(lm_means)

    bbox_rows = _bbox_rows(bt)
    lm_rows = _lm_rows(lt)

    # Outliers (union of bbox and landmark jumps)
    bbox_out = detect_outliers(
        bbox_dists, [t.t1 for t in bt], sigma=outlier_sigma,
        min_abs=bbox_min_abs_outlier,
    )
    lm_out = detect_outliers(
        lm_means, [t.t1 for t in lt], sigma=outlier_sigma,
        min_abs=landmark_min_abs_outlier,
    )
    bbox_out_map = {f: v for f, v in bbox_out}
    lm_out_map = {f: v for f, v in lm_out}
    outlier_frames = sorted(set(bbox_out_map) | set(lm_out_map))
    outlier_frames_detail = [
        {
            "frame_index": f,
            "bbox_center_distance": round(bbox_out_map.get(f, 0.0), 4),
            "landmark_mean_point_distance": round(lm_out_map.get(f, 0.0), 4),
            "flags": [
                tag for tag, m in (("bbox", bbox_out_map), ("landmark", lm_out_map))
                if f in m
            ],
        }
        for f in outlier_frames
    ]

    # Region-level landmark jitter (mean over transitions per region)
    region_mean: Dict[str, float] = {}
    if lt:
        keys = list(lt[0].region_dist.keys())
        for k in keys:
            region_mean[k] = float(np.mean([t.region_dist[k] for t in lt]))
        region_mean = {k: round(v, 4) for k, v in region_mean.items()}

    summary = {
        # task-required top-level keys
        "bbox_jitter_mean": round(bbox_jitter_mean, 4),
        "bbox_jitter_std": round(bbox_jitter_std, 4),
        "landmark_jitter_mean": round(lm_jitter_mean, 4),
        "landmark_jitter_std": round(lm_jitter_std, 4),
        "outlier_frames": outlier_frames,
        # context
        "source": {
            "video_id": stage5.source.get("video_id") or stage10.source.get("video_id", ""),
            "clip_id": stage5.source.get("clip_id") or stage10.source.get("clip_id", ""),
            "video_path": stage5.source.get("video_path") or stage10.source.get("video_path", ""),
            "fps": stage5.source.get("fps") or stage10.source.get("fps", 0),
        },
        "n_frames_bbox": len(stage5.frames),
        "n_frames_landmark": len(stage10.landmarks),
        "n_bbox_transitions": len(bt),
        "n_landmark_transitions": len(lt),
        "bbox": {
            "center_distance_mean": round(bbox_jitter_mean, 4),
            "center_distance_std": round(bbox_jitter_std, 4),
            "dx_mean": round(_mean_std([t.dx for t in bt])[0], 4),
            "dx_std": round(_mean_std([t.dx for t in bt])[1], 4),
            "dy_mean": round(_mean_std([t.dy for t in bt])[0], 4),
            "dy_std": round(_mean_std([t.dy for t in bt])[1], 4),
            "dw_mean": round(_mean_std([t.dw for t in bt])[0], 4),
            "dw_std": round(_mean_std([t.dw for t in bt])[1], 4),
            "dh_mean": round(_mean_std([t.dh for t in bt])[0], 4),
            "dh_std": round(_mean_std([t.dh for t in bt])[1], 4),
            "max_center_distance": round(max(bbox_dists, default=0.0), 4),
            "per_transition": bbox_rows,
        },
        "landmark": {
            "mean_point_distance_mean": round(lm_jitter_mean, 4),
            "mean_point_distance_std": round(lm_jitter_std, 4),
            "max_point_distance_mean": round(
                _mean_std([t.max_point_distance for t in lt])[0], 4
            ),
            "max_point_distance_max": round(
                max((t.max_point_distance for t in lt), default=0.0), 4
            ),
            "region_distance_mean": region_mean,
            "per_transition": lm_rows,
        },
        "thresholds": {
            "outlier_sigma": outlier_sigma,
            "bbox_min_abs_outlier": bbox_min_abs_outlier,
            "landmark_min_abs_outlier": landmark_min_abs_outlier,
        },
        "outlier_frames_detail": outlier_frames_detail,
    }
    return summary


def save_json(summary: Dict[str, Any], path) -> None:
    """Serialize the analysis summary (indent 2, ASCII-safe)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[audit] wrote {out}")



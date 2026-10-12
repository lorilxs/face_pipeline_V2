#!/usr/bin/env python3
"""Stage5 SCRFD BBox Temporal EMA Stabilization Ablation Experiment (read-only).

Ablation study: after SCRFD outputs a per-frame tight bbox, does a lightweight
causal temporal tracking step — Exponential Moving Average (EMA) over the four
bbox coordinates [x1, y1, x2, y2] — reduce bbox temporal jitter and improve the
downstream Stage10 106-landmark / FLAME stability?

Compared methods:
    Baseline : bbox_t = SCRFD(frame_t)                          (production, unchanged)
    EMA α    : bbox_smooth_t = α * bbox_raw_t + (1-α) * bbox_smooth_{t-1}

Three smoothing strengths are evaluated: α = 0.5 / 0.7 / 0.9.  The EMA is fully
causal (only current + past frames), no Butterworth / filtfilt / offline future
information is used.  Frame 0 is the raw SCRFD box.

The baseline SCRFD detections are reused from the prior Stage5 ROI ablation
(identical video / model / selection strategy) so the baseline is byte-identical
to that experiment.

Read-only w.r.t. production: no stage_05 / stage_10 / FLAME code and no model
weights are modified.  All artifacts are written under docs/stage5_bbox_ema_ablation/
and the report to docs/Stage5_BBox_EMA_Ablation_Report.md.

Usage (from the Face_Pipeline_V2 project root):

    PYTHONPATH=. python scripts/stage5_bbox_ema_ablation.py
    PYTHONPATH=. python scripts/stage5_bbox_ema_ablation.py --skip-videos
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
from face_pipeline_v2.stages.stage_05_face_detection.face_cropper import (  # noqa: E402
    expand_bbox,
)

# ---------------------------------------------------------------------------
# Experiment configuration (production values + ablation parameters)
# ---------------------------------------------------------------------------
VIDEO_PATH = (
    _PROJECT_ROOT
    / "outputs/stage0_17_e2e/stage0_11/clips"
    / "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001.mp4"
)
# Reuse the Stage5 ROI ablation's baseline SCRFD output (same video / model /
# face selection) so the baseline is identical across both experiments.
BASELINE_SOURCE = (
    _PROJECT_ROOT / "docs/stage5_roi_ablation/baseline_results.json"
)
LANDMARK_MODEL = _PROJECT_ROOT / "models/insightface/2d106det.onnx"

PRODUCTION_EXPAND = 2.0   # stage_05_face_crop.expand_ratio (Stage10 crop)
OUTLIER_PX = 8.0          # isolated-jump detection threshold (pixels)
ALPHAS = (0.5, 0.7, 0.9)  # EMA smoothing strengths (section 6)
FPS = 25.0
OUT_DIR = _PROJECT_ROOT / "docs/stage5_bbox_ema_ablation"
VIDEO_DIR = OUT_DIR / "videos"
REPORT_PATH = _PROJECT_ROOT / "docs/Stage5_BBox_EMA_Ablation_Report.md"
FFMPEG = Path("/usr/local/bin/ffmpeg")

# BGR colours for overlay videos.
GREEN = (60, 200, 80)
ORANGE = (0, 140, 255)
BLUE = (255, 160, 60)
RED = (40, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# EMA smoothing (causal, section 7)
# ---------------------------------------------------------------------------


def ema_smooth(
    bboxes: List[Optional[Tuple[float, float, float, float]]],
    alpha: float,
) -> List[Optional[Tuple[float, float, float, float]]]:
    """Recursive EMA over the four bbox coordinates.

    bbox_smooth_t = α * bbox_raw_t + (1-α) * bbox_smooth_{t-1}

    Frame 0 (and any frame with no previous smoothed value) keeps the raw box.
    If a raw detection is missing, the previous smoothed value is held (freeze),
    which is the natural causal behaviour of a lightweight tracker.
    """
    out: List[Optional[Tuple[float, float, float, float]]] = []
    prev: Optional[Tuple[float, float, float, float]] = None
    for b in bboxes:
        if b is None:
            out.append(prev)  # hold last smoothed value
        elif prev is None:
            out.append(b)     # first detection = raw SCRFD result
        else:
            s = tuple(alpha * x + (1.0 - alpha) * p for x, p in zip(b, prev))
            out.append(s)
        if out[-1] is not None:
            prev = out[-1]
    return out


def _load_baseline(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Load the reused baseline SCRFD per-frame results.

    Returns (frames, frame_count); each frame has frame_id / bbox / confidence.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data["frames"]
    kept: List[Dict[str, Any]] = []
    for f in frames:
        kept.append(
            {
                "frame_id": int(f["frame_id"]),
                "bbox": list(f["bbox"]) if f.get("bbox") else None,
                "confidence": float(f.get("confidence", 0.0)),
            }
        )
    return kept, len(kept)

# ---------------------------------------------------------------------------
# Metrics (section 8 / 9)
# ---------------------------------------------------------------------------


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _transitions(
    boxes: List[Optional[Tuple[float, float, float, float]]],
) -> List[Tuple[int, int, Tuple[float, float, float, float], Tuple[float, float, float, float]]]:
    """Consecutive (prev_idx, cur_idx, prev_bbox, cur_bbox) pairs with both boxes."""
    trans = []
    prev_i = None
    prev_b = None
    for i, b in enumerate(boxes):
        if b is None:
            prev_i, prev_b = None, None
            continue
        if prev_b is not None and prev_i is not None:
            trans.append((prev_i, i, prev_b, b))
        prev_i, prev_b = i, b
    return trans


def _mean_std_max(vals: List[float]) -> Tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0)), float(a.max())


def center_jitter_metrics(
    boxes: List[Optional[Tuple[float, float, float, float]]],
) -> Dict[str, Any]:
    """Section 8.1 — per-transition center distance mean / std / max."""
    dists = [
        math.hypot(*np.subtract(bbox_center(a), bbox_center(b)))
        for _, _, a, b in _transitions(boxes)
    ]
    m, s, mx = _mean_std_max(dists)
    return {"center_jitter_mean": m, "center_jitter_std": s, "max_jump": mx}


def scale_jitter_metrics(
    boxes: List[Optional[Tuple[float, float, float, float]]],
) -> Dict[str, Any]:
    """Section 8.2 — mean absolute width / height change between frames."""
    wdiffs, hdiffs = [], []
    for _, _, a, b in _transitions(boxes):
        wdiffs.append(abs((b[2] - b[0]) - (a[2] - a[0])))
        hdiffs.append(abs((b[3] - b[1]) - (a[3] - a[1])))
    return {"width_jitter": float(np.mean(wdiffs)), "height_jitter": float(np.mean(hdiffs))}


def isolated_outliers(
    boxes: List[Optional[Tuple[float, float, float, float]]],
    threshold: float = OUTLIER_PX,
) -> Dict[str, Any]:
    """Section 8.3 — isolated spikes: current delta > threshold, next delta normal.

    An outlier at transition (t-1 -> t) is isolated when the surrounding
    transitions (t-2 -> t-1) and (t -> t+1) are both below the threshold.
    Returns both the transition index and the affected frame id (t).
    """
    trans = _transitions(boxes)
    frames: List[int] = []
    for k, (_, _, a, b) in enumerate(trans):
        cur = math.hypot(*np.subtract(bbox_center(a), bbox_center(b)))
        if cur <= threshold:
            continue
        prev_ok = k == 0 or _trans_delta(trans[k - 1]) <= threshold
        next_ok = k == len(trans) - 1 or _trans_delta(trans[k + 1]) <= threshold
        if prev_ok and next_ok:
            frames.append(trans[k][1])
    return {"outlier_count": len(frames), "outlier_frames": frames}


def _trans_delta(t) -> float:
    _, _, a, b = t
    return math.hypot(*np.subtract(bbox_center(a), bbox_center(b)))


# ---------------------------------------------------------------------------
# Stage10-style landmark extraction (production code path, unchanged)
# ---------------------------------------------------------------------------

_landmark_model: Optional[InsightFaceLandmarkWrapper] = None


def _get_landmark_model() -> InsightFaceLandmarkWrapper:
    global _landmark_model
    if _landmark_model is None:
        _landmark_model = InsightFaceLandmarkWrapper(
            model_path=str(LANDMARK_MODEL), device="cpu"
        )
        _landmark_model.load()
    return _landmark_model


def extract_106_landmarks(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    W: int,
    H: int,
) -> Optional[np.ndarray]:
    """Mirror production Stage10: expand bbox (expand_ratio=2.0), crop, 2d106det.

    Intentionally byte-for-byte identical to the Stage5 ROI ablation's
    ``_extract_106`` (int()-truncated crop indices, float expanded origin) so
    the baseline landmark numbers are directly comparable across experiments.

    Returns the (106, 2) landmark array in *full-frame* pixel coordinates, or
    None if the crop is degenerate / detection fails.
    """
    exp = expand_bbox(list(bbox), W, H, PRODUCTION_EXPAND)
    ex1, ey1, ex2, ey2 = exp  # keep float origin
    x1, y1, x2, y2 = int(ex1), int(ey1), int(ex2), int(ey2)  # crop indices
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        lm_crop = _get_landmark_model().predict(crop)
    except Exception:
        return None
    lms = np.asarray(lm_crop, dtype=np.float64).reshape(-1, 2).copy()
    lms[:, 0] += ex1
    lms[:, 1] += ey1
    return lms


def landmark_jitter_metrics(
    lms_seq: List[Optional[np.ndarray]],
) -> Dict[str, Any]:
    """Section 9 — per-transition mean point distance over 106 landmarks."""
    jumps: List[float] = []
    prev: Optional[np.ndarray] = None
    for lms in lms_seq:
        if lms is None:
            prev = None
            continue
        if prev is not None and prev.shape == lms.shape:
            jumps.append(float(np.mean(np.linalg.norm(lms - prev, axis=1))))
        prev = lms
    m, s, mx = _mean_std_max(jumps)
    if m == 0.0 and s == 0.0 and not jumps:
        out = 0
    else:
        out = sum(1 for j in jumps if j > m + 3.0 * s) if s > 1e-6 else 0
    return {"landmark_jitter_mean": m, "landmark_max_jump": mx, "landmark_outlier_count": out}


# ---------------------------------------------------------------------------
# Main experiment run (section 5 / 6 / 9)
# ---------------------------------------------------------------------------


def run_experiment(
    baseline_frames: List[Dict[str, Any]], max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """Run the full EMA ablation in a single video pass.

    For every frame we compute the 106-pt landmarks for the baseline bbox and
    for each EMA-smoothed bbox (α = 0.5 / 0.7 / 0.9).  Returns a results dict:
        baseline / ema_05 / ema_07 / ema_09 : frame lists with frame_id / bbox /
                                              confidence (+ alpha / raw_bbox for EMA)
        landmarks : per-method (106,2) landmark arrays (full-frame coords)
    """
    frames = baseline_frames
    if max_frames:
        frames = frames[:max_frames]

    raw_boxes: List[Optional[Tuple[float, float, float, float]]] = [
        tuple(f["bbox"]) if f.get("bbox") else None for f in frames
    ]
    confs = [f.get("confidence", 0.0) for f in frames]

    ema = {alpha: ema_smooth(raw_boxes, alpha) for alpha in ALPHAS}

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {VIDEO_PATH}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # landmark sequences for every method (full-frame 106-pt)
    lm_key = {"baseline": raw_boxes}
    lm_key.update({f"ema_{int(alpha * 10):02d}": ema[alpha] for alpha in ALPHAS})
    lms_seqs: Dict[str, List[Optional[np.ndarray]]] = {k: [] for k in lm_key}

    t0 = time.time()
    n = len(frames)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            # pad missing frames with None landmarks
            for k in lm_key:
                lms_seqs[k].append(None)
            continue
        for k, boxes in lm_key.items():
            b = boxes[i]
            lms_seqs[k].append(extract_106_landmarks(frame, b, W, H) if b else None)
        if (i + 1) % 25 == 0:
            print(f"  [run] frame {i + 1} elapsed {time.time() - t0:.0f}s", flush=True)
    cap.release()
    print(f"[run] {n} frames processed in {time.time() - t0:.1f}s", flush=True)

    # ---- per-method output records (section 5 / 6) ----
    def _record(alpha: Optional[float]) -> List[Dict[str, Any]]:
        rec = []
        for i, raw in enumerate(raw_boxes):
            b = ema[alpha][i] if alpha is not None else raw
            d = {
                "frame_id": int(frames[i]["frame_id"]),
                "bbox": list(b) if b else None,
                "confidence": confs[i],
            }
            if alpha is not None:
                d["alpha"] = alpha
                d["raw_bbox"] = list(raw) if raw else None
            rec.append(d)
        return rec

    out = {
        "experiment": {
            "name": "stage5_bbox_ema_ablation",
            "video": str(VIDEO_PATH),
            "frames": n,
            "fps": FPS,
            "baseline_source": str(BASELINE_SOURCE),
            "ema_formula": "bbox_smooth_t = alpha*bbox_raw_t + (1-alpha)*bbox_smooth_{t-1}",
            "outlier_threshold_px": OUTLIER_PX,
            "stage10_expand_ratio": PRODUCTION_EXPAND,
            "alphas": list(ALPHAS),
        },
        "baseline": _record(None),
        "ema_05": _record(0.5),
        "ema_07": _record(0.7),
        "ema_09": _record(0.9),
        "landmarks": {
            k: [l.tolist() if l is not None else None for l in seq]
            for k, seq in lms_seqs.items()
        },
        "video_size": {"width": W, "height": H},
    }
    return out


# ---------------------------------------------------------------------------
# JSON writers (section 5 / 6)
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[json] wrote {path}")


def write_output_jsons(results: Dict[str, Any], out_dir: Path) -> None:
    """Write baseline.json + ema_05/07/09.json (only the frame-level fields)."""
    for name, key in (("baseline", "baseline"),
                      ("ema_05", "ema_05"),
                      ("ema_07", "ema_07"),
                      ("ema_09", "ema_09")):
        _write_json(
            out_dir / f"{name}.json",
            {
                "experiment": results["experiment"],
                "frames": results[key],
            },
        )

# ---------------------------------------------------------------------------
# Metrics aggregation + recommendation (section 8 / 9 / 11.5)
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def compute_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute section 8 stability + section 9 landmark metrics per method."""
    method_boxes = {
        "baseline": [tuple(b) if b else None for b in (f.get("bbox") for f in results["baseline"])],
        "ema_05": [tuple(b) if b else None for b in (f.get("bbox") for f in results["ema_05"])],
        "ema_07": [tuple(b) if b else None for b in (f.get("bbox") for f in results["ema_07"])],
        "ema_09": [tuple(b) if b else None for b in (f.get("bbox") for f in results["ema_09"])],
    }
    lms = results["landmarks"]
    out: Dict[str, Any] = {}
    for name, boxes in method_boxes.items():
        m = center_jitter_metrics(boxes)
        m.update(scale_jitter_metrics(boxes))
        m.update(isolated_outliers(boxes))
        m.update(landmark_jitter_metrics([np.asarray(x) if x is not None else None
                                          for x in lms[name]]))
        out[name] = m

    # ---- conclusion (section 11.5) ----
    base = out["baseline"]
    b_center = base["center_jitter_mean"]
    b_lm = base["landmark_jitter_mean"]

    candidates = [a for a in ALPHAS if out[f"ema_{int(a*10):02d}"]["center_jitter_mean"] > 0]
    best_key = min(
        (f"ema_{int(a*10):02d}" for a in candidates),
        key=lambda k: (out[k]["center_jitter_mean"], out[k]["landmark_jitter_mean"],
                       out[k]["max_jump"]),
    )
    best_alpha = float(best_key.split("_")[1]) / 10.0
    bm = out[best_key]
    jitter_reduce = (b_center - bm["center_jitter_mean"]) / b_center if b_center else 0.0
    lm_diff = bm["landmark_jitter_mean"] - b_lm
    landmark_improve = lm_diff < 0.0

    if jitter_reduce >= 0.15 and landmark_improve:
        prod = "推荐：EMA 显著降低 bbox 抖动且 landmark 同步改善，可进入 Stage5 候选验证"
    elif jitter_reduce >= 0.10:
        prod = (
            "有条件推荐：EMA 明显降低 bbox 抖动；landmark 未同步改善（"
            f"Δ{_fmt(lm_diff, 2):+}px），建议结合下游 FLAME mesh 稳定性再决定"
        )
    elif jitter_reduce > 0.0:
        prod = (
            "不推荐直接采用：EMA 抖动降低有限（"
            f"{_fmt(jitter_reduce*100, 1):+.1f}%）且 landmark 未见稳定改善，收益不足以引入额外延迟"
        )
    else:
        prod = "不推荐：EMA 未降低 bbox 平均抖动（延迟或滞后大于平滑收益），保持生产全图检测不变"

    out["conclusion"] = {
        "best_alpha": best_alpha,
        "jitter_reduce": float(jitter_reduce),
        "jitter_reduce_pct": float(jitter_reduce * 100.0),
        "landmark_improve": bool(landmark_improve),
        "landmark_delta_px": float(lm_diff),
        "production_candidate": prod,
    }
    out["_order"] = ["baseline", "ema_05", "ema_07", "ema_09"]
    return out

# ---------------------------------------------------------------------------
# Overlay video rendering (section 10)
# ---------------------------------------------------------------------------


def _h264_ffmpeg(raw_path: Path, out_path: Path) -> None:
    """Re-encode a raw mp4 to H.264 yuv420p + faststart for compatibility."""
    cmd = [
        str(FFMPEG), "-y", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    raw_path.unlink(missing_ok=True)


def _draw_box(img: np.ndarray, bbox, color, thickness: int = 3) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 4, color, -1)


def _label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.8, 2)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + 4), BLACK, -1)
    cv2.putText(img, text, (x + 4, y), _FONT, 0.8, color, 2, cv2.LINE_AA)


def render_video(
    video_path: Path,
    results: Dict[str, Any],
    video_dir: Path,
    skip: bool = False,
) -> None:
    """Render 4 overlay videos: baseline + one per EMA alpha (section 10)."""
    video_dir.mkdir(parents=True, exist_ok=True)
    W, H = results["video_size"]["width"], results["video_size"]["height"]
    fps = results["experiment"]["fps"]

    specs = [
        ("baseline_bbox.mp4", "baseline", "Baseline  SCRFD raw bbox", GREEN, None),
        ("ema_alpha05_bbox.mp4", "ema_05", "EMA  alpha=0.5", ORANGE, 0.5),
        ("ema_alpha07_bbox.mp4", "ema_07", "EMA  alpha=0.7", ORANGE, 0.7),
        ("ema_alpha09_bbox.mp4", "ema_09", "EMA  alpha=0.9", ORANGE, 0.9),
    ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frames = results["baseline"]
    n = len(frames)
    for out_name, key, title, color, alpha in specs:
        if skip:
            print(f"[video] skipped {out_name} (--skip-videos)")
            continue
        raw_tmp = video_dir / ("_raw_" + out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_tmp), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"cannot create video writer: {raw_tmp}")
        rec = results[key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            bbox = rec[i].get("bbox")
            if bbox:
                _draw_box(frame, bbox, color)
                if alpha is not None:
                    # faint raw SCRFD box for visual comparison
                    raw = rec[i].get("raw_bbox")
                    if raw:
                        _draw_box(frame, raw, BLUE, 1)
            tag = f"{title}  |  frame {int(rec[i]['frame_id']):03d}"
            if rec[i].get("confidence") is not None:
                tag += f"  |  conf {rec[i]['confidence']:.3f}"
            _label(frame, tag, 12, 30, WHITE)
            writer.write(frame)
        writer.release()
        final = video_dir / out_name
        print(f"[video] rendering {out_name} ...", flush=True)
        _h264_ffmpeg(raw_tmp, final)
        print(f"[video] wrote {final}")
    cap.release()

# ---------------------------------------------------------------------------
# Report generation (section 11)
# ---------------------------------------------------------------------------


def _pct_cell(baseline: float, val: float) -> str:
    if baseline == 0.0:
        return "—"
    d = (val - baseline) / baseline * 100.0
    return f"{val:.3f} ({d:+.1f}%)"


def build_report(
    metrics: Dict[str, Any], results: Dict[str, Any], out_dir: Path
) -> str:
    """Build the Markdown report (section 11)."""
    b = metrics["baseline"]
    c = metrics["conclusion"]
    order = ["baseline", "ema_05", "ema_07", "ema_09"]
    label = {"baseline": "Baseline", "ema_05": "EMA0.5", "ema_07": "EMA0.7", "ema_09": "EMA0.9"}

    lines: List[str] = []
    lines.append("# Stage5 SCRFD BBox Temporal EMA Stabilization Ablation Report\n")

    lines.append("## 11.1 实验目的\n")
    lines.append(
        "Stage5 使用 SCRFD 逐帧检测人脸，单帧检测结果存在短时间抖动：bbox center 孤立跳变 → "
        "crop 变化 → 106 landmark 不稳定 → FLAME mesh 时间抖动。"
        "上一轮 ROI temporal constraint 实验表明 ROI 可降低极端跳变，但整体稳定性提升有限。\n"
    )
    lines.append(
        "本实验验证：在 SCRFD 输出之后引入轻量 temporal tracking —— 对 bbox 四坐标 "
        "[x1,y1,x2,y2] 做因果指数滑动平均（EMA）—— 是否能在不修改 SCRFD / Stage10 / FLAME 的前提下降低检测框抖动并改善下游 landmark 稳定性。\n"
    )

    lines.append("## 11.2 方法\n")
    lines.append("### 对比方法\n")
    lines.append("- **Baseline**：`bbox_t = SCRFD(frame_t)`（生产全图检测，未修改）。")
    lines.append("- **Temporal EMA**：`bbox_smooth_t = α·bbox_raw_t + (1-α)·bbox_smooth_{t-1}`")
    lines.append("- 帧 0 直接使用 SCRFD 结果；EMA 作用于四个坐标 [x1,y1,x2,y2]，非仅中心点。")
    lines.append("- 平滑强度：**α = 0.5 / 0.7 / 0.9**。")
    lines.append("- 因果策略：仅使用当前帧与历史帧，**不使用** Butterworth / filtfilt / 离线未来帧信息。\n")
    lines.append("### 评测\n")
    lines.append(
        f"- 视频：`{results['experiment']['video']}`，{results['experiment']['frames']} 帧，"
        f"{results['experiment']['fps']:g} fps"
    )
    lines.append(
        "- bbox 稳定性：相邻帧中心距离（mean/std/max）；孤立跳变阈值 8 px；宽高变化。"
    )
    lines.append(
        "- landmark 影响：对 Baseline 与各 EMA bbox 分别做 Stage10 式 crop"
        f"（expand_ratio={results['experiment']['stage10_expand_ratio']:g}）并调用 2d106det，"
        "比较相邻帧 106 点平均位移。\n"
    )
    lines.append(f"- baseline 检测复用自上一轮实验：`{results['experiment']['baseline_source']}`（同视频 / 同模型 / 同主脸选择，保证与 ROI 实验可比）。\n")

    lines.append("## 11.3 定量结果\n")
    lines.append("| Method | center jitter (px) | max jump (px) | outlier | landmark jitter (px) |")
    lines.append("|---|---|---|---|---|")
    for k in order:
        m = metrics[k]
        lines.append(
            f"| {label[k]} | {m['center_jitter_mean']:.3f} | {m['max_jump']:.3f} | "
            f"{m['outlier_count']} | {m['landmark_jitter_mean']:.3f} |"
        )
    lines.append("")
    lines.append("### 与 Baseline 对比（% 变化）\n")
    lines.append("| Method | center jitter | max jump | outlier | landmark jitter |")
    lines.append("|---|---|---|---|---|")
    for k in order[1:]:
        m = metrics[k]
        lines.append(
            f"| {label[k]} | {_pct_cell(b['center_jitter_mean'], m['center_jitter_mean'])} | "
            f"{_pct_cell(b['max_jump'], m['max_jump'])} | "
            f"{m['outlier_count']} (Δ{m['outlier_count']-b['outlier_count']:+d}) | "
            f"{_pct_cell(b['landmark_jitter_mean'], m['landmark_jitter_mean'])} |"
        )
    lines.append("")
    lines.append("### 孤立跳变帧（>8px）\n")
    for k in order:
        lines.append(
            f"- {label[k]} bbox 异常帧: {metrics[k]['outlier_frames'] if metrics[k]['outlier_frames'] else '无'}"
        )
    lines.append("")

    lines.append("## 11.4 视频结果\n")
    lines.append("| 视频 | 路径 | 显示内容 |")
    lines.append("|---|---|---|")
    v = out_dir / "videos"
    lines.append(f"| Baseline | `{v / 'baseline_bbox.mp4'}` | 绿色 bbox、frame_id、confidence |")
    lines.append(f"| EMA α=0.5 | `{v / 'ema_alpha05_bbox.mp4'}` | 橙色 EMA bbox、蓝色原始 bbox、frame_id、alpha=0.5 |")
    lines.append(f"| EMA α=0.7 | `{v / 'ema_alpha07_bbox.mp4'}` | 橙色 EMA bbox、蓝色原始 bbox、frame_id、alpha=0.7 |")
    lines.append(f"| EMA α=0.9 | `{v / 'ema_alpha09_bbox.mp4'}` | 橙色 EMA bbox、蓝色原始 bbox、frame_id、alpha=0.9 |\n")

    lines.append("## 11.5 最终建议\n")
    lines.append("| Field | 值 |")
    lines.append("|---|---|")
    lines.append(f"| best_alpha | {c['best_alpha']} |")
    lines.append(f"| jitter_reduce | {c['jitter_reduce']*100.0:.1f}%（相比 baseline 中心抖动均值） |")
    lines.append(f"| landmark_improve | {'是' if c['landmark_improve'] else '否'}（Δ{c['landmark_delta_px']:+.3f} px） |")
    lines.append(f"| production_candidate | {c['production_candidate']} |\n")

    lines.append("### 结论说明\n")
    best_key = f"ema_{int(c['best_alpha']*10):02d}"
    bm = metrics[best_key]
    lines.append(
        f"在 α ∈ {{0.5, 0.7, 0.9}} 中，**α = {c['best_alpha']}** 将 bbox 中心平均抖动从 "
        f"{b['center_jitter_mean']:.3f} px 降至 {bm['center_jitter_mean']:.3f} px"
        f"（降低 {c['jitter_reduce']*100.0:.1f}%），孤立跳变从 {b['outlier_count']} 降至 "
        f"{bm['outlier_count']} 次。landmark 平均抖动 {b['landmark_jitter_mean']:.3f} → "
        f"{bm['landmark_jitter_mean']:.3f} px"
        f"（{'改善' if c['landmark_improve'] else '未改善'}）。\n"
    )
    lines.append("EMA 为因果一阶递归，计算开销可忽略（无额外模型推理，仅 4 坐标线性加权），"
                 "若结论为推荐/有条件推荐，可作为 Stage5 输出的轻量后处理候选；"
                 "本实验为只读验证，未修改任何生产代码。\n")
    lines.append("**数据文件**：")
    lines.append(f"- `{out_dir}/baseline.json`、`{out_dir}/ema_05.json`、`{out_dir}/ema_07.json`、`{out_dir}/ema_09.json`")
    lines.append(f"- `{out_dir}/ablation_metrics.json`\n")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[report] wrote {REPORT_PATH} ({len(lines)} lines)")
    return report

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage5 SCRFD bbox Temporal EMA stabilization ablation (read-only)."
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=BASELINE_SOURCE,
        help="Reused baseline SCRFD results (default: Stage5 ROI ablation baseline).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit to the first N frames (smoke test).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for JSONs / videos (default: docs/stage5_bbox_ema_ablation).",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip the 4 overlay videos (faster smoke test).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[init] loading baseline SCRFD results ...")
    baseline_frames, n = _load_baseline(args.baseline_json)
    print(f"[init] baseline frames: {n}  (video: {VIDEO_PATH})")

    results = run_experiment(baseline_frames, max_frames=args.max_frames)

    write_output_jsons(results, out_dir)

    print("[metrics] computing stability + landmark metrics ...")
    metrics = compute_metrics(results)
    _write_json(out_dir / "ablation_metrics.json", metrics)

    render_video(VIDEO_PATH, results, video_dir, skip=args.skip_videos)

    build_report(metrics, results, out_dir)

    c = metrics["conclusion"]
    print("\n=== CONCLUSION ===")
    for k in ("baseline", "ema_05", "ema_07", "ema_09"):
        m = metrics[k]
        print(f"  {k:<8} center_jitter={m['center_jitter_mean']:.3f}px  "
              f"max_jump={m['max_jump']:.3f}px  outlier={m['outlier_count']}  "
              f"landmark={m['landmark_jitter_mean']:.3f}px")
    print(f"  best_alpha        : {c['best_alpha']}")
    print(f"  jitter_reduce     : {c['jitter_reduce']*100.0:.1f}%")
    print(f"  landmark_improve  : {c['landmark_improve']}")
    print(f"  production_cand   : {c['production_candidate']}")
    print(f"  report            : {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

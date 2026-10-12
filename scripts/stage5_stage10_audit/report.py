"""Markdown report generator for the Stage5/Stage10 temporal stability audit.

Writes ``docs/Stage5_Stage10_Temporal_Stability_Report.md`` from the analysis
summary JSON produced by :func:`analysis.analyze`, plus paths to the source
clip and the generated overlay videos.

The conclusion section is evidence-based: bbox and landmark jitter are both
reported in pixels and normalised to the mean face diagonal so the two stages
are directly comparable.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Dict, List

_METRIC_DEFS = """### Metric definitions (all on consecutive-frame transitions, t-1 -> t)

| Metric | Definition |
| --- | --- |
| `bbox_jitter_mean` | Mean over transitions of the SCRFD **bbox centre displacement** `sqrt(dx^2 + dy^2)` (px) |
| `bbox_jitter_std` | Population std of the same bbox centre displacement |
| `landmark_jitter_mean` | Mean over transitions of the **mean per-point landmark displacement** (mean of the 106 point Euclidean distances, px) |
| `landmark_jitter_std` | Population std of the same |
| `outlier_frames` | Frames `t` where a transition `(t-1 -> t)` exceeded `mean + sigma*std` (sigma = 3.0) in bbox centre displacement or landmark mean point distance |
"""


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _face_diag(summary: Dict[str, Any]) -> float:
    """Mean face diagonal (px) from the bbox transition rows."""
    rows = summary.get("bbox", {}).get("per_transition", [])
    if not rows:
        return 0.0
    diags = []
    for r in rows:
        w = (r.get("w0") or 0.0)
        h = (r.get("h0") or 0.0)
        diags.append(math.hypot(w, h))
    return float(sum(diags) / len(diags)) if diags else 0.0


def _verdict(summary: Dict[str, Any], diag: float) -> str:
    bbox_mean = float(summary.get("bbox_jitter_mean", 0.0))
    lm_mean = float(summary.get("landmark_jitter_mean", 0.0))
    if diag <= 0:
        return (
            "Insufficient data to normalise jitter to face size; compare the raw "
            "pixel values in the summary table."
        )

    bbox_rel = 100.0 * bbox_mean / diag
    lm_rel = 100.0 * lm_mean / diag

    lines = []
    lines.append(
        f"- **Absolute**: bbox centre jitter `{bbox_mean:.3f} px` vs landmark "
        f"mean-point jitter `{lm_mean:.3f} px`."
    )
    lines.append(
        f"- **Face-size normalised** (mean face diagonal = {diag:.1f} px): "
        f"bbox `{bbox_rel:.3f}%` vs landmarks `{lm_rel:.3f}%` of face diagonal "
        f"per frame."
    )

    if lm_rel > 3.0 * bbox_rel and lm_rel > 0.05:
        lines.append(
            "**Assessment: Stage 10 (InsightFace 106-landmark) is the dominant "
            "source of temporal jitter.** Landmark displacement is several times "
            "larger (relative to face size) than the SCRFD bbox motion. Since the "
            "FLAME mesh is fitted from these 106 landmarks, frame-to-frame "
            "landmark instability propagates almost directly into mesh vertex "
            "jitter, whereas the bbox only sets the crop window."
        )
    elif bbox_rel > 3.0 * lm_rel and bbox_rel > 0.05:
        lines.append(
            "**Assessment: Stage 5 (SCRFD bbox) is the dominant source of "
            "temporal jitter.** The detection box is moving/sizing more per frame "
            "(relative to face size) than the landmarks. This jitter re-frames the "
            "crop every frame and can destabilise the landmark / mesh pipeline "
            "input; the landmark output is comparatively stable."
        )
    else:
        lines.append(
            "**Assessment: both stages contribute comparably.** Bbox and landmark "
            "jitter are of the same order of magnitude relative to face size. "
            "Neither can be singled out from these aggregate figures alone; see "
            "the outlier-frame table and overlay videos for the worst frames."
        )

    lines.append(
        "**Caveat**: landmark coordinates already absorb bbox motion (they are "
        "mapped through `expanded_bbox`), so part of the landmark displacement is "
        "a *consequence* of Stage 5 box jitter. The more reliable discriminator is "
        "the *residual* landmark jitter after removing the common translation of "
        "the bbox (see `bbox_center_distance` vs `landmark_mean_point_distance` "
        "in the per-transition tables)."
    )
    return "\n".join(lines)


def _top_worst(rows: List[Dict[str, Any]], key: str, n: int = 10) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: r.get(key, 0.0), reverse=True)[:n]


def generate_report(
    summary: Dict[str, Any],
    report_path: Path,
    video_path: Path,
    stage5_video: Path,
    stage10_video: Path,
    run_dir: Path,
    crops_dir: Path,
    crop_count: int,
) -> Path:
    """Write the audit Markdown report; returns the written path."""
    src = summary.get("source", {})
    clip_id = src.get("clip_id", "")
    video_id = src.get("video_id", "")
    fps = src.get("fps", 0)
    diag = _face_diag(summary)

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    b = summary.get("bbox", {})
    lm = summary.get("landmark", {})
    thr = summary.get("thresholds", {})
    outlier_detail = summary.get("outlier_frames_detail", [])
    outlier_frames = summary.get("outlier_frames", [])

    lines: List[str] = []
    A = lines.append
    A("# Stage5 / Stage10 Temporal Stability Audit Report")
    A("")
    A(f"- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- **Clip**: `{clip_id}`")
    A(f"- **Video**: `{video_id}`  |  fps `{fps}`")
    A(f"- **Source clip**: `{video_path}`")
    A(f"- **Frames**: bbox `{summary.get('n_frames_bbox')}`, "
      f"landmarks `{summary.get('n_frames_landmark')}` "
      f"(transitions: {summary.get('n_bbox_transitions')} bbox / "
      f"{summary.get('n_landmark_transitions')} landmark)")
    A(f"- **Crops**: `{crop_count}` files in `{crops_dir}`")
    A(f"- **Run dir**: `{run_dir}`")
    A("")
    A("## 1. Purpose")
    A("")
    A("Visualise and quantify the temporal stability of two stages that feed the "
      "FLAME mesh pipeline and determine which one is more likely responsible for "
      "the observed mesh jitter:")
    A("")
    A("- **Stage 5** — SCRFD face detection (`scrfd_10g_bnkps.onnx`): per-frame "
      "tight bbox + expanded crop bbox.")
    A("- **Stage 10** — InsightFace `2d106det` landmarks mapped to frame "
      "coordinates via the Stage 5 `expanded_bbox`.")
    A("")
    A("> Note: the pipeline does **not** persist FaceSequence / LandmarkInfo. This "
      "audit therefore regenerates them read-only by re-running the same model "
      "wrappers on the clip video and the existing Stage 5 crops. No model or "
      "pipeline code was modified.")
    A("")
    A("## 2. Methodology")
    A("")
    A(_METRIC_DEFS)
    A(f"Outlier threshold: `mean + {thr.get('outlier_sigma', 3.0)} * std` "
      f"(min abs: bbox {thr.get('bbox_min_abs_outlier', 0.0)} px, "
      f"landmark {thr.get('landmark_min_abs_outlier', 0.0)} px).")
    A("")
    A("## 3. Summary")
    A("")
    A("| Metric | Value (px) | Relative to face diagonal |")
    A("| --- | --- | --- |")
    diag_str = f"{diag:.1f}" if diag > 0 else "-"
    A(f"| Mean face diagonal | {diag_str} | 100% |")
    A(f"| **bbox_jitter_mean** (bbox centre displacement) | "
      f"{_fmt(summary.get('bbox_jitter_mean'))} | "
      f"{_fmt(100.0 * summary.get('bbox_jitter_mean', 0.0) / diag) if diag else '-'}% |")
    A(f"| **bbox_jitter_std** | {_fmt(summary.get('bbox_jitter_std'))} | - |")
    A(f"| **landmark_jitter_mean** (mean point distance) | "
      f"{_fmt(summary.get('landmark_jitter_mean'))} | "
      f"{_fmt(100.0 * summary.get('landmark_jitter_mean', 0.0) / diag) if diag else '-'}% |")
    A(f"| **landmark_jitter_std** | {_fmt(summary.get('landmark_jitter_std'))} | - |")
    A(f"| bbox max centre displacement | {_fmt(b.get('max_center_distance'))} | - |")
    A(f"| landmark max mean-point distance | {_fmt(lm.get('mean_point_distance_mean'))} (mean) / "
      f"{_fmt(lm.get('max_point_distance_max'))} (max) | - |")
    A("")
    A("## 4. Bbox detail (Stage 5)")
    A("")
    A("| Metric | Mean | Std |")
    A("| --- | --- | --- |")
    A(f"| centre displacement (px) | {_fmt(b.get('center_distance_mean'))} | "
      f"{_fmt(b.get('center_distance_std'))} |")
    A(f"| dx (px) | {_fmt(b.get('dx_mean'))} | {_fmt(b.get('dx_std'))} |")
    A(f"| dy (px) | {_fmt(b.get('dy_mean'))} | {_fmt(b.get('dy_std'))} |")
    A(f"| dw (px) | {_fmt(b.get('dw_mean'))} | {_fmt(b.get('dw_std'))} |")
    A(f"| dh (px) | {_fmt(b.get('dh_mean'))} | {_fmt(b.get('dh_std'))} |")
    A("")
    A("## 5. Landmark detail (Stage 10)")
    A("")
    A("| Metric | Value |")
    A("| --- | --- |")
    A(f"| mean point distance (px) | {_fmt(lm.get('mean_point_distance_mean'))} "
      f"± {_fmt(lm.get('mean_point_distance_std'))} |")
    A(f"| max point distance (px) | {_fmt(lm.get('max_point_distance_mean'))} "
      f"(mean of max) / {_fmt(lm.get('max_point_distance_max'))} (worst) |")
    A("")
    A("Per-region mean landmark displacement (px):")
    A("")
    A("| Region | Mean displacement (px) |")
    A("| --- | --- |")
    for k in ("eyes", "nose_tip", "mouth", "chin"):
        A(f"| {k} | {_fmt(lm.get('region_distance_mean', {}).get(k))} |")
    A("")
    A("## 6. Outlier frames")
    A("")
    if outlier_frames:
        A(f"{len(outlier_frames)} outlier frame(s) flagged "
          f"(threshold `mean + {thr.get('outlier_sigma', 3.0)}*std`):")
        A("")
        A("| frame | bbox centre dist (px) | landmark mean pt dist (px) | flags |")
        A("| --- | --- | --- | --- |")
        for o in outlier_detail:
            A(f"| {o.get('frame_index')} | {_fmt(o.get('bbox_center_distance'))} | "
              f"{_fmt(o.get('landmark_mean_point_distance'))} | "
              f"{', '.join(o.get('flags', []))} |")
        A("")
    else:
        A("No frames exceeded the outlier threshold.")
        A("")
    A("## 7. Worst transitions")
    A("")
    worst_bbox = _top_worst(b.get("per_transition", []), "center_distance", 8)
    worst_lm = _top_worst(lm.get("per_transition", []), "mean_point_distance", 8)
    A("Top-8 bbox centre displacements:")
    A("")
    A("| t0 | t1 | dx | dy | centre dist | dw | dh |")
    A("| --- | --- | --- | --- | --- | --- | --- |")
    for r in worst_bbox:
        A(f"| {r.get('t0')} | {r.get('t1')} | {_fmt(r.get('dx'))} | "
          f"{_fmt(r.get('dy'))} | **{_fmt(r.get('center_distance'))}** | "
          f"{_fmt(r.get('dw'))} | {_fmt(r.get('dh'))} |")
    A("")
    A("Top-8 landmark mean point distances:")
    A("")
    A("| t0 | t1 | mean pt dist | max pt dist | max pt idx | eyes | nose | mouth | chin |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in worst_lm:
        rd = r.get("region_distance", {})
        A(f"| {r.get('t0')} | {r.get('t1')} | **{_fmt(r.get('mean_point_distance'))}** | "
          f"{_fmt(r.get('max_point_distance'))} | {r.get('max_point_index')} | "
          f"{_fmt(rd.get('eyes'))} | {_fmt(rd.get('nose_tip'))} | "
          f"{_fmt(rd.get('mouth'))} | {_fmt(rd.get('chin'))} |")
    A("")
    A("## 8. Which stage is more likely causing the FLAME mesh jitter?")
    A("")
    A(_verdict(summary, diag))
    A("")
    A("## 9. Artifacts")
    A("")
    A("- Stage5 overlay video: `" + str(stage5_video) + "`")
    A("- Stage10 overlay video: `" + str(stage10_video) + "`")
    A("- Data: `stage5_frames.json`, `stage10_landmarks.json`, `temporal_stability.json`")
    A("- Tool: `scripts/stage5_stage10_audit/` (see its `README.md`)")
    A("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[audit] report -> {out}")
    return out



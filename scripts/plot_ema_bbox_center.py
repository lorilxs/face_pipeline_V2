#!/usr/bin/env python3
"""Plot EMA alpha=0.5 bbox center (cx, cy) trajectories.

Reads the Stage5 EMA ablation bboxes.json produced by
``scripts/stage5_ema_e2e_ablation.py`` and plots, per frame,

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

for the ``ema_alpha05`` branch (solid red), with the ``baseline`` branch
overlaid as a dashed blue reference so the smoothing effect is visible.

The trajectory / displacement / statistics helpers are reused from
``scripts/visualize_stage5_bbox_center.py`` (the Stage5 ROI visualizer); the
only adaptation is parsing the string-encoded ``bbox`` field of the EMA
ablation JSON into float lists.

Output (no videos):
    outputs/ema_motion_preservation/visualization/ema_alpha05_bbox_center.png
    outputs/ema_motion_preservation/visualization/bbox_center_statistics.json

Usage (from the Face_Pipeline_V2 project root):
    python3 scripts/plot_ema_bbox_center.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the trajectory / displacement / statistics helpers from the Stage5 ROI
# visualizer (cx=(x1+x2)/2, cy=(y1+y2)/2, isolated-outlier rule, stats).
import visualize_stage5_bbox_center as v  # noqa: E402

OUT_DIR = _ROOT / "outputs/ema_motion_preservation/visualization"
PNG_PATH = OUT_DIR / "ema_alpha05_bbox_center.png"
STATS_PATH = OUT_DIR / "bbox_center_statistics.json"

COL_EMA = "#d62728"  # red
COL_BASE = "#1f77b4"  # blue


def load_branch_frames(branch: str):
    """Load ``bboxes.json`` frames for one branch; decode string bboxes."""
    p = _ROOT / "outputs/ema_motion_preservation" / branch / "stage5" / "bboxes.json"
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    frames = []
    for fr in data["frames"]:
        bb = fr.get("bbox")
        if bb is None:
            continue
        if isinstance(bb, str):
            bb = json.loads(bb)
        frames.append(
            {"frame_id": int(fr["frame_id"]), "bbox": [float(x) for x in bb]}
        )
    return frames


def plot_centers(ema_traj, base_traj, out_path: Path) -> None:
    """Two-panel figure: cx (top) and cy (bottom), EMA vs baseline."""
    ema_ids = v._frame_id(ema_traj)
    base_ids = v._frame_id(base_traj)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), dpi=110, sharex=True)

    ax1.plot(base_ids, [t["cx"] for t in base_traj], color=COL_BASE, lw=1.1,
             ls="--", alpha=0.65, label="baseline cx")
    ax1.plot(ema_ids, [t["cx"] for t in ema_traj], color=COL_EMA, lw=1.6,
             label="EMA alpha=0.5 cx")
    ax1.set_ylabel("cx (px)")
    ax1.set_title("BBox center cx = (x1+x2)/2  —  Stage5 EMA ablation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(base_ids, [t["cy"] for t in base_traj], color=COL_BASE, lw=1.1,
             ls="--", alpha=0.65, label="baseline cy")
    ax2.plot(ema_ids, [t["cy"] for t in ema_traj], color=COL_EMA, lw=1.6,
             label="EMA alpha=0.5 cy")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("cy (px)")
    ax2.set_title("BBox center cy = (y1+y2)/2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ema_frames = load_branch_frames("ema_alpha05")
    base_frames = load_branch_frames("baseline")
    ema_traj = v.extract_trajectory(ema_frames)
    base_traj = v.extract_trajectory(base_frames)
    ema_deltas = v.compute_displacements(ema_traj)
    base_deltas = v.compute_displacements(base_traj)

    print(f"[load] ema_alpha05: {len(ema_frames)} frames -> {len(ema_traj)} traj")
    print(f"[load] baseline:    {len(base_frames)} frames -> {len(base_traj)} traj")

    b_stats = v.compute_statistics(base_traj, base_deltas, "baseline")
    e_stats = v.compute_statistics(ema_traj, ema_deltas, "ema_alpha05")
    for s in (b_stats, e_stats):
        print(f"[stats] {s['method']}: cx_mean={s['cx_mean']} cy_mean={s['cy_mean']} "
              f"cx_std={s['cx_std']} cy_std={s['cy_std']} "
              f"vel_mean={s['velocity_mean']}px/f vel_max={s['velocity_max']}px/f")

    # Per-frame jitter (same rule as the ablation report).
    thresh = 8.0
    b_out = v.isolated_outliers(base_deltas, thresh)
    e_out = v.isolated_outliers(ema_deltas, thresh)
    print(f"[outlier>{thresh}px] baseline={len(b_out)} ema_alpha05={len(e_out)}")

    plot_centers(ema_traj, base_traj, PNG_PATH)

    stats = {"baseline": b_stats, "ema_alpha05": e_stats,
             "outlier_threshold_px": thresh,
             "outlier_frames": {"baseline": b_out, "ema_alpha05": e_out}}
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"[json] wrote {STATS_PATH}")
    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

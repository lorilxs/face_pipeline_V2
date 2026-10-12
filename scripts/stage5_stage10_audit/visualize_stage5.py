"""Stage5 overlay video: SCRFD bbox + expanded crop bbox + jitter plots.

For every clip frame the video shows:
  - the SCRFD tight bbox (green) and the expanded crop bbox (blue),
  - the bbox centre (red dot) + confidence,
  - the frame index / timestamp,
  - a right-hand panel with the current transition (dx, dy, dw, dh,
    center_distance), a centre-trajectory plot and a per-transition
    centre-distance bar chart (outlier transitions highlighted in red).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from .analysis import BboxTransition, bbox_transitions
from .data_io import Stage5Data, select_main_face
from .visualize import (
    VideoWriter, GREEN, BLUE, RED, ORANGE, WHITE, GRAY, LIGHT, DARK,
    draw_text, draw_line_plot, draw_bar_plot,
)

TARGET_H = 720
PANEL_W = 560


def _build_panel(
    clip_id: str,
    fi: int,
    fps: float,
    trans_now: Optional[BboxTransition],
    cx_arr: np.ndarray,
    cy_arr: np.ndarray,
    dist_hist: List[float],
    frame_h: float,
    max_dist: float,
    mean_dist: float,
    std_dist: float,
) -> np.ndarray:
    panel_h = TARGET_H
    panel = np.zeros((panel_h, PANEL_W, 3), dtype=np.uint8)
    y = 16

    draw_text(panel, "STAGE 5 - SCRFD bbox", (14, y), 0.6, GREEN, bg=DARK)
    y += 30
    draw_text(panel, f"clip: {clip_id[:52]}", (14, y), 0.38, LIGHT)
    y += 22
    draw_text(panel, f"frame {fi:04d}   t = {fi / fps:.3f} s", (14, y), 0.45, WHITE)
    y += 28

    draw_text(panel, "current transition (prev -> this frame)", (14, y), 0.4, LIGHT)
    y += 20
    if trans_now is not None:
        draw_text(
            panel,
            f"center: dx={trans_now.dx:+8.2f}  dy={trans_now.dy:+8.2f} px",
            (14, y), 0.42, WHITE,
        )
        y += 20
        draw_text(
            panel,
            f"center distance = {trans_now.center_distance:7.2f} px",
            (14, y), 0.42, GREEN,
        )
        y += 20
        draw_text(
            panel,
            f"size: dw={trans_now.dw:+7.1f}  dh={trans_now.dh:+7.1f} px",
            (14, y), 0.42, WHITE,
        )
        y += 20
    else:
        draw_text(panel, "(first frame - no transition yet)", (14, y), 0.42, GRAY)
        y += 20
    y += 6

    # --- centre trajectory ---
    plot_h = 180
    draw_line_plot(
        panel, 12, y, PANEL_W - 24, plot_h,
        [cx_arr, cy_arr], ["cx", "cy"], [GREEN, ORANGE],
        ylabel="centre (px)", ymax=float(frame_h), highlight=fi,
    )
    y += plot_h + 18

    # --- per-transition centre distance bar chart ---
    bar_h = 180
    draw_bar_plot(
        panel, 12, y, PANEL_W - 24, bar_h,
        dist_hist, GREEN, label="bbox centre distance per transition (px)",
        ymax=max_dist if max_dist > 0 else None,
        highlight=(len(dist_hist) - 1) if dist_hist else None,
    )
    y += bar_h + 16

    draw_text(
        panel,
        f"jitter: mean {mean_dist:.2f} px   std {std_dist:.2f} px",
        (14, y), 0.42, LIGHT,
    )
    return panel


def render_stage5_video(
    stage5: Stage5Data,
    video_path: Path,
    out_path: Path,
    fps: Optional[float] = None,
    outlier_frames: Optional[Set[int]] = None,
    max_frames: Optional[int] = None,
) -> Path:
    """Render the Stage5 overlay video to ``out_path`` and return its path."""
    frames_by_idx = stage5.by_frame()
    trans_list = bbox_transitions(stage5.frames)
    trans_by_t1 = {t.t1: t for t in trans_list}

    fps = fps or stage5.source.get("fps") or 25.0
    sw = stage5.source.get("width") or 1920
    sh = stage5.source.get("height") or 1080
    clip_id = stage5.source.get("clip_id", "")

    scale = TARGET_H / float(sh)
    scaled_w = int(round(sw * scale))
    canvas_w = scaled_w + PANEL_W

    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_video = min(n_video, max_frames)

    cx_hist: List[float] = []
    cy_hist: List[float] = []
    dist_hist: List[float] = []
    max_dist = 0.0
    running_dist: List[float] = []

    out_path = Path(out_path)
    with VideoWriter(out_path, fps, canvas_w, TARGET_H) as writer:
        for fi in range(n_video):
            ret, frame = cap.read()
            if not ret:
                break
            disp = cv2.resize(
                frame, (scaled_w, TARGET_H), interpolation=cv2.INTER_AREA
            )

            rec = frames_by_idx.get(fi)
            if rec is not None:
                face = select_main_face([rec])
                if face is not None:
                    x1, y1, x2, y2 = [v * scale for v in face.bbox]
                    cv2.rectangle(
                        disp, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2
                    )
                    eb = face.expanded_bbox
                    if eb:
                        ex1, ey1, ex2, ey2 = [v * scale for v in eb]
                        cv2.rectangle(
                            disp, (int(ex1), int(ey1)), (int(ex2), int(ey2)),
                            BLUE, 1,
                        )
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    cv2.circle(disp, (int(cx), int(cy)), 4, RED, -1)
                    draw_text(
                        disp, f"conf {face.confidence:.2f}",
                        (int(x1), max(0, int(y1) - 8)), 0.45, GREEN, bg=DARK,
                    )
                    # history in ORIGINAL pixel coords
                    cx0 = (face.bbox[0] + face.bbox[2]) / 2.0
                    cy0 = (face.bbox[1] + face.bbox[3]) / 2.0
                    if cx_hist:
                        d = math.hypot(cx0 - cx_hist[-1], cy0 - cy_hist[-1])
                        dist_hist.append(d)
                        running_dist.append(d)
                        if d > max_dist:
                            max_dist = d
                    cx_hist.append(cx0)
                    cy_hist.append(cy0)

            draw_text(disp, f"frame {fi:04d}", (10, 28), 0.55, WHITE, bg=DARK)
            if outlier_frames and fi in outlier_frames:
                cv2.rectangle(
                    disp, (0, 0), (scaled_w - 1, TARGET_H - 1), RED, 5
                )

            trans_now = trans_by_t1.get(fi)
            mean_dist = float(np.mean(running_dist)) if running_dist else 0.0
            std_dist = float(np.std(running_dist)) if running_dist else 0.0

            panel = _build_panel(
                clip_id, fi, fps, trans_now,
                np.asarray(cx_hist, dtype=np.float64),
                np.asarray(cy_hist, dtype=np.float64),
                dist_hist, sh, max_dist, mean_dist, std_dist,
            )
            canvas = np.hstack([disp, panel])
            writer.write(canvas)

    cap.release()
    print(f"[audit] Stage5 overlay video -> {out_path} ({n_video} frames)")
    return out_path


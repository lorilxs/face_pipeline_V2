"""Stage10 overlay video: InsightFace 106-point landmarks + jitter plots.

For every clip frame the video shows (frame coordinates, scaled to fit):
  - all 106 landmark points (cyan) with their point index (0-105),
  - highlighted key points:
        eyes (pts 33-42, 87-96) in cyan, eye corners labelled,
        nose tip (pts 83-85) in red,
        mouth corners (min/max x of pts 52-71) in orange,
        chin (lowest contour point) in magenta,
  - the frame index / timestamp,
  - a right-hand panel with the current transition mean/max point distance,
    per-region distances (eyes / nose tip / mouth / chin) and plots of the
    per-transition landmark jitter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from .analysis import (
    FACE_CONTOUR, LEFT_EYE, RIGHT_EYE, MOUTH, NOSE_TIP, EYES,
    LandmarkTransition, landmark_transitions, chin_point_index,
)
from .data_io import Stage10Data
from .visualize import (
    VideoWriter, GREEN, BLUE, RED, CYAN, MAGENTA, ORANGE, WHITE, GRAY,
    LIGHT, DARK, draw_text, draw_line_plot, draw_bar_plot,
)

TARGET_H = 720
PANEL_W = 560


def _eye_corners(lms: np.ndarray, indices: List[int]) -> List[int]:
    """Return the min-x and max-x point indices within an eye region."""
    pts = lms[indices]
    return [indices[int(np.argmin(pts[:, 0]))], indices[int(np.argmax(pts[:, 0]))]]


def _mouth_corners(lms: np.ndarray) -> List[int]:
    pts = lms[MOUTH]
    return [MOUTH[int(np.argmin(pts[:, 0]))], MOUTH[int(np.argmax(pts[:, 0]))]]


def _draw_landmarks(disp: np.ndarray, lms: np.ndarray, scale: float) -> None:
    """Draw the 106 points (with indices) and highlighted key points on ``disp``."""
    xy = lms * scale
    # base points + index labels
    for i, (x, y) in enumerate(xy):
        px, py = int(round(x)), int(round(y))
        cv2.circle(disp, (px, py), 2, CYAN, -1, cv2.LINE_AA)
        draw_text(disp, str(i), (px + 2, py - 2), 0.28, GRAY, bg=DARK, pad=1)

    # eyes (all eye points, thicker)
    for i in EYES:
        px, py = int(round(xy[i, 0])), int(round(xy[i, 1]))
        cv2.circle(disp, (px, py), 3, CYAN, 1, cv2.LINE_AA)

    # eye corners
    for label, idx in (("LE_c", _eye_corners(lms, LEFT_EYE)),
                       ("RE_c", _eye_corners(lms, RIGHT_EYE))):
        for i in idx:
            px, py = int(round(xy[i, 0])), int(round(xy[i, 1]))
            cv2.circle(disp, (px, py), 4, WHITE, 1, cv2.LINE_AA)
            draw_text(disp, label, (px + 4, py - 4), 0.4, WHITE, bg=DARK)

    # nose tip
    for i in NOSE_TIP:
        px, py = int(round(xy[i, 0])), int(round(xy[i, 1]))
        cv2.circle(disp, (px, py), 3, RED, -1, cv2.LINE_AA)
    draw_text(disp, "nose", (int(round(xy[84, 0])) + 4, int(round(xy[84, 1])) - 4),
              0.4, RED, bg=DARK)

    # mouth corners
    for i in _mouth_corners(lms):
        px, py = int(round(xy[i, 0])), int(round(xy[i, 1]))
        cv2.circle(disp, (px, py), 3, ORANGE, -1, cv2.LINE_AA)
        draw_text(disp, "mouth", (px + 4, py + 10), 0.4, ORANGE, bg=DARK)

    # chin
    ci = chin_point_index(lms)
    px, py = int(round(xy[ci, 0])), int(round(xy[ci, 1]))
    cv2.circle(disp, (px, py), 4, MAGENTA, -1, cv2.LINE_AA)
    draw_text(disp, "chin", (px + 4, py - 4), 0.4, MAGENTA, bg=DARK)


def _build_panel(
    clip_id: str,
    fi: int,
    fps: float,
    trans_now: Optional[LandmarkTransition],
    mean_hist: List[float],
    max_hist: List[float],
    max_mean: float,
    max_max: float,
) -> np.ndarray:
    panel = np.zeros((TARGET_H, PANEL_W, 3), dtype=np.uint8)
    y = 16

    draw_text(panel, "STAGE 10 - 106 landmarks (2d106det)", (14, y), 0.58, CYAN,
              bg=DARK)
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
            f"mean point dist = {trans_now.mean_point_distance:7.2f} px",
            (14, y), 0.42, GREEN,
        )
        y += 20
        draw_text(
            panel,
            f"max  point dist = {trans_now.max_point_distance:7.2f} px "
            f"(pt {trans_now.max_point_index})",
            (14, y), 0.42, RED,
        )
        y += 20
        rd = trans_now.region_dist
        draw_text(
            panel,
            f"eyes {rd.get('eyes', 0):.2f} | nose {rd.get('nose_tip', 0):.2f} | "
            f"mouth {rd.get('mouth', 0):.2f} | chin {rd.get('chin', 0):.2f}",
            (14, y), 0.4, WHITE,
        )
        y += 24
    else:
        draw_text(panel, "(first frame - no transition yet)", (14, y), 0.42, GRAY)
        y += 20
    y += 6

    # mean + max point distance per transition
    plot_h = 190
    hi = (len(mean_hist) - 1) if mean_hist else None
    draw_line_plot(
        panel, 12, y, PANEL_W - 24, plot_h,
        [np.asarray(mean_hist, dtype=np.float64),
         np.asarray(max_hist, dtype=np.float64)],
        ["mean pt dist", "max pt dist"], [GREEN, RED],
        ylabel="point distance (px)",
        ymax=max(max_mean, max_max) if (max_mean or max_max) else None,
        highlight=hi,
    )
    y += plot_h + 16

    # bar chart of mean point distance per transition
    bar_h = 190
    draw_bar_plot(
        panel, 12, y, PANEL_W - 24, bar_h,
        mean_hist, GREEN, label="landmark mean point distance per transition (px)",
        ymax=max_mean if max_mean > 0 else None,
        highlight=hi,
    )
    y += bar_h + 12

    draw_text(
        panel,
        "legend: cyan=all 106 pts  red=nose tip  orange=mouth",
        (14, y), 0.34, GRAY,
    )
    y += 16
    draw_text(
        panel,
        "cyan eye pts + white corners  magenta=chin",
        (14, y), 0.34, GRAY,
    )
    return panel


def render_stage10_video(
    stage10: Stage10Data,
    video_path: Path,
    out_path: Path,
    fps: Optional[float] = None,
    outlier_frames: Optional[Set[int]] = None,
    max_frames: Optional[int] = None,
) -> Path:
    """Render the Stage10 overlay video to ``out_path`` and return its path."""
    lms_by_idx = stage10.by_frame()
    trans_list = landmark_transitions(stage10.landmarks)
    trans_by_t1 = {t.t1: t for t in trans_list}

    fps = fps or stage10.source.get("fps") or 25.0
    sw = stage10.source.get("width") or 1920
    sh = stage10.source.get("height") or 1080
    clip_id = stage10.source.get("clip_id", "")

    scale = TARGET_H / float(sh)
    scaled_w = int(round(sw * scale))
    canvas_w = scaled_w + PANEL_W

    mean_hist: List[float] = []
    max_hist: List[float] = []
    max_mean = max_max = 0.0

    cap = cv2.VideoCapture(str(video_path))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_video = min(n_video, max_frames)

    out_path = Path(out_path)
    with VideoWriter(out_path, fps, canvas_w, TARGET_H) as writer:
        for fi in range(n_video):
            ret, frame = cap.read()
            if not ret:
                break
            disp = cv2.resize(
                frame, (scaled_w, TARGET_H), interpolation=cv2.INTER_AREA
            )

            rec = lms_by_idx.get(fi)
            if rec is not None:
                _draw_landmarks(disp, rec.landmarks_106, scale)

            draw_text(disp, f"frame {fi:04d}", (10, 28), 0.55, WHITE, bg=DARK)
            if outlier_frames and fi in outlier_frames:
                cv2.rectangle(
                    disp, (0, 0), (scaled_w - 1, TARGET_H - 1), RED, 5
                )

            trans_now = trans_by_t1.get(fi)
            if trans_now is not None:
                mean_hist.append(trans_now.mean_point_distance)
                max_hist.append(trans_now.max_point_distance)
                max_mean = max(max_mean, trans_now.mean_point_distance)
                max_max = max(max_max, trans_now.max_point_distance)

            panel = _build_panel(
                clip_id, fi, fps, trans_now, mean_hist, max_hist,
                max_mean, max_max,
            )
            writer.write(np.hstack([disp, panel]))

    cap.release()
    print(f"[audit] Stage10 overlay video -> {out_path} ({n_video} frames)")
    return out_path


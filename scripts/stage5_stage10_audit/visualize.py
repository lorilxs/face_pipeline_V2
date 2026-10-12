"""Shared OpenCV drawing helpers for the audit overlay videos.

Everything is drawn with cv2 primitives only (no matplotlib dependency in
the render path) so the overlay videos can be produced headless and fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# BGR colours
GREEN = (60, 200, 80)
BLUE = (220, 160, 40)
RED = (40, 60, 230)
CYAN = (220, 220, 40)
MAGENTA = (220, 40, 220)
ORANGE = (40, 120, 240)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (40, 40, 40)
GRAY = (150, 150, 150)
LIGHT = (210, 210, 210)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(
    img: np.ndarray,
    text: str,
    org,
    scale: float = 0.5,
    color=WHITE,
    thickness: int = 1,
    bg=None,
    pad: int = 3,
) -> None:
    """Draw text with an optional dark background box for readability."""
    org = (int(org[0]), int(org[1]))
    if bg is not None:
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
        x0, y0 = org[0] - pad, org[1] - th - pad
        x1, y1 = org[0] + tw + pad, org[1] + pad
        cv2.rectangle(img, (x0, y0), (x1, y1), bg, -1)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def draw_line_plot(
    canvas: np.ndarray,
    x0: int, y0: int, w: int, h: int,
    series: List[np.ndarray],
    labels: List[str],
    colors: List,
    ylabel: str = "",
    ymax: Optional[float] = None,
    highlight: Optional[int] = None,
    n_ticks: int = 5,
) -> None:
    """Draw one or more line series in the box (x0,y0)-(x0+w,y0+h).

    Series are aligned by index (index = frame number). ``highlight`` is an
    x index whose bar/point is emphasised.
    """
    # background + border
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), DARK, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), GRAY, 1)

    n = max((len(s) for s in series), default=0)
    if n < 2:
        cv2.putText(canvas, "no data", (x0 + 8, y0 + h // 2), _FONT, 0.5, GRAY, 1)
        return

    all_vals = np.concatenate(series) if series else np.zeros(1)
    vmin = float(all_vals.min()) if len(all_vals) else 0.0
    vmax = ymax if ymax is not None else float(all_vals.max())
    if vmax == vmin:
        vmax = vmin + 1.0
    vmin = min(vmin, 0.0)

    pad_l, pad_r, pad_t, pad_b = 8, 8, 6, 14
    px = lambda i: int(x0 + pad_l + (w - pad_l - pad_r) * i / (n - 1))
    py = lambda v: int(y0 + h - pad_b - (h - pad_t - pad_b) * (v - vmin) / (vmax - vmin))

    # y ticks
    for k in range(n_ticks + 1):
        val = vmin + (vmax - vmin) * k / n_ticks
        yy = py(val)
        cv2.line(canvas, (x0 + pad_l - 3, yy), (x0 + pad_l, yy), GRAY, 1)
        cv2.putText(
            canvas, f"{val:.1f}", (x0 + 2, yy + 3), _FONT, 0.35, GRAY, 1
        )
    if ylabel:
        cv2.putText(canvas, ylabel, (x0 + 6, y0 + pad_t), _FONT, 0.4, LIGHT, 1)

    for s, col in zip(series, colors):
        pts = np.array([[px(i), py(float(v))] for i, v in enumerate(s)], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, col, 2, cv2.LINE_AA)

    if highlight is not None and 0 <= highlight < n:
        hx = px(highlight)
        cv2.line(canvas, (hx, y0 + pad_t), (hx, y0 + h - pad_b), WHITE, 1)


def draw_bar_plot(
    canvas: np.ndarray,
    x0: int, y0: int, w: int, h: int,
    values: List[float],
    color,
    label: str = "",
    ymax: Optional[float] = None,
    highlight: Optional[int] = None,
    show_last: Optional[int] = None,
) -> None:
    """Bar chart over transition indices (index i = transition i ends at frame i+1)."""
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), DARK, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), GRAY, 1)

    n = len(values)
    if n == 0:
        cv2.putText(canvas, "no data", (x0 + 8, y0 + h // 2), _FONT, 0.5, GRAY, 1)
        return
    if label:
        cv2.putText(canvas, label, (x0 + 6, y0 + 12), _FONT, 0.4, LIGHT, 1)

    vmax = ymax if ymax is not None else float(max(values))
    if vmax <= 0:
        vmax = 1.0
    pad_l, pad_r, pad_t, pad_b = 8, 8, 6, 14

    bar_w = max(1.0, (w - pad_l - pad_r) / n)
    for i, v in enumerate(values):
        bh = max(1.0, (h - pad_t - pad_b) * v / vmax)
        bx = int(x0 + pad_l + i * bar_w)
        by = int(y0 + h - pad_b - bh)
        col = RED if (highlight is not None and i == highlight) else color
        cv2.rectangle(canvas, (bx, by), (int(bx + bar_w), int(y0 + h - pad_b)), col, -1)


class VideoWriter:
    """Thin context manager around cv2.VideoWriter (mp4v / H.264 fallback)."""

    def __init__(self, path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self._writer = None

    def _open(self):
        codecs = [cv2.VideoWriter_fourcc(*"mp4v")]
        for codec in codecs:
            w = cv2.VideoWriter(
                str(self.path), codec, self.fps, (self.width, self.height)
            )
            if w.isOpened():
                return w
        raise RuntimeError(f"Could not open video writer for {self.path}")

    def __enter__(self):
        self._writer = self._open()
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(
                frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        self._writer.write(frame_bgr)

    def __exit__(self, *exc):
        if self._writer is not None:
            self._writer.release()
        return False


#!/usr/bin/env python3
"""Diagnose Stage 2 face-subtitle detection on the two verification clips.

Recomputes the exact Sobel edge-energy band series used by
``SceneDetector._analyze_video`` and prints where runs exceed the configured
threshold, so we can see why the subtitle-covering-face segments are not cut at
the moments manual review expects (~1s for video 1, ~5s for video 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VIDEOS = {
    "video1_4AVK": PROJECT_ROOT
    / "outputs/Talkvid_Chinese_videos_clips001/video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"
    / "verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4",
    "video2_59Ql": PROJECT_ROOT
    / "outputs/Talkvid_Chinese_videos_clips001/video_59QlHqVOCKw_11_471.02_482.27_open_domain"
    / "verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4",
}

OUT = PROJECT_ROOT / "test_data" / "stage2_debug"
BAND_TOP, BAND_BOTTOM = 0.33, 0.67
MIN_EDGE, EDGE_RATIO = 30.0, 2.0


def analyze(path: Path, save_frames: list[int]) -> None:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    y0 = max(0, int(round(height * BAND_TOP)))
    y1 = min(height, int(round(height * BAND_BOTTOM)))

    band_scores: list[float] = []
    full_scores: list[float] = []
    bottom_scores: list[float] = []
    i = 0
    saved: dict[int, Path] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        band = gray[y0:y1, :]
        gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
        band_scores.append(float(cv2.magnitude(gx, gy).mean()))

        gxf = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gyf = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        full_scores.append(float(cv2.magnitude(gxf, gyf).mean()))

        bot = gray[max(0, int(height * 0.67)):, :]
        gxb = cv2.Sobel(bot, cv2.CV_32F, 1, 0, ksize=3)
        gyb = cv2.Sobel(bot, cv2.CV_32F, 0, 1, ksize=3)
        bottom_scores.append(float(cv2.magnitude(gxb, gyb).mean()))

        if i in save_frames:
            p = OUT / f"{path.stem}_f{i:04d}.png"
            cv2.imwrite(str(p), frame)
            saved[i] = p
        i += 1
    cap.release()

    n = len(band_scores)
    median = float(np.median(band_scores))
    threshold = max(MIN_EDGE, EDGE_RATIO * median)
    print(f"\n{'=' * 76}")
    print(f"{path.name}")
    print(f"  {width}x{height} @ {fps:.1f} fps, {n} frames, band rows {y0}..{y1}")
    print(f"  band median={median:.2f}  threshold=max({MIN_EDGE}, {EDGE_RATIO}*median)"
          f" = {threshold:.2f}")
    print(f"  min={min(band_scores):.2f} max={max(band_scores):.2f} "
          f"mean={float(np.mean(band_scores)):.2f} "
          f"p99={float(np.percentile(band_scores, 99)):.2f}")

    # Identify runs above threshold
    runs = []
    start = None
    for idx, s in enumerate(band_scores):
        if s > threshold:
            if start is None:
                start = idx
        else:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
    if start is not None:
        runs.append((start, n - 1))

    print(f"  runs > threshold (>=5 frames):")
    shown = 0
    for a, b in runs:
        if b - a + 1 >= 5:
            print(f"    frames {a:5d}..{b:<5d}  ({a/fps:6.2f}s..{b/fps:6.2f}s)  "
                  f"peak={max(band_scores[a:b+1]):.2f}")
            shown += 1
    if shown == 0:
        print("    <none>")

    # Per-frame detail around the interesting times (all three bands)
    print(f"  key frames (band / full / bottom):")
    for f in save_frames:
        if 0 <= f < n:
            t = f / fps
            print(f"    f{f:4d} t={t:6.2f}s  band={band_scores[f]:7.2f} "
                  f"full={full_scores[f]:7.2f} bottom={bottom_scores[f]:7.2f} "
                  f"above_thr={band_scores[f] > threshold} -> {saved.get(f)}")
    print(f"  band_first_100s = {[round(x, 1) for x in band_scores[:100]]}")
    print(f"  band_first_100 = {' '.join(f'{x:5.1f}' for x in band_scores[:100])}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    analyze(VIDEOS["video1_4AVK"], save_frames=[10, 20, 25, 30, 40, 50])
    analyze(VIDEOS["video2_59Ql"], save_frames=[100, 110, 120, 125, 130, 140])


if __name__ == "__main__":
    main()
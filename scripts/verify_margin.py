#!/usr/bin/env python3
"""Verify the face-subtitle cut margin lands on clean frames.

Runs the SceneDetector's subtitle interval detection and then re-derives the
per-frame band edge/brightness series so we can print the raw values around each
detected cut to confirm the frame just after the boundary is clean.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.stages.stage_02_scene_detection.scene_detector import (
    SceneDetector,
)

OUT = PROJECT_ROOT / "outputs" / "Talkvid_Chinese_videos_clips001"
CLIPS = {
    "V1": OUT
    / "video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"
    / "verification_face_only"
    / "4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4",
    "V2": OUT
    / "video_59QlHqVOCKw_11_471.02_482.27_open_domain"
    / "verification_face_only"
    / "59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4",
}


def frame_series(path, band_top, band_bottom, bright_luma=235.0):
    cap = cv2.VideoCapture(str(path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    y0 = max(0, int(round(height * band_top)))
    y1 = min(height, int(round(height * band_bottom)))
    edges, brights = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = gray.astype(np.float32)[y0:y1, :]
        gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
        edges.append(float(cv2.magnitude(gx, gy).mean()))
        brights.append(float(np.mean(band >= bright_luma)))
    cap.release()
    return fps, edges, brights


def main():
    cfg_full = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text())
    cfg = dict(cfg_full["stage_02"]["face_subtitle"])
    det = SceneDetector(threshold=6.0, min_scene_len=15, adaptive_factor=2.0)

    for name, path in CLIPS.items():
        intervals, noise = det._analyze_video(path, cfg)
        fps, edges, brights = frame_series(
            path, cfg["band_top"], cfg["band_bottom"]
        )
        n = len(edges)
        print(f"\n=== {name} (fps={fps:.2f}, frames={n}) noise={noise!r} ===")
        print(f"  config: {cfg}")
        print(f"  intervals: {intervals}")
        for s, e in intervals:
            print(f"    {s}..{e}  ({s / fps:.3f}s .. {e / fps:.3f}s)")
        # Median baseline for reference.
        edges_a = np.asarray(edges)
        brights_a = np.asarray(brights)
        print(f"  med_edge={np.median(edges_a):.3f}  med_bright={np.median(brights_a):.5f}")
        # Show raw series around every interval end (the cut point is end+1).
        for s, e in intervals:
            lo = max(0, e - 4)
            hi = min(n - 1, e + 8)
            print(f"  tail of interval ending {e} (cut at {e + 1}):")
            for i in range(lo, hi + 1):
                mark = "<-- cut frame" if i == e + 1 else (
                    "interval end" if i == e else ""
                )
                print(
                    f"    f{i:4d} {i / fps:6.3f}s  edge={edges[i]:6.3f} "
                    f"bright={brights[i]:7.4%}  {mark}"
                )


if __name__ == "__main__":
    main()
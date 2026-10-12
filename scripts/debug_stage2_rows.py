#!/usr/bin/env python3
"""Row-profile analysis of frames to locate the face-covering subtitle."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VIDEOS = {
    "v1": PROJECT_ROOT
    / "outputs/Talkvid_Chinese_videos_clips001/video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"
    / "verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4",
    "v2": PROJECT_ROOT
    / "outputs/Talkvid_Chinese_videos_clips001/video_59QlHqVOCKw_11_471.02_482.27_open_domain"
    / "verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4",
}


def describe(path: Path, frames: list[int]) -> None:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"\n{'='*80}\n{path.name}  {w}x{h} n={n}")
    idx = 0
    want = set(frames)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lbl = f"  f{idx:4d} t={idx/fps:6.2f}s"
            # overall stats
            bright = float((gray > 200).mean()) * 100
            dark = float((gray < 60).mean()) * 100
            mean = float(gray.mean())
            std = float(gray.std())
            print(f"{lbl}  mean={mean:6.1f} std={std:5.1f} bright%={bright:5.2f} "
                  f"dark%={dark:5.2f}")
            # row profile in 18 bands
            rows = 18
            band_h = h // rows
            prof = []
            for r in range(rows):
                seg = gray[r*band_h:(r+1)*band_h, :]
                prof.append(float(seg.mean()))
            print("      row-mean (top->bottom, 18 bands): "
                  + " ".join(f"{v:5.1f}" for v in prof))
            # bright-row profile: fraction of bright pixels per row band
            bp = []
            for r in range(rows):
                seg = gray[r*band_h:(r+1)*band_h, :]
                bp.append(float((seg > 200).mean()) * 100)
            print("      bright%%  (top->bottom, 18 bands): "
                  + " ".join(f"{v:5.1f}" for v in bp))
        idx += 1
    cap.release()


def main() -> None:
    describe(VIDEOS["v1"], [5, 12, 20, 25, 30, 42, 50, 55])
    describe(VIDEOS["v2"], [5, 20, 42, 90, 100, 110, 120, 125, 130, 140])


if __name__ == "__main__":
    main()
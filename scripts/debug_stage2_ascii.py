#!/usr/bin/env python3
"""Render pairs of frames as ASCII (narrow, focused) to locate subtitle."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VD = PROJECT_ROOT / "outputs/Talkvid_Chinese_videos_clips001"
V1 = (VD / "video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"
      / "verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4")
V2 = (VD / "video_59QlHqVOCKw_11_471.02_482.27_open_domain"
      / "verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4")

CHARS = " .:-=+*#%@"


def art(path: Path, frame_idx: int, cols: int = 80, rows: int = 26) -> None:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i == frame_idx:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
            lo, hi = float(small.min()), float(small.max())
            print(f"\n--- {path.stem[:6]} f{frame_idx} (t={frame_idx/fps:.2f}s) luma {lo:.0f}..{hi:.0f}")
            rng = max(hi - lo, 1)
            for r in range(rows):
                line = "".join(
                    CHARS[min(len(CHARS) - 1, int((small[r, c] - lo) / rng * len(CHARS)))]
                    for c in range(cols)
                )
                print(line)
            break
        i += 1
    cap.release()


def main() -> None:
    art(V1, 5)
    art(V1, 25)
    art(V1, 45)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
VD = PROJECT_ROOT / "outputs/Talkvid_Chinese_videos_clips001"
V2 = (VD / "video_59QlHqVOCKw_11_471.02_482.27_open_domain"
      / "verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4")
CHARS = " .:-=+*#%@"
def art(path, frame_idx, cols=100, rows=30):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i == frame_idx:
            last = frame.copy()
            break
        i += 1
    cap.release()
    gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    lo, hi = float(small.min()), float(small.max())
    print(f"\n--- {path.stem[:6]} f{frame_idx} ({frame_idx/fps:.2f}s) luma {lo:.0f}..{hi:.0f}")
    rng = max(hi-lo, 1)
    for r in range(rows):
        print("".join(CHARS[min(len(CHARS)-1,int((small[r,c]-lo)/rng*len(CHARS)))] for c in range(cols)))
for f in [118, 122, 125]:
    art(V2, f)

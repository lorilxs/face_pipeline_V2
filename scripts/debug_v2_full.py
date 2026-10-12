#!/usr/bin/env python3
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
def load(path):
    cap=cv2.VideoCapture(str(path)); fs=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fs.append(f)
    cap.release(); return fs
fs=load(V2)
print("total frames:", len(fs))
# full-frame row-profile of bright/dark/edge for rows, comparing early vs late
for idx in (2, 60, 110, 118, 124, 126, 150, 200):
    g=cv2.cvtColor(fs[idx], cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo,hi=float(g.min()),float(g.max()); rng=max(hi-lo,1)
    print(f"\n=== V2 f{idx} {idx/25:.2f}s  full-luma {lo:.0f}..{hi:.0f} ===")
    s=cv2.resize(g,(96,30),interpolation=cv2.INTER_AREA)
    for r in range(30):
        print(''.join(CHARS[min(len(CHARS)-1,int((s[r,c]-lo)/rng*len(CHARS)))] for c in range(96)))

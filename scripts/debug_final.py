#!/usr/bin/env python3
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
def load(path):
    cap=cv2.VideoCapture(str(path)); fs=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fs.append(f)
    cap.release(); return fs
print("### V1 frame-by-frame band signals (0.24s..2.4s, step 2 frames = 0.08s)")
fs=load(V1); h=1080
bandy0,bandy1=356,723
for i in range(6,61,2):
    g=cv2.cvtColor(fs[i],cv2.COLOR_BGR2GRAY).astype(np.float32)
    band=g[bandy0:bandy1]
    bright=float((band>235).mean())*100
    edge=float(cv2.magnitude(cv2.Sobel(band,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(band,cv2.CV_32F,0,1,ksize=3)).mean())
    print(f"  f{i:3d} {i/25:.2f}s  bright%={bright:5.2f}  bandEdge={edge:5.2f}")
print()
print("### V1 full-frame ASCII f10 0.40s (before) / f25 1.00s / f45 1.80s / f60 2.40s (after)")
for i in (10,25,45,60):
    g=cv2.cvtColor(fs[i],cv2.COLOR_BGR2GRAY)
    s=cv2.resize(g,(80,25),interpolation=cv2.INTER_AREA)
    lo,hi=float(s.min()),float(s.max()); rng=max(hi-lo,1)
    print(f"\n-- V1 f{i} {i/25:.2f}s --")
    for r in range(25):
        print(''.join(CHARS[min(len(CHARS)-1,int((s[r,c]-lo)/rng*len(CHARS)))] for c in range(80)))

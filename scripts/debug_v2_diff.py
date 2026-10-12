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
def get(path, idx):
    cap=cv2.VideoCapture(str(path)); i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if i==idx:
            cap.release(); return f
        i+=1
    cap.release(); return None
def render(img, label, y0, y1, cols=100, rows=18):
    g = cv2.cvtColor(img[y0:y1], cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (cols, rows), interpolation=cv2.INTER_AREA)
    lo, hi = float(s.min()), float(s.max()); rng=max(hi-lo,1)
    print('\n%s  band rows %d..%d  luma %.0f..%.0f' % (label, y0, y1, lo, hi))
    for r in range(rows):
        print(''.join(CHARS[min(len(CHARS)-1,int((s[r,c]-lo)/rng*len(CHARS)))] for c in range(cols)))
y0,y1=713,1447
for idx in (118, 126):
    f = get(V2, idx)
    render(f, 'f%d (%.2fs)' % (idx, idx/25.0), y0, y1)
f1=get(V2,118); f2=get(V2,126)
g1=cv2.cvtColor(f1[y0:y1],cv2.COLOR_BGR2GRAY).astype(np.float32)
g2=cv2.cvtColor(f2[y0:y1],cv2.COLOR_BGR2GRAY).astype(np.float32)
d=np.abs(g1-g2)
s=cv2.resize(d,(100,18),interpolation=cv2.INTER_AREA)
lo,hi=float(s.min()),float(s.max()); rng=max(hi-lo,1)
print('\nABS-DIFF f118 vs f126  (diff luma %.0f..%.0f, mean area diff=%.2f)' % (lo, hi, d.mean()))
for r in range(18):
    print(''.join(CHARS[min(len(CHARS)-1,int((s[r,c]-lo)/rng*len(CHARS)))] for c in range(100)))

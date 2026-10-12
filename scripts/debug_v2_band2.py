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
def frames(path):
    cap = cv2.VideoCapture(str(path)); out=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out
def stats(fr, h, name):
    y0=int(round(h*0.33)); y1=int(round(h*0.67))
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32)
    band=g[y0:y1,:]
    gx=cv2.Sobel(band,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(band,cv2.CV_32F,0,1,ksize=3)
    mag=cv2.magnitude(gx,gy)
    mean=float(mag.mean()); med=float(np.median(mag)); p90=float(np.percentile(mag,90)); p99=float(np.percentile(mag,99))
    fracs={t: float((mag>t).mean())*100 for t in (40,60,80,100)}
    bright=float((band>180).mean())*100; dark=float((band<60).mean())*100
    print(f"  {name:20s} mean={mean:6.2f} med={med:6.2f} p90={p90:6.2f} p99={p99:7.2f} | "
          f">40={fracs[40]:5.2f}% >60={fracs[60]:5.2f}% >80={fracs[80]:5.2f}% >100={fracs[100]:5.2f}% | bright={bright:5.2f}% dark={dark:5.2f}%")
fs=frames(V2); h=2160
print(f"=== V2 ({h}px) ===")
for i,nm in [(90,"f90 3.6s"),(100,"f100 4.0s"),(118,"f118 4.72s"),(122,"f122 4.88s"),(124,"f124 4.96s"),(126,"f126 5.04s"),(130,"f130 5.2s"),(140,"f140 5.6s"),(200,"f200 8s")]:
    stats(fs[i],h,nm)

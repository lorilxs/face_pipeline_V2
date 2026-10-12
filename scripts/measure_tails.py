#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VD = PROJECT_ROOT / "outputs/Talkvid_Chinese_videos_clips001"
V1 = VD/"video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"/"verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4"
V2 = VD/"video_59QlHqVOCKw_11_471.02_482.27_open_domain"/"verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4"

def load(path):
    cap=cv2.VideoCapture(str(path)); fs=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fs.append(f)
    cap.release(); return fs

for name, path, lo, hi in (("V2",V2,110,175),("V1",V1,8,50)):
    fs=load(path)
    H=fs[0].shape[0]; y0=int(round(H*0.33)); y1=int(round(H*0.67))
    print(f"\n=== {name} H={H} band rows {y0}-{y1} ===")
    prev=None
    for i in range(lo, hi):
        g=cv2.cvtColor(fs[i],cv2.COLOR_BGR2GRAY); b=g.astype(np.float32)[y0:y1,:]
        ed=float(cv2.magnitude(cv2.Sobel(b,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(b,cv2.CV_32F,0,1,ksize=3)).mean())
        br=float((b>=235).mean()*100)
        print(f"  f{i:4d} t={i/25:5.2f}s  edge={ed:7.2f}  bright={br:6.2f}%")

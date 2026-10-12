#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VD = PROJECT_ROOT / "outputs/Talkvid_Chinese_videos_clips001"
V1 = VD/"video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"/"verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4"
CHARS = " .:-=+*#%@"
cap=cv2.VideoCapture(str(V1)); fs=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fs.append(f)
cap.release()
print("frames:", len(fs))
H, W = fs[0].shape[:2]
y0=int(round(H*0.33)); y1=int(round(H*0.67))
print("band rows", y0, y1, "size", W, "x", H, "W=", W)
for idx in (8, 14, 22, 25, 30, 35, 40, 44, 46, 50, 55):
    g=cv2.cvtColor(fs[idx], cv2.COLOR_BGR2GRAY)
    band=g[y0:y1, :].astype(np.float32)
    bright=float((band>=235).mean()*100)
    edge=float(cv2.magnitude(cv2.Sobel(band,cv2.CV_32F,1,0,ksize=3), cv2.Sobel(band,cv2.CV_32F,0,1,ksize=3)).mean())
    print(f"\n=== V1 f{idx} t={idx/25:.3f}s bright={bright:.2f}% edge={edge:.2f} ===")
    s=cv2.resize(band,(96,16),interpolation=cv2.INTER_AREA); lo=float(s.min()); hi=float(s.max()); rng=max(hi-lo,1)
    for r in range(16):
        print(''.join(CHARS[min(len(CHARS)-1,int((s[r,c]-lo)/rng*len(CHARS)))] for c in range(96)))

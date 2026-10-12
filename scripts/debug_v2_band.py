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
V1 = (VD / "video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"
      / "verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4")

def frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out

def stats(fr, h, name):
    y0 = int(round(h*0.33)); y1 = int(round(h*0.67))
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    band = g[y0:y1,:]
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mean = float(mag.mean()); med = float(np.median(mag)); p90 = float(np.percentile(mag,90)); p99 = float(np.percentile(mag,99))
    # high-edge fraction at several absolute thresholds
    fracs = {t: float((mag > t).mean())*100 for t in (40, 60, 80, 100)}
    bright = float((band > 180).mean())*100
    dark = float((band < 60).mean())*100
    print(f"  {name:24s} mean={mean:6.2f} med={med:6.2f} p90={p90:6.2f} p99={p99:6.2f} "
          f"| frac>40={fracs[40]:5.2f}% >60={fracs[60]:5.2f}% >80={fracs[80]:5.2f}% >100={fracs[100]:5.2f}% "
          f"| bright={bright:5.2f}% dark={dark:5.2f}%")

for tag, path, picks in [
    ("V1", V1, [(6,"baseline f6"), (25,"subtitle f25"), (60,"after f60")]),
    ("V2", V2, [(100,"pre f100 (4.0s)"), (118,"pre f118"), (126,"post f126 (5.04s)"), (140,"post f140")]),
]:
    fs = frames(path)
    h = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n=== {tag} ({h}px) ===")
    for i, nm in picks:
        stats(fs[i], h, nm)

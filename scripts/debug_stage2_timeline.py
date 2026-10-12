#!/usr/bin/env python3
"""Per-half-second summary of subtitle-relevant signals for both clips."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
VD = PROJECT_ROOT / "outputs/Talkvid_Chinese_videos_clips001"
VIDEOS = {
    "v1": VD/"video_4AVKdHSOLdQ_3_175.98_194.56_open_domain"/"verification_face_only/4AVKdHSOLdQ_3_175.98_194.56_scene_0000_clip_0000_faces_only.mp4",
    "v2": VD/"video_59QlHqVOCKw_11_471.02_482.27_open_domain"/"verification_face_only/59QlHqVOCKw_11_471.02_482.27_scene_0000_clip_0000_faces_only.mp4",
}

def analyze(path):
    cap = cv2.VideoCapture(str(path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    y0 = int(round(h*0.33)); y1 = int(round(h*0.67))
    yb = int(round(h*0.67))
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        band = gray[y0:y1,:]
        gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
        edge = float(cv2.magnitude(gx,gy).mean())
        bot = gray[yb:,:]
        bright_bot = float((bot > 180).mean())*100
        bright_band = float((band > 180).mean())*100
        dark_band = float((band < 60).mean())*100
        full_mean = float(gray.mean())
        rows.append((i, edge, bright_bot, bright_band, dark_band, full_mean))
        i += 1
    cap.release()
    n = len(rows)
    print(f"\n=== {path.stem[:6]}  ({h}px, {n} frames @ {fps:.1f})")
    print("  bin(0.5s)  bandEdge  botBright%  bandBright%  bandDark%  fullMean")
    step = max(1, int(round(fps*0.5)))
    for s in range(0, n, step):
        chunk = rows[s:s+step]
        e = np.mean([r[1] for r in chunk])
        bb = np.mean([r[2] for r in chunk])
        bbr = np.mean([r[3] for r in chunk])
        bd = np.mean([r[4] for r in chunk])
        fm = np.mean([r[5] for r in chunk])
        t0 = rows[s][0]/fps; t1 = rows[s+step-1][0]/fps if s+step <= n else rows[-1][0]/fps
        print(f"  {t0:5.2f}-{t1:5.2f}  {e:8.2f}  {bb:9.2f}  {bbr:10.2f}  {bd:8.2f}  {fm:8.1f}")

for k in ["v1", "v2"]:
    analyze(VIDEOS[k])

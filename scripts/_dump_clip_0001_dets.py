"""Probe: count SCRFD detections per frame in scene_0007 (frames 741..941)."""
import sys
import cv2
import numpy as np

sys.path.insert(0, "/workspace/Face_Pipeline_V2")
from pathlib import Path  # noqa: E402
from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper  # noqa: E402

VIDEO = (
    "/workspace/Face_Pipeline_V2/outputs/"
    "video_MS4wLjABAAAA1NdKwPxv_v0200fg10000d9u4lffog65u8thnp150_open_domain/"
    "stage0_11/videos/MS4wLjABAAAA1NdKwPxv_v0200fg10000d9u4lffog65u8thnp150/"
    "MS4wLjABAAAA1NdKwPxv_v0200fg10000d9u4lffog65u8thnp150_std.mp4"
)

w = SCRFDWrapper(model_path=Path("/workspace/Face_Pipeline_V2/models/SCRFD"),
                 device="cuda", execution_provider="auto")
w.load()
w.confidence_threshold = 0.3
cap = cv2.VideoCapture(VIDEO)

multi = []
counts = {}
for idx in range(741, 942):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    ds = w.detect(frame)
    n = len(ds)
    counts[n] = counts.get(n, 0) + 1
    if n > 1:
        detail = []
        for d in sorted(ds, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True):
            b = tuple(round(float(v), 1) for v in d["bbox"])
            detail.append((b, round(float(d.get("confidence", 0)), 3)))
        multi.append((idx, detail))

print("detection-count distribution:", counts)
print(f"\nframes with >1 detection: {len(multi)}")
for idx, detail in multi:
    print(f"  frame {idx}: {detail}")
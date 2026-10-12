"""Debug: run SCRFD on one Chinese test video, print per-frame detections."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "/workspace/Face_Pipeline_V2")
from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper

MODEL_PATH = Path("/workspace/Face_Pipeline_V2/models/SCRFD")
VIDEO = "/workspace/Face_Pipeline_V2/videos/Chinese_test_videos_kxy/001_24_F_JKYS/001_24_F_JKYS_011_FACE.mp4"

w = SCRFDWrapper(model_path=MODEL_PATH, device="cuda", execution_provider="auto")
w.load()
print("providers:", w.active_providers)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w_, h_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"video={VIDEO}\n  size={w_}x{h_} fps={fps} frames={total}")

step = max(1, int(round(fps / 1.0)))  # sample ~1 fps, same as stage 3
frame_idx = 0
saved = 0
while frame_idx < total:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"frame {frame_idx}: READ FAILED")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_b = float(gray.mean())
    dets = w.detect(frame)
    confs = [round(float(d["confidence"]), 3) for d in dets]
    print(f"frame {frame_idx:>4} (t={frame_idx/fps:5.1f}s): brightness={mean_b:6.1f}  faces={len(dets)}  conf={confs}")
    if saved < 3:
        cv2.imwrite(f"/workspace/Face_Pipeline_V2/debug_frame_{frame_idx:03d}.jpg", frame)
        saved += 1
    frame_idx += step

cap.release()
print("DONE")

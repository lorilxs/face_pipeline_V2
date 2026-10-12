"""Diagnose why Stage 3 rejects all videos: source vs standardized frame stats +
raw SCRFD candidate confidence (bypassing the wrapper's 0.5 internal floor).

Reads frames with cv2 only, prints numeric stats, NEVER saves/reads image files.

Usage:
    python scripts/diag_stage3_frames.py [video_id] [--frames N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper  # noqa: E402


def frame_stats(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def detect_with_floor(w: SCRFDWrapper, frame: np.ndarray, floor: float) -> list[dict]:
    """Run detect() with a temporarily lowered internal confidence floor."""
    orig_conf = w.confidence_threshold
    orig_pre = w.PRE_NMS_SCORE_FILTER
    w.confidence_threshold = floor
    w.PRE_NMS_SCORE_FILTER = min(0.001, floor / 10.0)
    try:
        return w.detect(frame)
    finally:
        w.confidence_threshold = orig_conf
        w.PRE_NMS_SCORE_FILTER = orig_pre


def analyze_capture(
    w: SCRFDWrapper,
    cap: cv2.VideoCapture,
    label: str,
    max_frames: int,
) -> None:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_, h_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(fps / 1.0))
    print(f"\n--- {label} ---")
    print(f"  size={w_}x{h_} fps={fps:.2f} frames={total} step={step}")
    idx = 0
    n = 0
    while idx < total and n < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {idx}: READ_FAILED")
            break
        mean_b, std_b = frame_stats(frame)
        dets_hi = w.detect(frame)  # wrapper default floor 0.5
        dets_lo = detect_with_floor(w, frame, 0.05)  # floor 0.05
        conf_hi = sorted((round(float(d["confidence"]), 3) for d in dets_hi), reverse=True)
        conf_lo = sorted((round(float(d["confidence"]), 3) for d in dets_lo), reverse=True)
        print(
            f"  t={idx/fps:5.1f}s  frame={idx:>4}  "
            f"bright={mean_b:6.1f}±{std_b:5.1f}  "
            f"faces(>=0.5)={len(dets_hi)} conf={conf_hi}  "
            f"faces(>=0.05)={len(dets_lo)} conf_top5={conf_lo[:5]}"
        )
        idx += step
        n += 1
    cap.release()


def main() -> None:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "001_24_F_JKYS_011_FACE"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

    model_dir = ROOT / "models" / "SCRFD"
    w = SCRFDWrapper(model_path=model_dir, device="cuda", execution_provider="auto")
    w.load()
    print(f"providers: {w.active_providers}")

    src = ROOT / "videos" / "Chinese_test_videos_kxy" / "001_24_F_JKYS" / f"{video_id}.mp4"
    std = ROOT / "outputs" / "chinese_001_24_F_JKYS" / f"video_{video_id}" / "stage0_11" / "videos" / video_id / f"{video_id}_std.mp4"
    print(f"source = {src} exists={src.exists()}")
    print(f"std    = {std} exists={std.exists()}")

    if src.exists():
        cap = cv2.VideoCapture(str(src))
        analyze_capture(w, cap, "SOURCE (original, 30fps)", max_frames)
    if std.exists():
        cap = cv2.VideoCapture(str(std))
        analyze_capture(w, cap, "STANDARDIZED (25fps)", max_frames)


if __name__ == "__main__":
    main()

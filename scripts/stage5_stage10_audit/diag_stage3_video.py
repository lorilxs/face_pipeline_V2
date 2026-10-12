"""Diagnose exactly why Stage 3 (SCRFD coarse face screening) rejects ONE video.

Replicates Stage 3 logic from
face_pipeline_v2/stages/stage_03_face_screening/stage.py + scrfd_processor.py:
  - sample frames at sampling_fps (1) -> step = int(fps / sampling_fps)
  - detection floor = detection_confidence_threshold (default 0.5)
  - reject if: no_face (face_detect_ratio < min_face_detect_ratio),
               multi_face (single_face_ratio < min_single_face_ratio),
               low_confidence (avg_face_confidence < min_face_confidence),
               too_short (duration < min_duration)

Usage:
    docker exec face_pipeline_lxs bash -lc "cd /workspace/Face_Pipeline_V2 && \\
        python3 scripts/stage5_stage10_audit/diag_stage3_video.py \\
        outputs/video__P1ZrBf7oNw_4_256.99_270.54_open_domain/stage0_11/videos/\\
        _P1ZrBf7oNw_4_256.99_270.54/_P1ZrBf7oNw_4_256.99_270.54_std.mp4"
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper  # noqa: E402

SAMPLING_FPS = 1
MIN_DURATION = 4.0
MIN_FACE_DETECT_RATIO = 0.90
MIN_SINGLE_FACE_RATIO = 0.95
MIN_FACE_CONFIDENCE = 0.5
DETECTION_FLOOR = 0.5


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not video_path or not Path(video_path).exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    # optional threshold overrides: python diag_stage3_video.py <video> [min_detect_ratio] [min_confidence]
    min_detect_ratio = MIN_FACE_DETECT_RATIO
    min_confidence = MIN_FACE_CONFIDENCE
    if len(sys.argv) > 2:
        min_detect_ratio = float(sys.argv[2])
    if len(sys.argv) > 3:
        min_confidence = float(sys.argv[3])

    print("=== Stage 3 diagnostic ===")
    print(f"video        = {video_path}")
    print(f"sampling_fps = {SAMPLING_FPS}")
    print(f"min_duration = {MIN_DURATION}s")
    print(f"min_face_detect_ratio = {min_detect_ratio}")
    print(f"min_single_face_ratio = {MIN_SINGLE_FACE_RATIO}")
    print(f"min_face_confidence   = {min_confidence}")
    print(f"detection_floor       = {DETECTION_FLOOR}")

    model_dir = ROOT / "models" / "SCRFD"
    w = SCRFDWrapper(model_path=model_dir, device="cuda", execution_provider="auto")
    w.load()
    print(f"providers: {w.active_providers}")
    w.confidence_threshold = DETECTION_FLOOR

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    step = max(1, int(fps / SAMPLING_FPS))
    print(f"fps={fps:.2f} frames={total} duration={duration:.2f}s step={step}")

    frame_indices = list(range(0, total, step))
    print(f"\nsampling {len(frame_indices)} frames at 1fps:")

    frames_with_face = 0
    frames_with_single = 0
    all_confidences: list[float] = []
    max_face_count = 0
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"  t={fi/fps:6.2f}s  frame={fi:>4}  READ_FAILED")
            continue
        dets = w.detect(frame)
        n = len(dets)
        confs = sorted((float(d["confidence"]) for d in dets), reverse=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mark = ""
        if n > 0:
            frames_with_face += 1
            if n == 1:
                frames_with_single += 1
            all_confidences.extend(confs)
            max_face_count = max(max_face_count, n)
        if n == 0:
            mark = "  <-- NO FACE"
        elif n > 1:
            mark = "  <-- MULTI FACE"
        elif confs and confs[0] < min_confidence:
            mark = f"  <-- LOW CONF (<{min_confidence})"
        print(
            f"  t={fi/fps:6.2f}s  frame={fi:>4}  faces={n} "
            f"conf={[round(c,3) for c in confs]}{mark}"
        )
    cap.release()

    sampled = len(frame_indices)
    face_detect_ratio = frames_with_face / sampled if sampled else 0.0
    single_face_ratio = frames_with_single / frames_with_face if frames_with_face else 0.0
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    print("\n=== Aggregate ===")
    print(f"sampled_frame_count    = {sampled}")
    print(f"frames_with_face       = {frames_with_face}")
    print(f"frames_with_single_face= {frames_with_single}")
    print(f"face_detect_ratio      = {face_detect_ratio:.4f}  (need >= {min_detect_ratio})")
    print(f"single_face_ratio      = {single_face_ratio:.4f}  (need >= {MIN_SINGLE_FACE_RATIO})")
    print(f"avg_face_confidence    = {avg_conf:.4f}  (need >= {min_confidence})")
    print(f"max_face_count         = {max_face_count}")
    print(f"duration               = {duration:.2f}s  (need >= {MIN_DURATION})")

    reasons = []
    if duration < MIN_DURATION:
        reasons.append("too_short")
    if face_detect_ratio < min_detect_ratio:
        reasons.append("no_face")
    if single_face_ratio < MIN_SINGLE_FACE_RATIO:
        reasons.append("multi_face")
    if avg_conf < min_confidence:
        reasons.append("low_confidence")
    print(f"\n=== VERDICT: {'PASS' if not reasons else 'REJECT'} -> {reasons}")


if __name__ == "__main__":
    main()

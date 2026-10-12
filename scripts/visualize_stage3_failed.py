#!/usr/bin/env python3
"""Visualize Stage-3 (SCRFD coarse face screening) results for FAILED videos.

For every failed video, this script re-runs the EXACT same detection that
Stage-3 used (SCRFD 10g, detection_floor=0.4) on the exact same sampled
frames (1 FPS sampling recorded in stage3_screening.json), then draws the
detection bbox + 5-point landmarks + confidence onto each sampled frame and
saves it as a JPG.  This lets a human inspect *why* a video was rejected
(no_face / low_confidence / multi_face) without re-running the whole pipeline.

Usage (inside docker, project at /workspace/Face_Pipeline_V2):
    python3 scripts/visualize_stage3_failed.py \
        --src-root outputs/flashtalk_batch_000000 \
        --out-root outputs/flashtalk_batch_000000/stage3_visualization \
        [--limit N] [--device cuda|cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper

# Must match configs/config_dataset_mode.yaml stage_03.detection_confidence_threshold.
DETECTION_FLOOR = 0.4

# 5 landmark indices (SCRFD 5-point: left eye, right eye, nose, left mouth, right mouth)
LANDMARK_NAMES = ["leye", "reye", "nose", "lmouth", "rmouth"]


def load_failed_videos(src_root: Path):
    """Return list of (video_id, video_dir) for final_status == FAIL."""
    failed = []
    for manifest in sorted(src_root.glob("video_*/pipeline_manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("run", {}).get("final_status") != "FAIL":
            continue
        failed.append((manifest.parent.name, manifest.parent))
    return failed


def load_screening(video_dir: Path):
    """Load stage3_screening.json (records sampled frame indices + per-scene metrics)."""
    sdir = video_dir / "stage0_11" / "stage3_screening"
    for jf in sorted(sdir.glob("*_stage3_screening.json")):
        return json.loads(jf.read_text(encoding="utf-8"))
    return None


def find_std_video(video_dir: Path):
    """Locate the standardized *_std.mp4 produced by Stage-1."""
    vdir = video_dir / "stage0_11" / "videos"
    for mp4 in vdir.rglob("*.mp4"):
        if mp4.name.endswith("_std.mp4"):
            return mp4
    for mp4 in vdir.rglob("*.mp4"):
        return mp4
    return None
def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    """Draw bbox + 5 landmarks + confidence for every detection."""
    for det in detections:
        bbox = det.get("bbox")
        conf = float(det.get("confidence", 0.0))
        landmarks = det.get("landmarks")

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"conf={conf:.3f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA,
            )

        if landmarks is not None:
            pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
            for i, (lx, ly) in enumerate(pts):
                cx, cy = int(round(lx)), int(round(ly))
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                if i < len(LANDMARK_NAMES):
                    cv2.putText(
                        frame, LANDMARK_NAMES[i], (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA,
                    )
    return frame


def annotate_header(frame: np.ndarray, video_id: str, scene_id: str, frame_index: int) -> np.ndarray:
    h, w = frame.shape[:2]
    band = np.zeros((46, w, 3), dtype=np.uint8)
    cv2.putText(band, video_id, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(band, f"{scene_id}  frame={frame_index}", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([band, frame])


def annotate_footer(frame: np.ndarray, scene: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    metrics = scene.get("metrics", {})
    reject = ", ".join(scene.get("reject_reasons", [])) or "PASSED"
    text = (
        f"reject=[{reject}]  face_detect_ratio={metrics.get('face_detect_ratio', 0):.3f} "
        f"avg_conf={metrics.get('avg_face_confidence', 0):.3f} "
        f"single_face_ratio={metrics.get('single_face_ratio', 0):.3f}"
    )
    band = np.zeros((46, w, 3), dtype=np.uint8)
    color = (0, 0, 255) if reject != "PASSED" else (0, 255, 0)
    cv2.putText(band, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack([frame, band])
def process_video(video_id: str, video_dir: Path, scrfd, out_root: Path) -> int:
    screening = load_screening(video_dir)
    if not screening:
        print(f"  [skip] {video_id}: no stage3_screening.json")
        return 0

    std_video = find_std_video(video_dir)
    if std_video is None:
        print(f"  [skip] {video_id}: no *_std.mp4")
        return 0

    cap = cv2.VideoCapture(str(std_video))
    if not cap.isOpened():
        print(f"  [skip] {video_id}: cannot open {std_video}")
        return 0

    vout = out_root / video_id
    vout.mkdir(parents=True, exist_ok=True)

    saved = 0
    for scene in screening.get("scenes", []):
        scene_id = scene.get("scene_id", "scene")
        frame_indexes = [int(f["frame_index"]) for f in scene.get("frames", [])]
        for fidx in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                print(f"  [warn] {video_id} frame {fidx}: read failed")
                continue

            detections = scrfd.detect(frame)
            if detections:
                draw_detections(frame, detections)
            else:
                h, w = frame.shape[:2]
                cv2.putText(frame, "NO FACE", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            frame = annotate_header(frame, video_id, scene_id, fidx)
            frame = annotate_footer(frame, scene)
            out_path = vout / f"{scene_id}_frame{fidx:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

    cap.release()
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="outputs/flashtalk_batch_000000")
    ap.add_argument("--out-root", default="outputs/flashtalk_batch_000000/stage3_visualization")
    ap.add_argument("--limit", type=int, default=0, help="only process first N failed videos (0=all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src_root = root / args.src_root
    out_root = root / args.out_root

    failed = load_failed_videos(src_root)
    print(f"failed videos: {len(failed)}")
    if args.limit and args.limit > 0:
        failed = failed[: args.limit]
        print(f"limited to first {len(failed)} videos")

    scrfd = SCRFDWrapper(
        model_path=root / "models" / "SCRFD",
        device=args.device,
        execution_provider="auto",
        confidence_threshold=DETECTION_FLOOR,
    )
    scrfd.load()
    print(f"SCRFD loaded, providers={scrfd.active_providers}, floor={scrfd.confidence_threshold}")

    total_saved = 0
    for i, (video_id, video_dir) in enumerate(failed, 1):
        print(f"[{i}/{len(failed)}] {video_id}")
        try:
            n = process_video(video_id, video_dir, scrfd, out_root)
            total_saved += n
            print(f"  -> {n} images")
        except Exception as exc:
            print(f"  [error] {video_id}: {exc}")

    print(f"DONE. total images: {total_saved}")
    print(f"output dir: {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Analyze Stage 3 (SCRFD coarse face screening) metrics across a batch of videos.

Re-runs the Stage 3 sampling/detection logic on already-standardized videos
(stage1 *_std.mp4) WITHOUT persisting or reading any images, and writes a
per-video CSV + prints distribution summaries so threshold tuning is data-driven.

KEY INSIGHT driving this script: the SCRFD wrapper drops every detection below
its internal CONFIDENCE_THRESHOLD (default 0.5) BEFORE Stage 3 ever sees it.
Faces whose heads are occluded can score ~0.40-0.49 and therefore NEVER reach
Stage 3 (face_detect_ratio=0 -> reject reason "no_face").  To study this we
detect once at a low floor (0.30) and then DERIVE per-floor metrics by
re-filtering the same detections at floors 0.50/0.40/0.35/0.30 — consistent and
fast (single inference pass per video).

Matching production logic:
  - sample step = max(1, int(fps / sampling_fps)), frames = range(0, total, step)
  - face_detect_ratio = frames_with_face / sampled_frames
  - single_face_ratio  = frames_with_single_face / frames_with_face (0 if none)
  - avg_face_confidence = mean(confidence) over ALL kept detections

Threshold scenarios evaluated per (video, floor): det / single / conf
  orig    : 0.80 / 0.95 / 0.60   (what the 2026-09-04 batch actually used)
  current : 0.80 / 0.80 / 0.40   (current configs/config_dataset_mode.yaml)
  lenient : 0.70 / 0.70 / 0.50   (candidate relaxed set)

Usage:
    python scripts/analyze_stage3_metrics.py [--videos-root outputs/chinese_001_24_F_JKYS]
                                             [--csv logs/stage3_metrics_batch.csv]
                                             [--detect-floor 0.30]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper  # noqa: E402

SAMPLING_FPS = 1

# Confidence floors (detection lower bounds) we derive metrics for.
# Detection runs once at the LOWEST floor; higher floors are re-filters.
FLOORS = [0.50, 0.40, 0.35, 0.30]

# Threshold scenarios: (name, min_face_detect_ratio, min_single_face_ratio, min_face_confidence)
SCENARIOS = [
    ("orig", 0.80, 0.95, 0.60),      # used by the 2026-09-04 batch run
    ("current", 0.80, 0.80, 0.40),   # current configs/config_dataset_mode.yaml
    ("lenient", 0.70, 0.70, 0.50),   # candidate
]


def evaluate_scene(
    dets_per_frame: list[list[dict]],
    det: float,
    single: float,
    conf: float,
) -> tuple[bool, list[str]]:
    """Replicate SCRFDProcessor.process_scene filtering decision."""
    sampled = len(dets_per_frame)
    frames_with_face = sum(1 for d in dets_per_frame if len(d) > 0)
    frames_with_single = sum(1 for d in dets_per_frame if len(d) == 1)
    all_conf: list[float] = []
    for d in dets_per_frame:
        for dd in d:
            all_conf.append(float(dd.get("confidence", 0.0)))
    face_detect_ratio = frames_with_face / sampled if sampled else 0.0
    single_face_ratio = (
        frames_with_single / frames_with_face if frames_with_face > 0 else 0.0
    )
    avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0

    reasons: list[str] = []
    if face_detect_ratio == 0.0:
        reasons.append("no_face")
    else:
        if face_detect_ratio < det:
            reasons.append("no_face")
        if single_face_ratio < single:
            reasons.append("multi_face")
        if avg_conf < conf:
            reasons.append("low_confidence")
    return (len(reasons) == 0), reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos-root",
        default=str(ROOT / "outputs" / "chinese_001_24_F_JKYS"),
        help="dir containing video_*/stage0_11/videos/*/*_std.mp4",
    )
    ap.add_argument(
        "--csv",
        default=str(ROOT / "logs" / "stage3_metrics_batch.csv"),
        help="output CSV path (project-root relative or absolute)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only process first N videos (0 = all) — for smoke testing",
    )
    args = ap.parse_args()

    videos_root = Path(args.videos_root)
    std_videos = sorted(
        p for p in videos_root.rglob("*_std.mp4") if p.is_file()
    )
    if not std_videos:
        print(f"No *_std.mp4 found under {videos_root}")
        sys.exit(1)
    if args.limit > 0:
        std_videos = std_videos[: args.limit]
    print(f"Found {len(std_videos)} standardized videos under {videos_root}")

    model_dir = ROOT / "models" / "SCRFD"
    w = SCRFDWrapper(model_path=model_dir, device="cuda", execution_provider="auto")
    w.load()
    detect_floor = min(FLOORS)  # detect once at the lowest floor
    w.confidence_threshold = detect_floor
    print(f"SCRFD providers: {w.active_providers}  detect floor={detect_floor}")

    rows: list[dict] = []
    t0 = time.monotonic()
    for n, vp in enumerate(std_videos, 1):
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"  ! cannot open {vp.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(fps / SAMPLING_FPS))
        frame_indices = list(range(0, total, step))

        raw_per_frame: list[list[dict]] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raw_per_frame.append([])
                continue
            raw_per_frame.append(w.detect(frame))
        cap.release()

        row: dict = {
            "video_id": vp.name.replace("_std.mp4", ""),
            "fps": round(fps, 2),
            "total_frames": total,
            "sampled_frames": len(frame_indices),
        }

        # --- derive per-floor metrics by re-filtering the same detections ---
        for fl in FLOORS:
            dets_per_frame = [
                [d for d in dets if float(d.get("confidence", 0.0)) >= fl]
                for dets in raw_per_frame
            ]
            sampled = len(dets_per_frame)
            frames_with_face = sum(1 for d in dets_per_frame if len(d) > 0)
            frames_with_single = sum(1 for d in dets_per_frame if len(d) == 1)
            all_conf: list[float] = []
            max_face_count = 0
            for d in dets_per_frame:
                max_face_count = max(max_face_count, len(d))
                for dd in d:
                    all_conf.append(float(dd.get("confidence", 0.0)))

            face_detect_ratio = frames_with_face / sampled if sampled else 0.0
            single_face_ratio = (
                frames_with_single / frames_with_face if frames_with_face > 0 else 0.0
            )
            avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0
            min_conf = min(all_conf) if all_conf else 0.0
            max_conf = max(all_conf) if all_conf else 0.0
            p50_conf = statistics.median(all_conf) if all_conf else 0.0

            p = f"f{int(fl * 100):03d}"
            row[f"{p}_frames_with_face"] = frames_with_face
            row[f"{p}_frames_with_single_face"] = frames_with_single
            row[f"{p}_face_detect_ratio"] = round(face_detect_ratio, 4)
            row[f"{p}_single_face_ratio"] = round(single_face_ratio, 4)
            row[f"{p}_n_face_dets"] = len(all_conf)
            row[f"{p}_avg_confidence"] = round(avg_conf, 4)
            row[f"{p}_min_confidence"] = round(min_conf, 4)
            row[f"{p}_p50_confidence"] = round(p50_conf, 4)
            row[f"{p}_max_confidence"] = round(max_conf, 4)
            row[f"{p}_max_face_count"] = max_face_count

            for name, det, single, conf in SCENARIOS:
                passed, reasons = evaluate_scene(dets_per_frame, det, single, conf)
                row[f"{p}_pass_{name}"] = "Y" if passed else "N"
                row[f"{p}_reasons_{name}"] = "+".join(reasons) if reasons else ""
        rows.append(row)
        if n % 8 == 0 or n == len(std_videos):
            print(f"  processed {n}/{len(std_videos)} ({time.monotonic()-t0:.0f}s)")

    # ---- write CSV ----
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["video_id", "fps", "total_frames", "sampled_frames"]
    for fl in FLOORS:
        p = f"f{int(fl * 100):03d}"
        cols += [
            f"{p}_frames_with_face", f"{p}_frames_with_single_face",
            f"{p}_face_detect_ratio", f"{p}_single_face_ratio",
            f"{p}_n_face_dets", f"{p}_avg_confidence", f"{p}_min_confidence",
            f"{p}_p50_confidence", f"{p}_max_confidence", f"{p}_max_face_count",
        ]
        for name, *_ in SCENARIOS:
            cols += [f"{p}_pass_{name}", f"{p}_reasons_{name}"]
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nCSV written -> {out_csv} ({len(rows)} rows)")

    # ---- distribution summary ----
    def _stats(vals: list[float]) -> str:
        vals = sorted(vals)
        n = len(vals)
        q = lambda p: vals[min(n - 1, int(p * n))]
        return (f"min={vals[0]:.4f} p25={q(0.25):.4f} p50={q(0.50):.4f} "
                f"p75={q(0.75):.4f} max={vals[-1]:.4f}")

    print("\n===== STAGE 3 METRIC DISTRIBUTION (n=%d) =====" % len(rows))
    for fl in FLOORS:
        p = f"f{int(fl * 100):03d}"
        print(f"\n--- confidence floor >= {fl:.2f} ---")
        for key in ["face_detect_ratio", "single_face_ratio",
                    "avg_confidence", "min_confidence", "p50_confidence"]:
            vals = [r[f"{p}_{key}"] for r in rows]
            print(f"  {key:<22} {_stats(vals)}")

        print("  face_detect_ratio buckets:")
        for lo, hi, label in [(0.0, 0.5, "<0.5"), (0.5, 0.8, "0.5-0.8"),
                              (0.8, 1.0, "0.8-1.0")]:
            c = sum(1 for r in rows if lo <= r[f"{p}_face_detect_ratio"] < hi)
            print(f"    [{lo:.1f},{hi:.1f}) {label:<8} {c}")
        print("  single_face_ratio buckets:")
        for lo, hi, label in [(0.0, 0.5, "<0.5"), (0.5, 0.8, "0.5-0.8"),
                              (0.8, 0.95, "0.8-0.95"), (0.95, 1.01, "0.95-1.0")]:
            c = sum(1 for r in rows if lo <= r[f"{p}_single_face_ratio"] < hi)
            print(f"    [{lo:.2f},{hi:.2f}) {label:<10} {c}")
        print("  avg_confidence buckets:")
        for lo, hi, label in [(0.0, 0.35, "<0.35"), (0.35, 0.4, "0.35-0.4"),
                              (0.4, 0.5, "0.4-0.5"), (0.5, 0.6, "0.5-0.6"),
                              (0.6, 1.01, ">=0.6")]:
            c = sum(1 for r in rows if lo <= r[f"{p}_avg_confidence"] < hi)
            print(f"    [{lo:.2f},{hi:.2f}) {label:<8} {c}")

        print("  pass counts by scenario:")
        for scen, *_ in SCENARIOS:
            pc = sum(1 for r in rows if r[f"{p}_pass_{scen}"] == "Y")
            print(f"    {scen:<10} pass={pc}/{len(rows)}")
        print("  rejection reason breakdown (orig = 0.80/0.95/0.60):")
        reason_counter: Counter = Counter()
        for r in rows:
            reason_counter[r[f"{p}_reasons_orig"] or "PASS"] += 1
        for k, v in reason_counter.most_common():
            print(f"    {k:<28} {v}")
    print("DONE")


if __name__ == "__main__":
    main()

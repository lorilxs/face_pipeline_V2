#!/usr/bin/env python3
"""Experiment: expand Stage 5 face crop bbox by 1.5x and test SyncNet.

Does NOT modify pipeline code.  Reads existing Stage 5 output, re-crops
with expanded bbox, builds SyncNet input video, and compares results.

Usage:
    python experiment_expand_crop.py
"""

import json, shutil, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[0] if "__file__" in dir() else Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
PROJECT_ROOT = Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "SyncNet"))

from SyncNetInstance_calc_scores import SyncNetInstance

# ── Config ──────────────────────────────────────────────────────────
VALIDATION_DIR = PROJECT_ROOT / "outputs/validation/stage7_rd_radio"
EXPAND_RATIO = 1.8
CLIP_IDX = 0  # clip_0000
# ─────────────────────────────────────────────────────────────────────

def expand_bbox(bbox, img_w, img_h, ratio):
    """Expand bbox from center by ratio, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    hw = (x2 - x1) * ratio / 2
    hh = (y2 - y1) * ratio / 2
    return (
        max(0, int(cx - hw)),
        max(0, int(cy - hh)),
        min(img_w, int(cx + hw)),
        min(img_h, int(cy + hh)),
    )


def build_syncnet_video(crop_dir, audio_path, output_path, fps=25, duration=8):
    """Build video from crop images + audio for SyncNet."""
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crop_files = sorted(crop_dir.glob("*.jpg"))
    for i, f in enumerate(crop_files):
        shutil.copy2(str(f), str(frames_dir / f"{i:06d}.jpg"))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
        "-i", str(audio_path), "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def run_syncnet(video_path, checkpoint, tmp_dir):
    """Run SyncNet and return (offset, lse_c, lse_d, median)."""
    model = SyncNetInstance(device="cpu")
    model.loadParameters(str(checkpoint))
    opt = SimpleNamespace(
        reference=video_path.stem,
        tmp_dir=str(tmp_dir),
        batch_size=32,
        vshift=10,
    )
    offset, conf, minval, median_val = model.evaluate(opt, str(video_path))
    return int(offset), float(conf), float(minval), float(median_val)


def main():
    # ── Load Stage 5 data ──────────────────────────────────────────
    with open(VALIDATION_DIR / "stage_outputs" / "stage5.json") as f:
        sequences = json.load(f)

    seq = sequences[CLIP_IDX]
    clip_id = seq["clip_id"]
    print(f"Clip: {clip_id}")
    print(f"Frames: {len(seq['frames'])}")

    # ── Video path ──────────────────────────────────────────────────
    video_path = VALIDATION_DIR / "videos" / seq["video_id"] / f"{seq['video_id']}_std.mp4"
    audio_path = VALIDATION_DIR / "videos" / seq["video_id"] / f"{seq['video_id']}_audio.wav"
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {img_w}x{img_h}")

    # ── Sample original bbox stats ──────────────────────────────────
    f0 = seq["frames"][0]["faces"][0]
    orig_bbox = f0["bbox"]
    print(f"Original bbox (frame 0): {[f'{v:.0f}' for v in orig_bbox]}")
    print(f"Original bbox size: {orig_bbox[2]-orig_bbox[0]:.0f} x {orig_bbox[3]-orig_bbox[1]:.0f}")

    # ── Build frame_index → bbox lookup ────────────────────────────
    frame_bboxes = {}
    for frame in seq["frames"]:
        faces = frame.get("faces", [])
        if faces:
            frame_bboxes[frame["frame_index"]] = faces[0]["bbox"]

    # ── Build expanded crops ────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="expand_crop_"))
    expanded_dir = tmp_dir / "expanded_crops"
    expanded_dir.mkdir()

    src_crop_dir = Path(seq["frames"][0]["faces"][0]["face_crop_path"]).parent

    frame_indices = sorted(frame_bboxes.keys())
    print(f"\nGenerating expanded crops (ratio={EXPAND_RATIO})...")

    for fi in frame_indices:
        bbox = frame_bboxes[fi]
        ex_bbox = expand_bbox(bbox, img_w, img_h, EXPAND_RATIO)
        x1, y1, x2, y2 = ex_bbox

        # Read frame from standardized video
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (256, 256))
        cv2.imwrite(str(expanded_dir / f"frame_{fi:06d}_face_000.jpg"), crop_resized)

    cap.release()

    count = len(list(expanded_dir.glob("*.jpg")))
    print(f"Generated {count} expanded crops")

    # ── Show first expanded crop stats ──────────────────────────────
    first_crop = cv2.imread(str(expanded_dir / f"frame_{frame_indices[0]:06d}_face_000.jpg"))
    print(f"Expanded crop mean pixel: {first_crop.mean():.1f}")
    print(f"Original crop mean pixel: ~169.6 (from previous test)")

    # ── Build video & run SyncNet ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Running SyncNet on EXPANDED crop...")
    print("=" * 60)
    exp_video = tmp_dir / "expanded_clip.mp4"
    build_syncnet_video(expanded_dir, audio_path, exp_video, fps=25, duration=8)
    exp_offset, exp_lse_c, exp_lse_d, exp_median = run_syncnet(
        exp_video, PROJECT_ROOT / "models/syncnet/syncnet_v2.model", tmp_dir
    )

    print("\n" + "=" * 60)
    print("Running SyncNet on ORIGINAL crop (existing output)...")
    print("=" * 60)
    orig_video = tmp_dir / "original_clip.mp4"
    build_syncnet_video(src_crop_dir, audio_path, orig_video, fps=25, duration=8)
    orig_offset, orig_lse_c, orig_lse_d, orig_median = run_syncnet(
        orig_video, PROJECT_ROOT / "models/syncnet/syncnet_v2.model", tmp_dir
    )

    # ── Compare ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Original':>12} {'Expanded (1.5x)':>18} {'Delta':>10}")
    print("-" * 60)
    print(f"{'lse_c':<20} {orig_lse_c:>12.4f} {exp_lse_c:>18.4f} {exp_lse_c - orig_lse_c:>+10.4f}")
    print(f"{'lse_d':<20} {orig_lse_d:>12.4f} {exp_lse_d:>18.4f} {exp_lse_d - orig_lse_d:>+10.4f}")
    print(f"{'offset':<20} {orig_offset:>12d} {exp_offset:>18d} {exp_offset - orig_offset:>+10d}")
    print(f"{'median':<20} {orig_median:>12.4f} {exp_median:>18.4f} {exp_median - orig_median:>+10.4f}")

    # ── Sample expanded crop stats ──────────────────────────────────
    exp_samples = []
    for fi in frame_indices[:5]:
        img = cv2.imread(str(expanded_dir / f"frame_{fi:06d}_face_000.jpg"))
        if img is not None:
            exp_samples.append(img.mean())
    if exp_samples:
        print(f"\nExpanded crop mean pixel (first 5): {exp_samples}")
        print(f"Expanded crop mean avg: {np.mean(exp_samples):.1f}")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Conclusion ──────────────────────────────────────────────────
    lse_c_improvement = (exp_lse_c - orig_lse_c) / max(orig_lse_c, 0.001) * 100
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if lse_c_improvement > 10:
        print(f"✅ Expanded bbox (1.5x) improves LSE-C by {lse_c_improvement:.0f}%")
        print("   Larger crop context helps SyncNet lip-reading.")
    elif lse_c_improvement > 0:
        print(f"→ Expanded bbox gives mild improvement ({lse_c_improvement:.0f}%)")
        print("  Effect is positive but not dramatic.")
    else:
        print(f"❌ Expanded bbox does NOT improve LSE-C ({lse_c_improvement:.0f}%)")
        print("   The issue is NOT crop tightness but something else.")


if __name__ == "__main__":
    main()

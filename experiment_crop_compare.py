#!/usr/bin/env python3
"""Experiment: compare 3 face crop methods for SyncNet.

A — bbox expand=1.5 → resize 224
B — landmark_5 alignment → 224×224
C — bbox expand=1.5 → landmark alignment → 224×224

Usage: python experiment_crop_compare.py
"""

import json, shutil, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "SyncNet"))
from SyncNetInstance_calc_scores import SyncNetInstance

# ── Config ──────────────────────────────────────────────────────────
VALIDATION_DIR = PROJECT_ROOT / "outputs/validation/stage7_rd_radio"
EXPAND_RATIO = 1.5
CLIP_IDX = 0
CROP_SIZE = 224  # SyncNet input size

# ArcFace 112×112 target points scaled to 224×224
ARC_SRC_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.6963],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.3655],   # right mouth
], dtype=np.float32)
ALIGN_TARGET = ARC_SRC_112 * (CROP_SIZE / 112.0)
# ─────────────────────────────────────────────────────────────────────


def expand_bbox(bbox, img_w, img_h, ratio):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) * ratio / 2
    hh = (y2 - y1) * ratio / 2
    return (max(0, int(cx - hw)), max(0, int(cy - hh)),
            min(img_w, int(cx + hw)), min(img_h, int(cy + hh)))


def crop_expand(frame, bbox, img_w, img_h, out_size=224):
    """A: expand bbox → resize."""
    x1, y1, x2, y2 = expand_bbox(bbox, img_w, img_h, EXPAND_RATIO)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (out_size, out_size))


def crop_align(frame, landmark_5, out_size=224):
    """B: 5-point affine alignment."""
    src = np.array(landmark_5, dtype=np.float32).reshape(5, 2)
    M = cv2.estimateAffinePartial2D(src, ALIGN_TARGET, method=cv2.RANSAC)[0]
    if M is None:
        # fallback: use first 3 points
        M = cv2.getAffineTransform(src[:3], ALIGN_TARGET[:3])
    return cv2.warpAffine(frame, M, (out_size, out_size),
                          borderMode=cv2.BORDER_REPLICATE)


def crop_expand_align(frame, bbox, landmark_5, img_w, img_h, out_size=224):
    """C: expand bbox → affine alignment."""
    x1, y1, x2, y2 = expand_bbox(bbox, img_w, img_h, EXPAND_RATIO)
    sub = frame[y1:y2, x1:x2]
    # Adjust landmarks to sub-image coordinates
    src = np.array(landmark_5, dtype=np.float32).reshape(5, 2)
    src[:, 0] -= x1
    src[:, 1] -= y1
    M = cv2.estimateAffinePartial2D(src, ALIGN_TARGET, method=cv2.RANSAC)[0]
    if M is None:
        M = cv2.getAffineTransform(src[:3], ALIGN_TARGET[:3])
    return cv2.warpAffine(sub, M, (out_size, out_size),
                          borderMode=cv2.BORDER_REPLICATE)


def build_syncnet_video(crop_dir, audio_path, output_path, fps=25, duration=8):
    frames_dir = output_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(sorted(Path(crop_dir).glob("*.jpg"))):
        shutil.copy2(str(f), str(frames_dir / f"{i:06d}.jpg"))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps), "-i", str(frames_dir / "%06d.jpg"),
        "-i", str(audio_path), "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-pix_fmt", "yuv420p", "-shortest",
        str(output_path),
    ], check=True)
    return output_path


def run_syncnet(video_path, checkpoint, tmp_dir):
    model = SyncNetInstance(device="cpu")
    model.loadParameters(str(checkpoint))
    opt = SimpleNamespace(reference=video_path.stem, tmp_dir=str(tmp_dir),
                          batch_size=32, vshift=10)
    offset, conf, minval, median_val = model.evaluate(opt, str(video_path))
    return int(offset), float(conf), float(minval), float(median_val)


def main():
    # ── Load data ───────────────────────────────────────────────────
    with open(VALIDATION_DIR / "stage_outputs" / "stage5.json") as f:
        sequences = json.load(f)
    seq = sequences[CLIP_IDX]
    video_path = VALIDATION_DIR / "videos" / seq["video_id"] / f"{seq['video_id']}_std.mp4"
    audio_path = VALIDATION_DIR / "videos" / seq["video_id"] / f"{seq['video_id']}_audio.wav"
    checkpoint = PROJECT_ROOT / "models/syncnet/syncnet_v2.model"

    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Build frame data lookup ─────────────────────────────────────
    frames_data = []
    for frame in seq["frames"]:
        faces = frame.get("faces", [])
        if faces:
            frames_data.append({
                "frame_index": frame["frame_index"],
                "bbox": faces[0]["bbox"],
                "landmark_5": faces[0]["landmark_5"],
            })

    print(f"Clip: {seq['clip_id']}")
    print(f"Frames with faces: {len(frames_data)}")
    print(f"Video: {img_w}x{img_h}")
    print(f"Target alignment points (224x224):\n{ALIGN_TARGET}")

    # ── Generate crops for all 3 methods ────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="crop_cmp_"))
    methods = {
        "A_expand1.5": crop_expand,
        "B_align": crop_align,
        "C_expand_align": crop_expand_align,
    }

    results = {}
    for name, crop_fn in methods.items():
        crop_dir = tmp_dir / name
        crop_dir.mkdir()
        print(f"\nGenerating {name}...")

        for fd in frames_data:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fd["frame_index"])
            ret, frame = cap.read()
            if not ret:
                continue

            if name == "A_expand1.5":
                crop = crop_fn(frame, fd["bbox"], img_w, img_h, CROP_SIZE)
            elif name == "B_align":
                crop = crop_fn(frame, fd["landmark_5"], CROP_SIZE)
            else:  # C_expand_align
                crop = crop_fn(frame, fd["bbox"], fd["landmark_5"], img_w, img_h, CROP_SIZE)

            cv2.imwrite(str(crop_dir / f"frame_{fd['frame_index']:06d}_face_000.jpg"), crop)

        # Stats
        samples = sorted(crop_dir.glob("*.jpg"))[:5]
        means = [cv2.imread(str(s)).mean() for s in samples if cv2.imread(str(s)) is not None]
        print(f"  {len(list(crop_dir.glob('*.jpg')))} crops, mean_pixel(first5)={np.mean(means):.1f}")

        # Build video + run SyncNet
        video = tmp_dir / f"{name}.mp4"
        build_syncnet_video(crop_dir, audio_path, video)
        offset, lse_c, lse_d, median = run_syncnet(video, checkpoint, tmp_dir)
        results[name] = {"offset": offset, "lse_c": lse_c, "lse_d": lse_d, "median": median}
        print(f"  → offset={offset}, lse_c={lse_c:.4f}, lse_d={lse_d:.4f}, median={median:.4f}")

    cap.release()

    # ── Save sample visualization crops ─────────────────────────────
    vis_dir = Path("/tmp/crop_compare_vis")
    vis_dir.mkdir(exist_ok=True)
    for name in methods:
        samples = sorted((tmp_dir / name).glob("*.jpg"))[:3]
        # Save first 3 frames as samples
        out_sub = vis_dir / name
        out_sub.mkdir(exist_ok=True)
        for s in samples:
            shutil.copy2(str(s), str(out_sub / s.name))
        # Also create a 3-frame horizontal strip for easy comparison
        imgs = [cv2.imread(str(s)) for s in samples if cv2.imread(str(s)) is not None]
        if len(imgs) >= 3:
            strip = np.hstack(imgs)
            cv2.imwrite(str(vis_dir / f"{name}_strip.jpg"), strip)

    # ── Comparison table ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(f"{'Method':<24} {'LSE-C':>8} {'LSE-D':>8} {'offset':>8} {'median':>8}")
    print("-" * 56)
    for name, r in results.items():
        label = {"A_expand1.5": "A: expand 1.5x",
                 "B_align": "B: align (5pt)",
                 "C_expand_align": "C: expand + align"}[name]
        print(f"{label:<24} {r['lse_c']:>8.4f} {r['lse_d']:>8.3f} {r['offset']:>8d} {r['median']:>8.3f}")

    print(f"\nVisualization samples: {vis_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Conclusion ──────────────────────────────────────────────────
    best = max(results.items(), key=lambda x: x[1]["lse_c"])
    print(f"\nBest: {best[0]} (lse_c={best[1]['lse_c']:.4f})")


if __name__ == "__main__":
    main()

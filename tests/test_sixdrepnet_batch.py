#!/usr/bin/env python3
"""Test script: compare 6DRepNet single-image vs batch inference.

Loads face crop images, runs predict() and predict_batch(),
verifies results are identical within tolerance, and benchmarks speed.

Usage:
    docker exec face_pipeline_lxs python3 \
        /workspace/Face_Pipeline_V2/tests/test_sixdrepnet_batch.py

    python3 tests/test_sixdrepnet_batch.py
"""

import argparse
import sys
import time
from pathlib import Path

# ---- path setup ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


def _collect_images(faces_dir: Path, max_images: int = 300) -> list:
    """Collect face crop images, apply resize_face (256x256) as pose_estimator does."""
    from face_pipeline_v2.utils.face_preprocess import resize_face

    images = []
    for crop_dir in faces_dir.rglob("crops"):
        for img_path in sorted(crop_dir.glob("*.jpg")):
            if len(images) >= max_images:
                break
            raw = cv2.imread(str(img_path))
            if raw is not None:
                image = resize_face(raw, 256)
                images.append(image)
        if len(images) >= max_images:
            break
    return images


def _load_model():
    """Load 6DRepNet model via wrapper."""
    from face_pipeline_v2.models.wrappers.sixdrepnet import SixDRepNetWrapper

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "6DRepNet"

    wrapper = SixDRepNetWrapper(model_path=model_path, device="auto")
    wrapper.load()
    print(f"Model loaded: device={wrapper.device}, "
          f"gpu={wrapper._model.gpu}")
    return wrapper


def compare_results(images, model, num_images):
    """Compare single-image predict() vs batch predict_batch()."""
    subset = images[:num_images]

    print(f"\n{'='*60}")
    print(f"Comparing {len(subset)} images: predict() vs predict_batch()")
    print(f"{'='*60}")

    # Single-image inference
    t0 = time.time()
    results_old = model.predict(subset)
    t_old = time.time() - t0
    print(f"\n  predict()      : {t_old:.3f}s  "
          f"({len(subset)/t_old:.1f} img/s)")

    # Batch inference
    t0 = time.time()
    results_new = model.predict_batch(subset, batch_size=32)
    t_new = time.time() - t0
    print(f"  predict_batch(): {t_new:.3f}s  "
          f"({len(subset)/t_new:.1f} img/s)")
    print(f"  Speedup: {t_old/t_new:.2f}x")

    # Compare per-frame
    max_diff_yaw = 0.0
    max_diff_pitch = 0.0
    max_diff_roll = 0.0
    diffs = []

    for i, (old, new) in enumerate(zip(results_old, results_new)):
        dy = abs(old["yaw"] - new["yaw"])
        dp = abs(old["pitch"] - new["pitch"])
        dr = abs(old["roll"] - new["roll"])
        diffs.append((i, old["yaw"], new["yaw"], dy,
                      old["pitch"], new["pitch"], dp,
                      old["roll"], new["roll"], dr))
        if dy > max_diff_yaw:
            max_diff_yaw = dy
        if dp > max_diff_pitch:
            max_diff_pitch = dp
        if dr > max_diff_roll:
            max_diff_roll = dr

    # Print first 10 diffs
    hdr = (f"  {'frame':>6s}  {'yaw_old':>8s}  {'yaw_new':>8s}  "
           f"{'diff_yaw':>9s}  {'pitch_old':>10s}  {'pitch_new':>10s}  "
           f"{'diff_pitch':>11s}  {'roll_old':>9s}  {'roll_new':>9s}  "
           f"{'diff_roll':>10s}")
    print(f"\n{hdr}")
    print("  " + "-" * 110)
    for i, yo, yn, dy, po, pn, dp, ro, rn, dr in diffs[:10]:
        print(f"  {i:6d}  {yo:8.3f}  {yn:8.3f}  {dy:9.5f}  "
              f"{po:10.3f}  {pn:10.3f}  {dp:11.5f}  "
              f"{ro:9.3f}  {rn:9.3f}  {dr:10.5f}")

    if len(diffs) > 10:
        print(f"  ... ({len(diffs) - 10} more frames)")

    print(f"\n  Max differences:")
    print(f"    yaw   : {max_diff_yaw:.6f}")
    print(f"    pitch : {max_diff_pitch:.6f}")
    print(f"    roll  : {max_diff_roll:.6f}")

    threshold = 0.1
    if (max_diff_yaw < threshold and max_diff_pitch < threshold
            and max_diff_roll < threshold):
        print(f"\n  PASS: All differences < {threshold} degree")
    else:
        print(f"\n  FAIL: Max difference exceeds {threshold} degree")


def benchmark(images, model, num_images):
    """Benchmark batch inference with different batch sizes."""
    subset = images[:num_images]

    print(f"\n{'='*60}")
    print(f"Benchmark: batch size vs throughput ({len(subset)} images)")
    print(f"{'='*60}")

    # Warmup
    _ = model.predict_batch(subset[:32], batch_size=32)

    for bs in [8, 16, 32, 64, 128]:
        t0 = time.time()
        _ = model.predict_batch(subset, batch_size=bs)
        elapsed = time.time() - t0
        print(f"  batch_size={bs:3d}: {elapsed:.3f}s  "
              f"({len(subset)/elapsed:.1f} img/s)")


def main():
    parser = argparse.ArgumentParser(
        description="Test 6DRepNet batch inference correctness and speed"
    )
    parser.add_argument(
        "--faces-dir", type=str,
        default="outputs/stage0_11_test_verify/faces",
        help="Directory containing face crop images",
    )
    parser.add_argument(
        "--num-images", type=int, default=300,
        help="Number of images to test (default: 300)",
    )
    parser.add_argument(
        "--collect-only", type=int, default=0,
        help="Max images to collect from disk (default: same as --num-images)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    faces_dir = project_root / args.faces_dir

    if not faces_dir.exists():
        print(f"ERROR: faces directory not found: {faces_dir}")
        print("Run the integration test first to generate face crops.")
        return

    max_collect = args.collect_only if args.collect_only > 0 else args.num_images
    print(f"Collecting up to {max_collect} images from {faces_dir} ...")
    images = _collect_images(faces_dir, max_images=max_collect)
    print(f"Collected {len(images)} images.")

    if len(images) < args.num_images:
        print(f"WARNING: Only {len(images)} images available "
              f"(requested {args.num_images}).")

    print("\nLoading 6DRepNet model ...")
    model = _load_model()

    n = min(args.num_images, len(images))
    compare_results(images, model, n)
    benchmark(images, model, n)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Stage13 Gaze Runtime Verification (host Python, MediaPipe FaceLandmarker).

Runs the *actual* Stage13 processor functions (create_mediapipe_detector /
extract_gaze_parameters) on 50 real aligned-face frames from Stage11 output,
then validates:

  1. GazeFeature count == 50 (T frames)
  2. gaze_vector shape [T, 6], left_eye/right_eye 3-tuples
  3. timestamps correct: timestamp == frame_index / 25.0
  4. no NaN / Inf; each eye vector near unit norm

Finally saves via Stage13's save_gaze_features() and round-trips through
read_gaze_features() to confirm the NPZ handoff for Stage17.

No Stage13 architecture changes — this only invokes the existing public API.
"""

import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config (mirrors configs/config.yaml -> stage13 + manifest.json fps=25)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
MODEL_PATH = PROJECT_ROOT / "models/mediapipe/face_landmarker.task"
ALIGNED_FACES_DIR = (
    PROJECT_ROOT
    / "outputs/stage0_11_test_verify/aligned_faces/"
    / "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs/stage13_gaze"
FPS = 25.0
IMAGE_SIZE = (256, 256)
NUM_FRAMES = 50

sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.data.models import AlignedFaceInfo, GazeFeature  # noqa: E402
from face_pipeline_v2.stages.stage_13_gaze.processor import (  # noqa: E402
    create_mediapipe_detector,
    extract_gaze_parameters,
)
from face_pipeline_v2.stages.stage_13_gaze.integration import (  # noqa: E402
    save_gaze_features,
    read_gaze_features,
)

FRAME_RE = re.compile(r"^aligned_(\d{6})_.+\.(?:jpg|png)$")


def parse_frame_index(img_path: Path) -> int:
    """Parse frame_index from aligned filename: aligned_000100_... -> 100."""
    m = FRAME_RE.match(img_path.name)
    if not m:
        raise ValueError(f"Cannot parse frame_index from filename: {img_path.name}")
    return int(m.group(1))


def main() -> int:
    clip_id = ALIGNED_FACES_DIR.name

    # --- 1. Build AlignedFaceInfo for the first NUM_FRAMES (sorted, consecutive) ---
    images = sorted(ALIGNED_FACES_DIR.glob("*.jpg")) + sorted(ALIGNED_FACES_DIR.glob("*.png"))
    images = sorted(images, key=parse_frame_index)
    if not images:
        print(f"[FAIL] No aligned face images in {ALIGNED_FACES_DIR}")
        return 1

    sampled = images[:NUM_FRAMES]
    faces: list[AlignedFaceInfo] = []
    for img in sampled:
        frame_idx = parse_frame_index(img)
        faces.append(
            AlignedFaceInfo(
                clip_id=clip_id,
                frame_index=frame_idx,
                face_id="face_0",
                timestamp=float(frame_idx) / FPS,
                aligned_face_path=img,
                affine_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                original_image_size=IMAGE_SIZE,
                aligned_resolution=IMAGE_SIZE,
            )
        )

    print("=" * 70)
    print("STAGE13 GAZE RUNTIME VERIFICATION (host Python /opt/hb/bin/python3)")
    print("=" * 70)
    print(f"clip          : {clip_id}")
    print(f"input frames  : {NUM_FRAMES} (frame {parse_frame_index(sampled[0])}.."
          f"{parse_frame_index(sampled[-1])})")
    print(f"model         : {MODEL_PATH}")
    print(f"mediapipe     : {__import__('mediapipe').__version__}")

    # --- 2. Run actual Stage13 processor ---
    detector = create_mediapipe_detector(model_path=str(MODEL_PATH), delegate="cpu")
    try:
        results: list[GazeFeature] = extract_gaze_parameters(
            aligned_faces=faces, detector=detector, image_size=IMAGE_SIZE
        )
    finally:
        detector.close()

    checks: list[tuple[str, bool, str]] = []

    # --- 3. Validation ---
    ok_count = len(results) == len(faces)
    checks.append(("extract count == 50", ok_count,
                   f"got {len(results)} / {len(faces)}"))

    all_gaze = np.array([f.gaze_vector for f in results], dtype=np.float64)
    checks.append(("gaze_vector shape [T,6]", all_gaze.ndim == 2 and all_gaze.shape[1] == 6,
                   f"shape={list(all_gaze.shape)}"))

    no_nan = bool(not np.any(np.isnan(all_gaze)))
    checks.append(("no NaN", no_nan, f"nan count={int(np.isnan(all_gaze).sum())}"))

    no_inf = bool(not np.any(np.isinf(all_gaze)))
    checks.append(("no Inf", no_inf, f"inf count={int(np.isinf(all_gaze).sum())}"))

    left = all_gaze[:, 0:3]
    right = all_gaze[:, 3:6]
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    norm_ok = bool(np.all(np.abs(left_norm - 1.0) < 0.05)
                   and np.all(np.abs(right_norm - 1.0) < 0.05))
    checks.append(("eye norms ~ 1.0 (±0.05)", norm_ok,
                   f"left=[{left_norm.min():.4f},{left_norm.max():.4f}] "
                   f"right=[{right_norm.min():.4f},{right_norm.max():.4f}]"))

    # timestamp correctness: timestamp == frame_index / 25.0, strictly increasing
    ts = np.array([f.timestamp for f in results], dtype=np.float64)
    fidx = np.array([f.frame_index for f in results], dtype=np.float64)
    ts_ok = bool(np.allclose(ts, fidx / FPS, atol=1e-6))
    mono_ok = bool(np.all(np.diff(fidx) > 0))
    checks.append(("timestamp == frame_index/25", ts_ok,
                   f"first={ts[0]:.4f}s last={ts[-1]:.4f}s"))
    checks.append(("frame_index strictly increasing", mono_ok,
                   f"range=[{int(fidx[0])},{int(fidx[-1])}]"))

    clip_ok = bool(all(f.clip_id == clip_id for f in results))
    checks.append(("clip_id consistent", clip_ok, clip_id))

    # --- 4. NPZ handoff round-trip (Stage13 -> Stage17) ---
    npz_path = save_gaze_features(results, OUTPUT_DIR, clip_id)
    restored = read_gaze_features(OUTPUT_DIR, clip_id)
    rt_ok = len(restored) == len(results) and all(
        np.allclose(np.array(a.gaze_vector, dtype=np.float64),
                    np.array(b.gaze_vector, dtype=np.float64))
        for a, b in zip(restored, results)
    )
    checks.append(("NPZ save/read round-trip", rt_ok,
                   f"{npz_path} (dtype={np.load(npz_path)['gaze_vector'].dtype})"))

    # --- 5. Report ---
    print("-" * 70)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")
    print("-" * 70)

    print(f"\nAggregate gaze stats:")
    print(f"  mean: {all_gaze.mean(axis=0)}")
    print(f"  std : {all_gaze.std(axis=0)}")
    print(f"  range: [{all_gaze.min():.4f}, {all_gaze.max():.4f}]")

    all_ok = all(ok for _, ok, _ in checks)
    print(f"\nStage13 runtime verification: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())


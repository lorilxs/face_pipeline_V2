"""Real-data test for Stage13 Gaze — runs on 50 random frames from Stage11 output.

Requires mediapipe to be installed.
"""

import random
from pathlib import Path
from typing import List

import numpy as np
import pytest

from face_pipeline_v2.data.models import AlignedFaceInfo, GazeFeature
from face_pipeline_v2.stages.stage_13_gaze.processor import (
    create_mediapipe_detector,
    extract_gaze_parameters,
    _HAS_MEDIAPIPE,
)

pytestmark = pytest.mark.skipif(
    not _HAS_MEDIAPIPE,
    reason="mediapipe not installed",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "models/mediapipe/face_landmarker.task"
ALIGNED_FACES_DIR = (
    "outputs/stage0_11_test_verify/aligned_faces/"
    "CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001"
)
NUM_FRAMES = 50


def _find_aligned_faces(base_dir: Path) -> List[Path]:
    """Find all PNG/JPG aligned face images in a directory."""
    images = sorted(base_dir.glob("*.png")) + sorted(base_dir.glob("*.jpg"))
    return images


def _build_aligned_face_info(img_path: Path, clip_id: str) -> AlignedFaceInfo:
    """Build AlignedFaceInfo from an image path."""
    frame_idx = int(img_path.stem) if img_path.stem.isdigit() else 0
    return AlignedFaceInfo(
        clip_id=clip_id,
        frame_index=frame_idx,
        face_id="face_0",
        timestamp=float(frame_idx) / 25.0,
        aligned_face_path=img_path,
        affine_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        original_image_size=(256, 256),
        aligned_resolution=(256, 256),
    )


class TestRealDataGaze:
    """Real data test with random 50 frames from Stage11 output."""

    def test_real_frames_gaze_extraction(self) -> None:
        base_dir = Path(ALIGNED_FACES_DIR)
        if not base_dir.exists():
            pytest.skip(f"Aligned faces dir not found: {base_dir}")

        all_images = _find_aligned_faces(base_dir)
        if len(all_images) == 0:
            pytest.skip("No aligned face images found")

        # Sample 50 random frames
        sample_size = min(NUM_FRAMES, len(all_images))
        sampled = random.sample(all_images, sample_size)

        clip_id = base_dir.name
        faces = [_build_aligned_face_info(img, clip_id) for img in sampled]

        # Create detector
        detector = create_mediapipe_detector(
            model_path=MODEL_PATH,
            delegate="cpu",
        )

        try:
            results = extract_gaze_parameters(
                aligned_faces=faces,
                detector=detector,
                image_size=(256, 256),
            )
        finally:
            detector.close()

        # --- Validations ---
        assert len(results) > 0, "No gaze features extracted"
        assert len(results) == sample_size, (
            f"Expected {sample_size} results, got {len(results)}"
        )

        # Check each result
        for i, f in enumerate(results):
            gv = np.array(f.gaze_vector, dtype=np.float64)

            # No NaN
            assert not np.any(np.isnan(gv)), (
                f"NaN in gaze_vector at frame {f.frame_index}: {gv}"
            )

            # No Inf
            assert not np.any(np.isinf(gv)), (
                f"Inf in gaze_vector at frame {f.frame_index}: {gv}"
            )

            # Norm ≈ 1.0 for each eye
            left = np.array(f.left_eye, dtype=np.float64)
            right = np.array(f.right_eye, dtype=np.float64)
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)

            assert abs(left_norm - 1.0) < 0.05, (
                f"Left eye norm {left_norm:.4f} at frame {f.frame_index}"
            )
            assert abs(right_norm - 1.0) < 0.05, (
                f"Right eye norm {right_norm:.4f} at frame {f.frame_index}"
            )

            # Field consistency
            assert len(f.gaze_vector) == 6
            assert len(f.left_eye) == 3
            assert len(f.right_eye) == 3
            assert f.clip_id == clip_id
            assert isinstance(f.timestamp, float)

        # Aggregate statistics
        all_gaze = np.array([f.gaze_vector for f in results], dtype=np.float64)
        print(f"\n[Stage13 Real Data Test] {len(results)} frames")
        print(f"  gaze_vector mean:  {all_gaze.mean(axis=0)}")
        print(f"  gaze_vector std:   {all_gaze.std(axis=0)}")
        print(f"  gaze_vector range: [{all_gaze.min():.4f}, {all_gaze.max():.4f}]")
"""Tests for Stage 11: Face Alignment.

Tests cover:
    1. Affine transform: key point selection, estimate, warp, inverse, matrix roundtrip.
    2. Processor: align_faces produces AlignedFaceInfo with correct fields.
    3. Stage: full run, disabled, validation, immutability.
    4. Error handling: missing crops, bad landmarks, affine failure.
    5. Coordinate recovery: affine inverse roundtrip.
    6. No SCRFD / InsightFace / model calls.
"""

from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    AlignedFaceInfo,
    FaceInfo,
    FaceSequence,
    FrameInfo,
    LandmarkInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_11_alignment.affine_transform import (
    TEMPLATE_256,
    TEMPLATE_EYES,
    build_template,
    decompose_affine,
    estimate_affine,
    invert_affine,
    matrix_to_tuple,
    select_eye_points,
    select_nose_point,
    tuple_to_matrix,
    warp_face,
)
from face_pipeline_v2.stages.stage_11_alignment.processor import (
    align_faces,
)
from face_pipeline_v2.stages.stage_11_alignment.stage import (
    Stage11Alignment,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "alignment": {
            "enabled": True,
            "resolution": 256,
            "save_aligned_faces": True,
            "output_dir": "outputs/aligned_faces",
        },
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_11", log_dir=None),
        model_manager=object(),
    )


def _face(
    frame_index: int = 0,
    face_id: str = "f0",
    crop_path: str = "/tmp/crop.jpg",
    bbox: tuple = (0.0, 0.0, 256.0, 256.0),
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index, face_id=face_id,
        bbox=bbox, landmark_5=(),
        confidence=0.9, face_crop_path=Path(crop_path),
    )


def _frame(idx: int, clip_id: str = "c0", faces: tuple = ()) -> FrameInfo:
    return FrameInfo(
        frame_index=idx, clip_id=clip_id, timestamp=idx / 25.0,
        image_path=Path(""), faces=faces,
    )


def _seq(clip_id: str = "c0", frames: tuple = ()) -> FaceSequence:
    return FaceSequence(clip_id=clip_id, video_id="v0", frames=frames)


def _make_landmarks(
    num_landmarks: int = 106,
    clip_id: str = "c0",
    frame_index: int = 0,
    face_id: str = "f0",
) -> LandmarkInfo:
    """Generate synthetic 106 landmarks forming a face shape in crop space.

    The bbox is (0,0,256,256) so crop space = frame space.
    """
    pts = np.zeros((num_landmarks, 2), dtype=np.float32)
    for i in range(num_landmarks):
        # Create plausible face-shaped landmarks in a 256×256 crop
        if i < 33:  # contour
            angle = (i / 32.0) * np.pi * 0.9 + np.pi * 0.05
            pts[i, 0] = 128 + 70 * np.cos(angle)
            pts[i, 1] = 130 + 80 * np.sin(angle)
        elif i < 43:  # left eyebrow
            pts[i, 0] = 85 + (i - 33) * 4
            pts[i, 1] = 90
        elif i < 51:  # right eyebrow
            pts[i, 0] = 155 + (i - 43) * 4
            pts[i, 1] = 90
        elif i < 63:  # nose
            pts[i, 0] = 120 + (i - 51) * 2
            pts[i, 1] = 100 + (i - 51) * 10
        elif i < 67:  # left eye upper
            pts[i, 0] = 95 + (i - 63) * 6
            pts[i, 1] = 100
        elif i < 71:  # right eye upper
            pts[i, 0] = 155 + (i - 67) * 6
            pts[i, 1] = 100
        elif i < 75:  # left eye lower
            pts[i, 0] = 95 + (i - 71) * 6
            pts[i, 1] = 105
        elif i < 79:  # right eye lower
            pts[i, 0] = 155 + (i - 75) * 6
            pts[i, 1] = 105
        elif i < 95:  # mouth
            pts[i, 0] = 105 + (i - 79) * 3
            pts[i, 1] = 170
        elif i < 99:  # pupils
            pts[i, 0] = 110 + (i - 95) * 50
            pts[i, 1] = 102
        else:
            pts[i, 0] = 128
            pts[i, 1] = 130

    # Flatten
    flat = tuple(float(v) for v in pts.flatten())
    return LandmarkInfo(
        clip_id=clip_id, frame_index=frame_index, face_id=face_id,
        timestamp=frame_index / 25.0, landmarks_106=flat,
        image_width=640, image_height=480,
    )


# ---------------------------------------------------------------------------
# affine_transform unit tests
# ---------------------------------------------------------------------------


class TestAffineTransform:
    def test_select_eye_points_returns_2_points(self) -> None:
        """select_eye_points returns (2, 2) array."""
        lm = np.zeros((106, 2), dtype=np.float32)
        lm[33:43] = [100.0, 120.0]  # left eye
        lm[87:97] = [180.0, 120.0]  # right eye
        pts = select_eye_points(lm)
        assert pts.shape == (2, 2)
        assert pts.dtype == np.float32
        assert np.allclose(pts[0], [100.0, 120.0])
        assert np.allclose(pts[1], [180.0, 120.0])

    def test_select_nose_point_returns_point(self) -> None:
        """select_nose_point returns (2,) array."""
        lm = np.zeros((106, 2), dtype=np.float32)
        lm[83:86] = [[140.0, 158.0], [140.0, 160.0], [140.0, 162.0]]
        pts = select_nose_point(lm)
        assert pts.shape == (2,)
        assert pts.dtype == np.float32
        assert np.allclose(pts, [140.0, 160.0])

    def test_select_eye_points_raises_on_insufficient_landmarks(self) -> None:
        """Less than 106 landmarks raises ValueError."""
        lm = np.zeros((50, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            select_eye_points(lm)

    def test_select_eye_points_raises_on_degenerate_eyes(self) -> None:
        """Coincident eye centers raise ValueError."""
        lm = np.zeros((106, 2), dtype=np.float32)
        # All zeros → left eye = right eye = (0, 0)
        with pytest.raises(ValueError):
            select_eye_points(lm)

    def test_estimate_affine_produces_valid_matrix(self) -> None:
        """estimate_affine returns valid 2×3 matrix."""
        src = TEMPLATE_EYES.copy()
        # Add slight perturbation
        src += np.random.randn(2, 2).astype(np.float32) * 2
        matrix = estimate_affine(src, TEMPLATE_EYES)
        assert matrix is not None
        assert matrix.shape == (2, 3)
        # Determinant should be positive
        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        assert det > 0

    def test_estimate_affine_returns_none_for_degenerate(self) -> None:
        """Degenerate points → None."""
        src = np.array([[0, 0], [0, 0]], dtype=np.float32)
        dst = TEMPLATE_EYES.copy()
        result = estimate_affine(src, dst)
        assert result is None

    def test_estimate_affine_ignores_nose_yaw(self) -> None:
        """A yawed nose (off-midline) must NOT tilt the face.

        Regression for the bug where estimateAffinePartial2D read the nose's
        horizontal offset as roll and rotated an otherwise-level face by ~15°.
        With level eyes (roll≈0) but an offset nose, the rotation must stay ~0.
        """
        # Level eyes, 80px apart, centred around x=128
        src = np.array([[88.0, 120.0], [168.0, 120.0]], dtype=np.float32)
        matrix = estimate_affine(src, TEMPLATE_EYES)
        assert matrix is not None
        rotation, _ = decompose_affine(matrix)
        assert abs(rotation) < 1.0  # eyes level → no tilt

    def test_estimate_affine_levels_tilted_eyes(self) -> None:
        """A genuinely tilted eye-line is corrected to level."""
        # Eyes tilted by ~10° (right eye lower)
        import math as _math
        ang = _math.radians(10.0)
        left = np.array([88.0, 120.0], dtype=np.float32)
        right = left + np.array([80.0 * _math.cos(ang), 80.0 * _math.sin(ang)],
                                dtype=np.float32)
        src = np.array([left, right], dtype=np.float32)
        matrix = estimate_affine(src, TEMPLATE_EYES)
        assert matrix is not None
        rotation, _ = decompose_affine(matrix)
        # Rotation should cancel the +10° eye-line tilt
        assert abs(rotation - (-10.0)) < 1.0

    def test_warp_face_produces_correct_resolution(self) -> None:
        """warp_face output has expected shape."""
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        src = TEMPLATE_EYES.copy()
        src += np.random.randn(2, 2).astype(np.float32) * 2
        matrix = estimate_affine(src, TEMPLATE_EYES)
        assert matrix is not None

        aligned = warp_face(img, matrix, (256, 256))
        assert aligned.shape == (256, 256, 3)

    def test_invert_affine_roundtrip(self) -> None:
        """Inverting an affine matrix and applying it recovers original."""
        src = TEMPLATE_EYES.copy() + np.random.randn(2, 2).astype(np.float32) * 3
        matrix = estimate_affine(src, TEMPLATE_EYES)
        assert matrix is not None

        inv = invert_affine(matrix)
        assert inv.shape == (2, 3)

        # Roundtrip: matrix × inv should be close to identity
        # Convert to 3×3 for multiplication
        m = np.eye(3, dtype=np.float32)
        m[:2, :] = matrix
        mi = np.eye(3, dtype=np.float32)
        mi[:2, :] = inv
        product = m @ mi
        # Should be close to identity
        assert np.allclose(product, np.eye(3), atol=1e-4)

    def test_matrix_to_tuple_and_back(self) -> None:
        """matrix_to_tuple → tuple_to_matrix roundtrip."""
        matrix = np.array([[1.5, 0.2, 100.0], [-0.2, 1.5, 50.0]], dtype=np.float32)
        t = matrix_to_tuple(matrix)
        assert len(t) == 6
        recovered = tuple_to_matrix(t)
        assert np.allclose(matrix, recovered)

    def test_estimate_affine_reflection_safety_net(self) -> None:
        """The 2-eye similarity transform never produces a reflection.

        det = scale² > 0 is guaranteed by construction; verify a valid
        transform is not incorrectly rejected.
        """
        src = TEMPLATE_EYES.copy() + np.random.randn(2, 2).astype(np.float32) * 2
        result = estimate_affine(src, TEMPLATE_EYES)
        assert result is not None
        det = result[0, 0] * result[1, 1] - result[0, 1] * result[1, 0]
        assert det > 0

    def test_estimate_affine_rejects_extreme_scale(self) -> None:
        """Extreme scale factors → None."""
        src = TEMPLATE_EYES.copy()
        dst = TEMPLATE_EYES * 100  # Scale factor ~100
        result = estimate_affine(src, dst)
        assert result is None

    def test_build_template_defaults_match_legacy(self) -> None:
        """build_template() with defaults returns the exact legacy template."""
        t = build_template()
        assert t.shape == (3, 2)
        assert t.dtype == np.float32
        assert np.array_equal(t, TEMPLATE_256)

    def test_build_template_explicit_defaults_match_legacy(self) -> None:
        """Explicit canonical defaults → identical to legacy TEMPLATE_256."""
        t = build_template(
            eye_distance=80.541, eye_center_y=118.0, scale_factor=1.0,
        )
        assert np.array_equal(t, TEMPLATE_256)

    def test_build_template_scale_factor_shrinks_eye_distance(self) -> None:
        """scale_factor scales the template eye distance by that factor."""
        t = build_template(
            eye_distance=80.541, eye_center_y=118.0, scale_factor=0.85,
        )
        ed = float(np.hypot(t[1, 0] - t[0, 0], t[1, 1] - t[0, 1]))
        assert ed == pytest.approx(80.541 * 0.85, abs=1e-4)
        # Eye line stays at eye_center_y
        assert t[0, 1] == pytest.approx(118.0)
        assert t[1, 1] == pytest.approx(118.0)
        # Symmetric around x=128
        assert (t[0, 0] + t[1, 0]) / 2.0 == pytest.approx(128.0, abs=1e-4)

    def test_build_template_custom_params(self) -> None:
        """Custom eye_distance / eye_center_y build a symmetric template."""
        t = build_template(eye_distance=70.0, eye_center_y=120.0)
        assert t.shape == (3, 2)
        assert (t[0, 0] + t[1, 0]) / 2.0 == pytest.approx(128.0)
        assert t[0, 1] == pytest.approx(120.0)
        assert t[1, 1] == pytest.approx(120.0)
        assert np.hypot(t[1, 0] - t[0, 0], t[1, 1] - t[0, 1]) == pytest.approx(
            70.0, abs=1e-4
        )


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------


class TestAlignmentProcessor:
    def test_align_faces_produces_aligned_face_info(self) -> None:
        """align_faces returns List[AlignedFaceInfo]."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0", crop_path="/tmp/crop.jpg")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            result = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=True, logger=logger,
            )

        assert len(result) == 1
        a = result[0]
        assert isinstance(a, AlignedFaceInfo)
        assert a.clip_id == "c0"
        assert a.frame_index == 0
        assert a.face_id == "f0"
        assert len(a.affine_matrix) == 6
        assert a.original_image_size == (256, 256)
        assert a.aligned_resolution == (256, 256)

    def test_affine_matrix_is_valid(self) -> None:
        """affine_matrix has 6 finite floats and positive determinant."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0", crop_path="/tmp/crop.jpg")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            result = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=True, logger=logger,
            )

        m = result[0].affine_matrix
        assert all(np.isfinite(v) for v in m)
        a, b = m[0], m[1]
        c, d = m[3], m[4]
        det = a * d - b * c
        assert det > 0

    def test_missing_face_info_skipped(self) -> None:
        """LandmarkInfo with no matching FaceInfo → skipped."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f_missing")
        f = _face(0, "f0")  # Different face_id
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("pathlib.Path.exists", return_value=True):
            result = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=False, logger=logger,
            )

        assert len(result) == 0

    def test_bad_landmarks_skipped(self) -> None:
        """LandmarkInfo with < 212 values → skipped."""
        bad_lm = LandmarkInfo(
            clip_id="c0", frame_index=0, face_id="f0",
            timestamp=0.0, landmarks_106=tuple([0.0] * 10),  # Only 10 values
            image_width=640, image_height=480,
        )
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("pathlib.Path.exists", return_value=True):
            result = align_faces(
                [bad_lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=False, logger=logger,
            )

        assert len(result) == 0

    def test_multiple_faces_aligned(self) -> None:
        """Multiple landmarks → multiple AlignedFaceInfo."""
        lms = [
            _make_landmarks(clip_id="c0", frame_index=0, face_id="f0"),
            _make_landmarks(clip_id="c0", frame_index=1, face_id="f1"),
        ]
        faces = (
            _frame(0, clip_id="c0", faces=(_face(0, "f0", crop_path="/tmp/c0.jpg"),)),
            _frame(1, clip_id="c0", faces=(_face(1, "f1", crop_path="/tmp/c1.jpg"),)),
        )
        seq = _seq(clip_id="c0", frames=faces)
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            result = align_faces(
                lms, [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=True, logger=logger,
            )

        assert len(result) == 2

    def test_missing_crop_file_skipped(self) -> None:
        """Non-existent crop file → skipped."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0", crop_path="/nonexistent/crop.jpg")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("pathlib.Path.exists", return_value=False):
            result = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=False, logger=logger,
            )

        assert len(result) == 0

    def test_no_scrfd_no_model_calls(self) -> None:
        """Stage 11 never calls any ML model."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        # Patch all model-related modules to fail if called
        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch("onnxruntime.InferenceSession", side_effect=RuntimeError("No ONNX in Stage11!")):
            result = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=True, logger=logger,
            )

        assert len(result) == 1

    def test_align_faces_eye_template_controls_scale(self) -> None:
        """eye_template controls the affine scale (final = original × factor)."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0", crop_path="/tmp/crop.jpg")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        scaled_eyes = build_template(
            eye_distance=80.541, eye_center_y=118.0, scale_factor=0.85,
        )[:2].copy()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            legacy = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=True, logger=logger,
                eye_template=None,
            )
            scaled = align_faces(
                [lm], [seq],
                output_dir=Path("/tmp/aligned"),
                output_resolution=(256, 256),
                save_images=True, logger=logger,
                eye_template=scaled_eyes,
            )

        def _scale_of(info: AlignedFaceInfo) -> float:
            m = info.affine_matrix
            matrix = np.array([
                [m[0], m[1], m[2]],
                [m[3], m[4], m[5]],
            ], dtype=np.float32)
            return decompose_affine(matrix)[1]

        # final_scale = original_scale * scale_factor holds exactly in terms of
        # template eye distances: scale = template_eye_dist / source_eye_dist.
        # (The canonical default eye distance 80.541 reproduces the legacy
        # template exactly, so legacy_scale here is the pre-config scale.)
        legacy_scale = _scale_of(legacy[0])
        scaled_scale = _scale_of(scaled[0])
        legacy_tmpl_dist = float(np.hypot(
            TEMPLATE_EYES[1, 0] - TEMPLATE_EYES[0, 0],
            TEMPLATE_EYES[1, 1] - TEMPLATE_EYES[0, 1],
        ))
        scaled_tmpl_dist = float(np.hypot(
            scaled_eyes[1, 0] - scaled_eyes[0, 0],
            scaled_eyes[1, 1] - scaled_eyes[0, 1],
        ))
        assert scaled_scale == pytest.approx(
            legacy_scale * (scaled_tmpl_dist / legacy_tmpl_dist),
            rel=1e-6,
        )
        assert scaled[0].aligned_resolution == (256, 256)


# ---------------------------------------------------------------------------
# Stage11Alignment integration tests
# ---------------------------------------------------------------------------


class TestStage11Alignment:
    def test_normal_run(self) -> None:
        """Full stage run produces AlignedFaceInfo list."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        stage = Stage11Alignment()
        context = _make_context()

        input_data = {
            "landmarks": [lm],
            "face_sequences": [seq],
        }

        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            output = stage.run(input_data=input_data, context=context)

        assert len(output) == 1
        assert isinstance(output[0], AlignedFaceInfo)

    def test_disabled_returns_empty(self) -> None:
        """alignment.enabled=false → empty list."""
        lm = _make_landmarks()
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, clip_id="c0", faces=(f,)),))
        stage = Stage11Alignment()
        context = _make_context({"alignment": {"enabled": False}})

        input_data = {"landmarks": [lm], "face_sequences": [seq]}
        output = stage.run(input_data=input_data, context=context)
        assert output == []

    def test_empty_landmarks_ok(self) -> None:
        """Empty landmarks list is valid (no faces to align)."""
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, clip_id="c0", faces=(f,)),))
        stage = Stage11Alignment()
        context = _make_context()

        input_data = {"landmarks": [], "face_sequences": [seq]}
        output = stage.run(input_data=input_data, context=context)
        assert output == []

    def test_input_not_mutated(self) -> None:
        """Input data is not modified."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        orig_id = lm.clip_id
        stage = Stage11Alignment()
        context = _make_context()

        input_data = {"landmarks": [lm], "face_sequences": [seq]}

        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            stage.run(input_data=input_data, context=context)

        assert lm.clip_id == orig_id

    def test_input_wrong_type_raises(self) -> None:
        """Wrong input type fails validation."""
        stage = Stage11Alignment()
        assert not stage.validate_input("wrong", _make_context())
        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_missing_keys_raises(self) -> None:
        """Missing required keys fails validation."""
        stage = Stage11Alignment()
        assert not stage.validate_input(
            {"landmarks": []}, _make_context()
        )

    def test_validate_output(self) -> None:
        """Output validation accepts List[AlignedFaceInfo]."""
        stage = Stage11Alignment()
        a = AlignedFaceInfo(
            clip_id="c0", frame_index=0, face_id="f0",
            timestamp=0.0, aligned_face_path=Path("/tmp/a.jpg"),
            affine_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            original_image_size=(256, 256),
            aligned_resolution=(256, 256),
        )
        assert stage.validate_output([a], _make_context())
        assert not stage.validate_output("bad", _make_context())

    def test_affine_matrix_invertible(self) -> None:
        """Saved affine_matrix can be inverted to recover original."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))
        stage = Stage11Alignment()
        context = _make_context()

        input_data = {"landmarks": [lm], "face_sequences": [seq]}

        with patch("cv2.imread", return_value=np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            output = stage.run(input_data=input_data, context=context)

        m = output[0].affine_matrix
        matrix = np.array([
            [m[0], m[1], m[2]],
            [m[3], m[4], m[5]],
        ], dtype=np.float32)

        import cv2
        inv = cv2.invertAffineTransform(matrix)
        assert inv.shape == (2, 3)
        assert np.all(np.isfinite(inv))

    # ------------------------------------------------------------------
    # Config-driven template (backward compatibility + scale_factor)
    # ------------------------------------------------------------------

    def _legacy_reference_matrix(self, lm: LandmarkInfo) -> np.ndarray:
        """Affine matrix computed by the pre-config code path (TEMPLATE_EYES).

        bbox=(0,0,256,256) + expanded_bbox=(0,0,0,0) → crop space == frame
        space, so landmarks_crop == landmarks_106 and the reference exactly
        reproduces the legacy internal computation.
        """
        pts = np.array(lm.landmarks_106, dtype=np.float32).reshape(106, 2)
        eye_pts = select_eye_points(pts)
        ref = estimate_affine(eye_pts, TEMPLATE_EYES)
        assert ref is not None
        return ref

    def _run_stage(
        self, lm: LandmarkInfo, seq: FaceSequence, config_overrides: dict | None,
    ) -> List[AlignedFaceInfo]:
        stage = Stage11Alignment()
        context = _make_context(config_overrides)
        input_data = {"landmarks": [lm], "face_sequences": [seq]}
        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("cv2.imwrite"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            return stage.run(input_data=input_data, context=context)

    def test_config_without_template_matches_legacy_output(self) -> None:
        """Legacy config (no alignment.template) → identical to pre-change."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))

        output = self._run_stage(lm, seq, None)

        assert len(output) == 1
        assert output[0].aligned_resolution == (256, 256)
        ref = self._legacy_reference_matrix(lm)
        assert np.allclose(
            np.array(output[0].affine_matrix, dtype=np.float32),
            ref.flatten(), atol=1e-6,
        )

    def test_scale_factor_one_matches_legacy_output(self) -> None:
        """Explicit template + scale_factor=1.0 → identical to legacy output."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))

        output = self._run_stage(lm, seq, {
            "alignment": {
                "template": {
                    "eye_distance": 80.541,
                    "eye_center_y": 118.0,
                },
                "scale_factor": 1.0,
            },
        })

        assert len(output) == 1
        assert output[0].aligned_resolution == (256, 256)
        ref = self._legacy_reference_matrix(lm)
        assert np.allclose(
            np.array(output[0].affine_matrix, dtype=np.float32),
            ref.flatten(), atol=1e-6,
        )

    def test_scale_factor_shrinks_face_keeps_output_size(self) -> None:
        """scale_factor=0.85 shrinks the canonical face; output stays 256×256."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))

        output = self._run_stage(lm, seq, {
            "alignment": {
                "template": {
                    "eye_distance": 80.541,
                    "eye_center_y": 118.0,
                },
                "scale_factor": 0.85,
            },
        })

        assert len(output) == 1
        assert output[0].aligned_resolution == (256, 256)
        m = output[0].affine_matrix
        matrix = np.array([
            [m[0], m[1], m[2]],
            [m[3], m[4], m[5]],
        ], dtype=np.float32)
        scale = decompose_affine(matrix)[1]
        legacy_scale = decompose_affine(
            self._legacy_reference_matrix(lm)
        )[1]
        # Exact scale relation in terms of template eye distances (the canonical
        # default 80.541 reproduces the legacy template exactly).
        scaled_eyes = build_template(
            eye_distance=80.541, eye_center_y=118.0, scale_factor=0.85,
        )[:2]
        legacy_tmpl_dist = float(np.hypot(
            TEMPLATE_EYES[1, 0] - TEMPLATE_EYES[0, 0],
            TEMPLATE_EYES[1, 1] - TEMPLATE_EYES[0, 1],
        ))
        scaled_tmpl_dist = float(np.hypot(
            scaled_eyes[1, 0] - scaled_eyes[0, 0],
            scaled_eyes[1, 1] - scaled_eyes[0, 1],
        ))
        assert scale == pytest.approx(
            legacy_scale * (scaled_tmpl_dist / legacy_tmpl_dist),
            rel=1e-6,
        )

    def test_scale_factor_0_8_keeps_256_output(self) -> None:
        """scale_factor=0.8 (spec #5) still produces 256×256 aligned output."""
        lm = _make_landmarks(clip_id="c0", frame_index=0, face_id="f0")
        f = _face(0, "f0")
        seq = _seq(clip_id="c0", frames=(_frame(0, clip_id="c0", faces=(f,)),))

        output = self._run_stage(lm, seq, {
            "alignment": {
                "template": {
                    "eye_distance": 80.541,
                    "eye_center_y": 118.0,
                },
                "scale_factor": 0.8,
            },
        })

        assert len(output) == 1
        assert output[0].aligned_resolution == (256, 256)
        m = output[0].affine_matrix
        matrix = np.array([
            [m[0], m[1], m[2]],
            [m[3], m[4], m[5]],
        ], dtype=np.float32)
        scale = decompose_affine(matrix)[1]
        legacy_scale = decompose_affine(
            self._legacy_reference_matrix(lm)
        )[1]
        # The canonical face shrinks by 0.8: final_scale = legacy_scale * 0.8
        # (exact in terms of template eye distances, cf. the 0.85 test above).
        scaled_eyes = build_template(
            eye_distance=80.541, eye_center_y=118.0, scale_factor=0.8,
        )[:2]
        legacy_tmpl_dist = float(np.hypot(
            TEMPLATE_EYES[1, 0] - TEMPLATE_EYES[0, 0],
            TEMPLATE_EYES[1, 1] - TEMPLATE_EYES[0, 1],
        ))
        scaled_tmpl_dist = float(np.hypot(
            scaled_eyes[1, 0] - scaled_eyes[0, 0],
            scaled_eyes[1, 1] - scaled_eyes[0, 1],
        ))
        assert scale == pytest.approx(
            legacy_scale * (scaled_tmpl_dist / legacy_tmpl_dist),
            rel=1e-6,
        )


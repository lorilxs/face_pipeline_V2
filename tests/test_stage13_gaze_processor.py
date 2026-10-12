"""Unit tests for Stage13 Gaze Processor — mocked MediaPipe output."""

import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from face_pipeline_v2.data.models import AlignedFaceInfo, GazeFeature
from face_pipeline_v2.stages.stage_13_gaze.processor import (
    get_gaze_vector_from_landmarks,
    extract_gaze_parameters,
    _HAS_MEDIAPIPE,
    LEFT_PUPIL_IDX,
    RIGHT_PUPIL_IDX,
    LEFT_CHEEK,
    RIGHT_CHEEK,
    FOREHEAD,
    CHIN,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)

# ---------------------------------------------------------------------------
# Mock MediaPipe
# ---------------------------------------------------------------------------

class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class MockLandmarkList:
    def __init__(self, landmarks: List[MockLandmark]) -> None:
        self._landmarks = landmarks

    def __getitem__(self, idx: int) -> MockLandmark:
        return self._landmarks[idx]

    def __len__(self) -> int:
        return len(self._landmarks)


class MockFaceLandmarkerResult:
    def __init__(self, landmarks: MockLandmarkList) -> None:
        self.face_landmarks = [landmarks]


class MockDetector:
    def __init__(self, landmarks: MockLandmarkList) -> None:
        self._landmarks = landmarks

    def detect(self, mp_image) -> MockFaceLandmarkerResult:
        return MockFaceLandmarkerResult(self._landmarks)

    def close(self) -> None:
        pass


def _make_dummy_face(path: Path, size: Tuple[int, int] = (256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _build_478_landmarks(
    left_iris: Tuple[float, float, float] = (0.35, 0.45, 0.02),
    right_iris: Tuple[float, float, float] = (0.65, 0.45, 0.02),
) -> MockLandmarkList:
    """Build 478 mock landmarks. Iris positions configurable."""
    landmarks = []
    for i in range(478):
        lm = MockLandmark(0.5, 0.5, 0.0)
        if i == LEFT_CHEEK:
            lm = MockLandmark(0.2, 0.5, 0.0)
        elif i == RIGHT_CHEEK:
            lm = MockLandmark(0.8, 0.5, 0.0)
        elif i == FOREHEAD:
            lm = MockLandmark(0.5, 0.2, 0.05)
        elif i == CHIN:
            lm = MockLandmark(0.5, 0.8, -0.05)
        if i in LEFT_EYE_INDICES:
            lm = MockLandmark(0.35, 0.45, 0.01)
        if i in RIGHT_EYE_INDICES:
            lm = MockLandmark(0.65, 0.45, 0.01)
        if i == LEFT_PUPIL_IDX:
            lm = MockLandmark(*left_iris)
        if i == RIGHT_PUPIL_IDX:
            lm = MockLandmark(*right_iris)
        landmarks.append(lm)
    return MockLandmarkList(landmarks)


# ---------------------------------------------------------------------------
# Tests: get_gaze_vector_from_landmarks
# ---------------------------------------------------------------------------

class TestGazeVectorFromLandmarks:

    def test_normal_output_has_6_keys(self) -> None:
        landmarks = _build_478_landmarks()
        result = get_gaze_vector_from_landmarks(landmarks, 256, 256)
        assert set(result.keys()) == {
            "gaze_0_x", "gaze_0_y", "gaze_0_z",
            "gaze_1_x", "gaze_1_y", "gaze_1_z",
        }

    def test_iris_gaze_is_normalized(self) -> None:
        landmarks = _build_478_landmarks(
            left_iris=(0.35, 0.45, 0.02),
            right_iris=(0.65, 0.45, 0.02),
        )
        result = get_gaze_vector_from_landmarks(landmarks, 256, 256)
        left = np.array([result["gaze_0_x"], result["gaze_0_y"], result["gaze_0_z"]])
        right = np.array([result["gaze_1_x"], result["gaze_1_y"], result["gaze_1_z"]])
        assert np.abs(np.linalg.norm(left) - 1.0) < 0.01
        assert np.abs(np.linalg.norm(right) - 1.0) < 0.01

    def test_fallback_no_iris_landmarks(self) -> None:
        landmarks = _build_478_landmarks()
        landmarks._landmarks = landmarks._landmarks[:470]  # remove iris
        result = get_gaze_vector_from_landmarks(landmarks, 256, 256)
        assert result["gaze_0_x"] == result["gaze_1_x"]
        assert result["gaze_0_y"] == result["gaze_1_y"]
        assert result["gaze_0_z"] == result["gaze_1_z"]

    def test_face_normal_z_negative(self) -> None:
        landmarks = _build_478_landmarks()
        result = get_gaze_vector_from_landmarks(landmarks, 256, 256)
        assert result["gaze_0_z"] < 0
        assert result["gaze_1_z"] < 0

    def test_left_right_gaze_different_with_iris_offset(self) -> None:
        landmarks = _build_478_landmarks(
            left_iris=(0.33, 0.45, 0.02),
            right_iris=(0.67, 0.45, 0.02),
        )
        result = get_gaze_vector_from_landmarks(landmarks, 256, 256)
        left = np.array([result["gaze_0_x"], result["gaze_0_y"], result["gaze_0_z"]])
        right = np.array([result["gaze_1_x"], result["gaze_1_y"], result["gaze_1_z"]])
        diff = np.linalg.norm(left - right)
        assert diff > 0.001, "Left/right gaze should differ with iris offset"


# ---------------------------------------------------------------------------
# Tests: extract_gaze_parameters
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_MEDIAPIPE, reason="mediapipe not installed")
class TestExtractGazeParameters:

    def test_empty_input(self) -> None:
        landmarks = _build_478_landmarks()
        detector = MockDetector(landmarks)
        results = extract_gaze_parameters([], detector)
        assert results == []

    def test_single_face(self) -> None:
        landmarks = _build_478_landmarks()
        detector = MockDetector(landmarks)
        with tempfile.TemporaryDirectory() as tmpdir:
            face_path = Path(tmpdir) / "face_0.png"
            _make_dummy_face(face_path)
            faces = [AlignedFaceInfo(
                clip_id="test", frame_index=0, face_id="face_0",
                timestamp=0.0, aligned_face_path=face_path,
                affine_matrix=(1., 0., 0., 0., 1., 0.),
                original_image_size=(256, 256),
                aligned_resolution=(256, 256),
            )]
            results = extract_gaze_parameters(faces, detector)
        assert len(results) == 1
        f = results[0]
        assert f.frame_index == 0
        assert f.timestamp == 0.0
        assert f.clip_id == "test"
        assert len(f.gaze_vector) == 6
        assert len(f.left_eye) == 3
        assert len(f.right_eye) == 3

    def test_multi_face(self) -> None:
        landmarks = _build_478_landmarks()
        detector = MockDetector(landmarks)
        with tempfile.TemporaryDirectory() as tmpdir:
            faces = []
            for i in range(5):
                path = Path(tmpdir) / f"face_{i}.png"
                _make_dummy_face(path)
                faces.append(AlignedFaceInfo(
                    clip_id="test", frame_index=i, face_id=f"face_{i}",
                    timestamp=float(i)/25.0, aligned_face_path=path,
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(256, 256),
                    aligned_resolution=(256, 256),
                ))
            results = extract_gaze_parameters(faces, detector)
        assert len(results) == 5
        for i, f in enumerate(results):
            assert f.frame_index == i
            assert len(f.gaze_vector) == 6

    def test_missing_image_skipped(self) -> None:
        landmarks = _build_478_landmarks()
        detector = MockDetector(landmarks)
        with tempfile.TemporaryDirectory() as tmpdir:
            face_path = Path(tmpdir) / "face_0.png"
            _make_dummy_face(face_path)
            faces = [
                AlignedFaceInfo(
                    clip_id="test", frame_index=0, face_id="face_0",
                    timestamp=0.0, aligned_face_path=face_path,
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(256, 256),
                    aligned_resolution=(256, 256),
                ),
                AlignedFaceInfo(
                    clip_id="test", frame_index=1, face_id="face_1",
                    timestamp=0.04,
                    aligned_face_path=Path(tmpdir) / "missing.png",
                    affine_matrix=(1., 0., 0., 0., 1., 0.),
                    original_image_size=(256, 256),
                    aligned_resolution=(256, 256),
                ),
            ]
            results = extract_gaze_parameters(faces, detector)
        assert len(results) == 1
        assert results[0].frame_index == 0
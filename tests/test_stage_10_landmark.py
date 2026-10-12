"""Tests for Stage 10: High-Precision Landmark Extraction.

Tests cover:
    1. Model loading (via ModelManager registry).
    2. Only processes accepted clips.
    3. Reads face_crop_path from FaceInfo.
    4. Output is List[LandmarkInfo].
    5. Each LandmarkInfo has exactly 106 landmarks (212 values).
    6. Landmark coordinates are in reasonable range (within frame bounds).
    7. Stage does NOT call SCRFD.
    8. Input validation.
    9. Disabled returns empty.
    10. Non-accepted clips are skipped.
    11. Missing crops handled gracefully.
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    FaceInfo,
    FaceSequence,
    FrameInfo,
    LandmarkInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_10_landmark.processor import (
    extract_landmarks,
)
from face_pipeline_v2.stages.stage_10_landmark.stage import (
    Stage10Landmark,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "landmark": {
            "enabled": True,
            "model_path": "models/insightface/2d106det.onnx",
            "device": "auto",
        },
        "models": {
            "insightface_landmark": {
                "path": "models/insightface/2d106det.onnx",
                "device": "auto",
            },
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
        logger=PipelineLogger(name="test_stage_10", log_dir=None),
        model_manager=object(),
    )


def _face(
    frame_index: int = 0,
    face_id: str = "f0",
    track_id: str | None = "track_000",
    crop_path: str = "/tmp/crop.jpg",
    bbox: tuple = (50.0, 30.0, 150.0, 170.0),
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index,
        face_id=face_id,
        track_id=track_id,
        bbox=bbox,
        landmark_5=(),
        confidence=0.9,
        face_crop_path=Path(crop_path),
    )


def _frame(
    idx: int,
    clip_id: str = "c0",
    faces: tuple = (),
) -> FrameInfo:
    return FrameInfo(
        frame_index=idx,
        clip_id=clip_id,
        timestamp=idx / 25.0,
        image_path=Path(""),
        faces=faces,
    )


def _seq(
    clip_id: str = "c0",
    video_id: str = "v0",
    frames: tuple = (),
) -> FaceSequence:
    return FaceSequence(
        clip_id=clip_id,
        video_id=video_id,
        frames=frames,
    )


class MockLandmarkModel:
    """Mock InsightFace 106 landmark model returning fixed landmarks."""

    NUM_LANDMARKS = 106

    def predict(self, face_crop: np.ndarray) -> np.ndarray:
        """Return fixed (106, 2) landmarks — grid pattern."""
        h, w = face_crop.shape[:2]
        landmarks = np.zeros((self.NUM_LANDMARKS, 2), dtype=np.float32)
        for i in range(self.NUM_LANDMARKS):
            landmarks[i, 0] = (i % 10) * w / 10.0
            landmarks[i, 1] = (i // 10) * h / 11.0
        return landmarks

    @property
    def is_loaded(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Processor unit tests
# ---------------------------------------------------------------------------


class TestLandmarkProcessor:
    def test_output_is_landmark_info_list(self) -> None:
        """Processor returns List[LandmarkInfo]."""
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, faces=(f,)),))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], LandmarkInfo)

    def test_landmarks_count_is_106(self) -> None:
        """Each LandmarkInfo has 212 values = 106 points × 2."""
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, faces=(f,)),))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        assert len(result[0].landmarks_106) == 212  # 106 × 2

    def test_coordinates_in_reasonable_range(self) -> None:
        """Landmark coords should be within ~2x bbox range of frame."""
        f = _face(0, "f0", bbox=(100.0, 50.0, 300.0, 350.0))
        seq = _seq(frames=(_frame(0, faces=(f,)),))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((300, 200, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        lms = result[0].landmarks_106
        for i in range(0, 212, 2):
            x, y = lms[i], lms[i + 1]
            # Should be near the bbox region
            assert 0 <= x <= 1000, f"x={x} out of reasonable range"
            assert 0 <= y <= 1000, f"y={y} out of reasonable range"

    def test_clip_id_preserved(self) -> None:
        """LandmarkInfo.clip_id matches source FaceSequence."""
        f = _face(0, "f0")
        seq = _seq(clip_id="test_clip_42", frames=(_frame(0, faces=(f,)),))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        assert result[0].clip_id == "test_clip_42"

    def test_frame_index_preserved(self) -> None:
        """LandmarkInfo.frame_index matches source FrameInfo."""
        f = _face(42, "f42")
        seq = _seq(frames=(_frame(42, faces=(f,)),))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        assert result[0].frame_index == 42

    def test_missing_crop_handled(self) -> None:
        """Missing crop → frame is skipped gracefully."""
        f = _face(0, "f0", crop_path="/nonexistent/crop.jpg")
        seq = _seq(frames=(_frame(0, faces=(f,)),))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("pathlib.Path.exists", return_value=False):
            result = extract_landmarks(seq, model, logger)

        assert len(result) == 0  # No landmarks for missing crop

    def test_empty_faces_skipped(self) -> None:
        """Frames with no faces are skipped."""
        f = _face(1, "f1")
        seq = _seq(frames=(
            _frame(0, faces=()),
            _frame(1, faces=(f,)),
        ))
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        assert len(result) == 1
        assert result[0].frame_index == 1

    def test_multi_frame_output(self) -> None:
        """All frames with valid faces produce landmarks."""
        frames = tuple(
            _frame(i, faces=(_face(i, f"f{i}"),)) for i in range(8)
        )
        seq = _seq(frames=frames)
        model = MockLandmarkModel()
        logger = PipelineLogger(name="test", log_dir=None)

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, model, logger)

        assert len(result) == 8
        frame_indices = [r.frame_index for r in result]
        assert frame_indices == list(range(8))

    def test_coordinate_mapping_uses_bbox(self) -> None:
        """Coordinates are mapped from crop space → frame space using bbox."""
        bbox = (100.0, 50.0, 300.0, 250.0)  # w=200, h=200
        f = _face(0, "f0", bbox=bbox)
        seq = _seq(frames=(_frame(0, faces=(f,)),))
        logger = PipelineLogger(name="test", log_dir=None)

        # Use a model that predicts a known point at crop center
        class CenterModel:
            def predict(self, face_crop):
                h, w = face_crop.shape[:2]
                return np.array([[w / 2, h / 2]], dtype=np.float32)

        with patch("cv2.imread", return_value=np.zeros((200, 200, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = extract_landmarks(seq, CenterModel(), logger)

        # Crop center (100, 100) in 200×200 crop
        # → frame: 100 * 200/200 + 100 = 200, 100 * 200/200 + 50 = 150
        lms = result[0].landmarks_106
        assert abs(lms[0] - 200.0) < 1.0, f"x={lms[0]}"
        assert abs(lms[1] - 150.0) < 1.0, f"y={lms[1]}"


# ---------------------------------------------------------------------------
# Stage10Landmark integration tests
# ---------------------------------------------------------------------------


class TestStage10Landmark:
    def test_normal_processing(self) -> None:
        """Full stage run with accepted clips."""
        seq = _seq(clip_id="c_accepted", frames=(
            _frame(0, clip_id="c_accepted", faces=(_face(0, "f0"),)),
            _frame(1, clip_id="c_accepted", faces=(_face(1, "f1"),)),
        ))
        stage = Stage10Landmark()
        context = _make_context()

        mock_model = MockLandmarkModel()
        input_data = {
            "face_sequences": [seq],
            "accepted_clip_ids": {"c_accepted"},
        }

        with patch.object(stage, "_get_model", return_value=mock_model), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=input_data, context=context)

        assert len(output) == 2
        assert all(isinstance(m, LandmarkInfo) for m in output)
        assert all(len(m.landmarks_106) == 212 for m in output)

    def test_only_accepted_clips_processed(self) -> None:
        """Non-accepted clips are skipped."""
        seq_good = _seq(clip_id="c_good", frames=(
            _frame(0, clip_id="c_good", faces=(_face(0, "f0"),)),
        ))
        seq_bad = _seq(clip_id="c_bad", frames=(
            _frame(0, clip_id="c_bad", faces=(_face(0, "f0"),)),
        ))
        stage = Stage10Landmark()
        context = _make_context()

        mock_model = MockLandmarkModel()
        input_data = {
            "face_sequences": [seq_good, seq_bad],
            "accepted_clip_ids": {"c_good"},  # Only c_good accepted
        }

        with patch.object(stage, "_get_model", return_value=mock_model), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=input_data, context=context)

        assert len(output) == 1
        assert output[0].clip_id == "c_good"

    def test_disabled_returns_empty(self) -> None:
        """When landmark.enabled=false, returns empty list."""
        seq = _seq(frames=(_frame(0, faces=(_face(0, "f0"),)),))
        stage = Stage10Landmark()
        context = _make_context({"landmark": {"enabled": False}})

        input_data = {
            "face_sequences": [seq],
            "accepted_clip_ids": {"c0"},
        }

        output = stage.run(input_data=input_data, context=context)
        assert output == []

    def test_no_accepted_clips_returns_empty(self) -> None:
        """When no clips are accepted, output is empty."""
        seq = _seq(clip_id="c_rejected", frames=(
            _frame(0, clip_id="c_rejected", faces=(_face(0, "f0"),)),
        ))
        stage = Stage10Landmark()
        context = _make_context()

        mock_model = MockLandmarkModel()
        input_data = {
            "face_sequences": [seq],
            "accepted_clip_ids": {"c_other"},  # Different clip id
        }

        with patch.object(stage, "_get_model", return_value=mock_model), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=input_data, context=context)

        assert output == []

    def test_input_not_mutated(self) -> None:
        """Input data structures are not modified."""
        seq = _seq(clip_id="c0", frames=(
            _frame(0, clip_id="c0", faces=(_face(0, "f0"),)),
        ))
        orig_clip_id = seq.clip_id
        stage = Stage10Landmark()
        context = _make_context()

        mock_model = MockLandmarkModel()
        input_data = {
            "face_sequences": [seq],
            "accepted_clip_ids": {"c0"},
        }

        with patch.object(stage, "_get_model", return_value=mock_model), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            stage.run(input_data=input_data, context=context)

        assert seq.clip_id == orig_clip_id

    def test_input_wrong_type_raises(self) -> None:
        """Wrong input type fails validation."""
        stage = Stage10Landmark()
        assert not stage.validate_input("wrong", _make_context())
        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_missing_keys_raises(self) -> None:
        """Missing required keys fails validation."""
        stage = Stage10Landmark()
        assert not stage.validate_input(
            {"face_sequences": []}, _make_context()
        )

    def test_validate_output(self) -> None:
        """Output validation accepts List[LandmarkInfo]."""
        stage = Stage10Landmark()
        lm = LandmarkInfo(
            clip_id="c0", frame_index=0, face_id="f0",
            timestamp=0.0, landmarks_106=tuple([float(i) for i in range(212)]),
            image_width=1920, image_height=1080,
        )
        assert stage.validate_output([lm], _make_context())
        assert not stage.validate_output("bad", _make_context())

    def test_does_not_call_scrfd(self) -> None:
        """Stage 10 must NEVER call SCRFD or re-detect faces."""
        seq = _seq(clip_id="c0", frames=(
            _frame(0, clip_id="c0", faces=(_face(0, "f0"),)),
        ))
        stage = Stage10Landmark()
        context = _make_context()

        mock_model = MockLandmarkModel()
        input_data = {
            "face_sequences": [seq],
            "accepted_clip_ids": {"c0"},
        }

        # Ensure no SCRFD-related imports or calls happen
        with patch.object(stage, "_get_model", return_value=mock_model), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("face_pipeline_v2.models.wrappers.scrfd.SCRFDWrapper", side_effect=RuntimeError("SCRFD must not be called")):
            output = stage.run(input_data=input_data, context=context)

        assert len(output) == 1

    def test_accepted_clip_ids_as_list(self) -> None:
        """accepted_clip_ids can be a list (not just set)."""
        seq = _seq(clip_id="c0", frames=(
            _frame(0, clip_id="c0", faces=(_face(0, "f0"),)),
        ))
        stage = Stage10Landmark()
        context = _make_context()

        mock_model = MockLandmarkModel()
        input_data = {
            "face_sequences": [seq],
            "accepted_clip_ids": ["c0"],  # List, not set
        }

        with patch.object(stage, "_get_model", return_value=mock_model), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=input_data, context=context)

        assert len(output) == 1

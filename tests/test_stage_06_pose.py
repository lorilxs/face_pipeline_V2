"""Tests for Stage 6: Head Pose Estimation.

Tests cover:
    1. Mock model → correct PoseInfo fields.
    2. Multi-frame → output count matches input.
    3. enabled=false → returns empty list.
    4. Missing face_crop_path → handled gracefully.
    5. Input immutability.
    6. Input validation.
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
    PoseInfo,
    PoseSequence,
    SingleFaceIdentityResult,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_06_pose.pose_estimator import (
    estimate_poses,
)
from face_pipeline_v2.stages.stage_06_pose.stage import (
    Stage6PoseEstimation,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_06": {"enabled": True, "batch_size": 32, "device": "cpu"},
        "models": {"sixdrepnet": {"path": "models/6DRepNet"}},
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_06", log_dir=None),
        model_manager=object(),
    )


def _face(
    frame_index: int, face_id: str, track_id: str | None = "track_000",
    crop_path: str = "/tmp/crop.jpg",
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index, face_id=face_id, track_id=track_id,
        bbox=(50.0, 30.0, 150.0, 170.0), landmark_5=(),
        confidence=0.9, face_crop_path=Path(crop_path),
    )


def _frame(idx: int, faces: tuple[FaceInfo, ...]) -> FrameInfo:
    return FrameInfo(
        frame_index=idx, clip_id="c0", timestamp=idx / 25.0,
        image_path=Path(""), faces=faces,
    )


def _seq(
    clip_id: str = "c0",
    frames: tuple[FrameInfo, ...] = (),
    identity_result: SingleFaceIdentityResult | None = None,
) -> FaceSequence:
    return FaceSequence(
        clip_id=clip_id, video_id="v", frames=frames,
        identity_result=identity_result,
    )


class MockPoseModel:
    """Mock 6DRepNet returning fixed poses."""

    def predict(self, images: List[Any]) -> List[dict]:
        return [
            {"yaw": 12.0, "pitch": -5.0, "roll": 2.0, "confidence": 0.98}
            for _ in images
        ]


# ---------------------------------------------------------------------------
# estimate_poses unit tests
# ---------------------------------------------------------------------------


class TestPoseEstimator:
    def test_single_frame_produces_pose(self) -> None:
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, (f,)),))
        model = MockPoseModel()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = estimate_poses(seq, model)

        assert isinstance(result, PoseSequence)
        assert result.clip_id == "c0"
        assert len(result.poses) == 1
        p = result.poses[0]
        assert p.frame_index == 0
        assert p.yaw == 12.0
        assert p.pitch == -5.0
        assert p.roll == 2.0
        assert p.confidence == 0.98

    def test_multi_frame_output_count_matches(self) -> None:
        frames = tuple(
            _frame(i, (_face(i, f"f{i}"),)) for i in range(8)
        )
        seq = _seq(frames=frames)
        model = MockPoseModel()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = estimate_poses(seq, model)

        assert len(result.poses) == 8

    def test_main_track_selected(self) -> None:
        """When identity_result exists, only main_track face is used."""
        main = _face(0, "main", track_id="track_A", crop_path="/tmp/main.jpg")
        junk = _face(0, "junk", track_id="track_B", crop_path="/tmp/junk.jpg")
        seq = _seq(
            frames=(_frame(0, (main, junk)),),
            identity_result=SingleFaceIdentityResult(
                passed=True, main_track_id="track_A",
                total_frames=1, anomaly_frames=0, anomaly_ratio=0.0,
            ),
        )
        model = MockPoseModel()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = estimate_poses(seq, model)

        assert len(result.poses) == 1
        assert result.poses[0].face_id == "main"

    def test_missing_crop_handled(self) -> None:
        """Missing/nonexistent crop → zero pose with confidence=0."""
        f = _face(0, "f0", crop_path="/nonexistent/crop.jpg")
        seq = _seq(frames=(_frame(0, (f,)),))
        model = MockPoseModel()

        with patch("pathlib.Path.exists", return_value=False):
            result = estimate_poses(seq, model)

        p = result.poses[0]
        assert p.confidence == 0.0

    def test_empty_faces_handled(self) -> None:
        """Frame with no faces → zero pose placeholder."""
        seq = _seq(frames=(_frame(0, ()), _frame(1, (_face(1, "f1"),))))
        model = MockPoseModel()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = estimate_poses(seq, model)

        assert len(result.poses) == 2
        assert result.poses[0].face_id == ""  # no_face placeholder

    def test_input_not_mutated(self) -> None:
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, (f,)),))
        orig_id = seq.clip_id
        model = MockPoseModel()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            estimate_poses(seq, model)

        assert seq.clip_id == orig_id


# ---------------------------------------------------------------------------
# Stage6PoseEstimation integration tests
# ---------------------------------------------------------------------------


class TestStage6PoseEstimation:
    def test_normal_processing(self) -> None:
        seq = _seq(frames=(
            _frame(0, (_face(0, "f0"),)),
            _frame(1, (_face(1, "f1"),)),
        ))
        stage = Stage6PoseEstimation()
        context = _make_context()

        with patch.object(stage, "_get_model", return_value=MockPoseModel()), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=[seq], context=context)

        assert len(output) == 1
        assert isinstance(output[0], PoseSequence)
        assert len(output[0].poses) == 2

    def test_disabled_returns_empty(self) -> None:
        seq = _seq(frames=(_frame(0, (_face(0, "f0"),)),))
        stage = Stage6PoseEstimation()
        context = _make_context({"stage_06": {"enabled": False}})

        output = stage.run(input_data=[seq], context=context)
        assert output == []

    def test_input_not_mutated(self) -> None:
        seq = _seq(frames=(_frame(0, (_face(0, "f0"),)),))
        orig_id = seq.clip_id
        stage = Stage6PoseEstimation()
        context = _make_context()

        with patch.object(stage, "_get_model", return_value=MockPoseModel()), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            stage.run(input_data=[seq], context=context)

        assert seq.clip_id == orig_id

    def test_input_wrong_type_raises(self) -> None:
        stage = Stage6PoseEstimation()
        assert stage.validate_input("wrong", _make_context()) is False
        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_empty_list_raises(self) -> None:
        stage = Stage6PoseEstimation()
        assert stage.validate_input([], _make_context()) is False

    def test_validate_output(self) -> None:
        stage = Stage6PoseEstimation()
        ps = PoseSequence(clip_id="c0", poses=(
            PoseInfo(frame_index=0, face_id="f0", yaw=0, pitch=0, roll=0),
        ))
        assert stage.validate_output([ps], _make_context()) is True
        assert stage.validate_output("bad", _make_context()) is False

"""Tests for Stage 8: Face Quality Assessment.

Tests cover:
    1. Mock HyperIQA → correct FaceQualityInfo.
    2. Multi-frame → output count matches.
    3. enabled=false → returns empty list.
    4. Missing crop → handled gracefully.
    5. Input immutability.
    6. Input validation.
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    FaceInfo,
    FaceQualityInfo,
    FaceQualitySequence,
    FaceSequence,
    FrameInfo,
    SingleFaceIdentityResult,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_08_quality.quality_estimator import (
    estimate_quality,
)
from face_pipeline_v2.stages.stage_08_quality.stage import (
    Stage8QualityAssessment,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_08": {"enabled": True, "batch_size": 32, "device": "cpu",
                      "checkpoint": "models/HyperIQA/"},
        "models": {"hyperiqa": {"path": "models/HyperIQA/"}},
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_08", log_dir=None),
        model_manager=object(),
    )


def _face(
    frame_index: int, face_id: str, track_id: str | None = "track_000",
    crop_path: str = "/tmp/crop.jpg",
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index, face_id=face_id, track_id=track_id,
        bbox=(0, 0, 100, 100), landmark_5=(),
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


class MockHyperIQA:
    """Mock HyperIQA returning fixed scores."""
    def predict(self, images: List[Any]) -> List[float]:
        return [72.5 for _ in images]


# ---------------------------------------------------------------------------
# estimate_quality unit tests
# ---------------------------------------------------------------------------


class TestQualityEstimator:
    def test_single_frame_sets_score(self) -> None:
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, (f,)),))
        model = MockHyperIQA()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = estimate_quality(seq, model)

        assert isinstance(result, FaceQualitySequence)
        assert result.clip_id == "c0"
        assert len(result.qualities) == 1
        q = result.qualities[0]
        assert q.frame_index == 0
        assert q.quality_score == 72.5
        assert q.confidence == 1.0

    def test_multi_frame_count_matches(self) -> None:
        frames = tuple(
            _frame(i, (_face(i, f"f{i}"),)) for i in range(5)
        )
        seq = _seq(frames=frames)
        model = MockHyperIQA()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            result = estimate_quality(seq, model)

        assert len(result.qualities) == 5

    def test_missing_crop_handled(self) -> None:
        f = _face(0, "f0", crop_path="/nonexistent.jpg")
        seq = _seq(frames=(_frame(0, (f,)),))
        model = MockHyperIQA()

        with patch("pathlib.Path.exists", return_value=False):
            result = estimate_quality(seq, model)

        assert result.qualities[0].quality_score == 0.0
        assert result.qualities[0].confidence == 0.0

    def test_input_not_mutated(self) -> None:
        f = _face(0, "f0")
        seq = _seq(frames=(_frame(0, (f,)),))
        orig_id = seq.clip_id
        model = MockHyperIQA()

        with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            estimate_quality(seq, model)

        assert seq.clip_id == orig_id


# ---------------------------------------------------------------------------
# Stage8QualityAssessment integration tests
# ---------------------------------------------------------------------------


class TestStage8QualityAssessment:
    def test_normal_processing(self) -> None:
        seq = _seq(frames=(
            _frame(0, (_face(0, "f0"),)),
            _frame(1, (_face(1, "f1"),)),
        ))
        stage = Stage8QualityAssessment()
        context = _make_context()

        with patch.object(stage, "_get_model", return_value=MockHyperIQA()), \
             patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=[seq], context=context)

        assert len(output) == 1
        assert isinstance(output[0], FaceQualitySequence)
        assert len(output[0].qualities) == 2

    def test_disabled_returns_empty(self) -> None:
        stage = Stage8QualityAssessment()
        context = _make_context({"stage_08": {"enabled": False}})
        output = stage.run(input_data=[_seq()], context=context)
        assert output == []

    def test_input_wrong_type_raises(self) -> None:
        stage = Stage8QualityAssessment()
        assert stage.validate_input("wrong", _make_context()) is False
        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_validate_output(self) -> None:
        stage = Stage8QualityAssessment()
        qs = FaceQualitySequence(clip_id="c0", qualities=(
            FaceQualityInfo(frame_index=0, face_id="f0", quality_score=80.0),
        ))
        assert stage.validate_output([qs], _make_context()) is True
        assert stage.validate_output("bad", _make_context()) is False

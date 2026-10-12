"""Tests for EnhancementStage base class.

Tests cover:
    1. enabled=false → run() returns input unchanged (passthrough).
    2. enabled=true → run() calls process().
    3. Different enhancement_key reads different config sections.
    4. Inheriting classes are instances of EnhancementStage.
    5. Invalid input still raises InputValidationError when enabled.
"""

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from face_pipeline_v2.data.models import (
    FaceInfo, FaceSequence, FrameInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.pipeline.enhancement_stage import EnhancementStage
from face_pipeline_v2.stages.stage_05_1_face_tracking.stage import (
    Stage5_1FaceTracking,
)
from face_pipeline_v2.stages.stage_05_2_single_face_filter.stage import (
    Stage5_2SingleFaceFilter,
)
from face_pipeline_v2.stages.stage_05_3_temporal_stability.stage import (
    Stage5_3TemporalStability,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Minimal concrete EnhancementStage for unit tests
# ---------------------------------------------------------------------------

class _TestEnhancement(EnhancementStage):
    """Minimal EnhancementStage implementation for testing."""

    _STAGE_NAME = "Test Enhancement"
    enhancement_key = "test_enhancement"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    def validate_input(self, input_data: Any, context: PipelineContext) -> bool:
        return isinstance(input_data, list) and len(input_data) > 0

    def process(self, input_data: Any, context: PipelineContext) -> Any:
        return [f"processed_{x}" for x in input_data]

    def validate_output(self, output_data: Any, context: PipelineContext) -> bool:
        return isinstance(output_data, list) and all(isinstance(x, str) for x in output_data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_context(enabled: bool = True, key: str = "test_enhancement") -> PipelineContext:
    return PipelineContext(
        config={
            "enhancements": {
                key: {"enabled": enabled},
            },
        },
        logger=PipelineLogger(name="test_enhancement", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnhancementStage:
    def test_disabled_passthrough(self) -> None:
        """enabled=false → input returned as-is, process() never called."""
        stage = _TestEnhancement()
        input_data = ["a", "b", "c"]
        context = _make_context(enabled=False)

        output = stage.run(input_data, context)

        # Must be the SAME object, not a copy
        assert output is input_data

    def test_enabled_calls_process(self) -> None:
        """enabled=true → process() is called and returns processed data."""
        stage = _TestEnhancement()
        input_data = ["a", "b"]
        context = _make_context(enabled=True)

        output = stage.run(input_data, context)

        assert output == ["processed_a", "processed_b"]
        assert output is not input_data

    def test_different_enhancement_keys(self) -> None:
        """EnhancementStage reads from its own enhancement_key."""
        stage = _TestEnhancement()
        stage.enhancement_key = "other_key"

        # Config has other_key.enabled=false, test_enhancement.enabled=true
        context = PipelineContext(
            config={
                "enhancements": {
                    "other_key": {"enabled": False},
                    "test_enhancement": {"enabled": True},
                },
            },
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
        )

        input_data = ["x"]
        output = stage.run(input_data, context)
        assert output is input_data  # passthrough because other_key.enabled=false

    def test_enabled_but_invalid_input_raises(self) -> None:
        """When enabled, invalid input still triggers InputValidationError."""
        stage = _TestEnhancement()
        context = _make_context(enabled=True)

        with pytest.raises(InputValidationError):
            stage.run(input_data=[], context=context)

    def test_stage_5_1_inherits_enhancement_stage(self) -> None:
        assert isinstance(Stage5_1FaceTracking(), EnhancementStage)

    def test_stage_5_2_inherits_enhancement_stage(self) -> None:
        assert isinstance(Stage5_2SingleFaceFilter(), EnhancementStage)

    def test_stage_5_3_inherits_enhancement_stage(self) -> None:
        assert isinstance(Stage5_3TemporalStability(), EnhancementStage)

    def test_stage_5_1_disabled_passthrough(self) -> None:
        """Stage 5.1 with enabled=false → passthrough via EnhancementStage."""
        stage = Stage5_1FaceTracking()
        face = FaceInfo(
            frame_index=0, face_id="f0", bbox=(0,0,100,100),
            landmark_5=(), confidence=0.9, face_crop_path=Path(""),
        )
        seq = FaceSequence(
            clip_id="c0", video_id="v",
            frames=(FrameInfo(
                frame_index=0, clip_id="c0", timestamp=0.0,
                image_path=Path(""), faces=(face,),
            ),),
        )
        context = PipelineContext(
            config={
                "enhancements": {
                    "stage_05_1_face_tracking": {"enabled": False},
                },
            },
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
        )
        output = stage.run(input_data=[seq], context=context)
        assert output is [seq] or output == [seq]

    def test_stage_5_2_disabled_passthrough(self) -> None:
        """Stage 5.2 with enabled=false → passthrough via EnhancementStage."""
        stage = Stage5_2SingleFaceFilter()
        seq = FaceSequence(clip_id="c0", video_id="v", frames=())
        context = PipelineContext(
            config={
                "enhancements": {
                    "stage_05_2_single_face_filter": {"enabled": False},
                },
            },
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
        )
        output = stage.run(input_data=[seq], context=context)
        assert len(output) == 1
        assert output[0].identity_result is None

    def test_stage_5_3_disabled_passthrough(self) -> None:
        """Stage 5.3 with enabled=false → passthrough via EnhancementStage."""
        stage = Stage5_3TemporalStability()
        seq = FaceSequence(clip_id="c0", video_id="v", frames=())
        context = PipelineContext(
            config={
                "enhancements": {
                    "stage_05_3_temporal_stability": {"enabled": False},
                },
            },
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
        )
        output = stage.run(input_data=[seq], context=context)
        assert len(output) == 1
        assert output[0].stability_result is None

    def test_enhancement_key_default_none_disabled(self) -> None:
        """enhancement_key=None → _is_enabled returns False."""
        stage = _TestEnhancement()
        stage.enhancement_key = None
        context = _make_context(enabled=True)
        assert stage._is_enabled(context) is False

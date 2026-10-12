"""Tests for PipelineContext.

Verifies that PipelineContext can be instantiated and that its
frozen dataclass properties behave correctly.
"""

import pytest

from face_pipeline_v2.pipeline.context import PipelineContext


class TestPipelineContext:
    """Tests for PipelineContext frozen dataclass."""

    def test_context_creation(self) -> None:
        """Verify PipelineContext can be created with required fields."""
        ctx = PipelineContext(
            config={},
            logger=object(),
            model_manager=object(),
        )
        assert ctx is not None
        assert ctx.config == {}
        assert ctx.runtime_info == {}

    def test_context_with_runtime_info(self) -> None:
        """Verify runtime_info stores and retrieves task state."""
        ctx = PipelineContext(
            config={"key": "value"},
            logger=object(),
            model_manager=object(),
            runtime_info={"video_id": "vid_001", "current_stage": "stage_00"},
        )
        assert ctx.video_id == "vid_001"
        assert ctx.current_stage == "stage_00"

    def test_context_is_immutable(self) -> None:
        """Verify PipelineContext is frozen — cannot be modified after creation."""
        ctx = PipelineContext(
            config={},
            logger=object(),
            model_manager=object(),
        )
        with pytest.raises(Exception):
            ctx.config = {}  # type: ignore[misc]

    def test_video_id_returns_none_when_not_set(self) -> None:
        """Verify video_id returns None when runtime_info has no video_id."""
        ctx = PipelineContext(
            config={},
            logger=object(),
            model_manager=object(),
        )
        assert ctx.video_id is None

    def test_current_stage_returns_none_when_not_set(self) -> None:
        """Verify current_stage returns None when not in runtime_info."""
        ctx = PipelineContext(
            config={},
            logger=object(),
            model_manager=object(),
        )
        assert ctx.current_stage is None

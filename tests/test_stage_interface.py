"""Tests for BaseStage abstract interface.

Verifies contract enforcement and lifecycle execution.
"""

from typing import Any

import pytest

from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
    StageExecutionError,
)


class _DummyStage(BaseStage):
    """Minimal concrete BaseStage implementation for testing."""

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        return input_data is not None

    def process(self, input_data: Any, context: PipelineContext) -> Any:
        return {"processed": True, "input": input_data}

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        return isinstance(output_data, dict) and "processed" in output_data


class _FailingInputStage(BaseStage):
    """Stage that fails input validation."""

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        return False

    def process(self, input_data: Any, context: PipelineContext) -> Any:
        return input_data

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        return True


class _FailingOutputStage(BaseStage):
    """Stage that fails output validation."""

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        return True

    def process(self, input_data: Any, context: PipelineContext) -> Any:
        return None

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        return False


class _CrashingStage(BaseStage):
    """Stage that crashes during process() with a non-PipelineError."""

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        return True

    def process(self, input_data: Any, context: PipelineContext) -> Any:
        raise RuntimeError("Simulated processing failure")

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        return True


class _PipelineErrorStage(BaseStage):
    """Stage that raises a PipelineError subclass during process()."""

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        return True

    def process(self, input_data: Any, context: PipelineContext) -> Any:
        raise InputValidationError("Simulated pipeline-level failure")

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        return True


class TestBaseStage:
    """Tests for the BaseStage abstract base class."""

    @pytest.fixture
    def context(self) -> PipelineContext:
        """Create a minimal PipelineContext for testing."""
        from face_pipeline_v2.utils.logger import PipelineLogger

        return PipelineContext(
            config={},
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
        )

    def test_stage_run_success(self, context: PipelineContext) -> None:
        """Verify a stage runs through the full lifecycle successfully."""
        stage = _DummyStage(stage_name="TestStage")
        result = stage.run(input_data={"data": 42}, context=context)
        assert result == {"processed": True, "input": {"data": 42}}

    def test_cannot_instantiate_abstract(self) -> None:
        """Verify BaseStage cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStage(stage_name="Bad")  # type: ignore[abstract]

    def test_input_validation_failure_raises_input_error(
        self, context: PipelineContext
    ) -> None:
        """Verify InputValidationError raised when input validation fails."""
        stage = _FailingInputStage(stage_name="FailInput")
        with pytest.raises(InputValidationError, match="Invalid input data"):
            stage.run(input_data={}, context=context)
        # InputValidationError is a PipelineError
        with pytest.raises(PipelineError):
            stage.run(input_data={}, context=context)

    def test_output_validation_failure_raises_stage_error(
        self, context: PipelineContext
    ) -> None:
        """Verify StageExecutionError raised when output validation fails."""
        stage = _FailingOutputStage(stage_name="FailOutput")
        with pytest.raises(
            StageExecutionError, match="Invalid output data"
        ):
            stage.run(input_data={}, context=context)

    def test_non_pipeline_exception_wrapped_as_stage_error(
        self, context: PipelineContext
    ) -> None:
        """Verify non-PipelineError exceptions are wrapped."""
        stage = _CrashingStage(stage_name="CrashStage")
        with pytest.raises(
            StageExecutionError, match="Unexpected error"
        ):
            stage.run(input_data={}, context=context)

    def test_pipeline_error_re_raised_as_is(
        self, context: PipelineContext
    ) -> None:
        """Verify PipelineError subclasses from process() pass through."""
        stage = _PipelineErrorStage(stage_name="PipelineFail")
        with pytest.raises(
            InputValidationError, match="Simulated pipeline-level failure"
        ):
            stage.run(input_data={}, context=context)

    def test_stage_name_property(self) -> None:
        """Verify stage_name property returns the constructor value."""
        stage = _DummyStage(stage_name="Stage 5 Face Detection")
        assert stage.stage_name == "Stage 5 Face Detection"

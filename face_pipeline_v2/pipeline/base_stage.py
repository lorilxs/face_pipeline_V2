"""Base Stage Interface — abstract base class for all Pipeline Stages.

All Stages MUST inherit from BaseStage and implement:
    - validate_input()
    - process()
    - validate_output()

The run() method provides the standard execution template with unified
logging and error handling per AISDD 2.10.
"""

from abc import ABC, abstractmethod
from typing import Any

from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
    StageExecutionError,
)


class BaseStage(ABC):
    """All Pipeline Stages must inherit from this abstract base class.

    Design constraints (AISDD 1.7 - 1.9):
        - Unidirectional data flow: output is a new object, never mutated input.
        - Stage communication via Data Objects only.
        - No direct access to other Stage internals.
        - Every Stage supports logging and error handling.

    Subclass contract:
        Implement validate_input(), process(), validate_output().
        Call run() to execute the standard pipeline.
    """

    def __init__(self, stage_name: str) -> None:
        """Initialize the BaseStage.

        Args:
            stage_name: Human-readable stage identifier
                        (e.g. "Stage 5 Face Detection").
        """
        self._stage_name: str = stage_name

    @property
    def stage_name(self) -> str:
        """Return the stage name."""
        return self._stage_name

    @abstractmethod
    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data meets this stage's requirements.

        Called before process(). If this returns False,
        run() raises InputValidationError.

        Args:
            input_data: The data object passed to this stage.
            context: Current PipelineContext.

        Returns:
            True if input is valid, False otherwise.
        """
        ...

    @abstractmethod
    def process(
        self, input_data: Any, context: PipelineContext
    ) -> Any:
        """Execute the core stage logic.

        The heart of the stage — performs the actual computation/transformation.
        Must return a NEW data object; must NOT mutate input_data.

        Args:
            input_data: Validated input data object.
            context: Current PipelineContext.

        Returns:
            A new data object representing the stage output.
        """
        ...

    @abstractmethod
    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output_data meets this stage's output contract.

        Called after process(). If this returns False,
        run() raises StageExecutionError.

        Args:
            output_data: The data object produced by process().
            context: Current PipelineContext.

        Returns:
            True if output is valid, False otherwise.
        """
        ...

    def run(self, input_data: Any, context: PipelineContext) -> Any:
        """Standard stage execution template.

        Executes the stage lifecycle:
            1. Log stage start
            2. validate_input()
            3. process()
            4. validate_output()
            5. Log stage completion

        Args:
            input_data: Input data object for this stage.
            context: PipelineContext carrying runtime environment.

        Returns:
            The stage's output data object.

        Raises:
            InputValidationError: If input validation fails.
            StageExecutionError: If output validation fails or
                                 a non-PipelineError exception occurs.
            PipelineError: Subclasses from process() are re-raised as-is.
        """
        logger = context.logger
        logger.info(f"[{self._stage_name}] Starting...")

        # Step 1: Validate input
        if not self.validate_input(input_data, context):
            logger.error(f"[{self._stage_name}] Input validation failed.")
            raise InputValidationError(
                f"[{self._stage_name}] Invalid input data: "
                f"{type(input_data).__name__}"
            )
        logger.debug(f"[{self._stage_name}] Input validation passed.")

        # Step 2: Process
        try:
            output_data = self.process(input_data, context)
        except PipelineError:
            # Re-raise PipelineError subclasses as-is (already meaningful)
            raise
        except Exception as exc:
            logger.error(
                f"[{self._stage_name}] Processing failed: {exc}"
            )
            raise StageExecutionError(
                f"[{self._stage_name}] Unexpected error during processing."
            ) from exc

        logger.debug(f"[{self._stage_name}] Processing complete.")

        # Step 3: Validate output
        if not self.validate_output(output_data, context):
            logger.error(
                f"[{self._stage_name}] Output validation failed."
            )
            raise StageExecutionError(
                f"[{self._stage_name}] Invalid output data: "
                f"{type(output_data).__name__}"
            )
        logger.debug(f"[{self._stage_name}] Output validation passed.")

        logger.info(f"[{self._stage_name}] Completed successfully.")
        return output_data

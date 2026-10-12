"""Enhancement Stage base class.

Per AISDD 3.9, Enhancement Modules (Stage 5.1/5.2/5.3/9.1) share a common
pattern: they are optional, config-driven, and must pass through input
unchanged when disabled.

EnhancementStage extends BaseStage with:
    - Unified enabled-check: reads context.config["enhancements"][key]["enabled"].
    - Disabled passthrough: returns input_data unchanged by default.
      Subclasses whose input and output types differ MUST override
      _get_passthrough_output() to return a type-consistent value.

Subclasses set `enhancement_key` and implement `process()` as usual.
They do NOT check enabled in process() — that's handled by run().
"""

from typing import Any

from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext


class EnhancementStage(BaseStage):
    """Base class for optional Enhancement Modules.

    Subclass contract:
        - Set `enhancement_key` class attribute (e.g. "stage_05_1_face_tracking").
        - Implement `validate_input()`, `process()`, `validate_output()`
          as usual (same as BaseStage).
        - Do NOT check enabled in process() — run() handles it.
        - If input type != output type, override _get_passthrough_output()
          to return a type-consistent empty/neutral value.

    When disabled, run() returns _get_passthrough_output(input_data).
    Default: input_data unchanged (works when input_type == output_type).
    When enabled, run() delegates to BaseStage.run().
    """

    # Set by subclass — matches a key under config["enhancements"]
    enhancement_key: str | None = None

    def __init__(self, stage_name: str) -> None:
        """Initialize EnhancementStage.

        Args:
            stage_name: Human-readable stage identifier
                        (e.g. "Stage 5.1 Face Tracking").
        """
        super().__init__(stage_name=stage_name)

    def _is_enabled(self, context: PipelineContext) -> bool:
        """Check whether this enhancement is enabled in config.

        Reads: context.config["enhancements"][self.enhancement_key]["enabled"].

        Args:
            context: PipelineContext.

        Returns:
            True if the enhancement is enabled.
        """
        if self.enhancement_key is None:
            return False
        enhancements = context.config.get("enhancements", {})
        cfg = enhancements.get(self.enhancement_key, {})
        return bool(cfg.get("enabled", False))

    def _get_passthrough_output(self, input_data: Any) -> Any:
        """Return the value to emit when this enhancement is disabled.

        Default: return input_data unchanged (passthrough).
        This works for stages where input type == output type
        (e.g. Stage 5.1/5.2/5.3: List[FaceSequence] → List[FaceSequence]).

        Override in subclasses where input type != output type
        (e.g. Stage 9.1: dict → List[QualityRankingInfo]).
        The override MUST return a value matching the enabled output type.

        Args:
            input_data: The original input data.

        Returns:
            A type-consistent passthrough value.
        """
        return input_data

    def run(self, input_data: Any, context: PipelineContext) -> Any:
        """Standard enhancement execution template.

        If disabled: return _get_passthrough_output(input_data).
        If enabled: delegate to BaseStage.run() → validate → process → validate.

        Args:
            input_data: Input data object.
            context: PipelineContext.

        Returns:
            Type-consistent passthrough value (disabled) or processed output (enabled).
        """
        if not self._is_enabled(context):
            context.logger.info(
                f"[{self._stage_name}] Enhancement disabled "
                f"(enhancements.{self.enhancement_key}.enabled=false). "
                f"Passing through unchanged."
            )
            return self._get_passthrough_output(input_data)

        return super().run(input_data, context)

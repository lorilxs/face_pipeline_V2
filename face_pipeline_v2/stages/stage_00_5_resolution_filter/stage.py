"""Stage 0.5: Resolution Filter.

Per AISDD 3.4, this stage performs a video-level resolution hard filter
immediately after Stage 0 metadata extraction and before any model inference.
Low-resolution videos are rejected early to avoid wasting downstream compute.

Input:  VideoInfo (from Stage 0)
Output: ResolutionValidationInfo
"""

from typing import Any

from face_pipeline_v2.data.models import ResolutionValidationInfo, VideoInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_00_5_resolution_filter.processor import (
    validate_resolution,
)


class Stage0_5ResolutionFilter(BaseStage):
    """Stage 0.5 — Video resolution hard filter (rule-based, no model).

    Input:  VideoInfo
    Output: ResolutionValidationInfo
    """

    _STAGE_NAME: str = "Stage 0.5 Resolution Filter"

    def __init__(self) -> None:
        """Initialize Stage 0.5."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a VideoInfo.

        Args:
            input_data: Expected to be a VideoInfo from Stage 0.
            context: PipelineContext.

        Returns:
            True if input_data is a VideoInfo.
        """
        if not isinstance(input_data, VideoInfo):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected VideoInfo, "
                f"got {type(input_data).__name__}"
            )
            return False
        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> ResolutionValidationInfo:
        """Apply the minimum-resolution hard filter.

        Steps (AISDD 3.4.4):
            1. Read enabled / min_width / min_height from
               stage_00_resolution_filter config.
            2. If disabled, pass through (is_valid=True).
            3. Otherwise compare width/height against the minimums.
            4. Log the accept/reject decision.

        Args:
            input_data: VideoInfo from Stage 0.
            context: PipelineContext with config and logger.

        Returns:
            ResolutionValidationInfo with the pass/fail decision.
        """
        logger = context.logger
        config = context.config
        video_info: VideoInfo = input_data

        cfg: dict = config.get("stage_00_resolution_filter", {})
        enabled: bool = bool(cfg.get("enabled", True))
        min_width: int = int(cfg.get("min_width", 640))
        min_height: int = int(cfg.get("min_height", 360))

        if not enabled:
            logger.info(
                f"[{self._STAGE_NAME}] Disabled — skipping resolution filter."
            )
            return ResolutionValidationInfo(
                width=video_info.width,
                height=video_info.height,
                min_width=min_width,
                min_height=min_height,
                is_valid=True,
                reject_reason=None,
            )

        result = validate_resolution(video_info, min_width, min_height)

        if result.is_valid:
            logger.info(
                f"[{self._STAGE_NAME}] Resolution accepted: "
                f"{result.width}x{result.height}"
            )
        else:
            logger.warning(
                f"[{self._STAGE_NAME}] Resolution rejected: "
                f"{result.width}x{result.height} "
                f"(minimum:{result.min_width}x{result.min_height})"
            )

        return result

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a ResolutionValidationInfo.

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output_data is a ResolutionValidationInfo.
        """
        return isinstance(output_data, ResolutionValidationInfo)

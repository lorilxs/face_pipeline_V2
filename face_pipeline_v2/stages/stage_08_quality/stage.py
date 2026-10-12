"""Stage 8: Face Quality Assessment.

Per AISDD 3.12, this stage evaluates per-frame face crop quality
using HyperIQA for each clip.

Input:  List[FaceSequence] (from Stage 5.x)
Output: List[FaceQualitySequence]
"""

from pathlib import Path
from typing import Any, List

from face_pipeline_v2.data.models import (
    FaceSequence,
    FaceQualityInfo,
    FaceQualitySequence,
)
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_08_quality.quality_estimator import (
    estimate_quality,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage8QualityAssessment(BaseStage):
    """Stage 8 — Face Quality Assessment (HyperIQA).

    Input:  List[FaceSequence]
    Output: List[FaceQualitySequence]
    """

    _STAGE_NAME: str = "Stage 8 Quality Assessment"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[FaceSequence], "
                f"got {type(input_data).__name__}"
            )
            return False
        if len(input_data) == 0:
            context.logger.warning(f"[{self._STAGE_NAME}] Empty sequence list.")
            return False
        return all(isinstance(s, FaceSequence) for s in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[FaceQualitySequence]:
        """Assess face quality for all sequences.

        Args:
            input_data: List[FaceSequence].
            context: PipelineContext.

        Returns:
            List[FaceQualitySequence], one per clip.
        """
        logger = context.logger
        config = context.config
        sequences: List[FaceSequence] = input_data

        stage_cfg: dict = config.get("stage_08", {})
        if not stage_cfg.get("enabled", True):
            logger.info(f"[{self._STAGE_NAME}] Stage 8 disabled.")
            return []

        batch_size: int = int(stage_cfg.get("batch_size", 32))

        # Preprocess config — resize only (expand now done at Stage 5 bbox level)
        preprocess_cfg = stage_cfg.get("preprocess", {})
        input_size = int(preprocess_cfg.get("input_size", 224))

        logger.info(
            f"[{self._STAGE_NAME}] Config: batch_size={batch_size}, "
            f"input_size={input_size}"
        )

        # --- Get HyperIQA model ---
        model = self._get_model(config, context, logger)

        # --- Process each sequence ---
        output: List[FaceQualitySequence] = []
        total_frames = 0

        for seq in sequences:
            qual_seq = estimate_quality(seq, model, target_size=input_size)
            output.append(qual_seq)
            total_frames += len(qual_seq.qualities)

        logger.info(
            f"[{self._STAGE_NAME}] Quality assessment complete: "
            f"{len(output)} sequences, {total_frames} frames"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            return False
        return all(isinstance(s, FaceQualitySequence) for s in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model(
        config: dict,
        context: PipelineContext,
        logger: Any,
    ) -> Any:
        """Get or load HyperIQA model."""
        try:
            return context.model_manager.get_model("hyperiqa")
        except (KeyError, AttributeError):
            stage_cfg: dict = config.get("stage_08", {})
            checkpoint = Path(stage_cfg.get(
                "checkpoint", "models/HyperIQA/"
            ))
            device: str = stage_cfg.get("device", "cpu")

            logger.info(
                f"[Stage 8] Loading HyperIQA: path={checkpoint}, device={device}"
            )
            try:
                return context.model_manager.load_model(
                    "hyperiqa", str(checkpoint)
                )
            except (NotImplementedError, AttributeError):
                from face_pipeline_v2.models.wrappers.hyperiqa import (
                    HyperIQAWrapper,
                )
                wrapper = HyperIQAWrapper(
                    model_path=checkpoint, device=device,
                )
                wrapper.load()
                return wrapper
            except Exception as exc:
                logger.error(f"[Stage 8] Failed to load HyperIQA: {exc}")
                raise PipelineError(
                    "[Stage 8] HyperIQA loading failed."
                ) from exc

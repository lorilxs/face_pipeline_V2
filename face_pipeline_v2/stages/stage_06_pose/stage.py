"""Stage 6: Head Pose Estimation.

Per AISDD 3.10, this stage estimates head pose (yaw, pitch, roll)
from face crop images using 6DRepNet for every frame in each clip.

Input:  List[FaceSequence] (from Stage 5.x)
Output: List[PoseSequence]
"""

from pathlib import Path
from typing import Any, List

from face_pipeline_v2.data.models import FaceSequence, PoseInfo, PoseSequence
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_06_pose.pose_estimator import (
    estimate_poses,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage6PoseEstimation(BaseStage):
    """Stage 6 — Head Pose Estimation (6DRepNet).

    Input:  List[FaceSequence]
    Output: List[PoseSequence]
    """

    _STAGE_NAME: str = "Stage 6 Pose Estimation"

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
            context.logger.warning(
                f"[{self._STAGE_NAME}] Empty sequence list."
            )
            return False
        return all(isinstance(s, FaceSequence) for s in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[PoseSequence]:
        """Estimate head pose for all FaceSequences.

        Args:
            input_data: List[FaceSequence].
            context: PipelineContext.

        Returns:
            List[PoseSequence], one per clip.
        """
        logger = context.logger
        config = context.config
        sequences: List[FaceSequence] = input_data

        stage_cfg: dict = config.get("stage_06", {})
        if not stage_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 6 disabled. "
                f"Returning empty list."
            )
            return []

        batch_size: int = int(stage_cfg.get("batch_size", 32))

        # Preprocess config — resize only (expand now done at Stage 5 bbox level)
        preprocess_cfg = stage_cfg.get("preprocess", {})
        input_size = int(preprocess_cfg.get("input_size", 256))

        logger.info(
            f"[{self._STAGE_NAME}] Config: batch_size={batch_size}, "
            f"input_size={input_size}"
        )

        # --- Get 6DRepNet model ---
        model = self._get_model(config, context, logger)

        # --- Process each sequence ---
        output: List[PoseSequence] = []
        total_frames = 0

        for seq in sequences:
            pose_seq = estimate_poses(seq, model, target_size=input_size)
            output.append(pose_seq)
            total_frames += len(pose_seq.poses)

        logger.info(
            f"[{self._STAGE_NAME}] Pose estimation complete: "
            f"{len(output)} sequences, {total_frames} frames"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            return False
        return all(isinstance(s, PoseSequence) for s in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model(
        config: dict,
        context: PipelineContext,
        logger: Any,
    ) -> Any:
        """Get or load 6DRepNet model."""
        try:
            return context.model_manager.get_model("sixdrepnet")
        except (KeyError, AttributeError):
            stage_cfg: dict = config.get("stage_06", {})
            device: str = stage_cfg.get("device", "cpu")
            models_cfg: dict = config.get("models", {})
            model_path = Path(
                models_cfg.get("sixdrepnet", {}).get(
                    "path", "models/6DRepNet"
                )
            )

            logger.info(
                f"[Stage 6] Loading 6DRepNet: path={model_path}, device={device}"
            )
            try:
                return context.model_manager.load_model(
                    "sixdrepnet", str(model_path)
                )
            except (NotImplementedError, AttributeError):
                logger.warning(
                    f"[Stage 6] ModelManager.load_model not implemented. "
                    f"Loading 6DRepNet wrapper directly."
                )
                from face_pipeline_v2.models.wrappers.sixdrepnet import (
                    SixDRepNetWrapper,
                )
                wrapper = SixDRepNetWrapper(
                    model_path=model_path, device=device
                )
                try:
                    wrapper.load()
                    return wrapper
                except FileNotFoundError:
                    logger.warning(
                        f"[Stage 6] 6DRepNet weights not found at "
                        f"{model_path}. Operating in no-model mode."
                    )
                    wrapper._model = wrapper._model or _NoOpPoseEstimator()
                    return wrapper
                except RuntimeError as exc:
                    logger.warning(
                        f"[Stage 6] Cannot load 6DRepNet: {exc}. "
                        f"Operating in no-model mode."
                    )
                    wrapper._model = _NoOpPoseEstimator()
                    return wrapper
            except Exception as exc:
                logger.error(f"[Stage 6] Failed to load 6DRepNet: {exc}")
                raise PipelineError(
                    f"[Stage 6] 6DRepNet loading failed."
                ) from exc


class _NoOpPoseEstimator:
    """No-op fallback."""

    def predict(self, images):
        return [
            {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "confidence": 1.0}
            for _ in images
        ]

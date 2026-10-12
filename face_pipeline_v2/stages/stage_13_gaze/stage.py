"""Stage 13: Gaze Parameter Extraction (MediaPipe FaceLandmarker).

Per AISDD 3.17, this stage extracts gaze vectors from aligned face
images using MediaPipe iris tracking (landmarks 468/473).

Input:  List[AlignedFaceInfo]  (from Stage 11)
Output: List[GazeFeature]

Does NOT modify FLAMEFeature.eye_pose — that is Stage17's responsibility.
"""

from pathlib import Path
from typing import Any, List

from face_pipeline_v2.data.models import AlignedFaceInfo, GazeFeature
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_13_gaze.processor import (
    create_mediapipe_detector,
    extract_gaze_parameters,
)
from face_pipeline_v2.stages.stage_13_gaze.integration import (
    save_gaze_features,
    read_gaze_features,
)


class Stage13Gaze(BaseStage):
    """Stage 13 — Gaze Parameter Extraction (MediaPipe).

    Input:  List[AlignedFaceInfo]
    Output: List[GazeFeature]
    """

    _STAGE_NAME: str = "Stage 13 Gaze Parameter Extraction"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[AlignedFaceInfo], "
                f"got {type(input_data).__name__}"
            )
            return False
        if len(input_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Input is empty: no faces to process."
            )
            return True
        return all(isinstance(m, AlignedFaceInfo) for m in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[GazeFeature]:
        aligned_faces: List[AlignedFaceInfo] = input_data
        config: dict = context.config
        logger = context.logger

        if len(aligned_faces) == 0:
            logger.info(
                f"[{self._STAGE_NAME}] No aligned faces to process. "
                f"Returning empty list."
            )
            return []

        stage13_cfg: dict = config.get("stage13", {})
        gaze_cfg: dict = stage13_cfg.get("gaze", {})
        model_path: str = gaze_cfg.get(
            "model_path", "models/mediapipe/face_landmarker.task",
        )
        delegate: str = gaze_cfg.get("delegate", "cpu")
        image_size = tuple(gaze_cfg.get("image_size", [256, 256]))

        detector = create_mediapipe_detector(
            model_path=model_path,
            delegate=delegate,
        )

        try:
            results = extract_gaze_parameters(
                aligned_faces=aligned_faces,
                detector=detector,
                image_size=image_size,
            )
        finally:
            detector.close()

        # Save to NPZ for Stage17
        if results:
            clip_id = results[0].clip_id
            output_dir = Path(
                stage13_cfg.get("output_dir", "outputs/stage13_gaze")
            )
            save_gaze_features(results, output_dir, clip_id)
            logger.info(
                f"[{self._STAGE_NAME}] Saved {len(results)} gaze features "
                f"to {output_dir / clip_id}"
            )

        logger.info(
            f"[{self._STAGE_NAME}] Gaze extraction complete: "
            f"{len(aligned_faces)} faces → {len(results)} GazeFeature entries"
        )
        return results

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[GazeFeature], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty: "
                f"no gaze features extracted."
            )
            return True
        if not all(isinstance(f, GazeFeature) for f in output_data):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements must be GazeFeature."
            )
            return False

        f0 = output_data[0]
        if len(f0.gaze_vector) != 6:
            context.logger.warning(
                f"[{self._STAGE_NAME}] gaze_vector has {len(f0.gaze_vector)} "
                f"dims, expected 6."
            )
            return False
        if len(f0.left_eye) != 3 or len(f0.right_eye) != 3:
            context.logger.warning(
                f"[{self._STAGE_NAME}] left_eye/right_eye must be 3-tuples."
            )
            return False
        return True
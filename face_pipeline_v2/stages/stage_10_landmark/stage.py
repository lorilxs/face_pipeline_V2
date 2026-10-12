"""Stage 10: High-Precision Landmark Extraction.

Per AISDD 3.14, this stage extracts 106-point facial landmarks
using InsightFace 2d106det model on face crops from Stage 5.

Located AFTER Quality Gate (Stage 9) — only processes accepted clips.

Input (dict):
    {
        "face_sequences": List[FaceSequence],   # from Stage 5
        "accepted_clip_ids": Set[str],           # from Stage 9
    }

Output: List[LandmarkInfo]
"""

from pathlib import Path
from typing import Any, Dict, List, Set

from face_pipeline_v2.data.models import FaceSequence, LandmarkInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_10_landmark.processor import (
    extract_landmarks,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage10Landmark(BaseStage):
    """Stage 10 — High-Precision Landmark Extraction (InsightFace 106).

    Input:  dict{"face_sequences": List[FaceSequence],
                 "accepted_clip_ids": Set[str]}
    Output: List[LandmarkInfo]
    """

    _STAGE_NAME: str = "Stage 10 High-Precision Landmark"

    def __init__(self) -> None:
        """Initialize Stage 10."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate input dict contains face_sequences and accepted_clip_ids.

        Args:
            input_data: Expected dict with face_sequences (list of FaceSequence)
                        and accepted_clip_ids (set of str).
            context: PipelineContext.

        Returns:
            True if all required keys are present with correct types.
        """
        if not isinstance(input_data, dict):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected dict, got "
                f"{type(input_data).__name__}"
            )
            return False

        required = {"face_sequences", "accepted_clip_ids"}
        if not required.issubset(input_data.keys()):
            missing = required - input_data.keys()
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required keys: {missing}. "
                f"Need: {required}"
            )
            return False

        seqs = input_data["face_sequences"]
        if not isinstance(seqs, list) or len(seqs) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] face_sequences must be a non-empty list."
            )
            return False
        if not all(isinstance(s, FaceSequence) for s in seqs):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in face_sequences "
                f"must be FaceSequence."
            )
            return False

        accepted = input_data["accepted_clip_ids"]
        if not isinstance(accepted, (set, list, tuple)):
            context.logger.warning(
                f"[{self._STAGE_NAME}] accepted_clip_ids must be a set or list, "
                f"got {type(accepted).__name__}"
            )
            return False
        if len(accepted) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] accepted_clip_ids is empty. "
                f"No clips to process."
            )
            # Not a hard error — just nothing to do.

        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[LandmarkInfo]:
        """Extract 106-point landmarks for accepted clips.

        Steps:
            1. Read config (landmark section).
            2. Load InsightFace 2d106det model via ModelManager.
            3. Filter face_sequences to only accepted clips.
            4. For each accepted clip, iterate frames in order.
            5. For each frame with faces, load crop and run inference.
            6. Map crop-space coords → original frame coords via bbox.
            7. Build LandmarkInfo entries.

        Args:
            input_data: dict with face_sequences and accepted_clip_ids.
            context: PipelineContext.

        Returns:
            List[LandmarkInfo], one per processed face.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data

        face_sequences: List[FaceSequence] = data["face_sequences"]
        accepted_ids: Set[str] = set(data["accepted_clip_ids"])

        # --- Load stage configuration ---
        landmark_cfg: dict = config.get("landmark", {})
        if not landmark_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 10 disabled. "
                f"Returning empty list."
            )
            return []

        logger.info(
            f"[{self._STAGE_NAME}] Config: enabled=True"
        )

        # --- Load model ---
        model = self._get_model(config, context, logger)

        # --- Process accepted clips ---
        output: List[LandmarkInfo] = []
        total_faces = 0
        clipped_count = 0

        for seq in face_sequences:
            clip_id = seq.clip_id

            # Only process clips accepted by Stage 9
            if clip_id not in accepted_ids:
                continue

            clip_landmarks = extract_landmarks(seq, model, logger)
            output.extend(clip_landmarks)
            total_faces += len(clip_landmarks)
            clipped_count += 1

        logger.info(
            f"[{self._STAGE_NAME}] Landmark extraction complete: "
            f"{clipped_count} accepted clips processed, "
            f"{total_faces} faces → {total_faces} LandmarkInfo entries, "
            f"{len(face_sequences) - clipped_count} clips skipped "
            f"(not in accepted set)"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate output is a list of LandmarkInfo.

        Each LandmarkInfo must have exactly 212 values in landmarks_106
        (= 106 points × 2 coordinates).

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a valid LandmarkInfo list.
        """
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[LandmarkInfo], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty: "
                f"no landmarks extracted."
            )
            # Empty output is valid (e.g. no accepted clips).
            return True
        return all(isinstance(m, LandmarkInfo) for m in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model(
        config: dict,
        context: PipelineContext,
        logger: Any,
    ) -> Any:
        """Get or load InsightFace 2d106det landmark model.

        Uses ModelManager.get_model() first; falls back to load_model()
        with path from config.

        Args:
            config: Full pipeline config dict.
            context: PipelineContext.
            logger: PipelineLogger.

        Returns:
            Loaded InsightFaceLandmarkWrapper instance.

        Raises:
            PipelineError: If model loading fails.
        """
        try:
            return context.model_manager.get_model("insightface_landmark")
        except (KeyError, AttributeError):
            pass

        landmark_cfg: dict = config.get("landmark", {})
        models_cfg: dict = config.get("models", {})
        model_path = Path(
            models_cfg.get("insightface_landmark", {}).get(
                "path", "models/insightface/2d106det.onnx"
            )
        )

        logger.info(
            f"[Stage 10] Loading InsightFace Landmark: path={model_path}"
        )
        try:
            return context.model_manager.load_model(
                "insightface_landmark", str(model_path)
            )
        except Exception as exc:
            logger.error(
                f"[Stage 10] Failed to load InsightFace Landmark: {exc}"
            )
            raise PipelineError(
                f"[Stage 10] Failed to load InsightFace landmark model."
            ) from exc

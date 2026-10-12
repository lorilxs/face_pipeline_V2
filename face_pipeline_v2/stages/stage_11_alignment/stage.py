"""Stage 11: Face Alignment.

Per AISDD 3.15, this stage normalizes faces to a standard 256×256
canonical space using landmark-based affine (similarity) transforms.

Consumes Stage 10 landmarks + Stage 5 face crops.
Produces AlignedFaceInfo with affine matrix for inverse recovery.

Located AFTER Stage 10 Landmark Extraction.

Input (dict):
    {
        "landmarks": List[LandmarkInfo],         # from Stage 10
        "face_sequences": List[FaceSequence],    # from Stage 5
    }

Output: List[AlignedFaceInfo]
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from face_pipeline_v2.data.models import (
    AlignedFaceInfo,
    FaceSequence,
    LandmarkInfo,
)
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_11_alignment.processor import (
    align_faces,
    resolve_alignment_template,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage11Alignment(BaseStage):
    """Stage 11 — Face Alignment (Landmark-based Affine Transform).

    Uses a 2-point (eye-center) similarity transform + cv2.warpAffine.
    No GPU, no PyTorch, no model inference.

    Input:  dict{"landmarks": List[LandmarkInfo],
                 "face_sequences": List[FaceSequence]}
    Output: List[AlignedFaceInfo]
    """

    _STAGE_NAME: str = "Stage 11 Face Alignment"

    def __init__(self) -> None:
        """Initialize Stage 11."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate input dict contains landmarks and face_sequences.

        Args:
            input_data: Expected dict with landmarks (list of LandmarkInfo)
                        and face_sequences (list of FaceSequence).
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

        required = {"landmarks", "face_sequences"}
        if not required.issubset(input_data.keys()):
            missing = required - input_data.keys()
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required keys: {missing}. "
                f"Need: {required}"
            )
            return False

        lm_list = input_data["landmarks"]
        if not isinstance(lm_list, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] landmarks must be a list, "
                f"got {type(lm_list).__name__}"
            )
            return False
        if len(lm_list) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] landmarks list is empty. "
                f"No faces to align."
            )
            # Not a hard error — just nothing to do.
        elif not all(isinstance(m, LandmarkInfo) for m in lm_list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in landmarks "
                f"must be LandmarkInfo."
            )
            return False

        seqs = input_data["face_sequences"]
        if not isinstance(seqs, list) or len(seqs) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] face_sequences must be a "
                f"non-empty list."
            )
            return False
        if not all(isinstance(s, FaceSequence) for s in seqs):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in face_sequences "
                f"must be FaceSequence."
            )
            return False

        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[AlignedFaceInfo]:
        """Align faces using Stage 10 landmarks.

        Steps:
            1. Read config (alignment section).
            2. Parse output resolution and save settings.
            3. Call align_faces() processor.
            4. Return list of AlignedFaceInfo.

        Args:
            input_data: dict with landmarks and face_sequences.
            context: PipelineContext.

        Returns:
            List[AlignedFaceInfo], one per successfully aligned face.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data

        landmarks: List[LandmarkInfo] = data["landmarks"]
        face_sequences: List[FaceSequence] = data["face_sequences"]

        # --- Load stage configuration ---
        alignment_cfg: dict = config.get("alignment", {})
        if not alignment_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 11 disabled. "
                f"Returning empty list."
            )
            return []

        resolution: int = int(
            alignment_cfg.get(
                "output_size",
                alignment_cfg.get("resolution", 256),
            )
        )
        save_images: bool = bool(
            alignment_cfg.get("save_aligned_faces", True)
        )
        # Output directory unified under storage.output_dir (NOT alignment.output_dir).
        # alignment.output_dir is deprecated — kept for backward compat only.
        storage_cfg: dict = config.get("storage", {})
        storage_output_dir = Path(storage_cfg.get("output_dir", "outputs"))
        output_dir = storage_output_dir / "aligned_faces"
        output_resolution: Tuple[int, int] = (resolution, resolution)

        logger.info(
            f"[{self._STAGE_NAME}] Config: resolution={resolution}, "
            f"save_images={save_images}, output_dir={output_dir}"
        )

        # --- Resolve canonical alignment template from config ---
        # Legacy configs without alignment.template keep the exact pre-change
        # template (byte-identical output); see resolve_alignment_template().
        template_params = resolve_alignment_template(alignment_cfg)
        eyes_template = template_params["eyes"]
        logger.info(
            f"[{self._STAGE_NAME}] alignment template: "
            f"eye_distance={template_params['eye_distance']:.2f}, "
            f"eye_center_y={template_params['eye_center_y']:.2f}, "
            f"scale_factor={template_params['scale_factor']:.3f}"
        )

        # --- Align faces ---
        aligned = align_faces(
            landmarks=landmarks,
            face_sequences=face_sequences,
            output_dir=output_dir,
            output_resolution=output_resolution,
            save_images=save_images,
            logger=logger,
            eye_template=eyes_template,
        )

        logger.info(
            f"[{self._STAGE_NAME}] Alignment complete: "
            f"{len(aligned)} faces aligned"
        )
        # Output summary log
        unique_clips = len({a.clip_id for a in aligned})
        logger.info(
            f"[{self._STAGE_NAME}] Alignment output: "
            f"output_dir={output_dir}, clips={unique_clips}, "
            f"aligned_images={len(aligned)}"
        )

        return aligned

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate output is a list of AlignedFaceInfo.

        Each AlignedFaceInfo must have exactly 6 values in affine_matrix
        and aligned_resolution must match config.

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a valid AlignedFaceInfo list.
        """
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[AlignedFaceInfo], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty: "
                f"no faces aligned."
            )
            return True
        return all(isinstance(m, AlignedFaceInfo) for m in output_data)

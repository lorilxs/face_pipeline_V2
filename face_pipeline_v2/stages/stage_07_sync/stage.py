"""Stage 7: Audio-Visual Synchronization Detection.

Per AISDD 3.11 (revised), this stage detects audio-video offset and sync
quality using SyncNet.  Visual input is face crops from Stage 5 — NOT
full-frame video.

Input:
    {
        "sequences": List[FaceSequence],
        "audio_info": AudioInfo,
        "clip_infos": List[ClipInfo],
        "std_video_path": Path,
    }

Output: List[AVSyncInfo]
"""

from pathlib import Path
from typing import Any, Dict, List

from face_pipeline_v2.data.models import AVSyncInfo, AudioInfo, ClipInfo, FaceSequence
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_07_sync.sync_estimator import (
    _check_syncnet_dependencies,
    estimate_sync,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage7SyncDetection(BaseStage):
    """Stage 7 — A/V Sync Detection (SyncNet).

    Uses face crops (Stage 5) as visual input for SyncNet.  The S3FD tight
    crop is reconstructed from the standardized full-frame video to avoid the
    gray-fill collapse caused by the Stage-5 2× crop.

    Input:  dict with "sequences", "audio_info", "clip_infos", "std_video_path"
    Output: List[AVSyncInfo]
    """

    _STAGE_NAME: str = "Stage 7 Sync Detection"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, dict):
            return False
        required = {"sequences", "audio_info", "clip_infos", "std_video_path"}
        if not required.issubset(input_data.keys()):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required keys. "
                f"Need: {required}"
            )
            return False
        seqs = input_data["sequences"]
        if not isinstance(seqs, list) or len(seqs) == 0:
            return False
        return all(isinstance(s, FaceSequence) for s in seqs)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[AVSyncInfo]:
        """Detect A/V sync for all clips using face crops.

        Args:
            input_data: dict with sequences, audio_info, clip_infos,
                std_video_path.
            context: PipelineContext.

        Returns:
            List[AVSyncInfo], one per clip.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data
        sequences: List[FaceSequence] = data["sequences"]
        clip_infos: List[ClipInfo] = data["clip_infos"]
        std_video_path: Path = Path(data["std_video_path"])

        stage_cfg: dict = config.get("stage_07", {})
        if not stage_cfg.get("enabled", True):
            logger.info(f"[{self._STAGE_NAME}] Stage 7 disabled.")
            return []

        batch_size: int = int(stage_cfg.get("batch_size", 32))
        vshift: int = int(stage_cfg.get("vshift", 10))
        crop_scale: float = float(stage_cfg.get("crop_scale", 0.40))
        preprocess_cfg: dict = stage_cfg.get("preprocess", {})
        input_size: int = int(preprocess_cfg.get("input_size", 256))

        logger.info(
            f"[{self._STAGE_NAME}] Config: batch_size={batch_size}, "
            f"vshift={vshift}, crop_scale={crop_scale}, input_size={input_size}"
        )

        # --- Check SyncNet python dependencies (torch/cv2/scipy/speech_features) ---
        _check_syncnet_dependencies(logger)

        # --- Resolve audio path ---
        audio_info: AudioInfo = data["audio_info"]
        audio_path = audio_info.audio_path

        # --- Resolve temp directory from storage config ---
        temp_base = self._resolve_temp_base(config)

        # --- Get SyncNet model ---
        model = self._get_model(config, context, logger, batch_size, vshift)

        # --- Process each clip ---
        clip_by_id: Dict[str, ClipInfo] = {c.clip_id: c for c in clip_infos}

        output: List[AVSyncInfo] = []
        for seq in sequences:
            clip = clip_by_id.get(seq.clip_id)
            if clip is None:
                logger.warning(
                    f"[{self._STAGE_NAME}] No ClipInfo found for {seq.clip_id}"
                )
                continue

            av_sync = estimate_sync(
                sequence=seq,
                clip=clip,
                audio_path=audio_path,
                model=model,
                std_video_path=std_video_path,
                temp_base=temp_base,
                logger=logger,
                crop_scale=crop_scale,
                input_size=input_size,
            )
            output.append(av_sync)
            logger.debug(
                f"[{self._STAGE_NAME}] {seq.clip_id}: "
                f"offset={av_sync.offset}, "
                f"lse_c={av_sync.sync_lse_c:.4f}, "
                f"lse_d={av_sync.sync_lse_d:.3f}, "
                f"median_dist={av_sync.median_distance:.3f}"
            )

        logger.info(
            f"[{self._STAGE_NAME}] Sync detection complete: "
            f"{len(output)} clips processed"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            return False
        return all(isinstance(s, AVSyncInfo) for s in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_temp_base(config: dict) -> Path | None:
        """Resolve temporary file base directory from storage config.

        Args:
            config: Full pipeline config dict.

        Returns:
            Path to use as temp base, or None for system default.
        """
        storage_cfg: dict = config.get("storage", {})
        output_dir = storage_cfg.get("output_dir")
        if output_dir:
            tmp_base = Path(output_dir) / "tmp" / "stage_07"
            tmp_base.mkdir(parents=True, exist_ok=True)
            return tmp_base
        return None

    @staticmethod
    def _get_model(
        config: dict,
        context: PipelineContext,
        logger: Any,
        batch_size: int,
        vshift: int,
    ) -> Any:
        stage_cfg: dict = config.get("stage_07", {})
        checkpoint = Path(stage_cfg.get(
            "checkpoint", "models/syncnet/syncnet_v2.model"
        ))

        # Try cached model first
        try:
            return context.model_manager.get_model("syncnet")
        except (KeyError, AttributeError):
            pass

        # Not cached — load via ModelManager
        logger.info(f"[Stage 7] Loading SyncNet via ModelManager: checkpoint={checkpoint}")
        try:
            return context.model_manager.load_model(
                "syncnet", str(checkpoint),
                batch_size=batch_size, vshift=vshift,
            )
        except Exception as exc:
            logger.error(f"[Stage 7] SyncNet loading failed: {exc}")
            raise PipelineError(
                f"[Stage 7] Failed to load SyncNet model via ModelManager. "
                f"Checkpoint: {checkpoint}"
            ) from exc

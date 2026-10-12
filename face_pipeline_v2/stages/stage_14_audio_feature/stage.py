"""Stage 14: Audio Feature Extraction (HuBERT).

Per AISDD 3.20, this stage extracts 768-dim audio embeddings from the
corrected (aligned) clip audio using HuBERT.

Input:  AVAlignmentInfo (from Stage 7.1) — corrected/aligned clip audio.
Output: List[AudioFeature]  (one per video frame, 768-dim)
        + NPZ serialization per clip for Stage 17.

Input MUST be AVAlignmentInfo (never raw audio, AISDD 3.20.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from face_pipeline_v2.data.models import AVAlignmentInfo, AudioFeature
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_14_audio_feature.integration import (
    save_audio_features,
)
from face_pipeline_v2.stages.stage_14_audio_feature.processor import (
    extract_audio_features_for_clip,
    load_hubert_wrapper,
)


class Stage14Audio(BaseStage):
    """Stage 14 — Audio Feature Extraction (HuBERT).

    Input:  AVAlignmentInfo (Stage 7.1 corrected audio).
    Output: List[AudioFeature] + NPZ per clip.
    """

    _STAGE_NAME: str = "Stage 14 Audio Feature Extraction"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, AVAlignmentInfo):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected AVAlignmentInfo, "
                f"got {type(input_data).__name__}"
            )
            return False
        audio_path = input_data.aligned_audio_path
        if audio_path is None:
            context.logger.warning(
                f"[{self._STAGE_NAME}] clip {input_data.clip_id}: "
                f"aligned_audio_path is None (severe offset / unavailable)."
            )
            return False
        if not Path(audio_path).exists():
            context.logger.warning(
                f"[{self._STAGE_NAME}] clip {input_data.clip_id}: "
                f"aligned audio missing: {audio_path}"
            )
            return False
        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[AudioFeature]:
        alignment: AVAlignmentInfo = input_data
        config: dict = context.config
        logger = context.logger

        stage14_cfg: dict = config.get("stage_14", {})
        audio_cfg: dict = config.get("audio", {})
        enabled: bool = stage14_cfg.get("enabled", True)
        if not enabled:
            logger.info(f"[{self._STAGE_NAME}] Disabled by config. Skipping.")
            return []

        # Model path resolution order:
        #   stage_14.model_path → audio.model_path → default.
        model_path: str = (
            stage14_cfg.get("model_path")
            or audio_cfg.get("model_path")
            or "models/hubert/chinese-hubert-base"
        )
        device: str = stage14_cfg.get("device", audio_cfg.get("device", "auto"))
        fps: int = int(stage14_cfg.get("fps", 25))
        output_dir: Path = Path(
            stage14_cfg.get("output_dir", "outputs/stage14_audio")
        )
        sample_rate: int = int(audio_cfg.get("sample_rate", 16000))

        audio_path = Path(alignment.aligned_audio_path)
        clip_id = alignment.clip_id
        start_time = float(alignment.clip_start or 0.0)

        logger.info(
            f"[{self._STAGE_NAME}] Extracting HuBERT features for clip "
            f"{clip_id} from {audio_path} (fps={fps})"
        )

        wrapper = load_hubert_wrapper(model_path=Path(model_path), device=device)

        try:
            features = extract_audio_features_for_clip(
                wrapper=wrapper,
                audio_path=audio_path,
                fps=fps,
                start_time=start_time,
            )
        finally:
            # No explicit teardown needed for the transformers model.
            pass

        logger.info(
            f"[{self._STAGE_NAME}] Extracted {len(features)} audio features "
            f"(dim={features[0].dimension if features else 'N/A'}) for clip {clip_id}"
        )

        # Save NPZ for Stage 17
        if features:
            npz_path = save_audio_features(features, output_dir, clip_id)
            logger.info(f"[{self._STAGE_NAME}] Saved audio features to {npz_path}")

        return features

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[AudioFeature], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Empty output (clip skipped)."
            )
            return True
        ok = all(isinstance(f, AudioFeature) for f in output_data)
        if ok:
            dims = {f.dimension for f in output_data}
            ok = len(dims) == 1
            if not ok:
                context.logger.warning(
                    f"[{self._STAGE_NAME}] Mixed feature dimensions: {dims}"
                )
        return ok

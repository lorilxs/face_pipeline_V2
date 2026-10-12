"""Stage 9.5: Clip Export — Final Training Data Asset Generation.

Exports Stage-9-accepted clips as self-contained video+audio assets
ready for downstream FLAME/Gaze/Audio-Feature extraction.

Input (dict):
    {
        "quality_metrics": List[QualityMetrics],       # from Stage 9
        "clip_infos": List[ClipInfo],                  # from Stage 4
        "std_video_info": StandardizedVideoInfo,       # from Stage 1
        "av_alignments": List[AVAlignmentInfo],        # from Stage 7.1
        "av_sync_infos": List[AVSyncInfo],             # optional, from Stage 7
    }

Output: Dict with export summary + manifest written to disk.
"""

from pathlib import Path
from typing import Any, Dict, List

from face_pipeline_v2.data.models import (
    AVAlignmentInfo,
    AVSyncInfo,
    ClipInfo,
    QualityMetrics,
    StandardizedVideoInfo,
)
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_09_5_clip_export.exporter import (
    export_accepted_clips,
)


class Stage9_5ClipExport(BaseStage):
    """Stage 9.5 — Export accepted clips as final training data assets.

    Video: ffmpeg cut from Stage 1 standardized video.
    Audio: Stage 7.1 per-clip corrected audio (offset-aligned).
    Only clips with QualityMetrics.is_accepted=True are exported.

    Input:  dict with quality_metrics, clip_infos, std_video_info,
            av_alignments, av_sync_infos (optional)
    Output: dict with export summary
    """

    _STAGE_NAME: str = "Stage 9.5 Clip Export"

    def __init__(self) -> None:
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(input_data, dict):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected dict, got "
                f"{type(input_data).__name__}"
            )
            return False

        required = {"quality_metrics", "clip_infos", "std_video_info", "av_alignments"}
        if not required.issubset(input_data.keys()):
            missing = required - input_data.keys()
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required keys: {missing}"
            )
            return False

        if not isinstance(input_data["quality_metrics"], list):
            return False
        if not isinstance(input_data["clip_infos"], list):
            return False
        if not isinstance(input_data["std_video_info"], StandardizedVideoInfo):
            return False

        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> Dict[str, Any]:
        """Export accepted clips.

        Args:
            input_data: dict with quality_metrics, clip_infos,
                        std_video_info, av_alignments, av_sync_infos.
            context: PipelineContext.

        Returns:
            Dict with export summary.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data

        metrics: List[QualityMetrics] = data["quality_metrics"]
        clip_infos: List[ClipInfo] = data["clip_infos"]
        std_info: StandardizedVideoInfo = data["std_video_info"]
        av_alignments: List[AVAlignmentInfo] = data["av_alignments"]
        av_sync_infos: List[AVSyncInfo] = data.get("av_sync_infos", [])

        # --- Filter accepted only ---
        accepted = [m for m in metrics if m.is_accepted]
        if not accepted:
            logger.info(
                f"[{self._STAGE_NAME}] No accepted clips. Nothing to export."
            )
            return {
                "accepted_count": 0, "exported_videos": 0,
                "exported_audio": 0, "manifest_entries": 0, "failures": [],
            }

        logger.info(
            f"[{self._STAGE_NAME}] Exporting {len(accepted)}/{len(metrics)} "
            f"accepted clips"
        )

        # --- Build lookups ---
        clip_by_id = {c.clip_id: c for c in clip_infos}
        align_by_id = {a.clip_id: a for a in av_alignments}
        sync_by_id = {s.clip_id: s for s in av_sync_infos}

        # --- Output directories ---
        storage_cfg = config.get("storage", {})
        output_base = Path(storage_cfg.get("output_dir", "outputs"))
        stage_cfg = config.get("stage_09_5", {}) or {}
        clips_dir = output_base / stage_cfg.get("clips_dir", "clips")
        # Final canonical audio dir — holds the aligned clip audio only
        # (raw clip audio stays as Stage 7.1 intermediate under
        # stage_outputs/av_alignment/<clip_id>/ and is never exported).
        audio_dir = output_base / stage_cfg.get("audio_dir", "audio")
        manifest_path = output_base / stage_cfg.get("manifest", "manifest.json")

        # --- Stage 7.1 → Stage 9.5 audio lifecycle management ---
        # audio_alignment.cleanup_after_export (default False):
        #   true  → after a clip's video + aligned audio both exported,
        #           remove its Stage 7.1 intermediate dir
        #           (stage_outputs/av_alignment/<clip_id>/);
        #   false → keep Stage 7.1 intermediate audio (default, safest).
        audio_alignment_cfg = config.get("audio_alignment", {}) or {}
        cleanup_after_export = bool(
            audio_alignment_cfg.get("cleanup_after_export", False)
        )
        if cleanup_after_export:
            logger.info(
                f"[{self._STAGE_NAME}] cleanup_after_export=True → Stage 7.1 "
                f"intermediate audio removed after successful export "
                f"(final clips/ + audio/ are protected)"
            )

        # --- Export ---
        result = export_accepted_clips(
            accepted_metrics=accepted,
            clip_by_id=clip_by_id,
            align_by_id=align_by_id,
            std_video_path=std_info.video_path,
            clips_dir=clips_dir,
            audio_dir=audio_dir,
            manifest_path=manifest_path,
            sync_by_id=sync_by_id,
            logger=logger,
            cleanup_intermediate=cleanup_after_export,
        )

        logger.info(
            f"[{self._STAGE_NAME}] Export complete: "
            f"{result['exported_videos']} videos, "
            f"{result['exported_audio']} audio, "
            f"{result['manifest_entries']} manifest entries, "
            f"{len(result['failures'])} failures"
        )
        c = result.get("cleanup") or {}
        if c.get("cleanup_after_export"):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 7.1 intermediate cleanup: "
                f"found={c.get('intermediate_found', 0)} "
                f"fully_exported={c.get('fully_exported', 0)} "
                f"removed={c.get('removed', 0)} "
                f"kept={len(c.get('kept', []))}"
            )
        if result.get("validation"):
            v = result["validation"]
            logger.info(
                f"[{self._STAGE_NAME}] Audio Alignment Validation: "
                f"{v['passed']}/{v['total_clips']} passed, "
                f"{v['failed']} failed, {v['skipped']} skipped "
                f"(tolerance={v['tolerance_ms']:.0f}ms)"
            )
        if result["failures"]:
            for f in result["failures"]:
                logger.warning(f"[{self._STAGE_NAME}] {f}")

        return result

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        if not isinstance(output_data, dict):
            return False
        required_keys = {"accepted_count", "exported_videos",
                         "exported_audio", "manifest_entries"}
        return required_keys.issubset(output_data.keys())

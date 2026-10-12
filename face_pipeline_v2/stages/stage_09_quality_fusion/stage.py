"""Stage 9: Quality Metrics Fusion & Hard Filtering.

Per AISDD 3.13, this stage fuses results from Stage 6 (Pose),
Stage 7 (AV Sync), and Stage 8 (Face Quality) into a unified
QualityMetrics per clip.

Hard Filter + Fusion Score.

Input:
    {
        "pose_sequences": List[PoseSequence],
        "av_sync_infos": List[AVSyncInfo],
        "quality_sequences": List[FaceQualitySequence],
        "video_id": str,           # optional; fallback: context.runtime_info
    }

Output: List[QualityMetrics]
"""

from typing import Any, Dict, List

from face_pipeline_v2.data.models import (
    AVSyncInfo,
    FaceQualitySequence,
    PoseSequence,
    QualityMetrics,
)
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_09_quality_fusion.fusion import (
    FusionThresholds,
    fuse_quality,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    PipelineError,
)


class Stage9QualityFusion(BaseStage):
    """Stage 9 — Quality Metrics Fusion.

    Pure rule-based logic (no ML model).

    Input:  dict with pose_sequences, av_sync_infos, quality_sequences
    Output: List[QualityMetrics]
    """

    _STAGE_NAME: str = "Stage 9 Quality Fusion"

    def __init__(self) -> None:
        """Initialize Stage 9."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate input is a dict with the three required sequence lists.

        Args:
            input_data: Expected dict{pose_sequences, av_sync_infos, quality_sequences}.
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

        required = {"pose_sequences", "av_sync_infos", "quality_sequences"}
        if not required.issubset(input_data.keys()):
            missing = required - input_data.keys()
            context.logger.warning(
                f"[{self._STAGE_NAME}] Missing required keys: {missing}. "
                f"Need: {required}"
            )
            return False

        pose_seqs = input_data["pose_sequences"]
        if not isinstance(pose_seqs, list) or len(pose_seqs) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] pose_sequences must be a non-empty list."
            )
            return False
        if not all(isinstance(s, PoseSequence) for s in pose_seqs):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in pose_sequences "
                f"must be PoseSequence."
            )
            return False

        av_infos = input_data["av_sync_infos"]
        if not isinstance(av_infos, list) or len(av_infos) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] av_sync_infos must be a non-empty list."
            )
            return False
        if not all(isinstance(a, AVSyncInfo) for a in av_infos):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in av_sync_infos "
                f"must be AVSyncInfo."
            )
            return False

        qual_seqs = input_data["quality_sequences"]
        if not isinstance(qual_seqs, list) or len(qual_seqs) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] quality_sequences must be a non-empty list."
            )
            return False
        if not all(isinstance(q, FaceQualitySequence) for q in qual_seqs):
            context.logger.warning(
                f"[{self._STAGE_NAME}] All elements in quality_sequences "
                f"must be FaceQualitySequence."
            )
            return False

        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[QualityMetrics]:
        """Execute quality fusion on all clips.

        Steps:
            1. Read config thresholds (pose/sync/quality top-level keys
               + stage_09 section).
            2. Resolve video_id.
            3. Build clip_id lookup dicts for AV sync and quality.
            4. For each PoseSequence, align with matching AVSyncInfo
               and FaceQualitySequence, then fuse.
            5. Return list of QualityMetrics.

        Args:
            input_data: dict with pose_sequences, av_sync_infos,
                        quality_sequences, and optionally video_id.
            context: PipelineContext.

        Returns:
            List[QualityMetrics], one per successfully aligned clip.

        Raises:
            PipelineError: If video_id cannot be resolved.
        """
        logger = context.logger
        config = context.config
        data: dict = input_data

        pose_sequences: List[PoseSequence] = data["pose_sequences"]
        av_sync_infos: List[AVSyncInfo] = data["av_sync_infos"]
        quality_sequences: List[FaceQualitySequence] = data["quality_sequences"]

        # --- Load stage configuration ---
        stage_cfg: dict = config.get("stage_09", {})
        if not stage_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 9 disabled. "
                f"Returning empty list."
            )
            return []

        # --- Resolve video_id ---
        video_id = self._resolve_video_id(data, context, logger)

        # --- Build FusionThresholds ---
        t = self._build_thresholds(config, stage_cfg)

        logger.info(
            f"[{self._STAGE_NAME}] Config: "
            f"yaw_th={t.yaw_threshold}, pitch_th={t.pitch_threshold}, "
            f"roll_th={t.roll_threshold}, "
            f"sync_offset_th={t.max_offset_frames}frames, "
            f"sync_lse_c_th={t.min_lse_c}, "
            f"quality_th={t.quality_threshold}, "
            f"min_pose_ratio={t.min_pose_ratio}, "
            f"fusion_th={t.fusion_threshold}, weights={t.weights}"
        )

        # --- Build lookup dicts ---
        av_by_id: Dict[str, AVSyncInfo] = {
            a.clip_id: a for a in av_sync_infos
        }
        qual_by_id: Dict[str, FaceQualitySequence] = {
            q.clip_id: q for q in quality_sequences
        }

        # --- Fuse each clip ---
        output: List[QualityMetrics] = []
        accepted_count = 0
        skipped_count = 0

        for pose_seq in pose_sequences:
            clip_id = pose_seq.clip_id

            av_sync = av_by_id.get(clip_id)
            if av_sync is None:
                logger.warning(
                    f"[{self._STAGE_NAME}] No AVSyncInfo for clip "
                    f"'{clip_id}'. Skipping."
                )
                skipped_count += 1
                continue

            quality_seq = qual_by_id.get(clip_id)
            if quality_seq is None:
                logger.warning(
                    f"[{self._STAGE_NAME}] No FaceQualitySequence for clip "
                    f"'{clip_id}'. Skipping."
                )
                skipped_count += 1
                continue

            metrics = fuse_quality(pose_seq, av_sync, quality_seq, t, video_id)
            output.append(metrics)

            if metrics.is_accepted:
                accepted_count += 1
                logger.info(
                    f"[{self._STAGE_NAME}] {clip_id}: ACCEPTED "
                    f"| pose={metrics.pose_score:.4f} (min={t.min_pose_ratio}) "
                    f"| offset={av_sync.offset} (max={t.max_offset_frames}frames) "
                    f"| lse_c={av_sync.sync_lse_c:.2f} (min={t.min_lse_c}) "
                    f"| quality={metrics.quality_score:.4f} (th={t.quality_threshold}) "
                    f"| fusion={metrics.fusion_score:.4f}"
                )
            else:
                # Detailed per-dimension reject logging for threshold tuning
                reject_detail = self._format_reject_detail(
                    clip_id, metrics, av_sync, t
                )
                logger.warning(reject_detail)

        logger.info(
            f"[{self._STAGE_NAME}] Fusion complete: "
            f"{len(output)} clips processed, {accepted_count} accepted, "
            f"{skipped_count} skipped (misaligned), "
            f"{len(output) - accepted_count} rejected"
        )

        return output

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate output is a list of QualityMetrics.

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a valid QualityMetrics list.
        """
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[QualityMetrics], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Output is empty: no clips fused."
            )
        return all(isinstance(m, QualityMetrics) for m in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_reject_detail(
        clip_id: str,
        metrics: QualityMetrics,
        av_sync: Any,
        t: Any,
    ) -> str:
        """Format detailed per-dimension reject reason with metric vs threshold.

        Sync gate uses raw AVSyncInfo values (offset + sync_lse_c),
        NOT the normalized metrics.sync_score.

        Args:
            clip_id: Clip identifier.
            metrics: Computed QualityMetrics for this clip.
            av_sync: AVSyncInfo (raw offset and sync_lse_c).
            t: FusionThresholds with configured thresholds.

        Returns:
            Multi-line formatted reject detail string.
        """
        lines = [f"[Stage 9] {clip_id}: REJECTED — per-dimension breakdown:"]

        # Pose dimension
        pose_ok = metrics.pose_score >= t.min_pose_ratio
        lines.append(
            f"  {'PASS' if pose_ok else 'FAIL'} | stage=pose "
            f"| metric=pose_ratio | value={metrics.pose_score:.4f} "
            f"| threshold={t.min_pose_ratio}"
        )

        # Sync dimension — raw offset + LSE-C (not normalized sync_score)
        offset_ok = abs(av_sync.offset) <= t.max_offset_frames
        lse_c_ok = av_sync.sync_lse_c >= t.min_lse_c
        sync_ok = offset_ok and lse_c_ok
        if not offset_ok:
            lines.append(
                f"  FAIL | stage=sync | reason=sync_offset_exceeded "
                f"| value=|offset|={abs(av_sync.offset)} frames "
                f"| threshold=<={t.max_offset_frames} frames"
            )
        if not lse_c_ok:
            lines.append(
                f"  FAIL | stage=sync | reason=sync_lse_c_too_low "
                f"| value=lse_c={av_sync.sync_lse_c:.2f} "
                f"| threshold=>={t.min_lse_c}"
            )
        if sync_ok:
            lines.append(
                f"  PASS | stage=sync "
                f"| offset={av_sync.offset} (max={t.max_offset_frames}) "
                f"| lse_c={av_sync.sync_lse_c:.2f} (min={t.min_lse_c})"
            )

        # Quality dimension
        quality_ok = metrics.quality_score >= t.quality_threshold
        lines.append(
            f"  {'PASS' if quality_ok else 'FAIL'} | stage=quality "
            f"| metric=hyperiqa | value={metrics.quality_score:.4f} "
            f"| threshold={t.quality_threshold}"
        )

        # Fusion score (informational)
        if t.fusion_threshold is not None:
            fusion_ok = metrics.fusion_score >= t.fusion_threshold
            lines.append(
                f"  {'PASS' if fusion_ok else 'FAIL'} | stage=fusion "
                f"| value={metrics.fusion_score:.4f} "
                f"| threshold={t.fusion_threshold}"
            )
        else:
            lines.append(
                f"  INFO | stage=fusion | value={metrics.fusion_score:.4f} "
                f"| threshold=disabled"
            )

        return "\n".join(lines)

    @staticmethod
    def _resolve_video_id(
        data: dict,
        context: PipelineContext,
        logger: Any,
    ) -> str:
        """Resolve video_id from input dict or context.

        Priority:
            1. ``data["video_id"]``
            2. ``context.runtime_info["video_id"]``
            3. Raise PipelineError.

        Args:
            data: Input dict.
            context: PipelineContext.
            logger: PipelineLogger.

        Returns:
            Resolved video_id string.

        Raises:
            PipelineError: If video_id cannot be resolved.
        """
        vid = data.get("video_id")
        if vid:
            return str(vid)

        try:
            vid = context.runtime_info.get("video_id")
        except AttributeError:
            vid = None

        if vid:
            logger.info(
                f"[Stage 9] Using video_id from context.runtime_info: '{vid}'"
            )
            return str(vid)

        raise PipelineError(
            "[Stage 9] Cannot resolve video_id. "
            "Pass 'video_id' in the input dict or set "
            "context.runtime_info['video_id']."
        )

    @staticmethod
    def _build_thresholds(
        config: dict,
        stage_cfg: dict,
    ) -> FusionThresholds:
        """Build FusionThresholds from config.

        Reads per-dimension thresholds from existing top-level keys
        (pose / sync / quality) and fusion parameters from stage_09.

        Args:
            config: Full pipeline config dict.
            stage_cfg: stage_09 config block.

        Returns:
            Populated FusionThresholds.
        """
        pose_cfg: dict = config.get("pose", {})
        quality_cfg: dict = config.get("quality", {})
        # Sync hard filter uses raw offset + LSE-C from quality.sync config
        sync_cfg: dict = quality_cfg.get("sync", {})

        # Fusion weights
        weights_cfg: dict = stage_cfg.get("fusion_weights", {})
        weights = (
            float(weights_cfg.get("pose", 0.4)),
            float(weights_cfg.get("sync", 0.3)),
            float(weights_cfg.get("quality", 0.3)),
        )

        # fusion_threshold: None means disabled
        fusion_th = stage_cfg.get("fusion_threshold")
        if fusion_th is not None:
            fusion_th = float(fusion_th)

        return FusionThresholds(
            yaw_threshold=float(pose_cfg.get("yaw_threshold", 30.0)),
            pitch_threshold=float(pose_cfg.get("pitch_threshold", 20.0)),
            roll_threshold=float(pose_cfg.get("roll_threshold", 30.0)),
            max_offset_frames=int(sync_cfg.get("max_offset_frames", 8)),
            min_lse_c=float(sync_cfg.get("min_lse_c", 5.0)),
            quality_threshold=float(quality_cfg.get("quality_threshold", 0.5)),
            min_pose_ratio=float(stage_cfg.get("min_pose_ratio", 0.9)),
            min_pose_confidence=float(stage_cfg.get("min_pose_confidence", 0.0)),
            fusion_threshold=fusion_th,
            weights=weights,
        )

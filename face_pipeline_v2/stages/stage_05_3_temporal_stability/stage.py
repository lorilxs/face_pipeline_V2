"""Stage 5.3: Temporal Stability Enhancement.

Per AISDD 3.9.3, this stage analyzes frame-to-frame face motion and
scale changes to filter out clips with unstable face trajectories.

Input:  List[FaceSequence] (post-5.2; independent fallback if 5.2 disabled)
Output: List[FaceSequence] (only sequences passing temporal stability;
        rejected sequences are dropped)
"""

from typing import Any, List

from face_pipeline_v2.data.models import FaceSequence, TemporalStabilityResult
from face_pipeline_v2.pipeline.enhancement_stage import EnhancementStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_3_temporal_stability.temporal_stability import (
    analyze_stability,
)
from face_pipeline_v2.utils.exceptions import InputValidationError


class Stage5_3TemporalStability(EnhancementStage):
    """Stage 5.3 — Temporal Stability Hard Filter.

    Input:  List[FaceSequence]
    Output: List[FaceSequence] (only sequences passing temporal stability)
    """

    _STAGE_NAME: str = "Stage 5.3 Temporal Stability"
    enhancement_key: str = "stage_05_3_temporal_stability"

    def __init__(self) -> None:
        """Initialize Stage 5.3."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a non-empty List[FaceSequence].

        Args:
            input_data: Expected List[FaceSequence].
            context: PipelineContext.

        Returns:
            True if input is valid.
        """
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
    ) -> List[FaceSequence]:
        """Analyze temporal stability of FaceSequences (HARD FILTER).

        Args:
            input_data: List[FaceSequence].
            context: PipelineContext.

        Returns:
            List[FaceSequence] containing only sequences that passed the
            temporal stability check (hard filter; rejected dropped).
        """
        logger = context.logger
        config = context.config
        sequences: List[FaceSequence] = input_data

        # --- Load thresholds ---
        enhancements_cfg: dict = config.get("enhancements", {})
        stability_cfg: dict = enhancements_cfg.get(
            "stage_05_3_temporal_stability", {}
        )
        max_center_motion: float = float(
            stability_cfg.get("max_center_motion", 15.0)
        )
        max_scale_change_ratio: float = float(
            stability_cfg.get("max_scale_change_ratio", 0.25)
        )
        max_stability_anomaly_ratio: float = float(
            stability_cfg.get("max_stability_anomaly_ratio", 0.05)
        )

        logger.info(
            f"[{self._STAGE_NAME}] Config: "
            f"max_center_motion={max_center_motion}px, "
            f"max_scale_change_ratio={max_scale_change_ratio}, "
            f"max_stability_anomaly_ratio={max_stability_anomaly_ratio}"
        )

        # --- Analyze each sequence (HARD FILTER) ---
        # Only sequences whose stability_result.passed is True continue downstream.
        # Note: this stage is independent of Stage 5.2 — analyze_stability() falls
        # back to all-bboxes logic when identity_result is absent.
        output_sequences: List[FaceSequence] = []
        input_count = 0
        passed_count = 0
        rejected_count = 0
        reject_reason_counts: dict[str, int] = {}

        for seq in sequences:
            input_count += 1
            result_seq = analyze_stability(
                sequence=seq,
                max_center_motion=max_center_motion,
                max_scale_change_ratio=max_scale_change_ratio,
                max_stability_anomaly_ratio=max_stability_anomaly_ratio,
            )

            sr = result_seq.stability_result
            if sr is not None and sr.passed:
                output_sequences.append(result_seq)
                passed_count += 1
            else:
                rejected_count += 1
                reasons = tuple(sr.reject_reasons) if sr else ("no_result",)
                for r in reasons:
                    reject_reason_counts[r] = reject_reason_counts.get(r, 0) + 1
                # --- pipeline_manifest audit (non-fatal, best-effort) ---
                try:
                    context.add_stage_audit(
                        self.enhancement_key,
                        seq.clip_id,
                        {
                            "reasons": list(reasons),
                            "bbox_jump_count": (
                                sr.bbox_jump_count if sr else None
                            ),
                            "average_center_motion": (
                                round(sr.average_center_motion, 6) if sr else None
                            ),
                            "max_center_motion": (
                                round(sr.max_center_motion, 6) if sr else None
                            ),
                            "average_scale_change": (
                                round(sr.average_scale_change, 6) if sr else None
                            ),
                            "max_scale_change": (
                                round(sr.max_scale_change, 6) if sr else None
                            ),
                            "anomaly_ratio": (
                                round(sr.anomaly_ratio, 6) if sr else None
                            ),
                        },
                    )
                except Exception:  # noqa: BLE001 - audit must never break filtering
                    logger.warning(
                        f"[{self._STAGE_NAME}] Failed to record audit entry for "
                        f"{seq.clip_id} (non-fatal)"
                    )
                logger.debug(
                    f"[{self._STAGE_NAME}] {seq.clip_id} rejected: "
                    f"jumps={sr.bbox_jump_count if sr else 0}, "
                    f"anomaly_ratio={(sr.anomaly_ratio if sr else 0.0):.4f}, "
                    f"reasons={reasons}"
                )

        logger.info(
            f"[{self._STAGE_NAME}] Stability hard filter complete: "
            f"{input_count} input, "
            f"{passed_count} passed, "
            f"{rejected_count} rejected, "
            f"reject_reason_counts={dict(reject_reason_counts)}"
        )

        return output_sequences

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a List[FaceSequence].

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is valid.
        """
        if not isinstance(output_data, list):
            return False
        if len(output_data) == 0:
            context.logger.info(f"[{self._STAGE_NAME}] Output is empty.")
        return all(isinstance(s, FaceSequence) for s in output_data)

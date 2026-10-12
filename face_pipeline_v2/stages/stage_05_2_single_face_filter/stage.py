"""Stage 5.2: Single Face Identity Consistency Filtering.

Per AISDD 3.9.2, this stage verifies that a clip's single-face frames
all belong to the same identity (track_id from Stage 5.1). This ensures
downstream FLAME/Gaze/Audio processing operates on a consistent subject.

Rules (AISDD 3.9.2):
    - Face Count: reject if too many multi-face or empty frames.
    - Track Continuity: reject if single-face frames belong to different
      track_ids (identity switch).

Input:  List[FaceSequence] (from Stage 5 or Stage 5.1)
Output: List[FaceSequence] (only sequences passing single-face identity
        consistency; rejected sequences are dropped)
"""

from typing import Any, List

from face_pipeline_v2.data.models import FaceSequence, SingleFaceIdentityResult
from face_pipeline_v2.pipeline.enhancement_stage import EnhancementStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_2_single_face_filter.single_face_filter import (
    filter_sequence,
)
from face_pipeline_v2.utils.exceptions import InputValidationError


class Stage5_2SingleFaceFilter(EnhancementStage):
    """Stage 5.2 — Single Face Identity Consistency Filtering.

    Input:  List[FaceSequence]
    Output: List[FaceSequence]
    """

    _STAGE_NAME: str = "Stage 5.2 Single Face Filter"
    enhancement_key: str = "stage_05_2_single_face_filter"

    def __init__(self) -> None:
        """Initialize Stage 5.2."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a non-empty List[FaceSequence].

        Args:
            input_data: Expected List[FaceSequence] from Stage 5/5.1.
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
                f"[{self._STAGE_NAME}] Empty sequence list: nothing to filter."
            )
            return False
        return all(isinstance(s, FaceSequence) for s in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[FaceSequence]:
        """Filter FaceSequences by single-face identity consistency (HARD FILTER).

        Steps:
            1. Check enhancements.stage_05_2_single_face_filter.enabled.
            2. If disabled, pass-through unchanged.
            3. Load thresholds from config.
            4. For each FaceSequence: evaluate identity consistency -> identity_result.
            5. Hard filter: append only sequences with identity_result.passed=True;
               rejected sequences are dropped (they do not reach Stage 6/7/8).

        Args:
            input_data: List[FaceSequence] from Stage 5/5.1.
            context: PipelineContext.

        Returns:
            List[FaceSequence] containing only sequences that passed the
            single-face identity consistency check.
        """
        logger = context.logger
        config = context.config
        sequences: List[FaceSequence] = input_data

        # --- Load thresholds ---
        enhancements_cfg: dict = config.get("enhancements", {})
        filter_cfg: dict = enhancements_cfg.get(
            "stage_05_2_single_face_filter", {}
        )
        max_identity_anomaly_ratio: float = float(
            filter_cfg.get("max_identity_anomaly_ratio", 0.005)
        )

        logger.info(
            f"[{self._STAGE_NAME}] Config: "
            f"max_identity_anomaly_ratio={max_identity_anomaly_ratio}"
        )

        # --- Filter each sequence (HARD FILTER) ---
        # Only sequences whose identity_result.passed is True continue downstream.
        output_sequences: List[FaceSequence] = []
        input_count = 0
        passed_count = 0
        rejected_count = 0
        reject_reason_counts: dict[str, int] = {}

        for seq in sequences:
            input_count += 1
            filtered = filter_sequence(
                sequence=seq,
                max_identity_anomaly_ratio=max_identity_anomaly_ratio,
            )

            if filtered.identity_result is not None and filtered.identity_result.passed:
                output_sequences.append(filtered)
                passed_count += 1
            else:
                rejected_count += 1
                reasons = (
                    filtered.identity_result.reject_reasons
                    if filtered.identity_result
                    else ()
                )
                for r in reasons:
                    reject_reason_counts[r] = reject_reason_counts.get(r, 0) + 1
                # --- pipeline_manifest audit (non-fatal, best-effort) ---
                # Record the per-clip reject reason in PipelineContext.stage_audit
                # so the manifest survives intermediate-output deletion.
                idr = filtered.identity_result
                try:
                    context.add_stage_audit(
                        self.enhancement_key,
                        seq.clip_id,
                        {
                            "reasons": list(reasons),
                            "main_track_id": idr.main_track_id if idr else None,
                            "total_frames": idr.total_frames if idr else None,
                            "anomaly_frames": idr.anomaly_frames if idr else None,
                            "anomaly_ratio": (
                                round(idr.anomaly_ratio, 6) if idr else None
                            ),
                        },
                    )
                except Exception:  # noqa: BLE001 - audit must never break filtering
                    logger.warning(
                        f"[{self._STAGE_NAME}] Failed to record audit entry for "
                        f"{seq.clip_id} (non-fatal)"
                    )
                if filtered.identity_result is not None:
                    logger.debug(
                        f"[{self._STAGE_NAME}] {seq.clip_id} rejected: "
                        f"anomaly_ratio={filtered.identity_result.anomaly_ratio:.4f} "
                        f"({filtered.identity_result.anomaly_frames}/"
                        f"{filtered.identity_result.total_frames}), "
                        f"reasons={reasons}"
                    )
                else:
                    logger.debug(
                        f"[{self._STAGE_NAME}] {seq.clip_id} rejected: "
                        f"no identity_result, reasons={reasons}"
                    )

        logger.info(
            f"[{self._STAGE_NAME}] Hard filter complete: "
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
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty."
            )
        return all(isinstance(s, FaceSequence) for s in output_data)

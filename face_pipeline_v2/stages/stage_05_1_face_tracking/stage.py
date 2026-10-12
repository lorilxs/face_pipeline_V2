"""Stage 5.1: Face Tracking Enhancement.

Per AISDD 3.9.1, this stage performs ByteTrack-based cross-frame face
identity association on Stage 5 output for open-domain (multi-person) videos.

Input:  List[FaceSequence] (from Stage 5)
Output: List[FaceSequence] (with track_id populated on each FaceInfo)
"""

from pathlib import Path
from typing import Any, List

from face_pipeline_v2.data.models import FaceInfo, FaceSequence, FrameInfo
from face_pipeline_v2.pipeline.enhancement_stage import EnhancementStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_1_face_tracking.tracker import (
    track_sequence,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
    PipelineError,
)


class Stage5_1FaceTracking(EnhancementStage):
    """Stage 5.1 — Face Tracking Enhancement (ByteTrack).

    Input:  List[FaceSequence]
    Output: List[FaceSequence]
    """

    _STAGE_NAME: str = "Stage 5.1 Face Tracking"
    enhancement_key: str = "stage_05_1_face_tracking"

    def __init__(self) -> None:
        """Initialize Stage 5.1."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a non-empty List[FaceSequence].

        Args:
            input_data: Expected List[FaceSequence] from Stage 5.
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
                f"[{self._STAGE_NAME}] Empty sequence list: nothing to track."
            )
            return False
        return all(isinstance(s, FaceSequence) for s in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[FaceSequence]:
        """Execute ByteTrack face tracking on all sequences.

        Steps (AISDD 3.9.1):
            1. Load stage_05_1 config.
            2. If disabled, pass-through unchanged.
            3. Get ByteTrack model from ModelManager or load directly.
            4. For each FaceSequence:
                a. Create a fresh tracker instance.
                b. Iterate frames: extract bboxes → tracker.update() → track_ids.
                c. Rebuild FaceInfo / FrameInfo with track_id.
                d. Collect into new FaceSequence.
            5. Return List[FaceSequence].

        Args:
            input_data: List[FaceSequence] from Stage 5.
            context: PipelineContext.

        Returns:
            List[FaceSequence] with track_id populated.

        Raises:
            PipelineError: If tracker loading fails.
        """
        logger = context.logger
        config = context.config
        sequences: List[FaceSequence] = input_data

        # --- Load tracker parameters ---
        stage_cfg: dict = config.get("stage_05_1", {})
        tracker_cfg: dict = stage_cfg.get("tracker", {})
        # Detection floor shared with Stage 3/5 (single source of truth).
        # Both ByteTrack gates default to this floor so that ANY face Stage 5
        # detected (score >= floor) is a "high-score" detection that can
        # INITIALISE a new track — not just extend an existing one.  Explicit
        # per-stage overrides are allowed via stage_05_1.tracker.
        detection_floor: float = float(
            config.get("stage_03", {}).get("detection_confidence_threshold", 0.5)
        )
        track_thresh: float = float(
            tracker_cfg.get("track_thresh", detection_floor)
        )
        track_buffer: int = int(tracker_cfg.get("track_buffer", 30))
        match_thresh: float = float(tracker_cfg.get("match_thresh", 0.8))
        det_thresh: float = float(tracker_cfg.get("det_thresh", detection_floor))

        # Resolve image shape
        video_cfg: dict = config.get("video", {})
        img_h: int = int(video_cfg.get("resolution", [640, 360])[1])
        img_w: int = int(video_cfg.get("resolution", [640, 360])[0])
        img_shape = (img_h, img_w)

        logger.info(
            f"[{self._STAGE_NAME}] Config: track_thresh={track_thresh}, "
            f"det_thresh={det_thresh}, track_buffer={track_buffer}, "
            f"match_thresh={match_thresh}, img_shape={img_shape}"
        )

        # --- Get ByteTrack model ---
        tracker_model = self._get_tracker(
            config,
            context,
            logger,
            track_thresh,
            track_buffer,
            match_thresh,
            det_thresh,
        )

        # --- Process each sequence ---
        output_sequences: List[FaceSequence] = []
        total_tracks = 0

        for seq in sequences:
            # Per AISDD 3.9.1 step 4a: each FaceSequence gets a fresh
            # tracker.  Without this, track IDs grow monotonically across
            # clips and the previous clip's lost/Kalman state leaks into
            # the next clip (clips must be tracked independently, not as
            # one continuous video stream).
            tracker_model.reset()

            tracked_seq = track_sequence(seq, tracker_model, img_shape)

            # Count unique track IDs
            track_ids_in_seq = set()
            for fi in tracked_seq.frames:
                for face in fi.faces:
                    if face.track_id is not None:
                        track_ids_in_seq.add(face.track_id)
            total_tracks += len(track_ids_in_seq)

            logger.debug(
                f"[{self._STAGE_NAME}] {seq.clip_id}: "
                f"{len(seq.frames)} frames, "
                f"{len(track_ids_in_seq)} unique track(s)"
            )

            output_sequences.append(tracked_seq)

        logger.info(
            f"[{self._STAGE_NAME}] Tracking complete: "
            f"{len(output_sequences)} sequences, "
            f"{total_tracks} unique tracks across all clips"
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tracker(
        config: dict,
        context: PipelineContext,
        logger: Any,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        det_thresh: float,
    ) -> Any:
        """Get or load the ByteTrack model via ModelManager with fallback.

        Args:
            config: Pipeline configuration dict.
            context: PipelineContext with model_manager.
            logger: PipelineLogger instance.
            track_thresh: Detection confidence threshold (high/low split).
            track_buffer: Frames to keep lost tracks alive.
            match_thresh: IoU matching threshold.
            det_thresh: New-track initialisation gate (aligned to the shared
                Stage 3/5 detection floor).

        Returns:
            Loaded ByteTrack tracker.

        Raises:
            PipelineError: If tracker loading fails.
        """
        try:
            return context.model_manager.get_model("bytetrack")
        except (KeyError, AttributeError):
            logger.info(
                f"[Stage 5.1] Loading ByteTrack directly: "
                f"track_thresh={track_thresh}, det_thresh={det_thresh}, "
                f"track_buffer={track_buffer}"
            )
            try:
                from face_pipeline_v2.models.wrappers.bytetrack import (
                    ByteTrackWrapper,
                )

                wrapper = ByteTrackWrapper(
                    track_thresh=track_thresh,
                    track_buffer=track_buffer,
                    match_thresh=match_thresh,
                    det_thresh=det_thresh,
                )
                wrapper.load()
                return wrapper
            except Exception as exc:
                logger.error(
                    f"[Stage 5.1] Failed to load ByteTrack: {exc}"
                )
                raise PipelineError(
                    f"[Stage 5.1] ByteTrack loading failed."
                ) from exc

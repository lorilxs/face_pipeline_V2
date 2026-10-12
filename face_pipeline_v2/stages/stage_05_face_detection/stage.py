"""Stage 5: Full Face Detection & Crop.

Per AISDD 3.8, this stage performs full-frame-rate SCRFD face detection
on each ClipInfo from Stage 4, producing per-frame FaceInfo + FrameInfo.

Performance optimizations (2026-08-12):
    - Frame detection cache: deduplicates SCRFD inference across
      overlapping clips (50% overlap → ~45% fewer inferences).
    - ffmpeg sequential reader: replaces cv2.VideoCapture seek with
      ffmpeg pipe for ~10× faster frame decoding on MP4.

Input:  List[ClipInfo] (from Stage 4)
Output: List[FaceSequence]  (one per clip, each carrying FrameInfo → FaceInfo)
"""

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2

from face_pipeline_v2.data.models import ClipInfo, FaceInfo, FaceSequence, FrameInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.runtime.video_reader import (
    FFmpegVideoReader, create_video_reader,
)
from face_pipeline_v2.stages.stage_05_face_detection.face_detector import (
    FaceDetector,
)
from face_pipeline_v2.stages.stage_05_face_detection.face_resolution_filter import (
    STATUS_FAIL_CROP,
    STATUS_FAIL_ERROR,
    STATUS_FAIL_INVALID_BBOX,
    FaceResolutionFilterConfig,
    FrameResolutionOutcome,
    classify_detected_faces,
    evaluate_video,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    FaceCropError,
    FaceResolutionFilterError,
    InputValidationError,
    InvalidBboxError,
    PipelineError,
    StageExecutionError,
)


class Stage5FaceDetection(BaseStage):
    """Stage 5 — Full Face Detection & Crop.

    Input:  List[ClipInfo]
    Output: List[FaceSequence]
    """

    _STAGE_NAME: str = "Stage 5 Face Detection"

    def __init__(self) -> None:
        """Initialize Stage 5."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a non-empty List[ClipInfo].

        Args:
            input_data: Expected List[ClipInfo] from Stage 4.
            context: PipelineContext.

        Returns:
            True if input is valid.
        """
        if not isinstance(input_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[ClipInfo], "
                f"got {type(input_data).__name__}"
            )
            return False
        if len(input_data) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Empty clip list: nothing to process."
            )
            return False
        return all(isinstance(c, ClipInfo) for c in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[FaceSequence]:
        """Execute full face detection on all clips.

        Steps (AISDD 3.8.4):
            1. Load stage_05 config.
            2. Get SCRFD model from ModelManager.
            3. Locate standardized video.
            4. Build FaceDetector with crop config.
            5. For each ClipInfo:
                a. Seek to start_frame.
                b. For each frame: SCRFD detect → FaceInfo.
                c. Optionally save frame image.
                d. Build FrameInfo.
                e. Collect into per-clip FaceSequence.
            6. Return List[FaceSequence].

        Args:
            input_data: List[ClipInfo] from Stage 4.
            context: PipelineContext.

        Returns:
            List[FaceSequence], one per clip, in chronological order.

        Raises:
            PipelineError: On model loading or detection failure.
        """
        logger = context.logger
        config = context.config
        clips: List[ClipInfo] = input_data

        # --- Load stage configuration ---
        stage_cfg: dict = config.get("stage_05", {})
        if not stage_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 5 is disabled in config. "
                f"Returning empty frame list."
            )
            return []

        selection_cfg: dict = stage_cfg.get("face_selection", {})
        selection_strategy: str = selection_cfg.get("strategy", "largest_area")

        crop_cfg: dict = stage_cfg.get("crop", {})
        crop_enabled: bool = crop_cfg.get("enabled", True)
        crop_size: int = int(crop_cfg.get("size", 256))

        # Bbox expansion on original frame (before crop)
        face_crop_cfg: dict = config.get("stage_05_face_crop", {})
        expand_ratio: float = float(face_crop_cfg.get("expand_ratio", 2.0))

        # Stage 5 face-resolution pre-filter (video-level gate, before 5.1–Stage9).
        resolution_filter = FaceResolutionFilterConfig.from_dict(
            config.get("stage_05_face_resolution_filter", {})
        )

        save_frames: bool = bool(stage_cfg.get("save_frames", False))

        # Resolve fps from config
        stage_04_cfg: dict = config.get("stage_04", {})
        target_fps: float = float(stage_04_cfg.get("target_fps", 25))

        logger.info(
            f"[{self._STAGE_NAME}] Config: selection={selection_strategy}, "
            f"crop={'enabled' if crop_enabled else 'disabled'}"
            f"{f'(expand={expand_ratio})' if crop_enabled else ''}, "
            f"fps={target_fps}"
        )

        # --- Get SCRFD model ---
        scrfd_model = self._get_scrfd_model(config, context, logger)

        # --- Resolve video ---
        video_path = self._resolve_video_path(clips, config, logger)
        video_id = clips[0].video_id

        # --- Build output directories ---
        storage_cfg: dict = config.get("storage", {})
        output_dir = Path(storage_cfg.get("output_dir", "outputs"))
        crops_base_dir: Path | None = None
        if crop_enabled:
            crops_base_dir = output_dir / "faces"

        # --- Build detector ---
        detector = FaceDetector(
            scrfd_model=scrfd_model,
            crop_enabled=crop_enabled,
            crop_size=crop_size,
            crops_base_dir=crops_base_dir,
            selection_strategy=selection_strategy,
            expand_ratio=expand_ratio,
        )

        # --- Determine frame range needed ---
        min_frame = min(c.start_frame for c in clips)
        max_frame = max(c.end_frame for c in clips)
        unique_frames_needed: Set[int] = set()
        for c in clips:
            unique_frames_needed.update(range(c.start_frame, c.end_frame))

        logger.info(
            f"[{self._STAGE_NAME}] Opening video: {video_path} "
            f"(frames {min_frame}–{max_frame}, {len(unique_frames_needed)} unique)"
        )

        # --- Phase 1: sequential read + detect (configurable backend) ---
        start_time = time.monotonic()
        face_detection_cache: Dict[int, List[FaceInfo]] = {}
        # Per-frame final status for the resolution pre-filter (only populated
        # when the filter is enabled; keyed by global frame index).
        frame_outcomes: Dict[int, FrameResolutionOutcome] = {}
        scrfd_inference_count = 0
        cache_hit_count = 0
        total_requests = 0

        # Resolve reader backend from config
        runtime_cfg = config.get("runtime", {})
        reader_backend = runtime_cfg.get("video_reader", "ffmpeg")
        reader = create_video_reader(reader_backend, video_path)

        logger.info(
            f"[{self._STAGE_NAME}] Video reader: backend={reader.backend_name}, "
            f"resolution={reader.width}x{reader.height}"
        )

        # Phase-1 (detection-only pass) writes crops into a temporary bucket,
        # NOT clip_0000: previously every frame in the full video was dumped
        # into clip_0000's crop directory (2000+ files). Phase-2 re-paths each
        # crop into its real per-clip directory via _clone_for_clip.
        placeholder_clip_id = "_detection_tmp"

        # Sequential read from min_frame to max_frame.
        #
        # IMPORTANT: the FFmpeg raw-pipe reader always decodes from the video's
        # frame 0 (it has no seek) and read_next() returns each frame together
        # with its TRUE 0-based index.  We must use that returned index rather
        # than relabeling from min_frame — otherwise, when min_frame > 0, every
        # frame is cached under a shifted (wrong) index and detection is read
        # from the wrong part of the video.  Frames below min_frame are decoded
        # but simply skipped (they are not in unique_frames_needed).
        #
        # OpenCV: read_frame() seeks per-frame, so it can start at min_frame.
        is_ffmpeg = isinstance(reader, FFmpegVideoReader)
        if is_ffmpeg:
            reader.start(frame_count=max_frame)
            frame_idx = 0
        else:
            frame_idx = min_frame

        while frame_idx < max_frame:
            if is_ffmpeg:
                result = reader.read_next()
                if result is None:
                    break
                frame_idx, frame = result
            else:
                frame = reader.read_frame(frame_idx)
                if frame is None:
                    break

            if frame_idx in unique_frames_needed:
                total_requests += 1
                if resolution_filter.enabled:
                    try:
                        face_infos = detector.process_frame(
                            frame_index=frame_idx, image=frame,
                            clip_id=placeholder_clip_id, video_id=video_id,
                        )
                        face_detection_cache[frame_idx] = face_infos
                        frame_outcomes[frame_idx] = classify_detected_faces(
                            face_infos, resolution_filter
                        )
                        scrfd_inference_count += 1
                    except InvalidBboxError:
                        face_detection_cache[frame_idx] = []
                        frame_outcomes[frame_idx] = FrameResolutionOutcome(
                            status=STATUS_FAIL_INVALID_BBOX
                        )
                    except FaceCropError as exc:
                        face_detection_cache[frame_idx] = []
                        frame_outcomes[frame_idx] = FrameResolutionOutcome(
                            status=STATUS_FAIL_CROP
                        )
                        logger.warning(
                            f"[{self._STAGE_NAME}] frame={frame_idx}: {exc}"
                        )
                    except Exception as exc:
                        face_detection_cache[frame_idx] = []
                        frame_outcomes[frame_idx] = FrameResolutionOutcome(
                            status=STATUS_FAIL_ERROR
                        )
                        logger.warning(
                            f"[{self._STAGE_NAME}] frame={frame_idx} "
                            f"processing error: {exc}"
                        )
                else:
                    face_infos = detector.process_frame(
                        frame_index=frame_idx, image=frame,
                        clip_id=placeholder_clip_id, video_id=video_id,
                    )
                    face_detection_cache[frame_idx] = face_infos
                    scrfd_inference_count += 1

            frame_idx += 1

        # Log reader stats
        if is_ffmpeg:
            dt = reader.decode_time()
            tf = reader.total_frames_read()
            fps_val = tf / dt if dt > 0 else 0
            logger.info(
                f"[{self._STAGE_NAME}] Video Reader: backend=ffmpeg "
                f"total_frames={tf} decode_time={dt:.1f}s fps={fps_val:.1f}"
            )

        reader.close()

        # --- Stage 5 face-resolution pre-filter (video-level gate) ---
        if resolution_filter.enabled:
            expected_frames = sorted(unique_frames_needed)
            resolution_result = evaluate_video(
                frame_outcomes, expected_frames, resolution_filter
            )
            _record_resolution_audit(context, video_id, resolution_result)
            logger.info(
                f"[{self._STAGE_NAME}] Face resolution pre-filter: "
                f"passed={resolution_result.passed} "
                f"reason={resolution_result.reject_reason} "
                f"expected={resolution_result.expected_frame_count} "
                f"pass={resolution_result.pass_frame_count} "
                f"valid_face={resolution_result.valid_face_frame_count} "
                f"ratio={resolution_result.qualified_ratio:.4f} "
                f"status_counts={resolution_result.status_counts}"
            )
            if not resolution_result.passed:
                msg = (
                    f"[{self._STAGE_NAME}] Face resolution pre-filter REJECTED "
                    f"video {video_id}: {resolution_result.reject_reason} "
                    f"(expected={resolution_result.expected_frame_count}, "
                    f"pass={resolution_result.pass_frame_count}, "
                    f"ratio={resolution_result.qualified_ratio:.4f})"
                )
                logger.error(msg)
                raise FaceResolutionFilterError(msg, result=resolution_result)

        # --- Phase 2: assemble per-clip FaceSequences from cache ---
        all_sequences: List[FaceSequence] = []
        total_faces_detected = 0

        for clip in clips:
            clip_frames: List[FrameInfo] = []

            for frame_idx in range(clip.start_frame, clip.end_frame):
                total_requests += 1
                cached = face_detection_cache.get(frame_idx)

                if cached is not None:
                    cache_hit_count += 1
                    # Clone FaceInfo with clip-specific face_id and crop_path
                    face_infos = _clone_for_clip(
                        cached, frame_idx, clip.clip_id, video_id,
                        crops_base_dir, crop_enabled,
                    )
                else:
                    # Should not happen: all frames in range were processed
                    face_infos = []

                total_faces_detected += len(face_infos)

                timestamp = frame_idx / target_fps
                frame_info = FrameInfo(
                    frame_index=frame_idx,
                    clip_id=clip.clip_id,
                    timestamp=timestamp,
                    image_path=Path(""),
                    faces=tuple(face_infos),
                )
                clip_frames.append(frame_info)

            sequence = FaceSequence(
                clip_id=clip.clip_id,
                video_id=video_id,
                frames=tuple(clip_frames),
            )
            all_sequences.append(sequence)

        elapsed = time.monotonic() - start_time
        total_frames_processed = sum(len(s.frames) for s in all_sequences)
        hit_ratio = (cache_hit_count / total_requests * 100) if total_requests > 0 else 0.0

        logger.info(
            f"[{self._STAGE_NAME}] Detection Cache: "
            f"total_requests={total_requests} "
            f"unique_inferences={scrfd_inference_count} "
            f"cache_hits={cache_hit_count} "
            f"hit_ratio={hit_ratio:.1f}%"
        )
        logger.info(
            f"[{self._STAGE_NAME}] Detection complete in {elapsed:.1f}s: "
            f"{total_frames_processed} frames, "
            f"{total_faces_detected} faces detected "
            f"({len(all_sequences)} FaceSequence objects)"
        )

        # Remove the Phase-1 temporary crop bucket (full-video detection pass).
        if crop_enabled and crops_base_dir is not None:
            tmp_bucket = crops_base_dir / video_id / placeholder_clip_id
            if tmp_bucket.exists():
                shutil.rmtree(tmp_bucket, ignore_errors=True)

        return all_sequences

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a List[FaceSequence].

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a valid FaceSequence list.
        """
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[FaceSequence], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty: no sequences generated."
            )
        return all(isinstance(s, FaceSequence) for s in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_scrfd_model(
        config: dict,
        context: PipelineContext,
        logger: Any,
    ) -> Any:
        """Get or load the SCRFD model via ModelManager with fallback.

        Args:
            config: Pipeline configuration dict.
            context: PipelineContext with model_manager.
            logger: PipelineLogger instance.

        Returns:
            Loaded SCRFD model with detect(image) -> List[dict] method.

        Raises:
            PipelineError: If model loading fails.
        """
        # Same detection floor as Stage 3 (stage_03.detection_confidence_threshold).
        # The config documents this floor as SHARED with Stage 5 ("该下限与 Stage5
        # 共享"), but that sharing only holds while the Stage-3-configured instance
        # survives in the ModelManager cache. When Stage 5 runs on its own (fresh
        # ModelManager / separate process) the cached instance is absent and a new
        # one is loaded with the SCRFD wrapper's default 0.5 — silently dropping
        # occluded / edge-cut faces (score ~0.35-0.5) and changing Stage 7 inputs.
        # Force the same per-instance floor on whichever instance we get.
        detection_floor: float = float(
            config.get("stage_03", {}).get("detection_confidence_threshold", 0.5)
        )

        try:
            scrfd_model = context.model_manager.get_model("scrfd")
        except (KeyError, AttributeError):
            models_cfg: dict = config.get("models", {})
            scrfd_cfg: dict = models_cfg.get("scrfd", {})
            scrfd_path = Path(scrfd_cfg.get("path", "models/SCRFD"))
            scrfd_device = scrfd_cfg.get("device", "cpu")
            scrfd_provider = scrfd_cfg.get("execution_provider", "auto")

            logger.info(
                f"[Stage 5] Loading SCRFD model: "
                f"path={scrfd_path}, device={scrfd_device}, "
                f"execution_provider={scrfd_provider}"
            )
            try:
                scrfd_model = context.model_manager.load_model(
                    "scrfd", str(scrfd_path), execution_provider=scrfd_provider
                )
            except (NotImplementedError, AttributeError):
                logger.warning(
                    f"[Stage 5] ModelManager.load_model not implemented. "
                    f"Loading SCRFD wrapper directly."
                )
                scrfd_model = _load_scrfd_directly_fallback(
                    scrfd_path, scrfd_device, logger, scrfd_provider
                )
            except Exception as exc:
                logger.error(f"[Stage 5] Failed to load SCRFD: {exc}")
                raise PipelineError(
                    f"[Stage 5] SCRFD model loading failed."
                ) from exc

        # Mirror Stage 3 (stage_03_face_screening/stage.py): set the per-instance
        # detection floor so Stage 5 and Stage 3 always share one threshold.
        if hasattr(scrfd_model, "confidence_threshold"):
            scrfd_model.confidence_threshold = detection_floor
            logger.info(
                f"[Stage 5] SCRFD detection floor set to {detection_floor}"
            )

        return scrfd_model

    @staticmethod
    def _resolve_video_path(
        clips: List[ClipInfo],
        config: dict,
        logger: Any,
    ) -> Path:
        """Resolve the standardized video path from config and clip data.

        Uses the same search strategy as Stage 3.
        """
        storage_cfg: dict = config.get("storage", {})
        output_dir = Path(storage_cfg.get("output_dir", "outputs"))

        video_id = clips[0].video_id
        candidates = [
            output_dir / "videos" / video_id / f"{video_id}_std.mp4",
            output_dir / video_id / f"{video_id}_std.mp4",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        test_output_dir = output_dir.parent / "test_output"
        for root_dir in [output_dir, Path("outputs"), test_output_dir]:
            for candidate in root_dir.rglob(f"*{video_id}*_std.mp4"):
                if candidate.is_file():
                    return candidate

        raise DataProcessingError(
            f"[Stage 5] Cannot find standardized video for "
            f"video_id={video_id}. Tried: {candidates}"
        )


def _record_resolution_audit(
    context: PipelineContext,
    video_id: str,
    result: Any,
) -> None:
    """Record the video-level resolution pre-filter decision in stage_audit."""
    context.stage_audit.setdefault(
        "stage_05_face_resolution_filter", {}
    ).update({
        "video_id": video_id,
        "passed": result.passed,
        "reject_reason": result.reject_reason,
        "expected_frame_count": result.expected_frame_count,
        "pass_frame_count": result.pass_frame_count,
        "valid_face_frame_count": result.valid_face_frame_count,
        "qualified_ratio": result.qualified_ratio,
        "status_counts": dict(result.status_counts),
        "face_short_side_min": result.face_short_side_min,
        "face_short_side_median": result.face_short_side_median,
        "face_short_side_max": result.face_short_side_max,
    })


def _load_scrfd_directly_fallback(
    model_path: Path,
    device: str,
    logger: Any,
    execution_provider: str = "auto",
) -> Any:
    """Fallback: load SCRFD wrapper directly when ModelManager is skeleton."""
    from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper

    wrapper = SCRFDWrapper(
        model_path=model_path,
        device=device,
        execution_provider=execution_provider,
    )
    try:
        wrapper.load()
        return wrapper
    except FileNotFoundError:
        logger.warning(
            f"[Stage 5] SCRFD weights not found at {model_path}. "
            f"Operating in no-model mode."
        )
        return _NoOpSCRFDModel()
    except RuntimeError as exc:
        logger.warning(
            f"[Stage 5] Cannot load SCRFD: {exc}. "
            f"Operating in no-model mode."
        )
        return _NoOpSCRFDModel()


def _clone_for_clip(
    cached_faces: List[FaceInfo],
    frame_idx: int,
    clip_id: str,
    video_id: str,
    crops_base_dir: Optional[Path],
    crop_enabled: bool,
) -> List[FaceInfo]:
    """Clone cached FaceInfo objects with clip-specific face_id and crop_path.

    Copies crop images to the new clip's directory structure.
    Does NOT re-run SCRFD — reuses cached detection results.

    Args:
        cached_faces: FaceInfo list from the first clip that processed this frame.
        frame_idx: Frame index.
        clip_id: Target clip identifier.
        video_id: Video identifier.
        crops_base_dir: Base directory for face crops.
        crop_enabled: Whether crop saving is enabled.

    Returns:
        New FaceInfo list with updated clip-specific fields.
    """
    new_faces: List[FaceInfo] = []
    for i, old in enumerate(cached_faces):
        face_id = f"{video_id}_{clip_id}_face_{i:03d}"
        new_crop_path = Path("")
        if crop_enabled and crops_base_dir is not None and old.face_crop_path != Path(""):
            crop_dir = crops_base_dir / video_id / clip_id / "crops"
            crop_filename = f"frame_{frame_idx:06d}_face_{i:03d}.jpg"
            new_crop_path = crop_dir / crop_filename
            if old.face_crop_path.exists() and old.face_crop_path != new_crop_path:
                new_crop_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(old.face_crop_path), str(new_crop_path))

        new_faces.append(FaceInfo(
            frame_index=frame_idx,
            face_id=face_id,
            bbox=old.bbox,
            landmark_5=old.landmark_5,
            confidence=old.confidence,
            face_crop_path=new_crop_path,
            track_id=old.track_id,
            expanded_bbox=old.expanded_bbox,
        ))
    return new_faces


class _NoOpSCRFDModel:
    """No-op SCRFD model for environments without GPU/weights."""

    def detect(self, image: Any) -> List[dict]:
        """Return empty detection list."""
        return []

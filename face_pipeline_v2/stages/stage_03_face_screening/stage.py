"""Stage 3: SCRFD Coarse Face Screening.

Per AISDD 3.6, this stage performs low-cost face screening on Stage 2's
SceneInfo list to filter out invalid scenes before expensive processing.

Processing flow:
    1. Duration validation — reject scenes shorter than min_duration.
    2. 1 FPS frame sampling from standardized video.
    3. SCRFD detection on sampled frames.
    4. Calculate CoarseFaceInfo per-scene metadata.
    5. Generate SceneFilterResult with reject_reasons.
    6. Return SceneInfo list, each carrying CoarseFaceInfo + SceneFilterResult.

Optional debug export (stage_03.debug.export_valid_scenes=true):
    Exports passed scenes as MP4 segments via FFmpeg for manual verification.

Important: Stage 3 does NOT produce aggregate statistics.
Stage 15 Quality Statistics handles all cross-stage aggregation.

Input:  List[SceneInfo] (from Stage 2)
Output: List[SceneInfo] (each with coarse_face_info + filter_result)
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from face_pipeline_v2.data.models import SceneInfo, StandardizedVideoInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_03_face_screening.scrfd_processor import (
    SCRFDProcessor,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
    PipelineError,
    StageExecutionError,
)


class Stage3CoarseFaceScreening(BaseStage):
    """Stage 3 — SCRFD Coarse Face Screening.

    Input:  List[SceneInfo]
    Output: List[SceneInfo] (each with coarse_face_info + filter_result)
    """

    _STAGE_NAME: str = "Stage 3 Coarse Face Screening"

    def __init__(self) -> None:
        """Initialize Stage 3."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a non-empty List[SceneInfo].

        Args:
            input_data: Expected List[SceneInfo] from Stage 2.
            context: PipelineContext.

        Returns:
            True if input is a non-empty list of SceneInfo objects.
        """
        if not isinstance(input_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[SceneInfo], "
                f"got {type(input_data).__name__}"
            )
            return False
        if len(input_data) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Empty scene list: nothing to screen."
            )
            return False
        return all(isinstance(s, SceneInfo) for s in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[SceneInfo]:
        """Execute coarse face screening on all scenes.

        Steps (AISDD 3.6.4):
            1. Load stage_03 config from context.config.
            2. Get SCRFD model from ModelManager.
            3. Locate standardized video from Stage 1 output.
            4. For each SceneInfo:
                a. Duration check → too_short reject.
                b. Sample frames at sampling_fps.
                c. Run SCRFD detection.
                d. Calculate CoarseFaceInfo.
                e. Generate SceneFilterResult.
                f. Return new SceneInfo (immutable).
            5. Optional: export passed scenes for debug verification.
            6. Return list of SceneInfo with metadata attached.

        Args:
            input_data: List[SceneInfo] from Stage 2.
            context: PipelineContext with config, logger, model_manager.

        Returns:
            List[SceneInfo], each carrying CoarseFaceInfo + SceneFilterResult.

        Raises:
            PipelineError: On model loading or detection failure.
        """
        logger = context.logger
        config = context.config
        scenes: List[SceneInfo] = input_data

        # --- Load stage configuration ---
        stage_cfg: dict = config.get("stage_03", {})
        if not stage_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 3 is disabled in config. "
                f"Passing through {len(scenes)} scenes unchanged."
            )
            return scenes

        sampling_fps: int = stage_cfg.get("sampling_fps", 1)
        min_duration: float = float(stage_cfg.get("min_duration", 8))
        min_face_detect_ratio: float = float(
            stage_cfg.get("min_face_detect_ratio", 0.95)
        )
        min_single_face_ratio: float = float(
            stage_cfg.get("min_single_face_ratio", 0.95)
        )
        min_face_confidence: float = float(
            stage_cfg.get("min_face_confidence", 0.6)
        )
        detection_floor: float = float(
            stage_cfg.get("detection_confidence_threshold", 0.5)
        )

        logger.info(
            f"[{self._STAGE_NAME}] Config: sampling_fps={sampling_fps}, "
            f"min_duration={min_duration}s, "
            f"min_face_detect_ratio={min_face_detect_ratio}, "
            f"min_single_face_ratio={min_single_face_ratio}, "
            f"min_face_confidence={min_face_confidence}, "
            f"detection_floor={detection_floor}"
        )

        # --- Get SCRFD model ---
        try:
            scrfd_model = context.model_manager.get_model("scrfd")
        except (KeyError, AttributeError):
            models_cfg: dict = config.get("models", {})
            scrfd_cfg: dict = models_cfg.get("scrfd", {})
            scrfd_path = Path(scrfd_cfg.get("path", "models/SCRFD"))
            scrfd_device = scrfd_cfg.get("device", "cpu")
            scrfd_provider = scrfd_cfg.get("execution_provider", "auto")

            logger.info(
                f"[{self._STAGE_NAME}] Loading SCRFD model: "
                f"path={scrfd_path}, device={scrfd_device}, "
                f"execution_provider={scrfd_provider}"
            )
            try:
                scrfd_model = context.model_manager.load_model(
                    "scrfd", str(scrfd_path), execution_provider=scrfd_provider
                )
            except (NotImplementedError, AttributeError):
                logger.warning(
                    f"[{self._STAGE_NAME}] ModelManager.load_model not "
                    f"implemented. Loading SCRFD wrapper directly."
                )
                scrfd_model = self._load_scrfd_directly(
                    scrfd_path, scrfd_device, logger, scrfd_provider
                )
            except Exception as exc:
                logger.error(
                    f"[{self._STAGE_NAME}] Failed to load SCRFD: {exc}"
                )
                raise PipelineError(
                    f"[{self._STAGE_NAME}] SCRFD model loading failed."
                ) from exc

        # Stage 3 is a coarse "is there a face" gate: lower the SCRFD wrapper's
        # internal detection floor (per-instance) so occluded / partially-cropped
        # real faces — whose SCRFD score lands in ~0.35-0.5 because the camera
        # cuts off the top of the head — are NOT silently dropped before the
        # ratio / confidence thresholds are evaluated.  Default 0.5 keeps prior
        # behavior when the config key is absent.  Note: ModelManager caches the
        # model instance, so this floor is shared with later SCRFD consumers
        # (e.g. Stage 5), which is desirable for this talking-face dataset.
        if hasattr(scrfd_model, "confidence_threshold"):
            scrfd_model.confidence_threshold = detection_floor
        logger.info(
            f"[{self._STAGE_NAME}] SCRFD detection floor set to {detection_floor}"
        )

        # --- Open standardized video ---
        video_path = self._resolve_video_path(scenes, config, logger)
        logger.info(
            f"[{self._STAGE_NAME}] Opening video: {video_path}"
        )
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoReadError(
                f"[{self._STAGE_NAME}] Cannot open video: {video_path}"
            )
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0
        logger.debug(
            f"[{self._STAGE_NAME}] Video FPS: {video_fps}"
        )

        # --- Build frame reader callable ---
        _frame_cache: dict = {}

        def read_frame(frame_index: int) -> np.ndarray:
            if frame_index in _frame_cache:
                return _frame_cache[frame_index]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                raise DataProcessingError(
                    f"[{self._STAGE_NAME}] Failed to read frame {frame_index}"
                )
            _frame_cache[frame_index] = frame
            if len(_frame_cache) > 1000:
                oldest = next(iter(_frame_cache))
                del _frame_cache[oldest]
            return frame

        # --- Create processor ---
        processor = SCRFDProcessor(
            scrfd_model=scrfd_model,
            video_reader=read_frame,
            video_fps=video_fps,
        )

        # --- Process each scene ---
        start_time = time.monotonic()
        output_scenes: List[SceneInfo] = []
        screening_records: List[dict] = []
        passed_count = 0
        rejected_count = 0

        for scene in scenes:
            coarse_info, filter_result = processor.process_scene(
                scene_start_frame=scene.start_frame,
                scene_end_frame=scene.end_frame,
                sampling_fps=sampling_fps,
                min_duration=min_duration,
                scene_duration=scene.duration,
                min_face_detect_ratio=min_face_detect_ratio,
                min_single_face_ratio=min_single_face_ratio,
                min_face_confidence=min_face_confidence,
            )

            screening_records.append(
                {
                    "video_id": scene.video_id,
                    "scene_id": scene.scene_id,
                    "start_frame": scene.start_frame,
                    "end_frame": scene.end_frame,
                    "start_time": scene.start_time,
                    "end_time": scene.end_time,
                    "duration_s": scene.duration,
                    "sampling_fps": sampling_fps,
                    "thresholds": {
                        "min_duration_s": min_duration,
                        "min_face_detect_ratio": min_face_detect_ratio,
                        "min_single_face_ratio": min_single_face_ratio,
                        "min_face_confidence": min_face_confidence,
                        "detection_floor": detection_floor,
                    },
                    "passed": filter_result.passed,
                    "reject_reasons": list(filter_result.reject_reasons),
                    "metrics": asdict(coarse_info),
                    "frames": list(processor.frame_summaries),
                }
            )

            if filter_result.passed:
                passed_count += 1
            else:
                rejected_count += 1
                logger.debug(
                    f"[{self._STAGE_NAME}] {scene.scene_id} rejected: "
                    f"{filter_result.reject_reasons}"
                )

            screened_scene = SceneInfo(
                scene_id=scene.scene_id,
                video_id=scene.video_id,
                start_frame=scene.start_frame,
                end_frame=scene.end_frame,
                start_time=scene.start_time,
                end_time=scene.end_time,
                duration=scene.duration,
                coarse_face_info=coarse_info,
                filter_result=filter_result,
            )
            output_scenes.append(screened_scene)

        cap.release()
        elapsed = time.monotonic() - start_time

        logger.info(
            f"[{self._STAGE_NAME}] Screening complete in {elapsed:.1f}s: "
            f"{len(output_scenes)} total, {passed_count} passed, "
            f"{rejected_count} rejected"
        )

        # --- Debug: export passed scenes ---
        debug_cfg: dict = stage_cfg.get("debug", {})
        if debug_cfg.get("export_valid_scenes", False):
            self._export_debug_scenes(
                output_scenes=output_scenes,
                video_path=video_path,
                debug_cfg=debug_cfg,
                logger=logger,
            )

        # --- Persist per-scene screening report for diagnostics ---
        self._write_screening_report(
            screening_records=screening_records,
            config=config,
            logger=logger,
        )

        return output_scenes

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a List[SceneInfo] with metadata.

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a valid SceneInfo list.
        """
        if not isinstance(output_data, list):
            return False
        if len(output_data) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Output is empty: all scenes rejected."
            )
        return all(isinstance(s, SceneInfo) for s in output_data)

    # ------------------------------------------------------------------
    # Debug: scene export
    # ------------------------------------------------------------------

    def _export_debug_scenes(
        self,
        output_scenes: List[SceneInfo],
        video_path: Path,
        debug_cfg: dict,
        logger: Any,
    ) -> None:
        """Export passed scene segments as MP4 files for manual verification.

        Only scenes with filter_result.passed=True are exported.
        Failed scenes are never exported.

        Args:
            output_scenes: Processed SceneInfo list with filter results.
            video_path: Path to the standardized source video.
            debug_cfg: Debug configuration dict (stage_03.debug).
            logger: PipelineLogger instance.
        """
        from face_pipeline_v2.utils.video_utils import cut_video_segment

        output_dir = Path(debug_cfg.get("output_dir", "outputs/debug/stage_03_scenes"))
        passed_scenes = [
            s for s in output_scenes
            if s.filter_result is not None and s.filter_result.passed
        ]

        if not passed_scenes:
            logger.info(
                f"[{self._STAGE_NAME}] Debug export: no passed scenes to export."
            )
            return

        logger.info(
            f"[{self._STAGE_NAME}] Debug export: exporting {len(passed_scenes)} "
            f"passed scenes to {output_dir}"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        exported = 0

        for scene in passed_scenes:
            segment_path = output_dir / f"{scene.scene_id}.mp4"
            try:
                cut_video_segment(
                    input_path=video_path,
                    output_path=segment_path,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                )
                exported += 1
                logger.debug(
                    f"[{self._STAGE_NAME}] Debug export: {segment_path.name} "
                    f"({scene.duration:.1f}s)"
                )
            except Exception as exc:
                logger.warning(
                    f"[{self._STAGE_NAME}] Debug export failed for "
                    f"{scene.scene_id}: {exc}"
                )

        logger.info(
            f"[{self._STAGE_NAME}] Debug export complete: "
            f"{exported}/{len(passed_scenes)} exported"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_video_path(
        scenes: List[SceneInfo],
        config: dict,
        logger: Any,
    ) -> Path:
        """Resolve the standardized video path from config and scene data."""
        storage_cfg: dict = config.get("storage", {})
        output_dir = Path(storage_cfg.get("output_dir", "outputs"))

        video_id = scenes[0].video_id
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
            f"[Stage 3] Cannot find standardized video for "
            f"video_id={video_id}. Tried: {candidates}"
        )

    def _write_screening_report(
        self,
        screening_records: List[dict],
        config: dict,
        logger: Any,
    ) -> None:
        """Persist per-scene Stage 3 screening decisions to a JSON file.

        The coarse screening results (metrics + reject reasons + per-frame
        detections) otherwise exist only in-memory / DEBUG logs, which makes
        debugging "why was this video rejected" require a full re-run.  Writing
        them out here lets that question be answered from disk afterwards.

        Writes to: <storage.output_dir>/stage3_screening/<video_id>_stage3_screening.json
        Never raises: a persistence failure is logged and ignored so it cannot
        break the pipeline.
        """
        if not screening_records:
            return
        storage_cfg: dict = config.get("storage", {})
        output_dir = Path(storage_cfg.get("output_dir", "outputs"))
        video_id = screening_records[0]["video_id"]
        report_dir = output_dir / "stage3_screening"
        report_path = report_dir / f"{video_id}_stage3_screening.json"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "video_id": video_id,
                "generated_by": "stage_03_coarse_face_screening",
                "scenes": screening_records,
            }
            report_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                f"[{self._STAGE_NAME}] Screening report saved: {report_path}"
            )
        except Exception as exc:
            logger.warning(
                f"[{self._STAGE_NAME}] Failed to write screening report "
                f"{report_path}: {exc}"
            )

    @staticmethod
    def _load_scrfd_directly(
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
                f"[Stage 3] SCRFD weights not found at {model_path}. "
                f"Operating in no-model mode — all scenes will be "
                f"marked as passed."
            )
            return _NoOpSCRFDModel()
        except RuntimeError as exc:
            logger.warning(
                f"[Stage 3] Cannot load SCRFD: {exc}. "
                f"Operating in no-model mode."
            )
            return _NoOpSCRFDModel()


class _NoOpSCRFDModel:
    """No-op SCRFD model for environments without GPU/weights."""

    def detect(self, image: Any) -> List[dict]:
        """Return empty detection list (no faces found)."""
        return []


class VideoReadError(PipelineError):
    """Video file cannot be opened or read for frame extraction."""

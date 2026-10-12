"""Stage 2: Scene Segmentation.

Per AISDD 3.5, this stage detects shot transitions in the standardized
video and partitions it into contiguous Scene segments.

Input:  StandardizedVideoInfo (from Stage 1)
Output: List[SceneInfo]
"""

import json
import time
from pathlib import Path
from typing import Any, List, Optional

from face_pipeline_v2.data.models import SceneInfo, StandardizedVideoInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_02_scene_detection.scene_detector import (
    SceneDetector,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
    MissingDependencyError,
    PipelineError,
)


class Stage2SceneSegmentation(BaseStage):
    """Stage 2 — Scene Segmentation using PySceneDetect ContentDetector.

    Input:  StandardizedVideoInfo
    Output: List[SceneInfo]
    """

    _STAGE_NAME: str = "Stage 2 Scene Segmentation"

    def __init__(self) -> None:
        """Initialize Stage 2."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a StandardizedVideoInfo with existing file.

        Args:
            input_data: Expected to be a StandardizedVideoInfo from Stage 1.
            context: PipelineContext.

        Returns:
            True if input_data is valid and its video exists.
        """
        if not isinstance(input_data, StandardizedVideoInfo):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected StandardizedVideoInfo, "
                f"got {type(input_data).__name__}"
            )
            return False
        video_path = input_data.video_path
        if not video_path.exists():
            context.logger.error(
                f"[{self._STAGE_NAME}] Standardized video not found: "
                f"{video_path}"
            )
            return False
        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[SceneInfo]:
        """Execute scene detection and produce SceneInfo list.

        Steps (AISDD 3.5.4):
            1. Read standardized video path from Stage 1 output.
            2. Load detector config from config.yaml.
            3. Run PySceneDetect ContentDetector.
            4. Convert boundaries to SceneInfo objects.
            5. Save scene metadata to outputs/scenes/.
            6. Return SceneInfo list.

        Args:
            input_data: StandardizedVideoInfo from Stage 1.
            context: PipelineContext with config and logger.

        Returns:
            List of SceneInfo objects.

        Raises:
            MissingDependencyError: If PySceneDetect is not installed.
            DataProcessingError: If scene detection fails.
        """
        logger = context.logger
        config = context.config
        std_info: StandardizedVideoInfo = input_data
        video_path: Path = std_info.video_path

        # Load stage configuration (AISDD 3.5)
        scene_cfg: dict = config.get("stage_02", {})
        threshold: float = scene_cfg.get("threshold", 27.0)
        min_scene_len: int = scene_cfg.get("min_scene_len", 15)

        adaptive_cfg: dict = scene_cfg.get("adaptive_threshold", {})
        adaptive_factor: Optional[float] = None
        if adaptive_cfg.get("enabled", False):
            adaptive_factor = float(adaptive_cfg.get("factor", 2.0))

        logger.info(
            f"[{self._STAGE_NAME}] Detecting scenes: {video_path.name} "
            f"(threshold={threshold}, min_scene_len={min_scene_len}, "
            f"adaptive_factor={adaptive_factor})"
        )

        start_time = time.monotonic()

        # Step 1-3: Run detector
        detector = SceneDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
            adaptive_factor=adaptive_factor,
        )
        face_subtitle_cfg: dict = scene_cfg.get("face_subtitle", {})
        if face_subtitle_cfg.get("enabled", False):
            logger.info(
                f"[{self._STAGE_NAME}] Face-subtitle detection enabled "
                f"(band {face_subtitle_cfg.get('band_top', 0.33):.2f}–"
                f"{face_subtitle_cfg.get('band_bottom', 0.67):.2f})"
            )
        try:
            raw_scenes = detector.detect(
                video_path, face_subtitle_cfg=face_subtitle_cfg
            )
        except MissingDependencyError:
            raise
        except PipelineError:
            raise
        except Exception as exc:
            logger.error(
                f"[{self._STAGE_NAME}] Scene detection failed: {exc}"
            )
            raise DataProcessingError(
                f"[{self._STAGE_NAME}] Failed to detect scenes in "
                f"'{video_path}'"
            ) from exc

        nc = detector.last_noise_ceiling
        nc_text = f"{nc:.3f}" if nc is not None else "N/A (disabled)"
        logger.debug(
            f"[{self._STAGE_NAME}] {video_path.name} adaptive threshold → "
            f"noise_ceiling={nc_text}, "
            f"effective_threshold={detector.last_effective_threshold:.3f}"
        )

        # Step 4: Convert to SceneInfo objects
        video_id = video_path.stem.replace("_std", "")
        scenes: List[SceneInfo] = []
        for raw in raw_scenes:
            scene_id = f"{video_id}_scene_{raw['scene_index']:04d}"
            si = SceneInfo(
                scene_id=scene_id,
                video_id=video_id,
                start_frame=raw["start_frame"],
                end_frame=raw["end_frame"],
                start_time=raw["start_time"],
                end_time=raw["end_time"],
                duration=raw["duration"],
            )
            scenes.append(si)

        elapsed = time.monotonic() - start_time
        logger.info(
            f"[{self._STAGE_NAME}] Detected {len(scenes)} scenes "
            f"in {elapsed:.1f}s"
        )

        # Step 5: Save scene metadata
        self._save_scene_metadata(
            scenes=scenes,
            video_id=video_id,
            config=config,
            logger=logger,
        )

        return scenes

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a non-empty list of SceneInfo.

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a valid SceneInfo list with at least one element.
        """
        if not isinstance(output_data, list):
            return False
        if len(output_data) == 0:
            context.logger.warning(
                f"[{self._STAGE_NAME}] Empty scene list: no scenes detected."
            )
            return False
        return all(isinstance(s, SceneInfo) for s in output_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_scene_metadata(
        self,
        scenes: List[SceneInfo],
        video_id: str,
        config: dict,
        logger: Any,
    ) -> None:
        """Save scene metadata to outputs/scenes/<video_id>_scenes.json.

        Args:
            scenes: List of detected SceneInfo objects.
            video_id: Video identifier for filename construction.
            config: Pipeline configuration dictionary.
            logger: PipelineLogger instance.
        """
        storage_cfg: dict = config.get("storage", {})
        output_base = Path(storage_cfg.get("output_dir", "outputs"))
        scene_dir = output_base / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = scene_dir / f"{video_id}_scenes.json"
        payload: list = []
        for s in scenes:
            payload.append({
                "video_id": s.video_id,
                "scene_id": s.scene_id,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
            })

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[{self._STAGE_NAME}] Scene metadata saved: {metadata_path}"
        )

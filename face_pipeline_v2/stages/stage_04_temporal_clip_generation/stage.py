"""Stage 4: Temporal Clip Generation / Clip Adapter.

Per AISDD 3.7, Stage 4 partitions SceneInfo segments into ClipInfo objects.

Two modes (config: stage_04.mode):
    - sliding_window (default, backward compatible): fixed-length windows
      (window_size / stride).
    - segment: one-to-one adapter — each SceneInfo becomes a single
      full-length ClipInfo (dynamic duration, no resampling).

Pure-logic stage — no model calls, video I/O, face detection, or audio.

Input:  List[SceneInfo] (from Stage 3, each carrying filter_result)
Output: List[ClipInfo]
"""

from typing import Any, List

from face_pipeline_v2.data.models import ClipInfo, SceneInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_04_temporal_clip_generation.clip_generator import (
    generate_clips,
    generate_segment_clips,
)
from face_pipeline_v2.utils.exceptions import InputValidationError

# Allowed stage_04.mode values.
VALID_MODES: tuple = ("sliding_window", "segment")


class Stage4TemporalClipGeneration(BaseStage):
    """Stage 4 — Clip Generation / Clip Adapter (sliding_window | segment).

    Input:  List[SceneInfo] (only filter_result.passed=True are processed)
    Output: List[ClipInfo]
    """

    _STAGE_NAME: str = "Stage 4 Temporal Clip Generation"

    def __init__(self) -> None:
        """Initialize Stage 4."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a non-empty List[SceneInfo].

        Args:
            input_data: Expected List[SceneInfo] from Stage 3.
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
                f"[{self._STAGE_NAME}] Empty scene list: nothing to clip."
            )
            return False
        return all(isinstance(s, SceneInfo) for s in input_data)

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> List[ClipInfo]:
        """Execute clip generation (mode: sliding_window | segment).

        Steps (AISDD 3.7):
            1. Read stage_04 config (incl. mode) from context.config.
            2. Filter scenes: only process filter_result.passed=True.
            3. For each passed scene, generate clips per configured mode
               (sliding_window: fixed-length windows; segment: 1:1 full length).
            4. Return flat list of ClipInfo.

        Args:
            input_data: List[SceneInfo] from Stage 3.
            context: PipelineContext with config, logger, model_manager.

        Returns:
            List[ClipInfo] in chronological order (by scene, then by clip).

        Raises:
            InputValidationError: If no passed scenes found.
        """
        logger = context.logger
        config = context.config
        scenes: List[SceneInfo] = input_data

        # --- Load stage configuration ---
        stage_cfg: dict = config.get("stage_04", {})
        if not stage_cfg.get("enabled", True):
            logger.info(
                f"[{self._STAGE_NAME}] Stage 4 is disabled in config. "
                f"Returning empty clip list."
            )
            return []

        window_size: float = float(stage_cfg.get("window_size", 8))
        stride: float = float(stage_cfg.get("stride", 4))
        target_fps: float = float(stage_cfg.get("target_fps", 25))
        keep_tail: bool = bool(stage_cfg.get("keep_tail", False))
        min_tail_duration: float = float(stage_cfg.get("min_tail_duration", 1.0))

        # --- Mode selection (default: sliding_window, backward compatible) ---
        mode: str = str(stage_cfg.get("mode", "sliding_window"))
        if mode not in VALID_MODES:
            logger.warning(
                f"[{self._STAGE_NAME}] Unknown mode '{mode}' in config. "
                f"Falling back to 'sliding_window'. "
                f"Allowed modes: {list(VALID_MODES)}"
            )
            mode = "sliding_window"

        logger.info(
            f"[{self._STAGE_NAME}] Config: mode={mode}, "
            f"window_size={window_size}s, stride={stride}s, "
            f"target_fps={target_fps}, keep_tail={keep_tail}, "
            f"min_tail_duration={min_tail_duration}s"
        )

        # --- Filter: only passed scenes ---
        passed_scenes = [
            s for s in scenes
            if s.filter_result is not None and s.filter_result.passed
        ]

        skipped = len(scenes) - len(passed_scenes)
        if skipped > 0:
            logger.info(
                f"[{self._STAGE_NAME}] Skipping {skipped} rejected scene(s), "
                f"processing {len(passed_scenes)} passed scene(s)"
            )

        if not passed_scenes:
            logger.warning(
                f"[{self._STAGE_NAME}] No passed scenes to process. "
                f"Returning empty clip list."
            )
            return []

        # --- Generate clips (mode-dependent) ---
        all_clips: List[ClipInfo] = []
        for scene in passed_scenes:
            if mode == "segment":
                # One-to-one adapter: full-length single clip per scene.
                clips = generate_segment_clips(scene=scene, target_fps=target_fps)
            else:
                # Sliding window: fixed-length clips (default, compatible).
                clips = generate_clips(
                    scene=scene,
                    window_size=window_size,
                    stride=stride,
                    target_fps=target_fps,
                    keep_tail=keep_tail,
                    min_tail_duration=min_tail_duration,
                )
            all_clips.extend(clips)

            if clips:
                logger.debug(
                    f"[{self._STAGE_NAME}] {scene.scene_id}: "
                    f"{len(clips)} clip(s) generated "
                    f"({scene.duration:.1f}s scene, mode={mode})"
                )
            else:
                logger.debug(
                    f"[{self._STAGE_NAME}] {scene.scene_id}: "
                    f"0 clips (duration {scene.duration:.1f}s "
                    f"not clip-able in mode={mode})"
                )

        logger.info(
            f"[{self._STAGE_NAME}] Generated {len(all_clips)} clip(s) "
            f"from {len(passed_scenes)} scene(s)"
        )

        return all_clips

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output is a List[ClipInfo].

        An empty list is valid (all scenes may be too short).

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if output is a List[ClipInfo] (may be empty).
        """
        if not isinstance(output_data, list):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected List[ClipInfo], "
                f"got {type(output_data).__name__}"
            )
            return False
        if len(output_data) == 0:
            context.logger.info(
                f"[{self._STAGE_NAME}] Output is empty: no clips generated."
            )
        return all(isinstance(c, ClipInfo) for c in output_data)

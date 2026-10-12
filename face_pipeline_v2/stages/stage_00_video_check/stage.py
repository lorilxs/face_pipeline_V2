"""Stage 0: Video Integrity Check.

Per AISDD 3.3, this stage validates that an input video:
    - Exists on disk.
    - Can be decoded by FFprobe.
    - Contains a Video Stream.
    - Optionally checks for an Audio Stream.

Output: VideoInfo (from face_pipeline_v2.data.models).
"""

from pathlib import Path
from typing import Any

from face_pipeline_v2.data.models import VideoInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_00_video_check.video_checker import (
    FFprobeChecker,
)
from face_pipeline_v2.utils.exceptions import (
    MissingDependencyError,
    MissingStreamError,
    VideoReadError,
)


class Stage0VideoIntegrityCheck(BaseStage):
    """Stage 0 — Video Integrity Check using FFprobe.

    Input:  Path (raw video file path)
    Output: VideoInfo
    """

    _STAGE_NAME: str = "Stage 0 Video Integrity Check"

    def __init__(self) -> None:
        """Initialize Stage 0."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a Path pointing to an existing file.

        Args:
            input_data: Expected to be a pathlib.Path or str convertible to Path.
            context: PipelineContext (unused here).

        Returns:
            True if input_data is a valid path to an existing file.
        """
        if input_data is None:
            return False
        if not isinstance(input_data, (Path, str)):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected Path or str, "
                f"got {type(input_data).__name__}"
            )
            return False
        video_path = Path(input_data)
        return video_path.exists() and video_path.is_file()

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> VideoInfo:
        """Execute video integrity check and build VideoInfo.

        Steps (AISDD 3.3.4):
            1. Verify file existence (already done in validate_input).
            2. Run FFprobe to extract metadata.
            3. Validate Video Stream presence.
            4. Extract: video_id, fps, duration, width, height,
               frame_count, video_codec, audio_codec.
            5. Return VideoInfo.

        Args:
            input_data: Path to the raw video file.
            context: PipelineContext with config and logger.

        Returns:
            VideoInfo populated with extracted metadata.

        Raises:
            VideoReadError: If ffprobe fails to read the video.
            MissingStreamError: If no video stream is detected.
        """
        logger = context.logger
        config = context.config
        video_path = Path(input_data)

        # Read stage-specific configuration (AISDD 3.3.6)
        video_cfg: dict = config.get("video", {})
        check_audio_stream: bool = video_cfg.get("check_audio_stream", True)

        # Resolve ffprobe path from config if available
        ffprobe_path: str | None = video_cfg.get("ffprobe_path")

        logger.info(
            f"[{self._STAGE_NAME}] Probing video: {video_path.name}"
        )

        # Step 1 + 2: Probe video metadata (ffprobe, with ffmpeg fallback).
        # If ffprobe is missing, is actually a stripped ffmpeg binary, or
        # fails, fall back to parsing `ffmpeg -i` stderr.
        source: str = "ffprobe"
        try:
            checker = FFprobeChecker(ffprobe_path=ffprobe_path)
            metadata, source = checker.probe_or_fallback(video_path)
        except MissingDependencyError:
            # ffprobe binary not found at all — fall back to ffmpeg directly.
            # Build a checker without running __init__ (which would re-raise
            # MissingDependencyError); all methods used below are static so a
            # bare instance is enough.
            checker = FFprobeChecker.__new__(FFprobeChecker)
            metadata = FFprobeChecker.probe_with_ffmpeg(video_path)
            source = "ffmpeg"
        except VideoReadError:
            raise
        except Exception as exc:
            logger.error(
                f"[{self._STAGE_NAME}] Failed to probe {video_path}: {exc}"
            )
            raise VideoReadError(
                f"[{self._STAGE_NAME}] Cannot read video: {video_path}"
            ) from exc

        if source == "ffmpeg":
            logger.warning(
                f"[{self._STAGE_NAME}] ffprobe unavailable or invalid, "
                f"using ffmpeg fallback"
            )
            logger.info(
                f"[{self._STAGE_NAME}] Video metadata extracted by ffmpeg fallback"
            )

        # Step 3: Validate video stream exists
        if not checker.has_video_stream(metadata):
            logger.error(
                f"[{self._STAGE_NAME}] No video stream in: {video_path}"
            )
            raise MissingStreamError(
                f"[{self._STAGE_NAME}] No video stream detected in "
                f"'{video_path}'"
            )

        # Step 4: Extract metadata
        video_meta = checker.extract_video_stream_metadata(metadata)
        format_duration = checker.extract_format_duration(metadata)
        duration: float = (
            format_duration if format_duration > 0 else video_meta["duration"]
        )

        has_audio: bool = checker.has_audio_stream(metadata)
        audio_codec: str | None = (
            checker.extract_audio_stream_codec(metadata)
            if (check_audio_stream and has_audio)
            else None
        )

        # Compute total_frames from fps * duration if nb_frames is zero
        total_frames: int = video_meta["nb_frames"]
        if total_frames == 0 and video_meta["fps"] > 0:
            total_frames = int(video_meta["fps"] * duration)

        # Step 5: Build VideoInfo
        video_info = VideoInfo(
            video_id=video_path.stem,
            video_path=video_path.resolve(),
            original_fps=video_meta["fps"],
            width=video_meta["width"],
            height=video_meta["height"],
            duration=duration,
            total_frames=total_frames,
            codec=video_meta["codec"],
            has_audio=has_audio and (audio_codec is not None),
        )

        # Warn if FFprobe could not determine dimensions (Stage 0.5 will reject).
        if video_info.width <= 0 or video_info.height <= 0:
            logger.warning(
                f"[{self._STAGE_NAME}] Could not determine video dimensions "
                f"({video_info.width}x{video_info.height}) — "
                f"Stage 0.5 will reject this video."
            )

        logger.info(
            f"[{self._STAGE_NAME}] VideoInfo created: "
            f"id={video_info.video_id}, "
            f"{video_info.width}x{video_info.height}, "
            f"{video_info.original_fps}fps, "
            f"{video_info.duration:.1f}s, "
            f"audio={video_info.has_audio}"
        )

        return video_info

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that the output is a populated VideoInfo.

        Stage 0 is a metadata-extraction stage only; it does NOT hard-filter
        on resolution. A VideoInfo with width/height == 0 (FFprobe could not
        determine dimensions) is still a valid Stage 0 output — Stage 0.5
        Resolution Filter is responsible for rejecting it.

        Args:
            output_data: The result of process().
            context: PipelineContext (unused here).

        Returns:
            True if output_data is a VideoInfo with a non-empty video_id.
        """
        if not isinstance(output_data, VideoInfo):
            return False
        return output_data.video_id != ""

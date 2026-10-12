"""Stage 1: Video / Audio Standardization.

Per AISDD 3.4, this stage converts validated video to a unified format:
    - Video: 25 FPS, standard codec, unified container.
    - Audio: 16kHz, mono WAV.

Input:  VideoInfo (from Stage 0)
Output: StandardizedVideoInfo + AudioInfo
"""

import time
from pathlib import Path
from typing import Any

from face_pipeline_v2.data.models import AudioInfo, StandardizedVideoInfo, VideoInfo
from face_pipeline_v2.pipeline.base_stage import BaseStage
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_01_standardization.ffmpeg_processor import (
    FFmpegProcessor,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
    MissingDependencyError,
    VideoReadError,
)


class Stage1AVStandardization(BaseStage):
    """Stage 1 — AV Standardization using FFmpeg.

    Input:  VideoInfo
    Output: StandardizedVideoInfo + AudioInfo
    """

    _STAGE_NAME: str = "Stage 1 AV Standardization"

    def __init__(self) -> None:
        """Initialize Stage 1."""
        super().__init__(stage_name=self._STAGE_NAME)

    # ------------------------------------------------------------------
    # BaseStage interface
    # ------------------------------------------------------------------

    def validate_input(
        self, input_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that input_data is a VideoInfo with an existing file.

        Args:
            input_data: Expected to be a VideoInfo from Stage 0.
            context: PipelineContext.

        Returns:
            True if input_data is a valid VideoInfo whose video_path exists.
        """
        if not isinstance(input_data, VideoInfo):
            context.logger.warning(
                f"[{self._STAGE_NAME}] Expected VideoInfo, "
                f"got {type(input_data).__name__}"
            )
            return False
        video_path = input_data.video_path
        if not video_path.exists():
            context.logger.error(
                f"[{self._STAGE_NAME}] Video file not found: {video_path}"
            )
            return False
        return True

    def process(
        self, input_data: Any, context: PipelineContext
    ) -> dict[str, Any]:
        """Execute AV standardization and return both output objects.

        Steps (AISDD 3.4.4):
            1. Verify input video exists.
            2. Load FFmpeg processor.
            3. Create output directories from config.
            4. Convert video to 25 FPS standard codec.
            5. Extract audio to 16kHz mono WAV.
            6. Return StandardizedVideoInfo + AudioInfo.

        Args:
            input_data: VideoInfo from Stage 0.
            context: PipelineContext with config and logger.

        Returns:
            Dictionary with keys "video_info" (StandardizedVideoInfo)
            and "audio_info" (AudioInfo).

        Raises:
            MissingDependencyError: If ffmpeg is not found.
            VideoReadError: If the input video cannot be read.
            DataProcessingError: If FFmpeg conversion fails.
        """
        logger = context.logger
        config = context.config
        video_info: VideoInfo = input_data

        # Load stage configuration (AISDD 3.4.6)
        stage_cfg: dict = config.get("stage_01", {})
        target_fps: int = stage_cfg.get("target_fps", 25)
        audio_sample_rate: int = stage_cfg.get("audio_sample_rate", 16000)
        video_codec: str = stage_cfg.get("video_codec", "libx264")
        audio_codec: str = stage_cfg.get("audio_codec", "pcm_s16le")
        ffmpeg_path: str | None = stage_cfg.get("ffmpeg_path")
        output_dir: Path = Path(
            stage_cfg.get("output_dir", "outputs/videos")
        )

        logger.info(
            f"[{self._STAGE_NAME}] Standardizing: {video_info.video_id} "
            f"(source={video_info.original_fps}fps → target={target_fps}fps)"
        )

        start_time = time.monotonic()

        # Step 1: video_path is already validated in validate_input

        # Step 2: Create FFmpeg processor
        processor = FFmpegProcessor(ffmpeg_path=ffmpeg_path)
        logger.debug(
            f"[{self._STAGE_NAME}] FFmpeg resolved: {processor.ffmpeg_path}"
        )

        # Step 3: Build output paths from config
        video_dir = output_dir / video_info.video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        std_video_path = video_dir / f"{video_info.video_id}_std.mp4"
        audio_path = video_dir / f"{video_info.video_id}_audio.wav"

        # Step 4: Convert video
        logger.info(
            f"[{self._STAGE_NAME}] Converting video: {video_info.video_path} "
            f"→ {std_video_path}"
        )
        processor.convert_video(
            input_path=video_info.video_path,
            output_path=std_video_path,
            target_fps=target_fps,
            video_codec=video_codec,
        )

        # Step 5: Extract audio
        logger.info(
            f"[{self._STAGE_NAME}] Extracting audio: {audio_path} "
            f"({audio_sample_rate}Hz {audio_codec})"
        )
        processor.extract_audio(
            input_path=video_info.video_path,
            output_path=audio_path,
            sample_rate=audio_sample_rate,
            audio_codec=audio_codec,
        )

        elapsed = time.monotonic() - start_time

        # Step 6: Build output objects
        std_info = StandardizedVideoInfo(
            video_path=std_video_path,
            audio_path=audio_path,
            target_fps=target_fps,
            resolution=(video_info.width, video_info.height),
            codec=video_codec,
        )
        aud_info = AudioInfo(
            audio_path=audio_path,
            sample_rate=audio_sample_rate,
            duration=video_info.duration,
            channels=1,
        )

        logger.info(
            f"[{self._STAGE_NAME}] Standardization complete in {elapsed:.1f}s: "
            f"video={std_video_path.name}, audio={audio_path.name}"
        )

        return {
            "video_info": std_info,
            "audio_info": aud_info,
        }

    def validate_output(
        self, output_data: Any, context: PipelineContext
    ) -> bool:
        """Validate that output contains both StandardizedVideoInfo and AudioInfo.

        Args:
            output_data: The result of process().
            context: PipelineContext.

        Returns:
            True if both output objects are valid and their files exist.
        """
        if not isinstance(output_data, dict):
            return False
        std_info = output_data.get("video_info")
        aud_info = output_data.get("audio_info")
        if not isinstance(std_info, StandardizedVideoInfo):
            return False
        if not isinstance(aud_info, AudioInfo):
            return False
        if not std_info.video_path.exists():
            context.logger.error(
                f"[{self._STAGE_NAME}] Standardized video not found: "
                f"{std_info.video_path}"
            )
            return False
        if not aud_info.audio_path.exists():
            context.logger.error(
                f"[{self._STAGE_NAME}] Audio file not found: "
                f"{aud_info.audio_path}"
            )
            return False
        return True

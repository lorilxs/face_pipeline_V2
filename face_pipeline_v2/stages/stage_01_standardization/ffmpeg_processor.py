"""FFmpeg wrapper for video/audio standardization.

Per AISDD 3.4, Stage 1 uses FFmpeg to:
    - Convert video to standard FPS (25).
    - Extract and resample audio (16kHz).
    - Unify codec and container format.

Uses subprocess — never OpenCV — for all media operations.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class FFmpegProcessor:
    """FFmpeg调用封装 — standardizes video and audio via FFmpeg CLI.

    Usage:
        proc = FFmpegProcessor()
        proc.convert_video(Path("in.mp4"), Path("out.mp4"), fps=25, codec="h264")
        proc.extract_audio(Path("in.mp4"), Path("out.wav"), sample_rate=16000)
    """

    _FFMPEG_CANDIDATES: tuple = ("ffmpeg",)

    def __init__(self, ffmpeg_path: Optional[str] = None) -> None:
        """Initialize FFmpegProcessor.

        Args:
            ffmpeg_path: Explicit path to ffmpeg executable.
                         If None, auto-detect from PATH.
        """
        self._ffmpeg_path: str = ffmpeg_path or self._detect_ffmpeg()

    @property
    def ffmpeg_path(self) -> str:
        """Return the resolved ffmpeg executable path."""
        return self._ffmpeg_path

    def _detect_ffmpeg(self) -> str:
        """Auto-detect ffmpeg executable from system PATH.

        Returns:
            Path to the found ffmpeg executable.

        Raises:
            MissingDependencyError: If ffmpeg is not found.
        """
        # Deferred import to avoid circular dependency at module level
        from face_pipeline_v2.utils.exceptions import MissingDependencyError

        for candidate in self._FFMPEG_CANDIDATES:
            resolved = shutil.which(candidate)
            if resolved is not None:
                return resolved
        raise MissingDependencyError(
            "FFmpeg not found on system PATH. "
            "Install ffmpeg package or set ffmpeg_path in config."
        )

    # ------------------------------------------------------------------
    # Video conversion
    # ------------------------------------------------------------------

    def convert_video(
        self,
        input_path: Path,
        output_path: Path,
        *,
        target_fps: int,
        video_codec: str = "libx264",
        pixel_format: str = "yuv420p",
    ) -> Path:
        """Convert video to standardized format via FFmpeg.

        Args:
            input_path: Source video file.
            output_path: Destination path for standardized video.
            target_fps: Target frames per second (e.g. 25).
            video_codec: FFmpeg codec name (e.g. "libx264").
            pixel_format: Output pixel format.

        Returns:
            Path to the standardized output video.

        Raises:
            VideoReadError: If input_path does not exist.
            DataProcessingError: If FFmpeg fails.
        """
        from face_pipeline_v2.utils.exceptions import (
            DataProcessingError,
            VideoReadError,
        )

        if not input_path.exists():
            raise VideoReadError(
                f"Input video not found: {input_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_video_command(
            input_path=input_path,
            output_path=output_path,
            target_fps=target_fps,
            video_codec=video_codec,
            pixel_format=pixel_format,
        )

        self._run_ffmpeg(cmd, input_path)

        if not output_path.exists():
            raise DataProcessingError(
                f"FFmpeg completed but output file not found: {output_path}"
            )

        return output_path

    def _build_video_command(
        self,
        input_path: Path,
        output_path: Path,
        target_fps: int,
        video_codec: str,
        pixel_format: str,
    ) -> list:
        """Build the FFmpeg command line for video standardization.

        Args:
            input_path: Source video file.
            output_path: Target output path.
            target_fps: Target FPS.
            video_codec: Codec name.
            pixel_format: Pixel format.

        Returns:
            List of command-line arguments for subprocess.
        """
        return [
            self._ffmpeg_path,
            "-y",                          # overwrite output
            "-i", str(input_path),
            "-r", str(target_fps),         # target FPS
            "-c:v", video_codec,           # video codec
            "-pix_fmt", pixel_format,      # pixel format
            "-an",                         # strip audio from video stream
            str(output_path),
        ]

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def extract_audio(
        self,
        input_path: Path,
        output_path: Path,
        *,
        sample_rate: int,
        audio_codec: str = "pcm_s16le",
        channels: int = 1,
    ) -> Path:
        """Extract and resample audio from video via FFmpeg.

        Args:
            input_path: Source video file.
            output_path: Destination path for extracted audio (WAV).
            sample_rate: Target sample rate in Hz (e.g. 16000).
            audio_codec: FFmpeg audio codec (e.g. "pcm_s16le").
            channels: Number of audio channels (1 = mono).

        Returns:
            Path to the extracted audio file.

        Raises:
            VideoReadError: If input_path does not exist.
            DataProcessingError: If FFmpeg fails.
        """
        from face_pipeline_v2.utils.exceptions import (
            DataProcessingError,
            VideoReadError,
        )

        if not input_path.exists():
            raise VideoReadError(
                f"Input video not found for audio extraction: {input_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-vn",                         # strip video
            "-acodec", audio_codec,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_path),
        ]

        self._run_ffmpeg(cmd, input_path)

        if not output_path.exists():
            raise DataProcessingError(
                f"FFmpeg completed but audio output not found: {output_path}"
            )

        return output_path

    # ------------------------------------------------------------------
    # Common
    # ------------------------------------------------------------------

    def _run_ffmpeg(self, cmd: list, input_path: Path) -> None:
        """Execute an FFmpeg command and handle errors.

        Args:
            cmd: Full FFmpeg command line as a list.
            input_path: The input file (for error context).

        Raises:
            DataProcessingError: If FFmpeg exits with non-zero code.
        """
        from face_pipeline_v2.utils.exceptions import DataProcessingError

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max for long videos
        )

        if result.returncode != 0:
            raise DataProcessingError(
                f"FFmpeg failed on '{input_path}': "
                f"stderr={result.stderr.strip()[:500]}"
            )

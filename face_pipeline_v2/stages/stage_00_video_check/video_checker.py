"""FFprobe wrapper for video integrity checking.

Per AISDD 3.3, Stage 0 uses FFprobe (not OpenCV) to:
    - Check video file validity.
    - Detect Video Stream and Audio Stream presence.
    - Extract basic video metadata (fps, resolution, duration, codec, frame count).
"""

import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from face_pipeline_v2.utils.exceptions import (
    MetadataExtractionError,
    MissingDependencyError,
    MissingStreamError,
    VideoReadError,
)


class FFprobeChecker:
    """FFprobe调用封装 — Reads video metadata via ffprobe JSON output.

    Usage:
        checker = FFprobeChecker()
        metadata = checker.probe(Path("/data/video.mp4"))
        has_video = checker.has_video_stream(metadata)
        has_audio = checker.has_audio_stream(metadata)
    """

    # Well-known ffprobe executable names
    _FFPROBE_CANDIDATES: tuple = ("ffprobe", "ffprobe-static")

    def __init__(self, ffprobe_path: Optional[str] = None) -> None:
        """Initialize FFprobeChecker.

        Args:
            ffprobe_path: Explicit path to ffprobe executable.
                          If None, auto-detect from PATH.

        Raises:
            MissingDependencyError: If ffprobe is not found on the system.
        """
        self._ffprobe_path: str = ffprobe_path or self._detect_ffprobe()

    @property
    def ffprobe_path(self) -> str:
        """Return the resolved ffprobe executable path."""
        return self._ffprobe_path

    def _detect_ffprobe(self) -> str:
        """Auto-detect ffprobe executable from system PATH.

        Returns:
            Path to the found ffprobe executable.

        Raises:
            MissingDependencyError: If ffprobe is not found on the system.
        """
        for candidate in self._FFPROBE_CANDIDATES:
            resolved = shutil.which(candidate)
            if resolved is not None:
                return resolved
        raise MissingDependencyError(
            "FFprobe not found on system PATH. "
            "Install ffmpeg package or set ffprobe_path in config."
        )

    def probe(self, video_path: Path) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON metadata.

        Args:
            video_path: Path to the video file.

        Returns:
            Parsed ffprobe JSON output as a dictionary.

        Raises:
            VideoReadError: If video_path does not exist or ffprobe fails.
            MetadataExtractionError: If ffprobe JSON output cannot be parsed.
        """
        if not video_path.exists():
            raise VideoReadError(f"Video file not found: {video_path}")

        cmd = [
            self._ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise VideoReadError(
                f"FFprobe failed on '{video_path}': {result.stderr.strip()}"
            )

        try:
            metadata: Dict[str, Any] = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MetadataExtractionError(
                f"Failed to parse ffprobe JSON output for '{video_path}'"
            ) from exc

        return metadata

    @staticmethod
    def has_video_stream(metadata: Dict[str, Any]) -> bool:
        """Check if the ffprobe metadata contains a video stream.

        Args:
            metadata: Parsed ffprobe JSON output.

        Returns:
            True if at least one video stream exists.
        """
        streams: list = metadata.get("streams", [])
        return any(s.get("codec_type") == "video" for s in streams)

    @staticmethod
    def has_audio_stream(metadata: Dict[str, Any]) -> bool:
        """Check if the ffprobe metadata contains an audio stream.

        Args:
            metadata: Parsed ffprobe JSON output.

        Returns:
            True if at least one audio stream exists.
        """
        streams: list = metadata.get("streams", [])
        return any(s.get("codec_type") == "audio" for s in streams)

    @staticmethod
    def extract_video_stream_metadata(
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key fields from the first video stream.

        Args:
            metadata: Parsed ffprobe JSON output.

        Returns:
            Dictionary with fps, width, height, codec, duration, nb_frames.

        Raises:
            MissingStreamError: If no video stream is found in metadata.
        """
        streams: list = metadata.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") != "video":
                continue

            # Parse frame rate (stored as fraction string "25/1" or "30000/1001")
            fps_str: str = stream.get("r_frame_rate", "0/1")
            fps: float = 0.0
            if "/" in fps_str:
                parts = fps_str.split("/")
                if int(parts[1]) != 0:
                    fps = float(parts[0]) / float(parts[1])

            # Parse frame count
            nb_frames_raw: str = stream.get("nb_frames", "0")
            nb_frames: int = int(nb_frames_raw) if nb_frames_raw.isdigit() else 0

            return {
                "fps": fps,
                "width": stream.get("width", 0),
                "height": stream.get("height", 0),
                "codec": stream.get("codec_name", "unknown"),
                "duration": float(stream.get("duration", 0.0)),
                "nb_frames": nb_frames,
            }

        raise MissingStreamError(
            "No video stream found in ffprobe metadata."
        )

    @staticmethod
    def extract_audio_stream_codec(metadata: Dict[str, Any]) -> Optional[str]:
        """Extract the codec name of the first audio stream, if any.

        Args:
            metadata: Parsed ffprobe JSON output.

        Returns:
            Audio codec name string, or None if no audio stream.
        """
        streams: list = metadata.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") == "audio":
                return stream.get("codec_name")
        return None

    @staticmethod
    def extract_format_duration(metadata: Dict[str, Any]) -> float:
        """Extract container-level duration from ffprobe format section.

        Args:
            metadata: Parsed ffprobe JSON output.

        Returns:
            Duration in seconds as float.
        """
        fmt: dict = metadata.get("format", {})
        return float(fmt.get("duration", 0.0))

    # ------------------------------------------------------------------
    # ffprobe availability check + ffmpeg fallback
    # ------------------------------------------------------------------

    def _check_ffprobe_available(self) -> bool:
        """Check that the configured ffprobe is a REAL ffprobe binary.

        Some environments ship a stripped-down ffmpeg (built with
        ``--disable-ffprobe``) under the name ``ffprobe``. That binary
        reports itself as "ffmpeg version ..." and fails on ``-print_format``.
        Merely checking that the file exists is not sufficient.

        Returns:
            True if ``ffprobe -version`` reports itself as ffprobe.
        """
        try:
            result = subprocess.run(
                [self._ffprobe_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        output = (result.stdout or "").lower()
        return result.returncode == 0 and "ffprobe" in output

    @staticmethod
    def probe_with_ffmpeg(video_path: Path) -> Dict[str, Any]:
        """Extract video metadata by parsing ``ffmpeg -i`` stderr output.

        Used as a fallback when ffprobe is unavailable or invalid. Returns a
        dict shaped like ffprobe JSON (``{"streams": [...], "format": {...}}``)
        so the existing ``extract_*`` methods work unchanged.

        Args:
            video_path: Path to the video file.

        Returns:
            Metadata dict in ffprobe-JSON shape.

        Raises:
            VideoReadError: If the file is missing or no dimensions could be
                            extracted (both ffprobe and ffmpeg failed).
        """
        if not video_path.exists():
            raise VideoReadError(f"Video file not found: {video_path}")

        result = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stderr = result.stderr or ""

        width = height = 0
        fps = 0.0
        duration = 0.0
        codec = "unknown"
        has_audio = False
        audio_codec: Optional[str] = None

        for line in stderr.split("\n"):
            if "Duration:" in line:
                m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", line)
                if m:
                    duration = (
                        int(m.group(1)) * 3600
                        + int(m.group(2)) * 60
                        + float(m.group(3))
                    )
            if "Stream #" in line and "Video:" in line:
                # Resolution appears as ", 640x360 [SAR ..." — the leading comma
                # disambiguates it from stream indices like "[0x1]" or hex codec
                # tags like "avc1 / 0x31637661".
                if m := re.search(r",\s*(\d+)x(\d+)", line):
                    width, height = int(m.group(1)), int(m.group(2))
                if m := re.search(r"([\d.]+)\s*fps", line):
                    fps = float(m.group(1))
                if m := re.search(r"Video:\s*(\w+)", line):
                    codec = m.group(1)
            if "Stream #" in line and "Audio:" in line:
                has_audio = True
                if m := re.search(r"Audio:\s*(\w+)", line):
                    audio_codec = m.group(1)

        if width <= 0 or height <= 0:
            raise VideoReadError(
                f"FFmpeg fallback could not extract dimensions from '{video_path}'"
            )

        # Store fps as a fraction string so extract_video_stream_metadata()
        # (which parses "num/den") resolves it correctly.
        fps_frac = Fraction(fps).limit_denominator(1000) if fps > 0 else Fraction(0)
        r_frame_rate = f"{fps_frac.numerator}/{fps_frac.denominator}"

        streams: list = [
            {
                "codec_type": "video",
                "codec_name": codec,
                "r_frame_rate": r_frame_rate,
                "width": width,
                "height": height,
                "duration": duration,
                "nb_frames": "0",
            },
        ]
        if has_audio:
            streams.append({
                "codec_type": "audio",
                "codec_name": audio_codec or "unknown",
                "duration": duration,
            })

        return {"streams": streams, "format": {"duration": duration}}

    def probe_or_fallback(self, video_path: Path) -> Tuple[Dict[str, Any], str]:
        """Probe video metadata via ffprobe, falling back to ffmpeg.

        Steps:
            1. If the configured ffprobe is a real ffprobe and succeeds,
               return its JSON metadata (source="ffprobe").
            2. Otherwise, parse ``ffmpeg -i`` stderr (source="ffmpeg").

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (metadata dict, source) where source is "ffprobe" or
            "ffmpeg".

        Raises:
            VideoReadError: If the file is missing or both backends fail.
        """
        if not video_path.exists():
            raise VideoReadError(f"Video file not found: {video_path}")

        if self._check_ffprobe_available():
            try:
                return self.probe(video_path), "ffprobe"
            except (VideoReadError, MetadataExtractionError, subprocess.SubprocessError):
                pass  # fall through to ffmpeg fallback

        return self.probe_with_ffmpeg(video_path), "ffmpeg"

"""Video utility functions for Face_Pipeline_V2.

Provides shared video operations used across Stages:
    - Segment extraction via FFmpeg (used by Stage 3 debug export, etc.)
"""

import subprocess
from pathlib import Path
from typing import Optional

from face_pipeline_v2.utils.exceptions import DataProcessingError, VideoReadError


def cut_video_segment(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    """Extract a video segment by time range using FFmpeg.

    Re-encodes (no stream copy) to guarantee frame-accurate cutting,
    independent of keyframe positions in the source video.
    Suitable for debug exports and scene visualization.

    Args:
        input_path: Source video file.
        output_path: Destination path for the extracted segment.
        start_time: Segment start time in seconds.
        end_time: Segment end time in seconds.
        ffmpeg_path: Optional explicit path to ffmpeg. Auto-detected if None.

    Returns:
        Path to the extracted segment file.

    Raises:
        VideoReadError: If input_path does not exist.
        DataProcessingError: If FFmpeg fails or output not created.
    """
    import shutil

    if not input_path.exists():
        raise VideoReadError(
            f"Input video not found: {input_path}"
        )

    duration = end_time - start_time
    if duration <= 0:
        raise DataProcessingError(
            f"Invalid segment range: start={start_time}, end={end_time}, "
            f"duration={duration}"
        )

    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DataProcessingError(
            "FFmpeg not found on system PATH. Install ffmpeg to use video utilities."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -ss before -i for fast input seeking (decoder discards pre-start frames
    # for accuracy); re-encode instead of -c copy for frame-precise cuts
    # regardless of keyframe positions.
    cmd = [
        ffmpeg,
        "-y",
        "-ss", str(start_time),
        "-i", str(input_path),
        "-t", str(duration),
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise DataProcessingError(
            f"FFmpeg segment extraction failed: "
            f"stderr={result.stderr.strip()[:500]}"
        )

    if not output_path.exists():
        raise DataProcessingError(
            f"FFmpeg completed but segment not found: {output_path}"
        )

    return output_path

"""Resolution filter processor for Stage 0.5.

Pure rule-based logic — no model, no IO. Applies the minimum-resolution
hard filter to a VideoInfo and produces a ResolutionValidationInfo.
"""

from typing import Optional

from face_pipeline_v2.data.models import ResolutionValidationInfo, VideoInfo

REJECT_REASON_RESOLUTION_TOO_LOW: str = "resolution_too_low"


def validate_resolution(
    video_info: VideoInfo,
    min_width: int,
    min_height: int,
) -> ResolutionValidationInfo:
    """Validate video resolution against minimum thresholds.

    Args:
        video_info: VideoInfo from Stage 0 (carries width/height).
        min_width: Minimum acceptable width in pixels.
        min_height: Minimum acceptable height in pixels.

    Returns:
        ResolutionValidationInfo with the pass/fail decision. A video is
        rejected if EITHER dimension is below its minimum.
    """
    is_valid: bool = (
        video_info.width >= min_width and video_info.height >= min_height
    )
    reject_reason: Optional[str] = (
        None if is_valid else REJECT_REASON_RESOLUTION_TOO_LOW
    )

    return ResolutionValidationInfo(
        width=video_info.width,
        height=video_info.height,
        min_width=min_width,
        min_height=min_height,
        is_valid=is_valid,
        reject_reason=reject_reason,
    )

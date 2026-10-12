"""Sliding-window clip generator for Stage 4.

Pure logic — no model, no video I/O, no face detection, no audio.

Given a SceneInfo (with filter_result.passed=True), window_size, stride,
and target_fps, this module produces a list of ClipInfo objects via
deterministic sliding-window partitioning.

Stage 4 also supports a `segment` mode: each SceneInfo maps 1:1 to a single
full-length ClipInfo (no resampling, no fixed window).
"""

from typing import List

from face_pipeline_v2.data.models import ClipInfo, SceneInfo
from face_pipeline_v2.utils.exceptions import DataProcessingError


def _build_clip(
    scene: SceneInfo,
    start_time: float,
    end_time: float,
    target_fps: float,
    index: int,
) -> ClipInfo:
    """Build a single ClipInfo covering [start_time, end_time) of a scene."""
    clip_duration = end_time - start_time
    clip_start_frame = int(round(start_time * target_fps))
    clip_end_frame = int(round(end_time * target_fps))
    frame_count = clip_end_frame - clip_start_frame
    clip_id = f"{scene.scene_id}_clip_{index:04d}"
    return ClipInfo(
        clip_id=clip_id,
        scene_id=scene.scene_id,
        video_id=scene.video_id,
        start_frame=clip_start_frame,
        end_frame=clip_end_frame,
        start_time=start_time,
        end_time=end_time,
        frame_count=frame_count,
        duration=clip_duration,
    )


def generate_clips(
    scene: SceneInfo,
    window_size: float,
    stride: float,
    target_fps: float,
    keep_tail: bool = False,
    min_tail_duration: float = 0.0,
) -> List[ClipInfo]:
    """Generate ClipInfo list from a single SceneInfo using sliding window.

    By default only complete windows are created — the tail shorter than
    window_size is discarded, and a scene shorter than one full window is
    dropped entirely.

    With ``keep_tail=True`` the trailing remainder is preserved as a shorter
    final clip, and a scene shorter than one full window is kept whole (as a
    single clip) instead of being dropped.

    Args:
        scene: Source SceneInfo (must have filter_result.passed=True).
        window_size: Target clip duration in seconds (e.g. 8.0).
        stride: Step between consecutive clip start times in seconds (e.g. 4.0).
        target_fps: Frames per second of the standardized video (e.g. 25.0).
        keep_tail: When True, keep the trailing remainder (shorter than
            ``window_size``) as one final clip, and keep a sub-window scene in
            full.  Intended for non-overlapping chunking
            (``stride == window_size``); with ``stride < window_size`` the
            trailing clip may overlap prior windows.
        min_tail_duration: Minimum duration in seconds for a trailing
            remainder clip to be kept when ``keep_tail=True``.  Tails strictly
            shorter than this are discarded.  Does not affect sub-window scenes
            kept whole.  Default 0.0 keeps every non-empty tail.

    Returns:
        List of ClipInfo objects in chronological order.

    Raises:
        DataProcessingError: If window_size <= 0, stride <= 0, or
            min_tail_duration < 0.
    """
    if window_size <= 0:
        raise DataProcessingError(
            f"window_size must be positive, got {window_size}"
        )
    if stride <= 0:
        raise DataProcessingError(
            f"stride must be positive, got {stride}"
        )
    if min_tail_duration < 0:
        raise DataProcessingError(
            f"min_tail_duration must be non-negative, got {min_tail_duration}"
        )

    scene_start = scene.start_time
    scene_end = scene.end_time
    scene_duration = scene.duration

    # Guard: degenerate scene (no duration)
    if scene_duration <= 0:
        return []

    # Scene shorter than one full window.
    if scene_duration < window_size:
        if keep_tail:
            return [_build_clip(scene, scene_start, scene_end, target_fps, 0)]
        return []

    clips: List[ClipInfo] = []
    index = 0

    # Sliding window: advance by stride, keep complete windows
    current_start = scene_start
    while current_start + window_size <= scene_end:
        clips.append(
            _build_clip(
                scene,
                current_start,
                current_start + window_size,
                target_fps,
                index,
            )
        )
        index += 1
        current_start += stride

    # Optionally keep the trailing remainder as a partial final clip,
    # provided it is at least min_tail_duration long.
    if keep_tail and current_start + 1e-9 < scene_end:
        tail_duration = scene_end - current_start
        if tail_duration >= min_tail_duration:
            clips.append(
                _build_clip(scene, current_start, scene_end, target_fps, index)
            )

    return clips


def generate_segment_clips(
    scene: SceneInfo,
    target_fps: float,
) -> List[ClipInfo]:
    """Generate a single full-length ClipInfo from a SceneInfo (segment mode).

    One-to-one adapter: each input SceneInfo produces exactly ONE ClipInfo
    covering the entire segment. No resampling and no fixed window.

    Timestamps and frame indices are preserved verbatim from the SceneInfo:
        - start_time / end_time / duration  →  copied from the scene
        - start_frame / end_frame  →  copied from the scene (full range)
        - frame_count  →  end_frame - start_frame (complete frame coverage)

    Args:
        scene: Source SceneInfo (must have filter_result.passed=True).
        target_fps: Frames per second of the standardized video (e.g. 25.0).
            Kept for API symmetry with generate_clips; the segment adapter
            uses the scene's own frame range and does not recompute frames.

    Returns:
        List containing exactly one ClipInfo (chronologically aligned with the
        scene). Returns an empty list if the scene has no frames / no duration
        (degenerate segment).

    Raises:
        DataProcessingError: If target_fps <= 0.
    """
    if target_fps <= 0:
        raise DataProcessingError(
            f"target_fps must be positive, got {target_fps}"
        )

    scene_duration = scene.duration
    frame_count = scene.end_frame - scene.start_frame

    # Guard: degenerate segment (no duration / no frames) → no clip.
    if scene_duration <= 0 or frame_count <= 0:
        return []

    # Stable clip id: exactly one clip per scene in segment mode.
    clip_id = f"{scene.scene_id}_clip_0000"

    return [
        ClipInfo(
            clip_id=clip_id,
            scene_id=scene.scene_id,
            video_id=scene.video_id,
            start_frame=scene.start_frame,
            end_frame=scene.end_frame,
            start_time=scene.start_time,
            end_time=scene.end_time,
            frame_count=frame_count,
            duration=scene_duration,
        )
    ]

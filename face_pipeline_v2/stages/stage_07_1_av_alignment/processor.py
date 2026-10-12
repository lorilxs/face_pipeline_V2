"""Audio alignment processor for Stage 7.1.

Pure logic — no model, no context, no I/O beyond ffmpeg subprocess call.

SyncNet offset convention:
  - offset > 0: audio is AHEAD of video → DELAY audio (add silence at start)
  - offset < 0: audio is BEHIND video  → ADVANCE audio (trim from start)
  - offset = 0: aligned → no action

Reference: SyncNetInstance.evaluate() computes offset = vshift - minidx,
where minidx is the index of min distance in the 21-offset window.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AlignmentConfig:
    """Correction thresholds.

    correction_threshold_ms: from stage_07_1_av_alignment (Stage 7.1 own param).
    max_offset_ms: derived from sync.max_offset_frames / fps * 1000 (unified).
    """

    correction_threshold_ms: float = 50.0
    max_offset_ms: float = 600.0


def compute_offset_ms(offset_frames: int, fps: float) -> float:
    """Convert frame offset to milliseconds.

    Args:
        offset_frames: SyncNet offset in frames.
        fps: Video frame rate.

    Returns:
        Offset in milliseconds.
    """
    return offset_frames / fps * 1000.0


# ---------------------------------------------------------------------------
# Alignment states (three explicit cases)
# ---------------------------------------------------------------------------

ALREADY_ALIGNED = "already_aligned"
CORRECTED = "corrected"
SEVERE_OFFSET_UNAVAILABLE = "severe_offset_unavailable"


def classify_alignment(
    offset_frames: int,
    fps: float,
    cfg: AlignmentConfig,
) -> str:
    """Classify a clip's A/V offset into one of three alignment states.

    Three explicit cases (NOT a boolean):
        1. already_aligned            — |offset| < correction_threshold_ms
            无需修正，裁剪原始 clip audio 即可。
        2. corrected                  — correction_threshold_ms <= |offset| <= max_offset_ms
            执行 offset shift 校正。
        3. severe_offset_unavailable  — |offset| > max_offset_ms
            严重音画错位，无法可靠修正，不生成可用音频。

    Args:
        offset_frames: SyncNet offset in frames.
        fps: Video frame rate.
        cfg: AlignmentConfig with thresholds.

    Returns:
        One of ALREADY_ALIGNED / CORRECTED / SEVERE_OFFSET_UNAVAILABLE.
    """
    abs_ms = abs(compute_offset_ms(offset_frames, fps))
    if abs_ms < cfg.correction_threshold_ms:
        return ALREADY_ALIGNED
    if abs_ms > cfg.max_offset_ms:
        return SEVERE_OFFSET_UNAVAILABLE
    return CORRECTED


def should_correct(offset_frames: int, fps: float, cfg: AlignmentConfig) -> bool:
    """Backward-compat wrapper — True only for the CORRECTED state.

    Kept for any external callers. New code should use classify_alignment().
    """
    return classify_alignment(offset_frames, fps, cfg) == CORRECTED


def align_audio(
    input_audio: Path,
    output_audio: Path,
    offset_frames: int,
    fps: float,
    clip_start: Optional[float] = None,
    clip_duration: Optional[float] = None,
) -> Path:
    """Align audio timeline to video, optionally trim to clip boundaries.

    SyncNet offset convention:
      offset > 0 → audio is AHEAD → DELAY audio (add silence at start)
      offset < 0 → audio is BEHIND → TRIM audio from start

    If clip_start and clip_duration are provided, the output is trimmed
    to [clip_start, clip_start + clip_duration] after offset correction.
    This produces per-clip corrected audio segments.

    Args:
        input_audio: Path to the original standardized audio (Stage 1).
        output_audio: Destination path for the corrected audio.
        offset_frames: SyncNet offset in frames.
        fps: Video frame rate.
        clip_start: Optional — clip start time in seconds for trimming.
        clip_duration: Optional — clip duration in seconds for trimming.

    Returns:
        Path to the corrected (and optionally trimmed) audio file.
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    offset_sec = abs(offset_frames) / fps

    if offset_frames > 0:
        # Audio ahead of video → delay audio, then trim to clip
        delay_ms = int(compute_offset_ms(offset_frames, fps))
        if clip_start is not None and clip_duration is not None:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{clip_start:.6f}",
                "-t", f"{clip_duration:.6f}",
                "-i", str(input_audio),
                "-filter:a", f"adelay={delay_ms}|{delay_ms}",
                "-ac", "1", "-ar", "16000",
                str(output_audio),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_audio),
                "-filter:a", f"adelay={delay_ms}|{delay_ms}",
                "-ac", "1", "-ar", "16000",
                str(output_audio),
            ]
    elif offset_frames < 0:
        # Audio behind video → trim from start, then trim to clip
        trim_start = offset_sec
        if clip_start is not None:
            trim_start += clip_start
        duration_arg = ["-t", f"{clip_duration:.6f}"] if clip_duration is not None else []
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{trim_start:.6f}",
            *duration_arg,
            "-i", str(input_audio),
            "-ac", "1", "-ar", "16000",
            str(output_audio),
        ]
    else:
        # offset == 0 → trim to clip or copy
        if clip_start is not None and clip_duration is not None:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{clip_start:.6f}",
                "-t", f"{clip_duration:.6f}",
                "-i", str(input_audio),
                "-ac", "1", "-ar", "16000",
                str(output_audio),
            ]
        else:
            shutil.copy2(str(input_audio), str(output_audio))
            return output_audio

    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    return output_audio


def extract_clip_audio(
    input_audio: Path,
    output_audio: Path,
    clip_start: float,
    clip_duration: float,
) -> Path:
    """Extract a clip-level audio segment without any offset correction.

    Used when |offset| <= correction_threshold — audio is considered
    already aligned, but we still produce a per-clip audio asset.

    Args:
        input_audio: Path to the original standardized audio (Stage 1).
        output_audio: Destination path for the clip audio.
        clip_start: Clip start time in seconds.
        clip_duration: Clip duration in seconds.

    Returns:
        Path to the extracted clip audio file.
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{clip_start:.6f}",
        "-t", f"{clip_duration:.6f}",
        "-i", str(input_audio),
        "-ac", "1", "-ar", "16000",
        str(output_audio),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    return output_audio


def shift_clip_audio(
    input_clip_audio: Path,
    output_audio: Path,
    offset_frames: int,
    fps: float,
    clip_duration: float,
) -> Path:
    """Apply an offset shift to an already-extracted CLIP audio, keeping its
    duration exactly equal to ``clip_duration``.

    This is the Stage 7.1 offset-correction step for clip-level audio.  It
    operates ONLY on the per-clip audio segment (never on the full video
    audio).  SyncNet convention:

      offset > 0: audio is AHEAD of video → DELAY audio.
                  Prepend ``offset`` ms of silence, drop the tail that no
                  longer fits in the window → output stays clip_duration.
      offset < 0: audio is BEHIND video → ADVANCE audio.
                  Drop ``offset`` from the start, pad silence at the end →
                  output stays clip_duration.
      offset = 0: copy unchanged.

    Args:
        input_clip_audio: Path to the extracted clip-level audio (raw, 16k mono).
        output_audio: Destination path for the shifted clip audio.
        offset_frames: SyncNet offset in frames.
        fps: Frame rate (per-clip fps from AVSyncInfo when available).
        clip_duration: Target clip duration in seconds; the output duration
                       is guaranteed to equal this value.

    Returns:
        Path to the shifted clip audio file.
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    dur = float(clip_duration)

    if offset_frames == 0:
        shutil.copy2(str(input_clip_audio), str(output_audio))
        return output_audio

    offset_sec = abs(offset_frames) / fps

    if offset_frames > 0:
        # Delay: silence at head, crop tail → duration preserved.
        delay_ms = int(round(offset_sec * 1000))
        audio_filter = (
            f"adelay={delay_ms}|{delay_ms},"
            f"atrim=0:{dur:.6f},asetpts=PTS-STARTPTS"
        )
    else:
        # Advance: drop head, pad silence at tail → duration preserved.
        audio_filter = (
            f"atrim={offset_sec:.6f}:,asetpts=PTS-STARTPTS,"
            f"apad,atrim=0:{dur:.6f}"
        )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_clip_audio),
        "-filter:a", audio_filter,
        "-ac", "1", "-ar", "16000",
        str(output_audio),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    return output_audio

"""Dataset-mode pipeline adapter (orchestration layer).

Allows skipping the quality-filtering stages (Stage 6 / 7 / 7.1 / 8 / 9)
for existing high-quality datasets.  Downstream stages (10 / 11 / 12 / 13 /
14 / 17) still receive the exact inputs Stage 9 would normally supply:

  * ``accepted_clip_ids`` for Stage 10  -> every valid Stage 5 clip
                                          (no quality filtering).
  * ``AVAlignmentInfo`` for Stage 14    -> the original clip wav extracted
                                          from the Stage 1 standardized
                                          audio (no SyncNet alignment;
                                          offset = 0 / already_aligned).

This module is an ADAPTER only.  It reuses Stage 7.1's clip-audio extraction
helper (``extract_clip_audio``) and builds ``AVAlignmentInfo`` records without
modifying any stage implementation or the Stage 14 audio-extraction algorithm.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Set

from face_pipeline_v2.data.models import (
    AVAlignmentInfo,
    ClipInfo,
    FaceSequence,
    StandardizedVideoInfo,
)
from face_pipeline_v2.stages.stage_07_1_av_alignment.processor import (
    extract_clip_audio,
)

# ---------------------------------------------------------------------------
# Pipeline modes
# ---------------------------------------------------------------------------

OPEN_DOMAIN_MODE = "open_domain"
"""Full Stage0-17 pipeline with quality filtering (default)."""

DATASET_MODE = "dataset"
"""Dataset pipeline: Stage0-5 -> Stage10-17, skipping quality stages."""

PIPELINE_MODES: frozenset = frozenset({OPEN_DOMAIN_MODE, DATASET_MODE})
"""Allowed values for the ``pipeline_mode`` config key."""

DEFAULT_PIPELINE_MODE: str = OPEN_DOMAIN_MODE
"""Backward-compatible default when ``pipeline_mode`` is absent."""


def resolve_pipeline_mode(config: dict) -> str:
    """Resolve and validate the configured pipeline mode.

    Args:
        config: Loaded pipeline configuration dict.

    Returns:
        ``"open_domain"`` (default) or ``"dataset"``.

    Raises:
        ValueError: If ``pipeline_mode`` is present but not an allowed value.
    """
    mode: str = config.get("pipeline_mode", DEFAULT_PIPELINE_MODE)
    if mode not in PIPELINE_MODES:
        raise ValueError(
            f"Invalid pipeline_mode={mode!r}. Allowed values: "
            f"{sorted(PIPELINE_MODES)}."
        )
    return mode


# ---------------------------------------------------------------------------
# Stage 10 adapter — accepted clip ids
# ---------------------------------------------------------------------------


def collect_accepted_clip_ids(
    face_sequences: Sequence[FaceSequence],
) -> Set[str]:
    """Collect the set of valid Stage 5 clip ids (no quality filtering).

    In dataset mode every clip that produced a Stage 5 face sequence is
    accepted — the ``accepted_clip_ids`` that Stage 9 would normally supply
    after filtering.

    Args:
        face_sequences: Stage 5 (post 5.1/5.2/5.3 passthrough) sequences.

    Returns:
        Set of clip ids, one per face sequence.
    """
    return {seq.clip_id for seq in face_sequences if seq.clip_id}


# ---------------------------------------------------------------------------
# Stage 14 adapter — audio alignment records
# ---------------------------------------------------------------------------


def _make_dataset_alignment(
    clip: ClipInfo,
    std_info: StandardizedVideoInfo,
    output_dir: Path,
) -> AVAlignmentInfo:
    """Build an ``already_aligned`` ``AVAlignmentInfo`` from the original
    clip wav (offset = 0, no SyncNet correction).

    Dataset mode has no SyncNet alignment (Stage 7 / 7.1 are skipped), so the
    clip-level audio extracted from the Stage 1 standardized audio IS the
    aligned audio.  Reuses ``extract_clip_audio`` exactly like Stage 7.1 does
    for the ``already_aligned`` case.

    Args:
        clip: Stage 4 clip whose audio window to extract.
        std_info: Stage 1 standardized video/audio info (source audio).
        output_dir: Directory to write the per-clip wav files.

    Returns:
        An ``AVAlignmentInfo`` whose ``aligned_audio_path`` points to the
        existing original clip wav.
    """
    clip_dir = output_dir / clip.clip_id
    clip_wav = clip_dir / f"{clip.clip_id}.wav"
    extract_clip_audio(
        input_audio=std_info.audio_path,
        output_audio=clip_wav,
        clip_start=clip.start_time,
        clip_duration=clip.duration,
    )
    return AVAlignmentInfo(
        clip_id=clip.clip_id,
        original_audio_path=std_info.audio_path,
        offset_frames=0,
        offset_ms=0.0,
        correction_applied=False,
        fps=float(std_info.target_fps),
        corrected_audio_path=None,
        aligned_audio_path=clip_wav,
        alignment_status="already_aligned",
        raw_audio_path=clip_wav,
        clip_start=clip.start_time,
        clip_duration=clip.duration,
    )


def build_dataset_av_alignments(
    clips: Sequence[ClipInfo],
    face_sequences: Sequence[FaceSequence],
    std_info: StandardizedVideoInfo,
    output_dir: Path,
) -> List[AVAlignmentInfo]:
    """Build Stage 14 input alignments for all accepted dataset-mode clips.

    For every Stage 4 clip that produced a Stage 5 face sequence, an
    ``already_aligned`` ``AVAlignmentInfo`` is produced whose aligned audio is
    the original clip wav cut from the Stage 1 standardized audio.

    Args:
        clips: All Stage 4 clips.
        face_sequences: Stage 5 sequences (defines the accepted clip ids).
        std_info: Stage 1 standardized video/audio info.
        output_dir: Directory to write the per-clip wav files.

    Returns:
        List of ``AVAlignmentInfo`` (one per accepted clip), or ``[]`` when
        no clip is accepted.
    """
    accepted = collect_accepted_clip_ids(face_sequences)
    alignments: List[AVAlignmentInfo] = []
    for clip in clips:
        if clip.clip_id in accepted:
            alignments.append(_make_dataset_alignment(clip, std_info, output_dir))
    return alignments


def resolve_stage14_alignment(
    existing: Optional[AVAlignmentInfo],
    clip: ClipInfo,
    std_info: StandardizedVideoInfo,
    output_dir: Path,
) -> AVAlignmentInfo:
    """Stage 14 audio adapter.

    If a Stage 7.1 ``AVAlignmentInfo`` with a usable ``aligned_audio_path``
    exists, use it (open-domain mode).  Otherwise fall back to the original
    clip wav as ``already_aligned`` audio (dataset mode — no SyncNet
    alignment).

    Args:
        existing: Optional Stage 7.1 alignment record (``None`` in dataset
            mode or when Stage 7.1 produced no usable audio).
        clip: Stage 4 clip info (used for the fallback window).
        std_info: Stage 1 standardized video/audio info.
        output_dir: Directory to write the fallback per-clip wav.

    Returns:
        The ``AVAlignmentInfo`` to feed Stage 14.
    """
    if (
        existing is not None
        and existing.aligned_audio_path is not None
        and Path(existing.aligned_audio_path).exists()
    ):
        return existing
    return _make_dataset_alignment(clip, std_info, output_dir)

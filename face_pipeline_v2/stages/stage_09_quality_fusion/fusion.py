"""Pure quality-fusion logic for Stage 9.

Per AISDD 3.13, this module fuses results from Stage 6 (Pose),
Stage 7 (AV Sync), and Stage 8 (Face Quality) into a unified
QualityMetrics per clip.

No model, no I/O, no PipelineContext — all thresholds are injected
via FusionThresholds.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from face_pipeline_v2.data.models import (
    AVSyncInfo,
    FaceQualityInfo,
    FaceQualitySequence,
    PoseInfo,
    PoseSequence,
    QualityMetrics,
)

# ---------------------------------------------------------------------------
# Threshold carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionThresholds:
    """All thresholds and weights for Stage 9 quality fusion.

    Constructed from config.yaml (pose/sync/quality top-level keys +
    stage_09 section) and injected into pure functions.

    Sync hard filter uses raw Stage 7 AVSyncInfo values:
        - max_offset_frames: |offset| threshold (default 8 frames)
        - min_lse_c: minimum LSE-C threshold (default 5.0, typical range 3-10)
    """

    yaw_threshold: float = 30.0
    pitch_threshold: float = 20.0
    roll_threshold: float = 30.0
    max_offset_frames: int = 8              # hard filter: |offset| <= N frames
    min_lse_c: float = 5.0                  # hard filter: raw LSE-C >= N
    quality_threshold: float = 0.5
    min_pose_ratio: float = 0.9
    min_pose_confidence: float = 0.0
    fusion_threshold: Optional[float] = None
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)


# ---------------------------------------------------------------------------
# Per-dimension score computation (all normalized to [0, 1])
# ---------------------------------------------------------------------------


def compute_pose_score(
    poses: Tuple[PoseInfo, ...],
    t: FusionThresholds,
) -> float:
    """Compute pose score as fraction of frames within angle thresholds.

    A frame passes when all three angles are within their respective
    thresholds AND the detection confidence exceeds the minimum.

    Args:
        poses: Per-frame head pose estimates, ordered by frame_index.
        t: Thresholds from config.

    Returns:
        Score in [0, 1] — higher = more frames within acceptable pose range.
    """
    if not poses:
        return 0.0

    pass_count = 0
    for p in poses:
        if p.confidence <= t.min_pose_confidence:
            continue  # no-detection frame → fail
        if (
            abs(p.yaw) <= t.yaw_threshold
            and abs(p.pitch) <= t.pitch_threshold
            and abs(p.roll) <= t.roll_threshold
        ):
            pass_count += 1

    return pass_count / len(poses)


def compute_sync_score_norm(av_sync: AVSyncInfo) -> float:
    """Normalize sync score to [0, 1] from LSE-C for fusion weighting.

    LSE-C = median(distance) - min(distance) is a confidence-like metric
    where higher values indicate more reliable sync detection.

    NOTE: This normalized score is used ONLY for the weighted fusion score
    and Stage 9.1 quality ranking. The hard filter gate uses raw AVSyncInfo
    values (offset + sync_lse_c) directly — see apply_hard_filter().

    Args:
        av_sync: AV sync detection result from Stage 7.

    Returns:
        Clamped LSE-C sync score in [0, 1].
    """
    return max(0.0, min(1.0, av_sync.sync_lse_c))


def compute_quality_score_norm(
    qualities: Tuple[FaceQualityInfo, ...],
) -> float:
    """Compute normalized face quality score as per-frame mean.

    HyperIQA returns scores in 0–100; normalise to [0, 1].

    Args:
        qualities: Per-frame quality scores, ordered by frame_index.

    Returns:
        Mean quality score in [0, 1].
    """
    if not qualities:
        return 0.0

    total = 0.0
    for q in qualities:
        total += max(0.0, min(1.0, q.quality_score / 100.0))
    return total / len(qualities)


# ---------------------------------------------------------------------------
# Hard filter
# ---------------------------------------------------------------------------


def apply_hard_filter(
    pose_score: float,
    sync_score_norm: float,
    quality_score_norm: float,
    av_sync: AVSyncInfo,
    t: FusionThresholds,
    fusion_score: float = 0.0,
) -> Tuple[bool, Tuple[str, ...]]:
    """Apply per-dimension hard filter gates.

    Each gate is evaluated independently.  A clip is accepted only
    when ALL gates pass.

    Sync gate uses raw AVSyncInfo values (NOT normalized sync_score):
        - |offset| <= max_offset_frames
        - sync_lse_c >= min_lse_c

    Args:
        pose_score: Pose pass ratio in [0, 1].
        sync_score_norm: Normalised sync score (unused in hard filter;
                         kept for Stage 9.1 ranking).
        quality_score_norm: Normalised quality score in [0, 1].
        av_sync: AV sync info — raw offset and sync_lse_c for hard filter.
        t: Thresholds from config.
        fusion_score: Pre-computed fusion score (for optional fusion gate).

    Returns:
        (is_accepted, reject_reasons) tuple.
    """
    reasons: list[str] = []

    # Pose gate
    if pose_score < t.min_pose_ratio:
        reasons.append("pose_failed")

    # Sync gate — uses raw AVSyncInfo offset + LSE-C (not normalized sync_score)
    if abs(av_sync.offset) > t.max_offset_frames:
        reasons.append("sync_offset_exceeded")
    if av_sync.sync_lse_c < t.min_lse_c:
        reasons.append("sync_lse_c_too_low")

    # Quality gate
    if quality_score_norm < t.quality_threshold:
        reasons.append("quality_failed")

    # Optional fusion gate
    if t.fusion_threshold is not None and fusion_score < t.fusion_threshold:
        reasons.append("fusion_failed")

    is_accepted = len(reasons) == 0
    return is_accepted, tuple(reasons)


# ---------------------------------------------------------------------------
# Fusion score
# ---------------------------------------------------------------------------


def compute_fusion_score(
    pose_score: float,
    sync_score_norm: float,
    quality_score_norm: float,
    t: FusionThresholds,
) -> float:
    """Compute weighted fusion score from three normalised per-dimension scores.

    If weights do not sum to ~1.0 they are normalised automatically
    (caller should log a warning when this occurs).

    Args:
        pose_score: Pose pass ratio in [0, 1].
        sync_score_norm: Normalised sync score in [0, 1].
        quality_score_norm: Normalised quality score in [0, 1].
        t: Thresholds / weights from config.

    Returns:
        Fusion score in [0, 1].
    """
    w_pose, w_sync, w_quality = t.weights
    weight_sum = w_pose + w_sync + w_quality

    if abs(weight_sum - 1.0) > 1e-6 and weight_sum > 0:
        w_pose /= weight_sum
        w_sync /= weight_sum
        w_quality /= weight_sum

    return w_pose * pose_score + w_sync * sync_score_norm + w_quality * quality_score_norm


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fuse_quality(
    pose_seq: PoseSequence,
    av_sync: AVSyncInfo,
    quality_seq: FaceQualitySequence,
    t: FusionThresholds,
    video_id: str,
) -> QualityMetrics:
    """Fuse one clip's pose + sync + quality into a QualityMetrics.

    This is the single entry point called by Stage9QualityFusion for
    each clip.  All thresholds are supplied via *t* — the function
    has no config or context dependencies.

    Args:
        pose_seq: Per-frame head pose sequence (Stage 6).
        av_sync: AV sync detection result (Stage 7).
        quality_seq: Per-frame quality scores (Stage 8).
        t: All thresholds and weights.
        video_id: Source video identifier.

    Returns:
        QualityMetrics for the clip.
    """
    # 1. Compute per-dimension normalised scores
    pose_score = compute_pose_score(pose_seq.poses, t)
    sync_score_norm = compute_sync_score_norm(av_sync)
    quality_score_norm = compute_quality_score_norm(quality_seq.qualities)

    # 2. Fusion score
    fusion_score = compute_fusion_score(
        pose_score, sync_score_norm, quality_score_norm, t
    )

    # 3. Hard filter
    is_accepted, reject_reasons = apply_hard_filter(
        pose_score,
        sync_score_norm,
        quality_score_norm,
        av_sync,
        t,
        fusion_score,
    )

    return QualityMetrics(
        clip_id=pose_seq.clip_id,
        video_id=video_id,
        pose_score=round(pose_score, 6),
        sync_score=round(sync_score_norm, 6),
        quality_score=round(quality_score_norm, 6),
        fusion_score=round(fusion_score, 6),
        is_accepted=is_accepted,
        reject_reasons=reject_reasons,
    )

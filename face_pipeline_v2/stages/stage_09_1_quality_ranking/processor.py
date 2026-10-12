"""Pure quality-ranking logic for Stage 9.1.

Per the Stage 9.1 design spec, this module computes a weighted multi-metric
score from existing QualityMetrics data and assigns a quality level.

No model, no I/O, no PipelineContext — all weights and thresholds are
injected via the RankingConfig carrier.
"""

from dataclasses import dataclass
from typing import Dict

from face_pipeline_v2.data.models import QualityMetrics, QualityRankingInfo


# ---------------------------------------------------------------------------
# Config carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingConfig:
    """All weights and level thresholds for Stage 9.1 quality ranking.

    Constructed from config.yaml quality_ranking section and injected
    into pure functions.
    """

    pose_weight: float = 0.3
    sync_weight: float = 0.3
    face_quality_weight: float = 0.4
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    normal_threshold: float = 0.50


# ---------------------------------------------------------------------------
# Weighted score computation
# ---------------------------------------------------------------------------


def compute_weighted_score(
    pose_score: float,
    sync_score: float,
    face_quality_score: float,
    cfg: RankingConfig,
) -> float:
    """Compute weighted multi-metric fusion score.

    Formula:
        weighted_score = pose_score * pose_weight
                       + sync_score * sync_weight
                       + face_quality_score * quality_weight

    If weights do not sum to ~1.0 they are normalised automatically.

    Args:
        pose_score: Normalised pose score from QualityMetrics, in [0, 1].
        sync_score: Normalised sync score from QualityMetrics, in [0, 1].
        face_quality_score: Normalised quality score from QualityMetrics, in [0, 1].
        cfg: RankingConfig with weights.

    Returns:
        Weighted score in [0, 1], rounded to 6 decimal places.
    """
    w_pose = cfg.pose_weight
    w_sync = cfg.sync_weight
    w_quality = cfg.face_quality_weight
    weight_sum = w_pose + w_sync + w_quality

    if abs(weight_sum - 1.0) > 1e-6 and weight_sum > 0:
        w_pose /= weight_sum
        w_sync /= weight_sum
        w_quality /= weight_sum

    raw = w_pose * pose_score + w_sync * sync_score + w_quality * face_quality_score
    return round(max(0.0, min(1.0, raw)), 6)


# ---------------------------------------------------------------------------
# Quality level assignment
# ---------------------------------------------------------------------------


def assign_quality_level(weighted_score: float, cfg: RankingConfig) -> str:
    """Assign a quality level based on the weighted score.

    Levels (from config):
        - excellent: score >= excellent_threshold
        - good:      score >= good_threshold
        - normal:    otherwise (score >= normal_threshold is implicit)

    Args:
        weighted_score: Weighted fusion score in [0, 1].
        cfg: RankingConfig with level thresholds.

    Returns:
        Quality level string: "excellent", "good", or "normal".
    """
    if weighted_score >= cfg.excellent_threshold:
        return "excellent"
    elif weighted_score >= cfg.good_threshold:
        return "good"
    else:
        return "normal"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def rank_clip(metrics: QualityMetrics, cfg: RankingConfig) -> QualityRankingInfo:
    """Compute quality ranking for a single clip.

    Uses existing QualityMetrics scores (from Stage 9) — no model execution.
    Computes a NEW weighted score with Stage 9.1's own weights and assigns
    a quality level.

    Args:
        metrics: QualityMetrics from Stage 9 for one clip.
        cfg: RankingConfig with weights and thresholds.

    Returns:
        QualityRankingInfo with weighted_score and quality_level.
    """
    weighted_score = compute_weighted_score(
        pose_score=metrics.pose_score,
        sync_score=metrics.sync_score,
        face_quality_score=metrics.quality_score,
        cfg=cfg,
    )

    quality_level = assign_quality_level(weighted_score, cfg)

    return QualityRankingInfo(
        clip_id=metrics.clip_id,
        pose_score=metrics.pose_score,
        sync_score=metrics.sync_score,
        face_quality_score=metrics.quality_score,
        weighted_score=weighted_score,
        quality_level=quality_level,
    )

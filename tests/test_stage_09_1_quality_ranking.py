"""Tests for Stage 9.1: Quality Ranking (Optional Enhancement).

Tests cover:
    1. enabled=false → Stage 9.1 passes through unchanged.
    2. enabled=true → normal QualityRankingInfo generation.
    3. Weights read from config (quality_ranking section).
    4. Weighted score calculation correctness.
    5. Quality level assignment correctness.
    6. All clips ranked regardless of is_accepted status.
    7. Input validation (missing keys, wrong types, empty lists).
    8. Output validation.
"""

from typing import Any, List
from unittest.mock import patch

import pytest

from face_pipeline_v2.data.models import QualityMetrics, QualityRankingInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_09_1_quality_ranking.processor import (
    RankingConfig,
    assign_quality_level,
    compute_weighted_score,
    rank_clip,
)
from face_pipeline_v2.stages.stage_09_1_quality_ranking.stage import (
    Stage9_1QualityRanking,
)
from face_pipeline_v2.utils.exceptions import InputValidationError, PipelineError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Default config for tests
# ---------------------------------------------------------------------------

_DEFAULT_RANKING_CFG = RankingConfig(
    pose_weight=0.3,
    sync_weight=0.3,
    face_quality_weight=0.4,
    excellent_threshold=0.85,
    good_threshold=0.70,
    normal_threshold=0.50,
)


def _make_context(
    enhancements_enabled: bool = False,
    ranking_overrides: dict | None = None,
) -> PipelineContext:
    """Build a PipelineContext with quality_ranking config.

    Args:
        enhancements_enabled: Whether stage_09_1_quality_ranking is enabled.
        ranking_overrides: Optional overrides for quality_ranking section.
    """
    ranking_config: dict = {
        "weights": {"pose": 0.3, "sync": 0.3, "face_quality": 0.4},
        "levels": {"excellent": 0.85, "good": 0.70, "normal": 0.50},
    }
    if ranking_overrides:
        for k, v in ranking_overrides.items():
            if isinstance(v, dict) and k in ranking_config:
                ranking_config[k].update(v)
            else:
                ranking_config[k] = v

    config: dict = {
        "enhancements": {
            "stage_09_1_quality_ranking": {
                "enabled": enhancements_enabled,
            },
        },
        "quality_ranking": ranking_config,
    }
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_09_1", log_dir=None),
        model_manager=object(),
    )


def _make_quality_metrics(
    clip_id: str = "c0",
    pose_score: float = 0.90,
    sync_score: float = 0.85,
    quality_score: float = 0.80,
    is_accepted: bool = True,
) -> QualityMetrics:
    return QualityMetrics(
        clip_id=clip_id,
        video_id="test_video",
        pose_score=pose_score,
        sync_score=sync_score,
        quality_score=quality_score,
        fusion_score=0.85,
        is_accepted=is_accepted,
    )


def _make_input(
    quality_metrics: List[QualityMetrics] | None = None,
) -> dict:
    if quality_metrics is None:
        quality_metrics = [
            _make_quality_metrics("c0", 0.90, 0.85, 0.80),
            _make_quality_metrics("c1", 0.75, 0.70, 0.65),
        ]
    return {"quality_metrics": quality_metrics}


# ===================================================================
# Processor pure-function tests — compute_weighted_score
# ===================================================================


class TestComputeWeightedScore:
    """Tests for compute_weighted_score()."""

    def test_weighted_average(self):
        """Standard weights → correct weighted average."""
        score = compute_weighted_score(0.8, 0.9, 0.7, _DEFAULT_RANKING_CFG)
        expected = 0.3 * 0.8 + 0.3 * 0.9 + 0.4 * 0.7
        assert score == pytest.approx(expected)

    def test_normalizes_non_summing_weights(self):
        """Weights that don't sum to 1.0 are auto-normalised."""
        cfg = RankingConfig(
            pose_weight=0.6,
            sync_weight=0.6,
            face_quality_weight=0.6,
        )  # sum = 1.8
        score = compute_weighted_score(0.8, 0.9, 0.7, cfg)
        # Normalised: each weight → 1/3
        expected = (0.8 + 0.9 + 0.7) / 3.0
        assert score == pytest.approx(expected)

    def test_all_zeros(self):
        """All scores zero → weighted score zero."""
        assert compute_weighted_score(0.0, 0.0, 0.0, _DEFAULT_RANKING_CFG) == 0.0

    def test_all_ones(self):
        """All scores one → weighted score one."""
        assert compute_weighted_score(1.0, 1.0, 1.0, _DEFAULT_RANKING_CFG) == 1.0

    def test_clamped_to_zero(self):
        """Negative input scores → clamped to 0.0."""
        score = compute_weighted_score(-0.5, -0.5, -0.5, _DEFAULT_RANKING_CFG)
        assert score == 0.0

    def test_clamped_to_one(self):
        """Scores above 1.0 → clamped to 1.0."""
        score = compute_weighted_score(1.5, 1.5, 1.5, _DEFAULT_RANKING_CFG)
        assert score == 1.0

    def test_rounded_to_6_decimals(self):
        """Result is rounded to 6 decimal places."""
        score = compute_weighted_score(1 / 3, 1 / 3, 1 / 3, _DEFAULT_RANKING_CFG)
        assert score == round(score, 6)
        # Verify it's not more than 6 decimal places
        assert score == round(score, 6)

    def test_custom_weights_from_config(self):
        """Custom weights produce correct weighted score."""
        cfg = RankingConfig(
            pose_weight=0.2,
            sync_weight=0.5,
            face_quality_weight=0.3,
        )
        score = compute_weighted_score(0.8, 0.9, 0.7, cfg)
        expected = 0.2 * 0.8 + 0.5 * 0.9 + 0.3 * 0.7
        assert score == pytest.approx(expected)


# ===================================================================
# Processor pure-function tests — assign_quality_level
# ===================================================================


class TestAssignQualityLevel:
    """Tests for assign_quality_level()."""

    def test_excellent(self):
        """Score >= excellent threshold → excellent."""
        assert assign_quality_level(0.90, _DEFAULT_RANKING_CFG) == "excellent"
        assert assign_quality_level(0.85, _DEFAULT_RANKING_CFG) == "excellent"

    def test_good(self):
        """Score >= good threshold but < excellent → good."""
        assert assign_quality_level(0.80, _DEFAULT_RANKING_CFG) == "good"
        assert assign_quality_level(0.70, _DEFAULT_RANKING_CFG) == "good"

    def test_normal(self):
        """Score >= normal threshold but < good → normal."""
        assert assign_quality_level(0.60, _DEFAULT_RANKING_CFG) == "normal"
        assert assign_quality_level(0.50, _DEFAULT_RANKING_CFG) == "normal"

    def test_below_normal(self):
        """Score below all thresholds → normal (lowest tier catches all)."""
        # The "normal" level is the catch-all; there's no "reject" level.
        assert assign_quality_level(0.30, _DEFAULT_RANKING_CFG) == "normal"
        assert assign_quality_level(0.0, _DEFAULT_RANKING_CFG) == "normal"

    def test_custom_thresholds(self):
        """Custom level thresholds change boundaries."""
        cfg = RankingConfig(
            excellent_threshold=0.90,
            good_threshold=0.60,
        )
        assert assign_quality_level(0.95, cfg) == "excellent"
        assert assign_quality_level(0.80, cfg) == "good"
        assert assign_quality_level(0.50, cfg) == "normal"


# ===================================================================
# Processor pure-function tests — rank_clip
# ===================================================================


class TestRankClip:
    """Tests for rank_clip()."""

    def test_returns_quality_ranking_info(self):
        """rank_clip returns a QualityRankingInfo with correct fields."""
        metrics = _make_quality_metrics("c0", 0.90, 0.85, 0.80)
        result = rank_clip(metrics, _DEFAULT_RANKING_CFG)
        assert isinstance(result, QualityRankingInfo)
        assert result.clip_id == "c0"
        assert result.pose_score == 0.90
        assert result.sync_score == 0.85
        assert result.face_quality_score == 0.80

    def test_computes_correct_weighted_score(self):
        """weighted_score matches manual calculation."""
        metrics = _make_quality_metrics("c0", 0.90, 0.85, 0.80)
        result = rank_clip(metrics, _DEFAULT_RANKING_CFG)
        expected = 0.3 * 0.90 + 0.3 * 0.85 + 0.4 * 0.80
        assert result.weighted_score == pytest.approx(expected)

    def test_excellent_level(self):
        """High scores → excellent."""
        metrics = _make_quality_metrics("c0", 0.95, 0.95, 0.95)
        result = rank_clip(metrics, _DEFAULT_RANKING_CFG)
        assert result.quality_level == "excellent"

    def test_good_level(self):
        """Medium scores → good."""
        metrics = _make_quality_metrics("c0", 0.75, 0.75, 0.75)
        result = rank_clip(metrics, _DEFAULT_RANKING_CFG)
        assert result.quality_level == "good"

    def test_normal_level(self):
        """Low scores → normal."""
        metrics = _make_quality_metrics("c0", 0.55, 0.55, 0.55)
        result = rank_clip(metrics, _DEFAULT_RANKING_CFG)
        assert result.quality_level == "normal"

    def test_accepted_status_not_affecting_ranking(self):
        """Rejected clips get ranked the same as accepted clips."""
        accepted = _make_quality_metrics("c0", 0.90, 0.85, 0.80, is_accepted=True)
        rejected = _make_quality_metrics("c0", 0.90, 0.85, 0.80, is_accepted=False)
        r1 = rank_clip(accepted, _DEFAULT_RANKING_CFG)
        r2 = rank_clip(rejected, _DEFAULT_RANKING_CFG)
        assert r1.weighted_score == r2.weighted_score
        assert r1.quality_level == r2.quality_level


# ===================================================================
# Stage integration tests — enabled=false (passthrough)
# ===================================================================


class TestStage9_1Disabled:
    """Tests for Stage 9.1 when enhancement is disabled."""

    def test_passthrough_when_disabled(self):
        """When enabled=false, input is passed through unchanged."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=False)
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        # Passthrough: returns the same input dict
        assert result is input_data
        assert result["quality_metrics"] == input_data["quality_metrics"]

    def test_does_not_compute_ranking_when_disabled(self):
        """Disabled stage does not create QualityRankingInfo."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=False)
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        # Still the original QualityMetrics, not QualityRankingInfo
        assert all(isinstance(m, QualityMetrics) for m in result["quality_metrics"])


# ===================================================================
# Stage integration tests — enabled=true
# ===================================================================


class TestStage9_1Enabled:
    """Tests for Stage 9.1 when enhancement is enabled."""

    def test_normal_run_generates_ranking(self):
        """Enabled stage returns List[QualityRankingInfo]."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, QualityRankingInfo) for r in result)

    def test_ranking_preserves_clip_ids(self):
        """Output clip_ids match input clip_ids."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        assert [r.clip_id for r in result] == ["c0", "c1"]

    def test_ranking_scores_match_input(self):
        """Pose/sync/quality scores are passed through from QualityMetrics."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        metrics_list = [
            _make_quality_metrics("c0", 0.90, 0.85, 0.80),
            _make_quality_metrics("c1", 0.75, 0.70, 0.65),
        ]
        input_data = _make_input(quality_metrics=metrics_list)
        result = stage.run(input_data=input_data, context=ctx)
        assert result[0].pose_score == 0.90
        assert result[0].sync_score == 0.85
        assert result[0].face_quality_score == 0.80
        assert result[1].pose_score == 0.75
        assert result[1].sync_score == 0.70
        assert result[1].face_quality_score == 0.65

    def test_all_clips_ranked_regardless_of_accepted(self):
        """Both accepted and rejected clips are ranked."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        metrics_list = [
            _make_quality_metrics("c_accepted", 0.90, 0.85, 0.80, is_accepted=True),
            _make_quality_metrics("c_rejected", 0.30, 0.30, 0.30, is_accepted=False),
        ]
        input_data = _make_input(quality_metrics=metrics_list)
        result = stage.run(input_data=input_data, context=ctx)
        assert len(result) == 2
        assert result[0].clip_id == "c_accepted"
        assert result[1].clip_id == "c_rejected"

    def test_weights_from_config(self):
        """Weights are read from quality_ranking config, not hard-coded."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(
            enhancements_enabled=True,
            ranking_overrides={"weights": {"pose": 0.5, "sync": 0.2, "face_quality": 0.3}},
        )
        metrics_list = [
            _make_quality_metrics("c0", 0.80, 0.90, 0.70),
        ]
        input_data = _make_input(quality_metrics=metrics_list)
        result = stage.run(input_data=input_data, context=ctx)
        expected = 0.5 * 0.80 + 0.2 * 0.90 + 0.3 * 0.70
        assert result[0].weighted_score == pytest.approx(expected)

    def test_levels_from_config(self):
        """Level thresholds are read from quality_ranking config."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(
            enhancements_enabled=True,
            ranking_overrides={"levels": {"excellent": 0.90, "good": 0.75, "normal": 0.50}},
        )
        metrics_list = [
            _make_quality_metrics("c0", 0.85, 0.85, 0.85),  # weighted ~0.85 → good (not excellent)
        ]
        input_data = _make_input(quality_metrics=metrics_list)
        result = stage.run(input_data=input_data, context=ctx)
        # With thresholds excellent=0.90, good=0.75:
        # weighted=0.85 → good (>=0.75, <0.90)
        assert result[0].quality_level == "good"


# ===================================================================
# Stage integration tests — input validation
# ===================================================================


class TestStage9_1InputValidation:
    """Tests for validate_input()."""

    def test_rejects_non_dict(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_input("not_a_dict", ctx) is False
        assert stage.validate_input([], ctx) is False

    def test_rejects_missing_quality_metrics_key(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_input({"other_key": []}, ctx) is False

    def test_rejects_empty_quality_metrics_list(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_input({"quality_metrics": []}, ctx) is False

    def test_rejects_wrong_element_type(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_input(
            {"quality_metrics": ["not_a_quality_metrics"]}, ctx
        ) is False

    def test_accepts_valid_input(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_input(_make_input(), ctx) is True

    def test_run_raises_on_invalid_input(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        with pytest.raises(InputValidationError):
            stage.run(input_data="bad", context=ctx)


# ===================================================================
# Stage integration tests — output validation
# ===================================================================


class TestStage9_1OutputValidation:
    """Tests for validate_output()."""

    def test_accepts_valid_ranking_list(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        rankings = [
            QualityRankingInfo("c0", 0.9, 0.85, 0.8, 0.845, "good"),
        ]
        assert stage.validate_output(rankings, ctx) is True

    def test_rejects_wrong_type(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_output("not_a_list", ctx) is False
        assert stage.validate_output(["not_a_ranking"], ctx) is False

    def test_empty_list_is_valid(self):
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        assert stage.validate_output([], ctx) is True


# ===================================================================
# Config loading tests
# ===================================================================


class TestConfigLoading:
    """Tests for config-driven behavior."""

    def test_missing_quality_ranking_section_raises(self):
        """PipelineError when quality_ranking section is absent from config."""
        stage = Stage9_1QualityRanking()
        config = {
            "enhancements": {
                "stage_09_1_quality_ranking": {"enabled": True},
            },
            # No quality_ranking section
        }
        ctx = PipelineContext(
            config=config,
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
        )
        input_data = _make_input()
        with pytest.raises(PipelineError, match="quality_ranking"):
            stage.run(input_data=input_data, context=ctx)

    def test_single_clip_input(self):
        """Single-clip input works correctly."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        input_data = _make_input(
            quality_metrics=[_make_quality_metrics("c_only", 0.88, 0.82, 0.79)]
        )
        result = stage.run(input_data=input_data, context=ctx)
        assert len(result) == 1
        assert result[0].clip_id == "c_only"


# ===================================================================
# Immutability tests
# ===================================================================


class TestImmutability:
    """Tests that Stage 9.1 does not modify input data."""

    def test_quality_metrics_not_modified(self):
        """Original QualityMetrics objects are unchanged after ranking."""
        stage = Stage9_1QualityRanking()
        ctx = _make_context(enhancements_enabled=True)
        metrics = _make_quality_metrics("c0", 0.90, 0.85, 0.80, is_accepted=True)
        original_accepted = metrics.is_accepted
        original_fusion = metrics.fusion_score

        input_data = _make_input(quality_metrics=[metrics])
        stage.run(input_data=input_data, context=ctx)

        # Original object untouched
        assert metrics.is_accepted == original_accepted
        assert metrics.fusion_score == original_fusion
        assert metrics.clip_id == "c0"

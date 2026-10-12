"""Tests for Stage 9: Quality Metrics Fusion.

Tests cover:
    1. Pure fusion functions — compute_pose_score, compute_sync_score_norm,
       compute_quality_score_norm, apply_hard_filter, compute_fusion_score.
    2. fuse_quality — full integration of pure functions.
    3. Stage9QualityFusion integration — dict input → List[QualityMetrics].
    4. enabled=false → returns empty list.
    5. Missing clip alignment → skipped with warning.
    6. video_id resolution (input dict vs context.runtime_info).
    7. Input validation.
"""

from typing import Any, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    AVSyncInfo,
    FaceQualityInfo,
    FaceQualitySequence,
    PoseInfo,
    PoseSequence,
    QualityMetrics,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_09_quality_fusion.fusion import (
    FusionThresholds,
    apply_hard_filter,
    compute_fusion_score,
    compute_pose_score,
    compute_quality_score_norm,
    compute_sync_score_norm,
    fuse_quality,
)
from face_pipeline_v2.stages.stage_09_quality_fusion.stage import (
    Stage9QualityFusion,
)
from face_pipeline_v2.utils.exceptions import InputValidationError, PipelineError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_T = FusionThresholds(
    yaw_threshold=30.0,
    pitch_threshold=20.0,
    roll_threshold=30.0,
    max_offset_frames=8,
    min_lse_c=5.0,
    quality_threshold=0.5,
    min_pose_ratio=0.9,
    min_pose_confidence=0.0,
    fusion_threshold=None,
    weights=(0.4, 0.3, 0.3),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "pose": {
            "yaw_threshold": 30.0,
            "pitch_threshold": 20.0,
            "roll_threshold": 30.0,
        },
        "quality": {
            "quality_threshold": 0.5,
            "sync": {
                "max_offset_frames": 8,
                "min_lse_c": 5.0,
            },
        },
        "stage_09": {
            "enabled": True,
            "min_pose_ratio": 0.9,
            "fusion_threshold": None,
            "fusion_weights": {"pose": 0.4, "sync": 0.3, "quality": 0.3},
        },
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_09", log_dir=None),
        model_manager=object(),
    )


def _pose_info(
    frame_index: int = 0,
    yaw: float = 10.0,
    pitch: float = 5.0,
    roll: float = 3.0,
    confidence: float = 1.0,
) -> PoseInfo:
    return PoseInfo(
        frame_index=frame_index,
        face_id="face_0",
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        confidence=confidence,
    )


def _pose_seq(
    clip_id: str = "c0",
    poses: Tuple[PoseInfo, ...] | None = None,
) -> PoseSequence:
    if poses is None:
        poses = tuple(_pose_info(i) for i in range(10))
    return PoseSequence(clip_id=clip_id, poses=poses)


def _av_sync_info(
    clip_id: str = "c0",
    offset: int = 2,
    sync_lse_c: float = 0.85,
    sync_lse_d: float = 14.5,
    median_distance: float = 14.6,
) -> AVSyncInfo:
    return AVSyncInfo(
        clip_id=clip_id,
        offset=offset,
        sync_lse_c=sync_lse_c,
        sync_lse_d=sync_lse_d,
        median_distance=median_distance,
    )


def _quality_info(
    frame_index: int = 0,
    quality_score: float = 75.0,
    confidence: float = 0.9,
) -> FaceQualityInfo:
    return FaceQualityInfo(
        frame_index=frame_index,
        face_id="face_0",
        quality_score=quality_score,
        confidence=confidence,
    )


def _quality_seq(
    clip_id: str = "c0",
    qualities: Tuple[FaceQualityInfo, ...] | None = None,
) -> FaceQualitySequence:
    if qualities is None:
        qualities = tuple(_quality_info(i, quality_score=75.0) for i in range(10))
    return FaceQualitySequence(clip_id=clip_id, qualities=qualities)


def _make_input(
    pose_sequences=None,
    av_sync_infos=None,
    quality_sequences=None,
    video_id="test_video",
):
    return {
        "pose_sequences": pose_sequences or [_pose_seq("c0"), _pose_seq("c1")],
        "av_sync_infos": av_sync_infos or [_av_sync_info("c0"), _av_sync_info("c1")],
        "quality_sequences": quality_sequences or [_quality_seq("c0"), _quality_seq("c1")],
        "video_id": video_id,
    }


# ===================================================================
# Pure function tests — compute_pose_score
# ===================================================================


class TestComputePoseScore:
    """Tests for compute_pose_score()."""

    def test_all_good_frames(self):
        """All frames within thresholds → score near 1.0."""
        poses = tuple(_pose_info(i, yaw=10, pitch=5, roll=3) for i in range(10))
        score = compute_pose_score(poses, _DEFAULT_T)
        assert score == 1.0

    def test_some_exceeding_yaw(self):
        """Some frames exceed yaw → reduced score."""
        poses = tuple(
            _pose_info(i, yaw=10 if i < 5 else 40.0) for i in range(10)
        )
        score = compute_pose_score(poses, _DEFAULT_T)
        assert score == 0.5

    def test_zero_confidence_fails(self):
        """Frames with confidence <= min_pose_confidence count as fail."""
        poses = (
            _pose_info(0, yaw=5, confidence=0.0),
            _pose_info(1, yaw=5, confidence=1.0),
            _pose_info(2, yaw=5, confidence=1.0),
        )
        score = compute_pose_score(poses, _DEFAULT_T)
        assert score == 2.0 / 3.0

    def test_empty_poses(self):
        """Empty poses → 0.0."""
        assert compute_pose_score((), _DEFAULT_T) == 0.0

    def test_all_exceeding(self):
        """All frames exceed thresholds → 0.0."""
        poses = tuple(_pose_info(i, yaw=50, pitch=40, roll=50) for i in range(10))
        score = compute_pose_score(poses, _DEFAULT_T)
        assert score == 0.0

    def test_custom_thresholds(self):
        """Custom thresholds change pass/fail boundary."""
        poses = tuple(_pose_info(i, yaw=25) for i in range(10))
        # Default threshold=30 → pass
        assert compute_pose_score(poses, _DEFAULT_T) == 1.0
        # Stricter threshold=20 → fail
        strict = FusionThresholds(yaw_threshold=20.0)
        assert compute_pose_score(poses, strict) == 0.0


# ===================================================================
# Pure function tests — compute_sync_score_norm
# ===================================================================


class TestComputeSyncScoreNorm:
    """Tests for compute_sync_score_norm()."""

    def test_normal_value(self):
        av = _av_sync_info(sync_lse_c=0.85)
        assert compute_sync_score_norm(av) == 0.85

    def test_clamp_above_one(self):
        av = _av_sync_info(sync_lse_c=1.5)
        assert compute_sync_score_norm(av) == 1.0

    def test_clamp_below_zero(self):
        av = _av_sync_info(sync_lse_c=-0.3)
        assert compute_sync_score_norm(av) == 0.0


# ===================================================================
# Pure function tests — compute_quality_score_norm
# ===================================================================


class TestComputeQualityScoreNorm:
    """Tests for compute_quality_score_norm()."""

    def test_normalizes_0_100_to_0_1(self):
        qualities = tuple(_quality_info(i, quality_score=75.0) for i in range(4))
        score = compute_quality_score_norm(qualities)
        assert score == pytest.approx(0.75)

    def test_empty(self):
        assert compute_quality_score_norm(()) == 0.0

    def test_clamp_above_100(self):
        qualities = (_quality_info(0, quality_score=150.0),)
        assert compute_quality_score_norm(qualities) == 1.0

    def test_clamp_below_zero(self):
        qualities = (_quality_info(0, quality_score=-10.0),)
        assert compute_quality_score_norm(qualities) == 0.0


# ===================================================================
# Pure function tests — apply_hard_filter
# ===================================================================


class TestApplyHardFilter:
    """Tests for apply_hard_filter()."""

    def test_all_pass(self):
        av = _av_sync_info(offset=2, sync_lse_c=0.85)
        accepted, reasons = apply_hard_filter(
            pose_score=0.95, sync_score_norm=0.85,
            quality_score_norm=0.75, av_sync=av, t=_DEFAULT_T,
        )
        assert accepted is True
        assert reasons == ()

    def test_pose_failed(self):
        av = _av_sync_info(offset=2, sync_lse_c=0.85)
        accepted, reasons = apply_hard_filter(
            pose_score=0.5, sync_score_norm=0.85,
            quality_score_norm=0.75, av_sync=av, t=_DEFAULT_T,
        )
        assert accepted is False
        assert "pose_failed" in reasons

    def test_sync_failed_offset(self):
        av = _av_sync_info(offset=16, sync_lse_c=0.85)
        accepted, reasons = apply_hard_filter(
            pose_score=0.95, sync_score_norm=0.85,
            quality_score_norm=0.75, av_sync=av, t=_DEFAULT_T,
        )
        assert accepted is False
        assert "sync_failed" in reasons

    def test_sync_failed_score(self):
        av = _av_sync_info(offset=2, sync_lse_c=0.3)
        accepted, reasons = apply_hard_filter(
            pose_score=0.95, sync_score_norm=0.3,
            quality_score_norm=0.75, av_sync=av, t=_DEFAULT_T,
        )
        assert accepted is False
        assert "sync_failed" in reasons

    def test_quality_failed(self):
        av = _av_sync_info(offset=2, sync_lse_c=0.85)
        accepted, reasons = apply_hard_filter(
            pose_score=0.95, sync_score_norm=0.85,
            quality_score_norm=0.3, av_sync=av, t=_DEFAULT_T,
        )
        assert accepted is False
        assert "quality_failed" in reasons

    def test_multiple_failures(self):
        av = _av_sync_info(offset=16, sync_lse_c=0.3)
        accepted, reasons = apply_hard_filter(
            pose_score=0.5, sync_score_norm=0.3,
            quality_score_norm=0.3, av_sync=av, t=_DEFAULT_T,
        )
        assert accepted is False
        assert len(reasons) == 3
        assert "pose_failed" in reasons
        assert "sync_failed" in reasons
        assert "quality_failed" in reasons

    def test_fusion_gate_when_set(self):
        """When fusion_threshold is set, fusion score is gated."""
        t = FusionThresholds(fusion_threshold=0.7)
        av = _av_sync_info(offset=2, sync_lse_c=0.85)
        accepted, reasons = apply_hard_filter(
            pose_score=0.95, sync_score_norm=0.85,
            quality_score_norm=0.75, av_sync=av, t=t,
            fusion_score=0.5,  # below 0.7 threshold
        )
        assert accepted is False
        assert "fusion_failed" in reasons

    def test_fusion_gate_disabled_when_none(self):
        """When fusion_threshold is None, fusion gate is skipped."""
        t = FusionThresholds(fusion_threshold=None)
        av = _av_sync_info(offset=2, sync_lse_c=0.85)
        accepted, reasons = apply_hard_filter(
            pose_score=0.95, sync_score_norm=0.85,
            quality_score_norm=0.75, av_sync=av, t=t,
            fusion_score=0.3,
        )
        assert accepted is True


# ===================================================================
# Pure function tests — compute_fusion_score
# ===================================================================


class TestComputeFusionScore:
    """Tests for compute_fusion_score()."""

    def test_weighted_average(self):
        score = compute_fusion_score(0.8, 0.9, 0.7, _DEFAULT_T)
        expected = 0.4 * 0.8 + 0.3 * 0.9 + 0.3 * 0.7
        assert score == pytest.approx(expected)

    def test_normalizes_non_summing_weights(self):
        t = FusionThresholds(weights=(0.6, 0.6, 0.6))  # sum=1.8
        score = compute_fusion_score(0.8, 0.9, 0.7, t)
        # Normalised: each weight → 1/3
        expected = (0.8 + 0.9 + 0.7) / 3.0
        assert score == pytest.approx(expected)

    def test_all_zeros(self):
        assert compute_fusion_score(0.0, 0.0, 0.0, _DEFAULT_T) == 0.0

    def test_all_ones(self):
        assert compute_fusion_score(1.0, 1.0, 1.0, _DEFAULT_T) == 1.0


# ===================================================================
# Pure function tests — fuse_quality (integration of all pure helpers)
# ===================================================================


class TestFuseQuality:
    """Tests for fuse_quality()."""

    def test_returns_quality_metrics(self):
        ps = _pose_seq("c0")
        av = _av_sync_info("c0")
        qs = _quality_seq("c0")
        result = fuse_quality(ps, av, qs, _DEFAULT_T, "v0")
        assert isinstance(result, QualityMetrics)
        assert result.clip_id == "c0"
        assert result.video_id == "v0"

    def test_accepted_with_good_data(self):
        """Good pose + sync + quality → accepted."""
        ps = _pose_seq("c0")  # all yaw=10, pitch=5, roll=3 → all within thresholds
        av = _av_sync_info("c0", offset=2, sync_lse_c=0.85)
        qs = _quality_seq("c0")  # all quality_score=75 → norm=0.75
        result = fuse_quality(ps, av, qs, _DEFAULT_T, "v0")
        assert result.is_accepted is True
        assert result.reject_reasons == ()
        assert result.pose_score == 1.0
        assert result.sync_score == 0.85
        assert result.quality_score == pytest.approx(0.75)
        assert 0.0 < result.fusion_score <= 1.0

    def test_rejected_bad_pose(self):
        """All frames exceed pose thresholds → rejected with pose_failed."""
        poses = tuple(_pose_info(i, yaw=50, pitch=40, roll=50) for i in range(10))
        ps = _pose_seq("c0", poses=poses)
        av = _av_sync_info("c0", offset=2, sync_lse_c=0.85)
        qs = _quality_seq("c0")
        result = fuse_quality(ps, av, qs, _DEFAULT_T, "v0")
        assert result.is_accepted is False
        assert "pose_failed" in result.reject_reasons

    def test_rejected_bad_sync(self):
        """Large offset + low score → rejected with sync_failed."""
        ps = _pose_seq("c0")
        av = _av_sync_info("c0", offset=20, sync_lse_c=0.2)
        qs = _quality_seq("c0")
        result = fuse_quality(ps, av, qs, _DEFAULT_T, "v0")
        assert result.is_accepted is False
        assert "sync_failed" in result.reject_reasons

    def test_rejected_bad_quality(self):
        """Very low quality → rejected with quality_failed."""
        ps = _pose_seq("c0")
        av = _av_sync_info("c0", offset=2, sync_lse_c=0.85)
        qualities = tuple(_quality_info(i, quality_score=10.0) for i in range(10))
        qs = _quality_seq("c0", qualities=qualities)
        result = fuse_quality(ps, av, qs, _DEFAULT_T, "v0")
        assert result.is_accepted is False
        assert "quality_failed" in result.reject_reasons

    def test_scores_rounded_to_6_decimals(self):
        ps = _pose_seq("c0")
        av = _av_sync_info("c0")
        qs = _quality_seq("c0")
        result = fuse_quality(ps, av, qs, _DEFAULT_T, "v0")
        # 6 decimal places max
        assert result.pose_score == round(result.pose_score, 6)
        assert result.fusion_score == round(result.fusion_score, 6)


# ===================================================================
# Stage integration tests
# ===================================================================


class TestStage9QualityFusion:
    """Integration tests for Stage9QualityFusion."""

    def test_normal_run(self):
        """Typical input → returns QualityMetrics list."""
        stage = Stage9QualityFusion()
        ctx = _make_context()
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(m, QualityMetrics) for m in result)
        assert all(m.video_id == "test_video" for m in result)

    def test_disabled_returns_empty(self):
        """stage_09.enabled=false → returns [].

        Note: validate_input must also pass — the stage checks
        enabled inside process(), not before validation.
        """
        stage = Stage9QualityFusion()
        ctx = _make_context({"stage_09": {"enabled": False}})
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        assert result == []

    def test_skips_misaligned_clip(self):
        """Clip present in pose but missing in av/quality → skipped."""
        stage = Stage9QualityFusion()
        ctx = _make_context()
        input_data = _make_input(
            pose_sequences=[_pose_seq("c0"), _pose_seq("c_missing")],
            av_sync_infos=[_av_sync_info("c0")],
            quality_sequences=[_quality_seq("c0")],
        )
        result = stage.run(input_data=input_data, context=ctx)
        assert len(result) == 1
        assert result[0].clip_id == "c0"

    def test_validate_input_rejects_wrong_type(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        assert stage.validate_input("not_a_dict", ctx) is False
        assert stage.validate_input([], ctx) is False

    def test_validate_input_rejects_missing_keys(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        assert stage.validate_input({"pose_sequences": []}, ctx) is False

    def test_validate_input_rejects_empty_lists(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        assert stage.validate_input(
            {"pose_sequences": [], "av_sync_infos": [_av_sync_info()],
             "quality_sequences": [_quality_seq()]}, ctx
        ) is False

    def test_validate_input_rejects_wrong_element_types(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        assert stage.validate_input(
            {"pose_sequences": ["not_a_pose_sequence"],
             "av_sync_infos": [_av_sync_info()],
             "quality_sequences": [_quality_seq()]}, ctx
        ) is False

    def test_run_raises_on_invalid_input(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        with pytest.raises(InputValidationError):
            stage.run(input_data="bad", context=ctx)

    def test_validate_output_empty_list(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        # Empty list is valid (all clips could be skipped)
        assert stage.validate_output([], ctx) is True

    def test_validate_output_rejects_wrong_type(self):
        stage = Stage9QualityFusion()
        ctx = _make_context()
        assert stage.validate_output("not_a_list", ctx) is False
        assert stage.validate_output([_av_sync_info()], ctx) is False

    def test_video_id_from_input_dict(self):
        """video_id from input dict is used."""
        stage = Stage9QualityFusion()
        ctx = _make_context()
        input_data = _make_input(video_id="from_input_dict")
        result = stage.run(input_data=input_data, context=ctx)
        assert result[0].video_id == "from_input_dict"

    def test_video_id_from_context_runtime_info(self):
        """Fallback to context.runtime_info when input dict has no video_id."""
        stage = Stage9QualityFusion()
        config = {
            "pose": {"yaw_threshold": 30.0, "pitch_threshold": 20.0, "roll_threshold": 30.0},
            "quality": {"quality_threshold": 0.5, "sync": {"max_offset_frames": 8, "min_lse_c": 5.0}},
            "stage_09": {
                "enabled": True, "min_pose_ratio": 0.9,
                "fusion_threshold": None,
                "fusion_weights": {"pose": 0.4, "sync": 0.3, "quality": 0.3},
            },
        }
        ctx = PipelineContext(
            config=config,
            logger=PipelineLogger(name="test", log_dir=None),
            model_manager=object(),
            runtime_info={"video_id": "from_context"},
        )
        input_data = _make_input()
        del input_data["video_id"]
        result = stage.run(input_data=input_data, context=ctx)
        assert result[0].video_id == "from_context"

    def test_missing_video_id_raises(self):
        """No video_id anywhere → PipelineError."""
        stage = Stage9QualityFusion()
        ctx = _make_context()
        input_data = _make_input()
        del input_data["video_id"]
        with pytest.raises(PipelineError, match="video_id"):
            stage.run(input_data=input_data, context=ctx)

    def test_all_accepted_count_logging(self):
        """Verify accepted count matches when all clips are good."""
        stage = Stage9QualityFusion()
        ctx = _make_context()
        input_data = _make_input()
        result = stage.run(input_data=input_data, context=ctx)
        accepted = sum(1 for m in result if m.is_accepted)
        assert accepted == 2

    def test_fusion_threshold_gate(self):
        """When fusion_threshold is set, very low scores can be rejected."""
        stage = Stage9QualityFusion()
        ctx = _make_context({"stage_09": {"fusion_threshold": 0.95}})
        # Use poor data so fusion score is low
        poses = tuple(_pose_info(i, yaw=50, pitch=40, roll=50) for i in range(10))
        input_data = _make_input(
            pose_sequences=[_pose_seq("c0", poses=poses)],
            av_sync_infos=[_av_sync_info("c0", offset=20, sync_lse_c=0.2)],
            quality_sequences=[_quality_seq("c0")],
        )
        result = stage.run(input_data=input_data, context=ctx)
        assert result[0].is_accepted is False
        # Should include fusion_failed (plus pose_failed, sync_failed, quality_failed)
        assert "fusion_failed" in result[0].reject_reasons

"""Tests for Stage 5.3: Temporal Stability Enhancement.

Tests cover:
    1. disabled → passthrough.
    2. Stable clip (small motion) → passed.
    3. Fast motion → rejected (center_motion_abnormal).
    4. Scale change → rejected (scale_abnormal).
    5. Multiple bbox jumps → rejected.
    6. Main track used when identity_result present.
    7. Empty/single frame clips handled.
    8. Input immutability.
    9. Input validation.
"""

from pathlib import Path

import pytest

from face_pipeline_v2.data.models import (
    FaceInfo,
    FaceSequence,
    FrameInfo,
    SingleFaceIdentityResult,
    TemporalStabilityResult,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_3_temporal_stability.temporal_stability import (
    analyze_stability,
    REASON_CENTER_MOTION,
    REASON_SCALE,
)
from face_pipeline_v2.stages.stage_05_3_temporal_stability.stage import (
    Stage5_3TemporalStability,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _face(
    frame_index: int, face_id: str, track_id: str | None,
    bbox: tuple[float, ...] = (50.0, 30.0, 150.0, 170.0),
    confidence: float = 0.9,
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index, face_id=face_id, track_id=track_id,
        bbox=bbox, landmark_5=(), confidence=confidence,
        face_crop_path=Path(""),
    )


def _frame(idx: int, faces: tuple[FaceInfo, ...]) -> FrameInfo:
    return FrameInfo(
        frame_index=idx, clip_id="c0", timestamp=idx / 25.0,
        image_path=Path(""), faces=faces,
    )


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "enhancements": {
            "stage_05_3_temporal_stability": {
                "enabled": True,
                "max_center_motion": 15.0,
                "max_scale_change_ratio": 0.25,
                "max_stability_anomaly_ratio": 0.05,
            },
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
        logger=PipelineLogger(name="test_stage_05_3", log_dir=None),
        model_manager=object(),
    )


def _seq(
    clip_id: str = "c0", video_id: str = "v",
    frames: tuple[FrameInfo, ...] = (),
    identity_result: SingleFaceIdentityResult | None = None,
) -> FaceSequence:
    return FaceSequence(
        clip_id=clip_id, video_id=video_id, frames=frames,
        identity_result=identity_result,
    )


# ---------------------------------------------------------------------------
# Builders for controlled motion sequences
# ---------------------------------------------------------------------------

def _stable_sequence(n: int = 100) -> FaceSequence:
    """Face moves slowly and smoothly → should pass."""
    frames: list[FrameInfo] = []
    for i in range(n):
        # Slow drift: 1px per 10 frames
        x1 = 50.0 + i * 0.1
        y1 = 30.0 + i * 0.05
        x2 = 150.0 + i * 0.1
        y2 = 170.0 + i * 0.05
        f = _face(i, f"f_{i}", "track_000", bbox=(x1, y1, x2, y2))
        frames.append(_frame(i, (f,)))
    return _seq(frames=tuple(frames))


def _fast_motion_sequence(n: int = 100) -> FaceSequence:
    """Face jumps 20px every frame from frame 50 onward → should fail."""
    frames: list[FrameInfo] = []
    for i in range(n):
        if i < 50:
            x1, y1, x2, y2 = 50.0, 30.0, 150.0, 170.0
        else:
            # Jump 20px per frame
            jump = (i - 49) * 20.0
            x1, y1, x2, y2 = 50.0 + jump, 30.0, 150.0 + jump, 170.0
        f = _face(i, f"f_{i}", "track_000", bbox=(x1, y1, x2, y2))
        frames.append(_frame(i, (f,)))
    return _seq(frames=tuple(frames))


def _scale_jump_sequence(n: int = 100) -> FaceSequence:
    """Face area doubles suddenly at frame 50 → should fail."""
    frames: list[FrameInfo] = []
    for i in range(n):
        if i < 50:
            x1, y1, x2, y2 = 50.0, 30.0, 150.0, 170.0
        else:
            # Double width/height → 4x area
            x1, y1, x2, y2 = 0.0, 0.0, 200.0, 280.0
        f = _face(i, f"f_{i}", "track_000", bbox=(x1, y1, x2, y2))
        frames.append(_frame(i, (f,)))
    return _seq(frames=tuple(frames))


def _few_jumps_sequence(n: int = 100) -> FaceSequence:
    """1 jump out of 99 transitions → 0.01 anomaly ratio → passes."""
    frames: list[FrameInfo] = []
    for i in range(n):
        if i == 50:
            # Single jump
            x1, y1, x2, y2 = 200.0, 200.0, 300.0, 300.0
        else:
            x1, y1, x2, y2 = 50.0, 30.0, 150.0, 170.0
        f = _face(i, f"f_{i}", "track_000", bbox=(x1, y1, x2, y2))
        frames.append(_frame(i, (f,)))
    return _seq(frames=tuple(frames))


def _multi_track_sequence(n: int = 100) -> FaceSequence:
    """Has identity_result with main_track_id → only that track's bboxes used."""
    frames: list[FrameInfo] = []
    for i in range(n):
        main = _face(i, f"main_{i}", "track_main", bbox=(50.0, 30.0, 150.0, 170.0))
        # Also a junk face that jumps wildly — should be ignored
        junk = _face(i, f"junk_{i}", "track_junk", bbox=(float(i * 10), 0.0, float(i * 10 + 100), 100.0))
        frames.append(_frame(i, (main, junk)))
    return _seq(
        frames=tuple(frames),
        identity_result=SingleFaceIdentityResult(
            passed=True, main_track_id="track_main", total_frames=n,
            anomaly_frames=0, anomaly_ratio=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# analyze_stability unit tests
# ---------------------------------------------------------------------------


class TestTemporalStability:
    def test_stable_clip_passes(self) -> None:
        seq = _stable_sequence(100)
        result = analyze_stability(seq)

        assert result.stability_result is not None
        assert result.stability_result.passed is True
        assert result.stability_result.bbox_jump_count == 0
        assert result.stability_result.average_center_motion < 5.0

    def test_fast_motion_rejected(self) -> None:
        seq = _fast_motion_sequence(100)
        result = analyze_stability(seq)

        sr = result.stability_result
        assert sr is not None
        assert sr.passed is False
        assert sr.max_center_motion > 15.0
        assert REASON_CENTER_MOTION in sr.reject_reasons

    def test_scale_change_rejected(self) -> None:
        seq = _scale_jump_sequence(100)
        # Use strict threshold: any anomaly rejects
        result = analyze_stability(
            seq, max_stability_anomaly_ratio=0.0,
        )

        sr = result.stability_result
        assert sr is not None
        assert sr.passed is False
        assert sr.max_scale_change > 0.25
        assert REASON_SCALE in sr.reject_reasons

    def test_few_jumps_within_threshold_passes(self) -> None:
        """1 jump + 1 return to normal = 2 jumps / 99 transitions → passes at 0.05."""
        seq = _few_jumps_sequence(100)
        result = analyze_stability(seq)

        sr = result.stability_result
        assert sr is not None
        assert sr.passed is True
        # Jump at frame 50 (away) + frame 51 (back) = 2 anomalous transitions
        assert sr.anomaly_ratio == pytest.approx(2 / 99)

    def test_main_track_filtered(self) -> None:
        """Only main_track_id bboxes used; junk track ignored."""
        seq = _multi_track_sequence(100)
        result = analyze_stability(seq)

        sr = result.stability_result
        assert sr is not None
        assert sr.passed is True  # main track is stable
        assert sr.bbox_jump_count == 0

    def test_single_frame_passes(self) -> None:
        f = _face(0, "f0", None)
        seq = _seq(frames=(_frame(0, (f,)),))
        result = analyze_stability(seq)

        assert result.stability_result is not None
        assert result.stability_result.passed is True

    def test_empty_sequence(self) -> None:
        seq = _seq()
        result = analyze_stability(seq)

        sr = result.stability_result
        assert sr is not None
        assert sr.passed is True

    def test_no_faces_in_frames(self) -> None:
        seq = _seq(frames=(
            _frame(0, ()),
            _frame(1, ()),
        ))
        result = analyze_stability(seq)

        sr = result.stability_result
        assert sr is not None
        assert sr.passed is False  # no bboxes to analyze

    def test_input_not_mutated(self) -> None:
        seq = _fast_motion_sequence(100)
        orig_clip_id = seq.clip_id

        result = analyze_stability(seq)

        assert seq.clip_id == orig_clip_id
        assert seq.stability_result is None
        assert result.stability_result is not None

    def test_custom_thresholds(self) -> None:
        """Relaxed thresholds → fast motion passes."""
        seq = _fast_motion_sequence(100)
        result = analyze_stability(
            seq,
            max_center_motion=100.0,
            max_scale_change_ratio=1.0,
            max_stability_anomaly_ratio=1.0,
        )
        assert result.stability_result.passed is True


# ---------------------------------------------------------------------------
# Stage5_3TemporalStability integration tests
# ---------------------------------------------------------------------------


class TestStage5_3TemporalStability:
    def test_enabled_mode_analyzes(self) -> None:
        seq = _stable_sequence(100)
        stage = Stage5_3TemporalStability()
        output = stage.run(input_data=[seq], context=_make_context())

        assert output[0].stability_result is not None
        assert output[0].stability_result.passed is True

    def test_disabled_mode_passthrough(self) -> None:
        seq = _fast_motion_sequence(100)
        stage = Stage5_3TemporalStability()
        context = _make_context({
            "enhancements": {
                "stage_05_3_temporal_stability": {"enabled": False},
            },
        })
        output = stage.run(input_data=[seq], context=context)

        assert output[0].stability_result is None

    def test_mixed_pass_and_fail(self) -> None:
        good = _stable_sequence(50)
        bad = _fast_motion_sequence(100)  # 100 frames so fast motion activates at frame 50
        stage = Stage5_3TemporalStability()
        output = stage.run(input_data=[good, bad], context=_make_context())

        # Hard filter: only the passing sequence survives downstream.
        assert len(output) == 1
        assert output[0].stability_result is not None
        assert output[0].stability_result.passed is True

    def test_all_rejected_returns_empty(self) -> None:
        bad = _fast_motion_sequence(100)
        stage = Stage5_3TemporalStability()
        output = stage.run(input_data=[bad], context=_make_context())

        # No passing sequences -> empty output (downstream stages skip).
        assert output == []

    def test_disabled_passthrough_keeps_all(self) -> None:
        good = _stable_sequence(50)
        bad = _fast_motion_sequence(100)
        stage = Stage5_3TemporalStability()
        context = _make_context({
            "enhancements": {
                "stage_05_3_temporal_stability": {"enabled": False},
            },
        })
        output = stage.run(input_data=[good, bad], context=context)

        # Disabled = pure passthrough: no dropping, no stability_result attached.
        assert len(output) == 2
        assert all(s.stability_result is None for s in output)

    def test_input_not_mutated(self) -> None:
        seq = _stable_sequence(50)
        orig_id = seq.clip_id
        stage = Stage5_3TemporalStability()
        stage.run(input_data=[seq], context=_make_context())

        assert seq.clip_id == orig_id
        assert seq.stability_result is None

    def test_input_wrong_type_raises(self) -> None:
        stage = Stage5_3TemporalStability()
        assert stage.validate_input("wrong", _make_context()) is False
        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_empty_list_raises(self) -> None:
        stage = Stage5_3TemporalStability()
        assert stage.validate_input([], _make_context()) is False

    def test_input_wrong_element_type(self) -> None:
        stage = Stage5_3TemporalStability()
        assert stage.validate_input(["bad"], _make_context()) is False

    def test_validate_output(self) -> None:
        stage = Stage5_3TemporalStability()
        assert stage.validate_output([_seq()], _make_context()) is True
        assert stage.validate_output("bad", _make_context()) is False

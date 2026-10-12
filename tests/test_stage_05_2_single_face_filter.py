"""Tests for Stage 5.2: Single Face Identity Anomaly Filtering.

Unified anomaly ratio: every frame that is NOT "single face of main_track"
counts as an anomaly. One threshold controls all rejection.

Tests cover:
    1. Mixed anomaly (empty + identity_switch) → rejected.
    2. All same track → passed.
    3. 1 multi-face frame within threshold → passed.
    4. Threshold change → result follows config.
    5. disabled → passthrough.
    6. Empty sequence.
    7. Identity switch.
    8. Input immutability.
    9. Input validation.
"""

from pathlib import Path

import pytest

from face_pipeline_v2.data.models import FaceInfo, FaceSequence, FrameInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_2_single_face_filter.single_face_filter import (
    filter_sequence,
    REASON_EMPTY_FACE,
    REASON_IDENTITY_SWITCH,
    REASON_MULTI_FACE,
)
from face_pipeline_v2.stages.stage_05_2_single_face_filter.stage import (
    Stage5_2SingleFaceFilter,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A second, clearly non-overlapping face box used to model a *genuine*
# identity switch (a different person at a different screen position).  It is
# spatially incompatible with the default box below, so Stage 5.2's
# fragment-merge pre-pass keeps it as a separate identity instead of collapsing
# it into the primary track.
DIFF_BBOX = (300.0, 250.0, 400.0, 390.0)


def _face(
    frame_index: int, face_id: str, track_id: str | None,
    bbox: tuple = (50.0, 30.0, 150.0, 170.0),
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index,
        face_id=face_id,
        bbox=bbox,
        landmark_5=(),
        confidence=0.9,
        face_crop_path=Path(""),
        track_id=track_id,
    )


def _seq(
    clip_id: str = "v_s0_c0",
    video_id: str = "v",
    frames: list[FrameInfo] | None = None,
) -> FaceSequence:
    return FaceSequence(
        clip_id=clip_id, video_id=video_id,
        frames=tuple(frames or []),
    )


def _seq_single(
    count: int, track_id: str = "track_000", start_idx: int = 0,
    bbox: tuple = (50.0, 30.0, 150.0, 170.0),
) -> list[FrameInfo]:
    return [
        FrameInfo(
            frame_index=i, clip_id="c0", timestamp=i / 25.0, image_path=Path(""),
            faces=(_face(i, f"f_{i}", track_id, bbox=bbox),),
        )
        for i in range(start_idx, start_idx + count)
    ]


def _seq_empty(count: int, start_idx: int = 0) -> list[FrameInfo]:
    return [
        FrameInfo(
            frame_index=i, clip_id="c0", timestamp=i / 25.0, image_path=Path(""),
            faces=(),
        )
        for i in range(start_idx, start_idx + count)
    ]


def _seq_multi(
    count: int, track_a: str = "track_A", track_b: str = "track_B", start_idx: int = 0
) -> list[FrameInfo]:
    frames: list[FrameInfo] = []
    for i in range(start_idx, start_idx + count):
        frames.append(FrameInfo(
            frame_index=i, clip_id="c0", timestamp=i / 25.0, image_path=Path(""),
            faces=(_face(i, f"fa_{i}", track_a), _face(i, f"fb_{i}", track_b)),
        ))
    return frames


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "enhancements": {
            "stage_05_2_single_face_filter": {
                "enabled": True,
                "max_identity_anomaly_ratio": 0.005,
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
        logger=PipelineLogger(name="test_stage_05_2", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# filter_sequence unit tests
# ---------------------------------------------------------------------------


class TestAnomalyFilter:
    def test_mixed_anomaly_rejected(self) -> None:
        """198 track_A + 1 empty + 1 track_B → 2/200 anomaly → rejected."""
        seq = _seq(frames=(
            _seq_single(198, "track_A")
            + _seq_empty(1)
            + _seq_single(1, "track_B", start_idx=199, bbox=DIFF_BBOX)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.005)

        assert result.identity_result is not None
        assert result.identity_result.passed is False
        assert result.identity_result.total_frames == 200
        assert result.identity_result.anomaly_frames == 2
        assert result.identity_result.anomaly_ratio == pytest.approx(0.01)
        assert result.identity_result.main_track_id == "track_A"
        assert REASON_EMPTY_FACE in result.identity_result.reject_reasons
        assert REASON_IDENTITY_SWITCH in result.identity_result.reject_reasons

    def test_all_same_track_passes(self) -> None:
        """200 frames all track_000 → 0 anomaly → passed."""
        seq = _seq(frames=_seq_single(200, "track_000"))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.005)

        assert result.identity_result.passed is True
        assert result.identity_result.anomaly_frames == 0
        assert result.identity_result.anomaly_ratio == 0.0
        assert result.identity_result.reject_reasons == ()

    def test_one_multi_frame_within_threshold_passes(self) -> None:
        """199 track_A + 1 multi → 1/200=0.005 ≤ 0.005 → passed."""
        seq = _seq(frames=(
            _seq_single(199, "track_A")
            + _seq_multi(1, start_idx=199)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.005)

        assert result.identity_result.passed is True
        assert result.identity_result.anomaly_frames == 1
        assert result.identity_result.anomaly_ratio == pytest.approx(0.005)

    def test_one_empty_frame_within_threshold_passes(self) -> None:
        """199 track_A + 1 empty → 1/200=0.005 ≤ 0.005 → passed."""
        seq = _seq(frames=(
            _seq_single(199, "track_A")
            + _seq_empty(1, start_idx=199)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.005)

        assert result.identity_result.passed is True

    def test_one_identity_switch_within_threshold_passes(self) -> None:
        """199 track_A + 1 track_B → 1/200=0.005 ≤ 0.005 → passed."""
        seq = _seq(frames=(
            _seq_single(199, "track_A")
            + _seq_single(1, "track_B", start_idx=199, bbox=DIFF_BBOX)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.005)

        assert result.identity_result.passed is True
        assert REASON_IDENTITY_SWITCH in result.identity_result.reject_reasons

    def test_identity_switch_rejected(self) -> None:
        """100 track_A + 100 track_B → 100 anomaly → rejected."""
        seq = _seq(frames=(
            _seq_single(100, "track_A")
            + _seq_single(100, "track_B", start_idx=100, bbox=DIFF_BBOX)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.005)

        assert result.identity_result.passed is False
        assert result.identity_result.anomaly_frames == 100
        assert REASON_IDENTITY_SWITCH in result.identity_result.reject_reasons

    def test_tighter_threshold_rejects(self) -> None:
        """Same data, max_ratio=0 → any anomaly rejects."""
        seq = _seq(frames=(
            _seq_single(199, "track_A")
            + _seq_single(1, "track_B", start_idx=199, bbox=DIFF_BBOX)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.0)

        assert result.identity_result.passed is False

    def test_relaxed_threshold_accepts(self) -> None:
        """max_ratio=0.02 → 2/200 passes."""
        seq = _seq(frames=(
            _seq_single(198, "track_A")
            + _seq_empty(1, start_idx=198)
            + _seq_single(1, "track_B", start_idx=199)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=0.02)

        assert result.identity_result.passed is True

    def test_empty_sequence_fails(self) -> None:
        seq = _seq()
        result = filter_sequence(seq)

        assert result.identity_result.passed is False
        assert REASON_EMPTY_FACE in result.identity_result.reject_reasons

    def test_main_track_correct(self) -> None:
        """Main track is most frequent even with anomalies."""
        seq = _seq(frames=(
            _seq_single(150, "track_A")
            + _seq_multi(5, start_idx=150)
            + _seq_single(45, "track_B", start_idx=155)
        ))
        result = filter_sequence(seq, max_identity_anomaly_ratio=1.0)
        assert result.identity_result.main_track_id == "track_A"

    def test_no_track_ids_means_all_anomaly(self) -> None:
        """Stage 5.1 not run → all face frames lack track_id → anomaly."""
        # Build frames without track_id
        frames = [
            FrameInfo(
                frame_index=i, clip_id="c0", timestamp=i / 25.0, image_path=Path(""),
                faces=(FaceInfo(
                    frame_index=i, face_id=f"f_{i}", bbox=(0,0,100,100),
                    landmark_5=(), confidence=0.9, face_crop_path=Path(""),
                    track_id=None,
                ),),
            )
            for i in range(100)
        ]
        seq = _seq(frames=frames)
        result = filter_sequence(seq)

        assert result.identity_result.passed is False
        assert result.identity_result.anomaly_frames == 100
        assert REASON_IDENTITY_SWITCH in result.identity_result.reject_reasons

    def test_input_not_mutated(self) -> None:
        seq = _seq(frames=_seq_single(200, "track_000"))
        original_clip_id = seq.clip_id

        result = filter_sequence(seq)

        assert seq.clip_id == original_clip_id
        assert seq.identity_result is None
        assert result.identity_result is not None


# ---------------------------------------------------------------------------
# Stage5_2SingleFaceFilter integration tests
# ---------------------------------------------------------------------------


class TestStage5_2SingleFaceFilter:
    def test_enabled_filters(self) -> None:
        seq = _seq(frames=_seq_single(200, "track_000"))
        stage = Stage5_2SingleFaceFilter()
        output = stage.run(input_data=[seq], context=_make_context())

        assert output[0].identity_result.passed is True

    def test_disabled_passthrough(self) -> None:
        seq = _seq(frames=_seq_single(200, "track_000"))
        stage = Stage5_2SingleFaceFilter()
        context = _make_context({
            "enhancements": {
                "stage_05_2_single_face_filter": {"enabled": False},
            },
        })
        output = stage.run(input_data=[seq], context=context)

        assert output[0].identity_result is None

    def test_mixed_pass_and_fail(self) -> None:
        good = _seq(clip_id="c_good", frames=_seq_single(200, "track_000"))
        bad = _seq(clip_id="c_bad", frames=(
            _seq_single(100, "track_A")
            + _seq_single(100, "track_B", start_idx=100, bbox=DIFF_BBOX)
        ))
        stage = Stage5_2SingleFaceFilter()
        output = stage.run(input_data=[good, bad], context=_make_context())

        # Hard filter: only the passing sequence survives downstream.
        assert len(output) == 1
        assert output[0].clip_id == "c_good"
        assert output[0].identity_result.passed is True

    def test_all_rejected_returns_empty(self) -> None:
        bad = _seq(clip_id="c_bad", frames=(
            _seq_single(100, "track_A")
            + _seq_single(100, "track_B", start_idx=100, bbox=DIFF_BBOX)
        ))
        stage = Stage5_2SingleFaceFilter()
        output = stage.run(input_data=[bad], context=_make_context())

        # No passing sequences -> empty output (downstream stages skip).
        assert output == []

    def test_disabled_passthrough_keeps_all(self) -> None:
        good = _seq(clip_id="c_good", frames=_seq_single(200, "track_000"))
        bad = _seq(clip_id="c_bad", frames=(
            _seq_single(100, "track_A")
            + _seq_single(100, "track_B", start_idx=100)
        ))
        stage = Stage5_2SingleFaceFilter()
        context = _make_context({
            "enhancements": {
                "stage_05_2_single_face_filter": {"enabled": False},
            },
        })
        output = stage.run(input_data=[good, bad], context=context)

        # Disabled = pure passthrough: no dropping, no identity_result attached.
        assert len(output) == 2
        assert all(s.identity_result is None for s in output)

    def test_input_not_mutated(self) -> None:
        seq = _seq(frames=_seq_single(100, "track_000"))
        orig_id = seq.clip_id
        stage = Stage5_2SingleFaceFilter()
        stage.run(input_data=[seq], context=_make_context())

        assert seq.clip_id == orig_id
        assert seq.identity_result is None

    def test_input_wrong_type_raises(self) -> None:
        stage = Stage5_2SingleFaceFilter()
        assert stage.validate_input("wrong", _make_context()) is False
        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_empty_list_raises(self) -> None:
        stage = Stage5_2SingleFaceFilter()
        assert stage.validate_input([], _make_context()) is False

    def test_input_wrong_element_type(self) -> None:
        stage = Stage5_2SingleFaceFilter()
        assert stage.validate_input(["bad"], _make_context()) is False

    def test_validate_output(self) -> None:
        stage = Stage5_2SingleFaceFilter()
        assert stage.validate_output([_seq()], _make_context()) is True
        assert stage.validate_output("bad", _make_context()) is False

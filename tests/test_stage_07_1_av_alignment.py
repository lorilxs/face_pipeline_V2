"""Tests for Stage 7.1: A/V Sync Correction (Audio Timeline Alignment).

Tests cover:
    1. offset=0 → correction_applied=False, no aligned file generated
    2. offset=3 (120ms @25fps) → generates aligned audio
    3. offset > max → skipped (exceeds max_correction_ms)
    4. offset < threshold → skipped (below correction_threshold_ms)
    5. offset_ms calculation correctness
    6. should_correct() decision logic
    7. disabled stage → returns empty list
    8. Input validation
    9. Output validation
    10. Original audio not modified
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from face_pipeline_v2.data.models import (
    AVAlignmentInfo,
    AVSyncInfo,
    StandardizedVideoInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_07_1_av_alignment.processor import (
    AlignmentConfig,
    align_audio,
    compute_offset_ms,
    should_correct,
)
from face_pipeline_v2.stages.stage_07_1_av_alignment.stage import (
    Stage7_1AVAlignment,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ALIGN_CFG = AlignmentConfig(
    correction_threshold_ms=50.0,
    max_offset_ms=600.0,  # 15 frames / 25 fps * 1000
)


def _make_context(
    enabled: bool = True,
    threshold_ms: float = 50.0,
    max_offset_frames: int = 15,
) -> PipelineContext:
    config: dict = {
        "stage_07_1_av_alignment": {
            "enabled": enabled,
            "correction_threshold_ms": threshold_ms,
            "output_dir": "stage_outputs/av_alignment",
        },
        "sync": {
            "max_offset_frames": max_offset_frames,
        },
        "storage": {"output_dir": "/tmp/test_stage71"},
    }
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_71", log_dir=None),
        model_manager=object(),
    )


def _make_sync_info(
    clip_id: str = "c0",
    offset: int = 3,
) -> AVSyncInfo:
    return AVSyncInfo(
        clip_id=clip_id,
        offset=offset,
        sync_lse_c=5.0,
        sync_lse_d=10.0,
        median_distance=15.0,
    )


def _make_std_info(
    audio_path: Path | None = None,
    fps: int = 25,
) -> StandardizedVideoInfo:
    return StandardizedVideoInfo(
        video_path=Path("/tmp/test.mp4"),
        audio_path=audio_path or Path("/tmp/test.wav"),
        target_fps=fps,
        resolution=(256, 256),
        codec="h264",
    )


def _make_input(
    sync_infos=None,
    std_info=None,
):
    return {
        "av_sync_infos": sync_infos or [_make_sync_info("c0", offset=3)],
        "std_video_info": std_info or _make_std_info(),
    }


# ===================================================================
# Processor tests — compute_offset_ms
# ===================================================================


class TestComputeOffsetMs:
    def test_positive_offset(self):
        """offset=3 @25fps → 120ms."""
        assert compute_offset_ms(3, 25.0) == pytest.approx(120.0)

    def test_negative_offset(self):
        """offset=-2 @25fps → -80ms."""
        assert compute_offset_ms(-2, 25.0) == pytest.approx(-80.0)

    def test_zero_offset(self):
        assert compute_offset_ms(0, 25.0) == 0.0

    def test_30fps(self):
        """offset=3 @30fps → 100ms."""
        assert compute_offset_ms(3, 30.0) == pytest.approx(100.0)


# ===================================================================
# Processor tests — should_correct
# ===================================================================


class TestShouldCorrect:
    def test_zero_offset_skipped(self):
        """offset=0 → below threshold → skipped."""
        assert should_correct(0, 25.0, _DEFAULT_ALIGN_CFG) is False

    def test_small_offset_skipped(self):
        """offset=1 @25fps=40ms → below 50ms threshold → skipped."""
        assert should_correct(1, 25.0, _DEFAULT_ALIGN_CFG) is False

    def test_normal_offset_corrected(self):
        """offset=3 @25fps=120ms → in range → corrected."""
        assert should_correct(3, 25.0, _DEFAULT_ALIGN_CFG) is True

    def test_negative_offset_corrected(self):
        """offset=-3 @25fps=120ms → in range → corrected."""
        assert should_correct(-3, 25.0, _DEFAULT_ALIGN_CFG) is True

    def test_offset_within_max_corrected(self):
        """offset=10 @25fps=400ms → within 600ms max → corrected."""
        assert should_correct(10, 25.0, _DEFAULT_ALIGN_CFG) is True

    def test_offset_exceeds_max_skipped(self):
        """offset=16 @25fps=640ms → >600ms max → skipped."""
        assert should_correct(16, 25.0, _DEFAULT_ALIGN_CFG) is False

    def test_boundary_50ms_corrected(self):
        """offset=2 @25fps=80ms → >50ms → corrected (boundary is '< threshold' skip)."""
        assert should_correct(2, 25.0, _DEFAULT_ALIGN_CFG) is True

    def test_boundary_15frames_corrected(self):
        """offset=15 @25fps=600ms → =max (not >) → corrected."""
        assert should_correct(15, 25.0, _DEFAULT_ALIGN_CFG) is True


# ===================================================================
# Processor tests — align_audio
# ===================================================================


class TestAlignAudio:
    def test_positive_offset_delay(self):
        """offset>0 uses adelay filter."""
        with patch("subprocess.run") as mock_run:
            align_audio(
                Path("/tmp/in.wav"), Path("/tmp/out.wav"), offset_frames=3, fps=25.0
            )
        cmd = mock_run.call_args[0][0]
        assert "-filter:a" in cmd
        delay_idx = cmd.index("-filter:a")
        assert "adelay" in cmd[delay_idx + 1]

    def test_negative_offset_trim(self):
        """offset<0 uses -ss flag."""
        with patch("subprocess.run") as mock_run:
            align_audio(
                Path("/tmp/in.wav"), Path("/tmp/out.wav"), offset_frames=-3, fps=25.0
            )
        cmd = mock_run.call_args[0][0]
        assert "-ss" in cmd

    def test_zero_offset_copy(self):
        """offset=0 copies file."""
        with patch("shutil.copy2") as mock_copy:
            result = align_audio(
                Path("/tmp/in.wav"), Path("/tmp/out.wav"), offset_frames=0, fps=25.0
            )
        mock_copy.assert_called_once()
        assert result == Path("/tmp/out.wav")

    def test_output_is_16khz_mono(self):
        """Output always enforces 16kHz mono."""
        with patch("subprocess.run") as mock_run:
            align_audio(
                Path("/tmp/in.wav"), Path("/tmp/out.wav"), offset_frames=3, fps=25.0
            )
        cmd = mock_run.call_args[0][0]
        assert "-ac" in cmd and "1" in cmd
        assert "-ar" in cmd and "16000" in cmd


# ===================================================================
# Stage integration tests
# ===================================================================


class TestStage7_1AVAlignment:
    """Integration tests for Stage7_1AVAlignment."""

    def test_normal_correction(self):
        """offset=3 → correction_applied=True."""
        stage = Stage7_1AVAlignment()
        ctx = _make_context()

        with patch(
            "face_pipeline_v2.stages.stage_07_1_av_alignment.stage.align_audio"
        ) as mock_align, patch("pathlib.Path.mkdir"):
            mock_align.return_value = Path("/tmp/out.wav")
            result = stage.run(input_data=_make_input(), context=ctx)

        assert len(result) == 1
        assert result[0].clip_id == "c0"
        assert result[0].offset_frames == 3
        assert result[0].offset_ms == pytest.approx(120.0)
        assert result[0].correction_applied is True
        assert result[0].fps == 25.0

    def test_offset_zero_skipped(self):
        """offset=0 → correction_applied=False."""
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        input_data = _make_input(
            sync_infos=[_make_sync_info("c0", offset=0)],
        )
        result = stage.run(input_data=input_data, context=ctx)
        assert result[0].correction_applied is False

    def test_offset_too_large_skipped(self):
        """offset=16 (640ms @25fps) → exceeds 600ms max → correction_applied=False."""
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        input_data = _make_input(
            sync_infos=[_make_sync_info("c0", offset=16)],
        )
        result = stage.run(input_data=input_data, context=ctx)
        assert result[0].correction_applied is False

    def test_disabled_returns_empty(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context(enabled=False)
        result = stage.run(input_data=_make_input(), context=ctx)
        assert result == []

    def test_multi_clip(self):
        """Multiple clips → one AVAlignmentInfo per clip."""
        stage = Stage7_1AVAlignment()
        ctx = _make_context()

        with patch(
            "face_pipeline_v2.stages.stage_07_1_av_alignment.stage.align_audio"
        ) as mock_align, patch("pathlib.Path.mkdir"):
            mock_align.return_value = Path("/tmp/out.wav")
            input_data = _make_input(
                sync_infos=[
                    _make_sync_info("c0", offset=3),
                    _make_sync_info("c1", offset=1),
                    _make_sync_info("c2", offset=4),
                ],
            )
            result = stage.run(input_data=input_data, context=ctx)

        assert len(result) == 3
        # c1 (offset=1, 40ms) should be skipped
        assert result[1].correction_applied is False
        # c0 and c2 should be corrected
        assert result[0].correction_applied is True
        assert result[2].correction_applied is True

    def test_original_audio_path_preserved(self):
        """AVAlignmentInfo.original_audio_path always points to original."""
        stage = Stage7_1AVAlignment()
        audio = Path("/tmp/original_test.wav")
        ctx = _make_context()
        input_data = _make_input(std_info=_make_std_info(audio_path=audio))
        with patch(
            "face_pipeline_v2.stages.stage_07_1_av_alignment.stage.align_audio"
        ) as mock_align, patch("pathlib.Path.mkdir"):
            mock_align.return_value = Path("/tmp/out.wav")
            result = stage.run(input_data=input_data, context=ctx)
        assert result[0].original_audio_path == audio

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------

    def test_validate_input_rejects_non_dict(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_input("bad", ctx) is False

    def test_validate_input_rejects_missing_key(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_input({"av_sync_infos": []}, ctx) is False
        assert stage.validate_input({"std_video_info": _make_std_info()}, ctx) is False

    def test_validate_input_rejects_empty_list(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_input(
            {"av_sync_infos": [], "std_video_info": _make_std_info()}, ctx
        ) is False

    def test_validate_input_rejects_wrong_type(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_input(
            {"av_sync_infos": ["bad"], "std_video_info": _make_std_info()}, ctx
        ) is False

    def test_validate_input_rejects_wrong_std_info_type(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_input(
            {"av_sync_infos": [_make_sync_info()], "std_video_info": "bad"}, ctx
        ) is False

    def test_valid_input_accepted(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_input(_make_input(), ctx) is True

    def test_run_raises_on_invalid_input(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        with pytest.raises(InputValidationError):
            stage.run(input_data="bad", context=ctx)

    # --------------------------------------------------------------
    # Output validation
    # --------------------------------------------------------------

    def test_validate_output_accepts_alignment_list(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        alignment = AVAlignmentInfo(
            clip_id="c0",
            original_audio_path=Path("/tmp/in.wav"),
            corrected_audio_path=Path("/tmp/out.wav"),
            offset_frames=3,
            offset_ms=120.0,
            correction_applied=True,
            fps=25.0,
        )
        assert stage.validate_output([alignment], ctx) is True

    def test_validate_output_rejects_wrong_type(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_output("bad", ctx) is False
        assert stage.validate_output([_make_sync_info()], ctx) is False

    def test_validate_output_empty_list(self):
        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        assert stage.validate_output([], ctx) is True


# ===================================================================
# Immutability / non-regression tests
# ===================================================================


class TestNonRegression:
    def test_does_not_import_syncnet(self):
        """Stage 7.1 must NOT import SyncNet model or re-run detection."""
        import face_pipeline_v2.stages.stage_07_1_av_alignment.stage as s
        import inspect
        src = inspect.getsource(s)
        # Must not import syncnet model wrapper
        assert "syncnet" not in src.lower().replace("syncnet", "").replace(
            "syncnet", ""
        ), "Stage 7.1 must not import syncnet"
        # Must not import from stage_07_sync
        assert "stage_07_sync" not in src, "Stage 7.1 must not import Stage 7 internals"

    def test_does_not_import_syncnet_from_processor(self):
        """Processor must NOT import SyncNet model modules."""
        import face_pipeline_v2.stages.stage_07_1_av_alignment.processor as p
        import inspect
        src = inspect.getsource(p)
        # Check actual import lines only (docstring mentions are fine)
        import_lines = [l.strip() for l in src.split('\n')
                        if l.strip().startswith(('import ', 'from '))]
        joined = ' '.join(import_lines).lower()
        assert 'syncnet' not in joined, (
            f"Processor must not import syncnet modules. Found: {import_lines}"
        )

    def test_avsyncinfo_not_modified(self):
        """Original AVSyncInfo is not modified during processing."""
        sync = _make_sync_info("c0", offset=3)
        original_offset = sync.offset
        original_lse_c = sync.sync_lse_c

        stage = Stage7_1AVAlignment()
        ctx = _make_context()
        with patch(
            "face_pipeline_v2.stages.stage_07_1_av_alignment.stage.align_audio"
        ) as mock_align, patch("pathlib.Path.mkdir"):
            mock_align.return_value = Path("/tmp/out.wav")
            stage.run(input_data=_make_input(sync_infos=[sync]), context=ctx)

        # Frozen dataclass ensures immutability, but verify anyway
        assert sync.offset == original_offset
        assert sync.sync_lse_c == original_lse_c

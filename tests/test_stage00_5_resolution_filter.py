"""Tests for Stage 0.5: Resolution Filter.

Tests cover:
    1. Processor: pass/fail decision for HD / boundary / low-resolution /
       zero-dimension inputs.
    2. Stage: full run, disabled passthrough, input/output validation,
       reject warning logging.
"""

from pathlib import Path
from unittest.mock import MagicMock

from face_pipeline_v2.data.models import ResolutionValidationInfo, VideoInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_00_5_resolution_filter.processor import (
    REJECT_REASON_RESOLUTION_TOO_LOW,
    validate_resolution,
)
from face_pipeline_v2.stages.stage_00_5_resolution_filter.stage import (
    Stage0_5ResolutionFilter,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video_info(width: int, height: int) -> VideoInfo:
    """Build a minimal VideoInfo with the given dimensions."""
    return VideoInfo(
        video_id="test_video",
        video_path=Path("/fake/test_video.mp4"),
        original_fps=25.0,
        width=width,
        height=height,
        duration=10.0,
        total_frames=250,
        codec="h264",
        has_audio=True,
    )


def _make_context(
    config: dict | None = None,
    logger: PipelineLogger | MagicMock | None = None,
) -> PipelineContext:
    """Build a PipelineContext with stage_00_resolution_filter config."""
    if config is None:
        config = {
            "stage_00_resolution_filter": {
                "enabled": True,
                "min_width": 640,
                "min_height": 360,
            },
        }
    return PipelineContext(
        config=config,
        logger=logger if logger is not None else PipelineLogger(
            name="test_stage00_5", log_dir=None
        ),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# Processor unit tests
# ---------------------------------------------------------------------------

class TestResolutionProcessor:
    def test_1920x1080_valid(self) -> None:
        """HD resolution passes."""
        r = validate_resolution(_make_video_info(1920, 1080), 640, 360)
        assert isinstance(r, ResolutionValidationInfo)
        assert r.is_valid is True
        assert r.reject_reason is None

    def test_640x360_valid(self) -> None:
        """Exact minimum resolution passes (boundary)."""
        r = validate_resolution(_make_video_info(640, 360), 640, 360)
        assert r.is_valid is True
        assert r.reject_reason is None

    def test_320x240_invalid(self) -> None:
        """Low resolution is rejected."""
        r = validate_resolution(_make_video_info(320, 240), 640, 360)
        assert r.is_valid is False
        assert r.reject_reason == REJECT_REASON_RESOLUTION_TOO_LOW

    def test_zero_dimensions_invalid(self) -> None:
        """Missing dimensions (0x0) are rejected."""
        r = validate_resolution(_make_video_info(0, 0), 640, 360)
        assert r.is_valid is False
        assert r.reject_reason == REJECT_REASON_RESOLUTION_TOO_LOW

    def test_only_width_below_invalid(self) -> None:
        """Width below minimum rejects even when height is fine."""
        r = validate_resolution(_make_video_info(320, 720), 640, 360)
        assert r.is_valid is False

    def test_only_height_below_invalid(self) -> None:
        """Height below minimum rejects even when width is fine."""
        r = validate_resolution(_make_video_info(1920, 240), 640, 360)
        assert r.is_valid is False

    def test_result_fields_roundtrip(self) -> None:
        """ResolutionValidationInfo carries width/height/min/decision."""
        r = validate_resolution(_make_video_info(1920, 1080), 640, 360)
        assert r.width == 1920
        assert r.height == 1080
        assert r.min_width == 640
        assert r.min_height == 360


# ---------------------------------------------------------------------------
# Stage unit tests
# ---------------------------------------------------------------------------

class TestStage0_5ResolutionFilter:
    def test_run_accepts_hd(self) -> None:
        stage = Stage0_5ResolutionFilter()
        out = stage.run(_make_video_info(1920, 1080), _make_context())
        assert isinstance(out, ResolutionValidationInfo)
        assert out.is_valid is True

    def test_run_rejects_low_res(self) -> None:
        stage = Stage0_5ResolutionFilter()
        out = stage.run(_make_video_info(320, 240), _make_context())
        assert out.is_valid is False
        assert out.reject_reason == REJECT_REASON_RESOLUTION_TOO_LOW

    def test_run_missing_dimensions_rejected(self) -> None:
        stage = Stage0_5ResolutionFilter()
        out = stage.run(_make_video_info(0, 0), _make_context())
        assert out.is_valid is False

    def test_run_reject_logs_warning(self) -> None:
        logger = MagicMock()
        stage = Stage0_5ResolutionFilter()
        out = stage.run(_make_video_info(320, 240), _make_context(logger=logger))
        assert out.is_valid is False
        assert logger.warning.called

    def test_run_accept_logs_no_warning(self) -> None:
        logger = MagicMock()
        stage = Stage0_5ResolutionFilter()
        out = stage.run(_make_video_info(1920, 1080), _make_context(logger=logger))
        assert out.is_valid is True
        assert not logger.warning.called

    def test_disabled_passes_through(self) -> None:
        stage = Stage0_5ResolutionFilter()
        ctx = _make_context(config={
            "stage_00_resolution_filter": {
                "enabled": False,
                "min_width": 640,
                "min_height": 360,
            },
        })
        out = stage.run(_make_video_info(320, 240), ctx)
        assert out.is_valid is True
        assert out.reject_reason is None

    def test_default_thresholds_when_config_missing(self) -> None:
        """Missing config section falls back to 640x360 defaults."""
        stage = Stage0_5ResolutionFilter()
        ctx = _make_context(config={})
        assert stage.run(_make_video_info(640, 360), ctx).is_valid is True
        assert stage.run(_make_video_info(320, 240), ctx).is_valid is False

    def test_validate_input_rejects_non_videoinfo(self) -> None:
        stage = Stage0_5ResolutionFilter()
        assert stage.validate_input({"wrong": "type"}, _make_context()) is False
        assert stage.validate_input(_make_video_info(1920, 1080), _make_context()) is True

    def test_validate_output(self) -> None:
        stage = Stage0_5ResolutionFilter()
        valid = ResolutionValidationInfo(
            width=1920, height=1080, min_width=640, min_height=360,
            is_valid=True, reject_reason=None,
        )
        assert stage.validate_output(valid, _make_context()) is True
        assert stage.validate_output({"wrong": "type"}, _make_context()) is False

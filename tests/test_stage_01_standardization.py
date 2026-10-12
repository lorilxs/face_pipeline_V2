"""Tests for Stage 1: AV Standardization.

Tests cover:
    1. Normal video conversion + audio extraction (mocked ffmpeg).
    2. FFmpeg not found → MissingDependencyError.
    3. Input file not found → InputValidationError.
    4. Output file missing after "successful" ffmpeg run → DataProcessingError.
    5. FFmpegProcessor unit tests.
"""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from face_pipeline_v2.data.models import AudioInfo, StandardizedVideoInfo, VideoInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_01_standardization.ffmpeg_processor import (
    FFmpegProcessor,
)
from face_pipeline_v2.stages.stage_01_standardization.stage import (
    Stage1AVStandardization,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
    MissingDependencyError,
    PipelineError,
    VideoReadError,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_video_info(
    video_path: Path = Path("/fake/input/video.mp4"),
    video_id: str = "test_video",
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    duration: float = 10.0,
) -> VideoInfo:
    """Build a realistic VideoInfo for testing Stage 1."""
    return VideoInfo(
        video_id=video_id,
        video_path=video_path,
        original_fps=fps,
        width=width,
        height=height,
        duration=duration,
        total_frames=int(fps * duration),
        codec="h264",
        has_audio=True,
    )


def _make_context(
    config_overrides: dict | None = None,
) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    config: dict[str, Any] = {
        "stage_01": {
            "target_fps": 25,
            "audio_sample_rate": 16000,
            "video_codec": "libx264",
            "audio_codec": "pcm_s16le",
            "ffmpeg_path": None,
            "output_dir": "/fake/outputs/videos",
        },
    }
    if config_overrides:
        config.update(config_overrides)
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_01", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# FFmpegProcessor unit tests
# ---------------------------------------------------------------------------


class TestFFmpegProcessor:
    """Unit tests for FFmpegProcessor."""

    def test_detect_ffmpeg_missing_raises(self) -> None:
        """Verify MissingDependencyError when ffmpeg not on PATH."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(
                MissingDependencyError, match="FFmpeg not found"
            ):
                FFmpegProcessor(ffmpeg_path=None)

    def test_convert_video_input_missing(self) -> None:
        """Verify VideoReadError when input doesn't exist."""
        proc = FFmpegProcessor(ffmpeg_path="/fake/ffmpeg")
        with pytest.raises(VideoReadError, match="Input video not found"):
            proc.convert_video(
                input_path=Path("/nonexistent/v.mp4"),
                output_path=Path("/fake/out.mp4"),
                target_fps=25,
            )

    def test_convert_video_ffmpeg_failure(self) -> None:
        """Verify DataProcessingError when ffmpeg returns non-zero."""
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        proc = FFmpegProcessor(ffmpeg_path="/fake/ffmpeg")
        with patch("subprocess.run", return_value=fake_result), patch(
            "pathlib.Path.exists", return_value=True
        ):
            with pytest.raises(
                DataProcessingError, match="FFmpeg failed"
            ):
                proc.convert_video(
                    input_path=Path("/fake/in.mp4"),
                    output_path=Path("/fake/out.mp4"),
                    target_fps=25,
                )

    def test_convert_video_output_missing_after_run(self) -> None:
        """Verify DataProcessingError when ffmpeg exits 0 but output not created."""
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        proc = FFmpegProcessor(ffmpeg_path="/fake/ffmpeg")

        # First call to exists(): input_path check → True
        # After subprocess, output_path check: both fail
        exists_values = iter([True, False])

        with patch("subprocess.run", return_value=fake_result), patch(
            "pathlib.Path.exists",
            side_effect=lambda: next(exists_values),
        ), patch("pathlib.Path.mkdir"):
            with pytest.raises(
                DataProcessingError, match="output file not found"
            ):
                proc.convert_video(
                    input_path=Path("/fake/in.mp4"),
                    output_path=Path("/fake/out.mp4"),
                    target_fps=25,
                )

    def test_extract_audio_input_missing(self) -> None:
        """Verify VideoReadError when audio input doesn't exist."""
        proc = FFmpegProcessor(ffmpeg_path="/fake/ffmpeg")
        with pytest.raises(
            VideoReadError, match="Input video not found for audio extraction"
        ):
            proc.extract_audio(
                input_path=Path("/nonexistent/v.mp4"),
                output_path=Path("/fake/out.wav"),
                sample_rate=16000,
            )

    def test_ffmpeg_path_property(self) -> None:
        """Verify ffmpeg_path property returns the resolved path."""
        proc = FFmpegProcessor(ffmpeg_path="/custom/ffmpeg")
        assert proc.ffmpeg_path == "/custom/ffmpeg"


# ---------------------------------------------------------------------------
# Stage1AVStandardization tests
# ---------------------------------------------------------------------------


class TestStage1AVStandardization:
    """Tests for Stage1AVStandardization with mocked ffmpeg."""

    @staticmethod
    def _make_successful_ffmpeg() -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

    def test_normal_conversion(self) -> None:
        """Verify Stage 1 successfully standardizes a valid video."""
        fake_result = self._make_successful_ffmpeg()
        video_info = _make_video_info()

        with patch("subprocess.run", return_value=fake_result), patch(
            "shutil.which", return_value="/fake/ffmpeg"
        ), patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.mkdir"
        ):
            stage = Stage1AVStandardization()
            context = _make_context()
            output = stage.run(input_data=video_info, context=context)

            assert isinstance(output, dict)
            assert isinstance(output["video_info"], StandardizedVideoInfo)
            assert isinstance(output["audio_info"], AudioInfo)
            std_info: StandardizedVideoInfo = output["video_info"]
            aud_info: AudioInfo = output["audio_info"]
            assert std_info.target_fps == 25
            assert aud_info.sample_rate == 16000
            assert aud_info.channels == 1

    def test_input_wrong_type_rejected(self) -> None:
        """Verify validate_input returns False for non-VideoInfo."""
        stage = Stage1AVStandardization()
        assert (
            stage.validate_input({"wrong": "type"}, _make_context()) is False
        )

    def test_input_file_missing_raises(self) -> None:
        """Verify InputValidationError when video file doesn't exist."""
        video_info = _make_video_info(
            video_path=Path("/nonexistent/v.mp4")
        )
        stage = Stage1AVStandardization()
        context = _make_context()
        with pytest.raises(
            InputValidationError, match="Invalid input data"
        ):
            stage.run(input_data=video_info, context=context)

    def test_validate_output_rejects_non_dict(self) -> None:
        """Verify validate_output returns False for non-dict output."""
        stage = Stage1AVStandardization()
        assert stage.validate_output("not_a_dict", _make_context()) is False

    def test_validate_output_rejects_missing_keys(self) -> None:
        """Verify validate_output returns False when keys are missing."""
        stage = Stage1AVStandardization()
        assert stage.validate_output(
            {"video_info": None}, _make_context()
        ) is False

    def test_ffmpeg_path_from_config(self) -> None:
        """Verify custom ffmpeg_path from config is used."""
        fake_result = self._make_successful_ffmpeg()
        video_info = _make_video_info()

        with patch("subprocess.run", return_value=fake_result) as mock_run, patch(
            "pathlib.Path.exists", return_value=True
        ), patch("pathlib.Path.mkdir"):
            stage = Stage1AVStandardization()
            context = _make_context(
                config_overrides={
                    "stage_01": {
                        "target_fps": 25,
                        "audio_sample_rate": 16000,
                        "video_codec": "libx264",
                        "audio_codec": "pcm_s16le",
                        "ffmpeg_path": "/custom/ffmpeg",
                        "output_dir": "/fake/outputs/videos",
                    },
                }
            )
            output = stage.run(input_data=video_info, context=context)
            assert output["video_info"].target_fps == 25

            # Check ffmpeg was called with the custom path
            video_calls = [
                c for c in mock_run.call_args_list
                if "-c:v" in c[0][0]
            ]
            assert len(video_calls) > 0
            assert video_calls[0][0][0][0] == "/custom/ffmpeg"

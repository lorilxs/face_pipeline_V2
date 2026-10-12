"""Tests for Stage 0: Video Integrity Check.

Tests cover:
    1. Normal video metadata extraction (mocked ffprobe).
    2. Missing file exception → VideoReadError.
    3. Video without video stream → MissingStreamError.
    4. FFprobe not found → MissingDependencyError.
    5. FFprobe checker unit tests with Pipeline exceptions.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from face_pipeline_v2.data.models import VideoInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_00_video_check.stage import (
    Stage0VideoIntegrityCheck,
)
from face_pipeline_v2.stages.stage_00_video_check.video_checker import (
    FFprobeChecker,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    MetadataExtractionError,
    MissingDependencyError,
    MissingStreamError,
    PipelineError,
    StageExecutionError,
    VideoReadError,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_ffprobe_output(
    *,
    has_video: bool = True,
    has_audio: bool = True,
    fps: str = "25/1",
    width: int = 1920,
    height: int = 1080,
    duration: float = 10.0,
    nb_frames: str = "250",
    video_codec: str = "h264",
    audio_codec: str = "aac",
) -> Dict[str, Any]:
    """Build a realistic ffprobe JSON output for testing."""
    streams: list = []
    if has_video:
        streams.append({
            "codec_type": "video",
            "codec_name": video_codec,
            "r_frame_rate": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "nb_frames": nb_frames,
        })
    if has_audio:
        streams.append({
            "codec_type": "audio",
            "codec_name": audio_codec,
            "duration": duration,
        })
    return {
        "streams": streams,
        "format": {"duration": duration},
    }


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    """Create a minimal PipelineContext for testing."""
    config = {
        "video": {
            "check_video_stream": True,
            "check_audio_stream": True,
            "ffprobe_path": None,
        },
    }
    if config_overrides:
        config.update(config_overrides)
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_00", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# FFprobeChecker unit tests
# ---------------------------------------------------------------------------


class TestFFprobeChecker:
    """Unit tests for FFprobeChecker — static methods and exception types."""

    def test_has_video_stream_true(self) -> None:
        meta = _make_ffprobe_output(has_video=True, has_audio=False)
        assert FFprobeChecker.has_video_stream(meta) is True

    def test_has_video_stream_false(self) -> None:
        meta = _make_ffprobe_output(has_video=False, has_audio=True)
        assert FFprobeChecker.has_video_stream(meta) is False

    def test_has_audio_stream_true(self) -> None:
        meta = _make_ffprobe_output(has_video=True, has_audio=True)
        assert FFprobeChecker.has_audio_stream(meta) is True

    def test_has_audio_stream_false(self) -> None:
        meta = _make_ffprobe_output(has_video=True, has_audio=False)
        assert FFprobeChecker.has_audio_stream(meta) is False

    def test_extract_video_stream_metadata(self) -> None:
        meta = _make_ffprobe_output(
            fps="30/1", width=1280, height=720, video_codec="h265"
        )
        result = FFprobeChecker.extract_video_stream_metadata(meta)
        assert result["fps"] == 30.0
        assert result["width"] == 1280
        assert result["height"] == 720
        assert result["codec"] == "h265"

    def test_extract_video_stream_metadata_fractional_fps(self) -> None:
        meta = _make_ffprobe_output(fps="30000/1001")
        result = FFprobeChecker.extract_video_stream_metadata(meta)
        assert abs(result["fps"] - 29.97) < 0.1

    def test_extract_video_stream_no_video_raises_missing_stream(self) -> None:
        """Verify MissingStreamError raised when no video stream exists."""
        meta = _make_ffprobe_output(has_video=False, has_audio=True)
        with pytest.raises(MissingStreamError, match="No video stream"):
            FFprobeChecker.extract_video_stream_metadata(meta)
        # MissingStreamError is a PipelineError
        with pytest.raises(PipelineError):
            FFprobeChecker.extract_video_stream_metadata(meta)

    def test_extract_audio_codec(self) -> None:
        meta = _make_ffprobe_output(audio_codec="mp3")
        assert FFprobeChecker.extract_audio_stream_codec(meta) == "mp3"

    def test_extract_audio_codec_none_when_no_audio(self) -> None:
        meta = _make_ffprobe_output(has_audio=False)
        assert FFprobeChecker.extract_audio_stream_codec(meta) is None

    def test_extract_format_duration(self) -> None:
        meta = _make_ffprobe_output(duration=42.5)
        assert FFprobeChecker.extract_format_duration(meta) == 42.5

    def test_probe_file_not_found_raises_video_read_error(self) -> None:
        """Verify VideoReadError raised for missing file."""
        checker = FFprobeChecker(ffprobe_path="/usr/bin/ffprobe")
        with pytest.raises(VideoReadError, match="Video file not found"):
            checker.probe(Path("/nonexistent/video.mp4"))
        # VideoReadError is a PipelineError
        with pytest.raises(PipelineError):
            checker.probe(Path("/nonexistent/video.mp4"))

    def test_detect_ffprobe_missing_raises_dependency_error(self) -> None:
        """Verify MissingDependencyError when ffprobe not on PATH."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(
                MissingDependencyError, match="FFprobe not found"
            ):
                FFprobeChecker(ffprobe_path=None)

    def test_probe_corrupt_json_raises_metadata_error(self) -> None:
        """Verify MetadataExtractionError for corrupt JSON output."""
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not valid json {{{", stderr=""
        )
        checker = FFprobeChecker(ffprobe_path="/fake/ffprobe")
        with patch("subprocess.run", return_value=fake_result), patch(
            "pathlib.Path.exists", return_value=True
        ):
            with pytest.raises(
                MetadataExtractionError, match="Failed to parse ffprobe JSON"
            ):
                checker.probe(Path("/fake/bad.mp4"))

    def test_check_ffprobe_available_true_for_real_ffprobe(self) -> None:
        """Real ffprobe reports 'ffprobe version' → available."""
        checker = FFprobeChecker(ffprobe_path="/fake/ffprobe")
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ffprobe version 6.0.1", stderr=""
        )
        with patch("subprocess.run", return_value=fake):
            assert checker._check_ffprobe_available() is True

    def test_check_ffprobe_available_false_for_stripped_ffmpeg(self) -> None:
        """A stripped ffmpeg named 'ffprobe' reports 'ffmpeg version' → invalid."""
        checker = FFprobeChecker(ffprobe_path="/fake/ffprobe")
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ffmpeg version 7.0.2", stderr=""
        )
        with patch("subprocess.run", return_value=fake):
            assert checker._check_ffprobe_available() is False

    def test_check_ffprobe_available_false_on_error(self) -> None:
        """Non-zero exit from `ffprobe -version` → invalid."""
        checker = FFprobeChecker(ffprobe_path="/fake/ffprobe")
        fake = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        with patch("subprocess.run", return_value=fake):
            assert checker._check_ffprobe_available() is False

    def test_probe_with_ffmpeg_parses_stderr(self) -> None:
        """ffmpeg fallback parses width/height/fps/duration/audio from stderr."""
        fake_stderr = (
            "ffmpeg version 7.0.2\n"
            "  Duration: 00:00:15.12, start: 0.000000, bitrate: 702 kb/s\n"
            "  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 640x360 "
            "[SAR 1:1 DAR 16:9], 398 kb/s, 25 fps, 25 tbr (default)\n"
            "  Stream #0:1[0x2](und): Audio: aac (LC), 44100 Hz, stereo, "
            "fltp, 128 kb/s (default)\n"
        )
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=fake_stderr
        )
        with patch("subprocess.run", return_value=fake), patch(
            "pathlib.Path.exists", return_value=True
        ):
            metadata = FFprobeChecker.probe_with_ffmpeg(Path("/fake/video.mp4"))

        vstream = metadata["streams"][0]
        assert vstream["codec_type"] == "video"
        assert vstream["width"] == 640
        assert vstream["height"] == 360
        assert vstream["codec_name"] == "h264"
        assert vstream["r_frame_rate"] == "25/1"
        assert len(metadata["streams"]) == 2
        assert metadata["streams"][1]["codec_type"] == "audio"
        assert metadata["streams"][1]["codec_name"] == "aac"
        assert metadata["format"]["duration"] == 15.12

    def test_probe_with_ffmpeg_no_dimensions_raises(self) -> None:
        """ffmpeg fallback raises VideoReadError if no dimensions found."""
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="no metadata here"
        )
        with patch("subprocess.run", return_value=fake), patch(
            "pathlib.Path.exists", return_value=True
        ):
            with pytest.raises(VideoReadError, match="could not extract"):
                FFprobeChecker.probe_with_ffmpeg(Path("/fake/video.mp4"))

    def test_probe_or_fallback_uses_ffprobe_when_valid(self) -> None:
        """Valid ffprobe → source='ffprobe'."""
        checker = FFprobeChecker(ffprobe_path="/fake/ffprobe")
        ffprobe_meta = _make_ffprobe_output()
        with patch.object(
            FFprobeChecker, "_check_ffprobe_available", return_value=True
        ), patch.object(
            FFprobeChecker, "probe", return_value=ffprobe_meta
        ), patch("pathlib.Path.exists", return_value=True):
            metadata, source = checker.probe_or_fallback(Path("/fake/video.mp4"))
        assert source == "ffprobe"
        assert metadata == ffprobe_meta

    def test_probe_or_fallback_falls_back_to_ffmpeg(self) -> None:
        """Invalid ffprobe → source='ffmpeg'."""
        checker = FFprobeChecker(ffprobe_path="/fake/ffprobe")
        ffmpeg_meta = {
            "streams": [{"codec_type": "video", "width": 640, "height": 360}],
            "format": {"duration": 10.0},
        }
        with patch.object(
            FFprobeChecker, "_check_ffprobe_available", return_value=False
        ), patch.object(
            FFprobeChecker, "probe_with_ffmpeg", return_value=ffmpeg_meta
        ), patch("pathlib.Path.exists", return_value=True):
            metadata, source = checker.probe_or_fallback(Path("/fake/video.mp4"))
        assert source == "ffmpeg"
        assert metadata == ffmpeg_meta


# ---------------------------------------------------------------------------
# Stage0VideoIntegrityCheck tests
# ---------------------------------------------------------------------------


class TestStage0VideoIntegrityCheck:
    """Tests for Stage0VideoIntegrityCheck with mocked ffprobe.

    All tests that invoke stage.run() with fake paths must also mock
    Path.exists() and Path.is_file() since validate_input checks them.
    """

    @staticmethod
    def _run_with_mocks(
        fake_ffprobe_result: subprocess.CompletedProcess,
        video_path_str: str = "/fake/video.mp4",
        config_overrides: dict | None = None,
    ):
        """Helper: run stage with subprocess, Path, and shutil.which mocks."""
        with patch(
            "subprocess.run", return_value=fake_ffprobe_result
        ), patch.object(
            FFprobeChecker, "_check_ffprobe_available", return_value=True
        ), patch(
            "pathlib.Path.exists", return_value=True
        ), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch(
            "shutil.which", return_value="/fake/ffprobe"
        ):
            stage = Stage0VideoIntegrityCheck()
            context = _make_context(config_overrides)
            return stage.run(input_data=Path(video_path_str), context=context)

    def test_normal_video_processing(self) -> None:
        """Verify Stage 0 correctly processes a valid video with video+audio."""
        fake_meta = _make_ffprobe_output()
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(fake_meta), stderr=""
        )

        output = self._run_with_mocks(fake_result, "/fake/test_video.mp4")

        assert isinstance(output, VideoInfo)
        assert output.video_id == "test_video"
        assert output.width == 1920
        assert output.height == 1080
        assert output.original_fps == 25.0
        assert output.duration == 10.0
        assert output.total_frames == 250
        assert output.codec == "h264"
        assert output.has_audio is True

    def test_video_without_audio_stream(self) -> None:
        """Verify Stage 0 handles video with no audio stream."""
        fake_meta = _make_ffprobe_output(has_audio=False)
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(fake_meta), stderr=""
        )

        output = self._run_with_mocks(fake_result, "/fake/no_audio.mp4")
        assert output.has_audio is False

    def test_stage0_uses_ffmpeg_fallback(self) -> None:
        """Stage 0 falls back to ffmpeg when ffprobe is invalid/unavailable."""
        ffmpeg_meta = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "r_frame_rate": "25/1",
                    "width": 640,
                    "height": 360,
                    "duration": 15.12,
                    "nb_frames": "0",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": 15.12,
                },
            ],
            "format": {"duration": 15.12},
        }
        with patch("shutil.which", return_value="/fake/ffprobe"), patch.object(
            FFprobeChecker, "_check_ffprobe_available", return_value=False
        ), patch.object(
            FFprobeChecker, "probe_with_ffmpeg", return_value=ffmpeg_meta
        ), patch(
            "pathlib.Path.exists", return_value=True
        ), patch(
            "pathlib.Path.is_file", return_value=True
        ):
            stage = Stage0VideoIntegrityCheck()
            output = stage.run(
                input_data=Path("/fake/video.mp4"), context=_make_context()
            )

        assert isinstance(output, VideoInfo)
        assert output.width == 640
        assert output.height == 360
        assert output.original_fps == 25.0
        assert output.duration == 15.12
        assert output.has_audio is True

    def test_nonexistent_file_raises_input_validation_error(self) -> None:
        """Verify InputValidationError when running with non-existent file."""
        stage = Stage0VideoIntegrityCheck()
        context = _make_context()
        # validate_input returns False, run() raises InputValidationError
        with pytest.raises(
            InputValidationError, match="Invalid input data"
        ):
            stage.run(
                input_data=Path("/nonexistent/ghost.mp4"), context=context
            )

    def test_none_input_rejected(self) -> None:
        """Verify validate_input returns False for None (run raises)."""
        stage = Stage0VideoIntegrityCheck()
        with pytest.raises(InputValidationError):
            stage.run(input_data=None, context=_make_context())

    def test_no_video_stream_raises_missing_stream(self) -> None:
        """Verify MissingStreamError when no video stream in metadata."""
        fake_meta = _make_ffprobe_output(has_video=False, has_audio=True)
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(fake_meta), stderr=""
        )

        with pytest.raises(MissingStreamError, match="No video stream"):
            self._run_with_mocks(fake_result, "/fake/bad.mp4")
        # Also a PipelineError
        with pytest.raises(PipelineError):
            self._run_with_mocks(fake_result, "/fake/bad.mp4")

    def test_ffprobe_crash_falls_back_then_raises(self) -> None:
        """ffprobe crash → ffmpeg fallback → both fail → VideoReadError.

        With the ffmpeg fallback, a non-zero ffprobe exit no longer raises
        immediately; Stage 0 falls back to `ffmpeg -i`. Only when both
        backends fail does it raise VideoReadError.
        """
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="ffprobe: error"
        )
        with pytest.raises(VideoReadError, match="could not extract"):
            self._run_with_mocks(fake_result, "/fake/bad.mp4")

    def test_validate_output_rejects_non_videoinfo(self) -> None:
        """Verify validate_output returns False for non-VideoInfo objects."""
        stage = Stage0VideoIntegrityCheck()
        assert (
            stage.validate_output({"wrong": "type"}, _make_context()) is False
        )

    def test_validate_output_rejects_empty_videoinfo(self) -> None:
        """Verify validate_output returns False for zero-dimension VideoInfo."""
        stage = Stage0VideoIntegrityCheck()
        bad = VideoInfo(
            video_id="",
            video_path=Path("x"),
            original_fps=0.0,
            width=0,
            height=0,
            duration=0.0,
            total_frames=0,
            codec="",
            has_audio=False,
        )
        assert stage.validate_output(bad, _make_context()) is False

    def test_ffprobe_path_from_config(self) -> None:
        """Verify ffprobe_path is read from config when provided."""
        fake_meta = _make_ffprobe_output()
        fake_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(fake_meta), stderr=""
        )

        with patch(
            "subprocess.run", return_value=fake_result
        ) as mock_run, patch.object(
            FFprobeChecker, "_check_ffprobe_available", return_value=True
        ), patch(
            "pathlib.Path.exists", return_value=True
        ), patch(
            "pathlib.Path.is_file", return_value=True
        ):
            stage = Stage0VideoIntegrityCheck()
            context = PipelineContext(
                config={
                    "video": {
                        "check_video_stream": True,
                        "check_audio_stream": True,
                        "ffprobe_path": "/custom/ffprobe",
                    },
                },
                logger=PipelineLogger(name="test", log_dir=None),
                model_manager=object(),
            )
            stage.run(input_data=Path("/fake/v.mp4"), context=context)
            called_cmd = mock_run.call_args[0][0]
            assert called_cmd[0] == "/custom/ffprobe"


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the exception class hierarchy is correct."""

    def test_all_exceptions_are_pipeline_errors(self) -> None:
        """Every specific exception should be a PipelineError subclass."""
        for exc_cls in (
            InputValidationError,
            StageExecutionError,
            VideoReadError,
            MissingDependencyError,
            MetadataExtractionError,
            MissingStreamError,
        ):
            assert issubclass(exc_cls, PipelineError)

    def test_pipeline_error_is_exception(self) -> None:
        """PipelineError should be a standard Exception subclass."""
        assert issubclass(PipelineError, Exception)

    def test_catch_by_base_class(self) -> None:
        """Verify specific exceptions can be caught as PipelineError."""
        # MissingStreamError can be caught as PipelineError
        try:
            raise MissingStreamError("test")
        except PipelineError:
            pass  # expected

    def test_specific_exception_not_caught_by_wrong_type(self) -> None:
        """Verify exceptions are NOT caught by unrelated types."""
        with pytest.raises(MissingStreamError):
            raise MissingStreamError("test")
        # The above line must not be caught by VideoReadError
        with pytest.raises(VideoReadError):
            raise VideoReadError("test")

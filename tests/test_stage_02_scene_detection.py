"""Tests for Stage 2: Scene Segmentation.

Tests cover:
    1. Normal scene detection with mocked PySceneDetect.
    2. Input video not found → InputValidationError.
    3. PySceneDetect not installed → MissingDependencyError.
    4. SceneInfo output validation.
    5. SceneDetector unit tests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from face_pipeline_v2.data.models import SceneInfo, StandardizedVideoInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_02_scene_detection.scene_detector import (
    SceneDetector,
)
from face_pipeline_v2.stages.stage_02_scene_detection.stage import (
    Stage2SceneSegmentation,
)
from face_pipeline_v2.utils.exceptions import (
    InputValidationError,
    MissingDependencyError,
    VideoReadError,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_STD_VIDEO = StandardizedVideoInfo(
    video_path=Path("/fake/outputs/videos/test_vid/test_vid_std.mp4"),
    audio_path=Path("/fake/outputs/videos/test_vid/test_vid_audio.wav"),
    target_fps=25,
    resolution=(1920, 1080),
    codec="libx264",
)


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_02": {
            "detector": "ContentDetector",
            "threshold": 27.0,
            "min_scene_len": 15,
        },
        "storage": {
            "output_dir": "/fake/outputs",
        },
    }
    if config_overrides:
        config.update(config_overrides)
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_02", log_dir=None),
        model_manager=object(),
    )


def _make_mock_scene_list(fps: float = 25.0) -> list:
    """Build a mock scene list mimicking PySceneDetect FrameTimecode pairs."""
    scenes_data = [(0.0, 5.0), (5.0, 12.0), (12.0, 20.0)]
    result = []
    for start_sec, end_sec in scenes_data:
        start_tc = MagicMock()
        start_tc.frame_num = int(start_sec * fps)
        start_tc.get_seconds.return_value = start_sec

        end_tc = MagicMock()
        end_tc.frame_num = int(end_sec * fps)
        end_tc.get_seconds.return_value = end_sec

        result.append((start_tc, end_tc))
    return result


def _make_mock_video(fps: float = 25.0, total_frames: int = 500) -> MagicMock:
    video = MagicMock()
    video.frame_rate = fps
    video.duration.frame_num = total_frames
    return video


def _install_mock_scenedetect(module_mock: MagicMock) -> None:
    """Inject a mock scenedetect module into sys.modules."""
    sys.modules["scenedetect"] = module_mock


def _remove_scenedetect() -> None:
    """Remove scenedetect from sys.modules to simulate not-installed."""
    sys.modules.pop("scenedetect", None)


# ---------------------------------------------------------------------------
# SceneDetector unit tests
# ---------------------------------------------------------------------------


class TestSceneDetector:
    """Unit tests for SceneDetector with mocked PySceneDetect via sys.modules."""

    def setup_method(self) -> None:
        """Save and remove the real scenedetect before each test."""
        self._real_sd = sys.modules.get("scenedetect")
        sys.modules.pop("scenedetect", None)

    def teardown_method(self) -> None:
        """Restore scenedetect after each test."""
        sys.modules.pop("scenedetect", None)
        if self._real_sd is not None:
            sys.modules["scenedetect"] = self._real_sd

    def test_detect_normal_video(self) -> None:
        """Verify SceneDetector returns correct scene dicts."""
        mock_sd = MagicMock()
        mock_sd.open_video.return_value = _make_mock_video()
        mock_sm = MagicMock()
        mock_sm.get_scene_list.return_value = _make_mock_scene_list()
        mock_sd.SceneManager.return_value = mock_sm
        _install_mock_scenedetect(mock_sd)

        detector = SceneDetector(threshold=27.0, min_scene_len=15)
        with patch("pathlib.Path.exists", return_value=True):
            scenes = detector.detect(Path("/fake/video.mp4"))

        assert len(scenes) == 3
        assert scenes[0]["scene_index"] == 0
        assert scenes[0]["start_frame"] == 0
        assert scenes[0]["start_time"] == 0.0
        assert scenes[0]["end_time"] == 5.0
        assert scenes[0]["duration"] == 5.0

    def test_detect_no_scenes_returns_single_scene(self) -> None:
        """Verify single scene returned when detector finds no cuts."""
        mock_sd = MagicMock()
        mock_sd.open_video.return_value = _make_mock_video()
        mock_sm = MagicMock()
        mock_sm.get_scene_list.return_value = []
        mock_sd.SceneManager.return_value = mock_sm
        _install_mock_scenedetect(mock_sd)

        detector = SceneDetector(threshold=27.0, min_scene_len=15)
        with patch("pathlib.Path.exists", return_value=True):
            scenes = detector.detect(Path("/fake/video.mp4"))

        assert len(scenes) == 1
        assert scenes[0]["scene_index"] == 0
        assert scenes[0]["end_frame"] == 500

    def test_video_not_found(self) -> None:
        """Verify VideoReadError when input video does not exist."""
        detector = SceneDetector()
        with pytest.raises(VideoReadError, match="Video file not found"):
            detector.detect(Path("/nonexistent/video.mp4"))

    def test_scenedetect_not_installed(self) -> None:
        """Verify MissingDependencyError when PySceneDetect is unavailable."""
        import builtins

        _orig = builtins.__import__

        def _block_sd(name, *args, **kwargs):
            if name == "scenedetect":
                raise ImportError("No module named 'scenedetect'")
            return _orig(name, *args, **kwargs)

        detector = SceneDetector()
        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.__import__", side_effect=_block_sd
        ):
            with pytest.raises(
                MissingDependencyError, match="PySceneDetect is not installed"
            ):
                detector.detect(Path("/fake/video.mp4"))

    def test_threshold_and_min_scene_len_properties(self) -> None:
        """Verify configuration values are stored and accessible."""
        detector = SceneDetector(threshold=30.0, min_scene_len=20)
        assert detector.threshold == 30.0
        assert detector.min_scene_len == 20


# ---------------------------------------------------------------------------
# Stage2SceneSegmentation tests
# ---------------------------------------------------------------------------


class TestStage2SceneSegmentation:
    """Tests for Stage2SceneSegmentation with mocked PySceneDetect."""

    def setup_method(self) -> None:
        self._real_sd = sys.modules.get("scenedetect")
        sys.modules.pop("scenedetect", None)

    def teardown_method(self) -> None:
        sys.modules.pop("scenedetect", None)
        if self._real_sd is not None:
            sys.modules["scenedetect"] = self._real_sd

    def test_normal_scene_detection(self) -> None:
        """Verify Stage 2 produces valid SceneInfo list from mocked detector."""
        mock_sd = MagicMock()
        mock_sd.open_video.return_value = _make_mock_video()
        mock_sm = MagicMock()
        mock_sm.get_scene_list.return_value = _make_mock_scene_list()
        mock_sd.SceneManager.return_value = mock_sm
        _install_mock_scenedetect(mock_sd)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.mkdir"
        ), patch("builtins.open", MagicMock()):
            stage = Stage2SceneSegmentation()
            context = _make_context()
            output = stage.run(input_data=_STD_VIDEO, context=context)

            assert isinstance(output, list)
            assert len(output) == 3
            assert all(isinstance(s, SceneInfo) for s in output)
            assert output[0].scene_id == "test_vid_scene_0000"
            assert output[0].video_id == "test_vid"
            assert output[0].duration == 5.0

    def test_input_wrong_type_rejected(self) -> None:
        """Verify validate_input returns False for non-StandardizedVideoInfo."""
        stage = Stage2SceneSegmentation()
        assert (
            stage.validate_input({"wrong": "type"}, _make_context()) is False
        )

    def test_input_file_missing_raises(self) -> None:
        """Verify InputValidationError when standardized video missing."""
        bad_input = StandardizedVideoInfo(
            video_path=Path("/nonexistent/v.mp4"),
            audio_path=Path("/nonexistent/a.wav"),
            target_fps=25,
            resolution=(1920, 1080),
            codec="libx264",
        )
        stage = Stage2SceneSegmentation()
        with pytest.raises(
            InputValidationError, match="Invalid input data"
        ):
            stage.run(input_data=bad_input, context=_make_context())

    def test_validate_output_rejects_non_list(self) -> None:
        """Verify validate_output returns False for non-list."""
        stage = Stage2SceneSegmentation()
        assert stage.validate_output("not_a_list", _make_context()) is False

    def test_validate_output_rejects_empty_list(self) -> None:
        """Verify validate_output returns False for empty list."""
        stage = Stage2SceneSegmentation()
        assert stage.validate_output([], _make_context()) is False

    def test_pyscenedetect_not_installed_raises(self) -> None:
        """Verify MissingDependencyError propagated through the stage."""
        import builtins

        _orig = builtins.__import__

        def _block_sd(name, *args, **kwargs):
            if name == "scenedetect":
                raise ImportError("No module named 'scenedetect'")
            return _orig(name, *args, **kwargs)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.__import__", side_effect=_block_sd
        ), patch("pathlib.Path.mkdir"):
            stage = Stage2SceneSegmentation()
            with pytest.raises(
                MissingDependencyError, match="PySceneDetect is not installed"
            ):
                stage.run(input_data=_STD_VIDEO, context=_make_context())

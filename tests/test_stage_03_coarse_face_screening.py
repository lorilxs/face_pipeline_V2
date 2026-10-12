"""Tests for Stage 3: SCRFD Coarse Face Screening.

Tests cover:
    1. Duration < min_duration → reject_reasons includes "too_short".
    2. No face detected → reject_reasons includes "no_face".
    3. Multiple faces → reject_reasons includes "multi_face".
    4. Low confidence → reject_reasons includes "low_confidence".
    5. CoarseFaceInfo fields are correctly calculated.
    6. reject_reasons supports multiple values simultaneously.
    7. Stage 3 does NOT generate aggregate statistics reports.
    8. Input/output validation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    CoarseFaceInfo,
    SceneFilterResult,
    SceneInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_03_face_screening.scrfd_processor import (
    REJECT_LOW_CONFIDENCE,
    REJECT_MULTI_FACE,
    REJECT_NO_FACE,
    REJECT_TOO_SHORT,
    SCRFDProcessor,
)
from face_pipeline_v2.stages.stage_03_face_screening.stage import (
    Stage3CoarseFaceScreening,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_scene(
    scene_id: str = "test_scene_0000",
    video_id: str = "test_vid",
    start_frame: int = 0,
    end_frame: int = 200,
    duration: float = 8.0,
) -> SceneInfo:
    """Create a test SceneInfo."""
    return SceneInfo(
        scene_id=scene_id,
        video_id=video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=float(start_frame) / 25.0,
        end_time=float(end_frame) / 25.0,
        duration=duration,
    )


def _make_scenes(*durations: float) -> list:
    """Create multiple SceneInfo objects with varying durations."""
    scenes = []
    for i, dur in enumerate(durations):
        end_frame = int(dur * 25)
        scenes.append(
            _make_scene(
                scene_id=f"test_scene_{i:04d}",
                duration=dur,
                end_frame=end_frame,
            )
        )
    return scenes


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    """Create a PipelineContext with stage_03 config."""
    config: dict = {
        "stage_03": {
            "enabled": True,
            "sampling_fps": 1,
            "min_duration": 8,
            "min_face_detect_ratio": 0.95,
            "min_single_face_ratio": 0.95,
            "min_face_confidence": 0.6,
        },
        "models": {
            "scrfd": {
                "path": "models/SCRFD",
                "device": "cpu",
            },
        },
        "storage": {
            "output_dir": "/fake/outputs",
        },
    }
    if config_overrides:
        config.update(config_overrides)
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_03", log_dir=None),
        model_manager=MagicMock(),
    )


def _make_mock_scrfd_model(detections_per_frame: list = None):
    """Create a mock SCRFD model.

    Args:
        detections_per_frame: List of detection lists, one per sampled frame.
            Each element is a list of face dicts with bbox, confidence.
            If None, returns empty detections (no faces).
    """
    model = MagicMock()
    if detections_per_frame is None:
        model.detect.return_value = []
    else:
        model.detect.side_effect = detections_per_frame
    return model


def _make_face_detection(confidence: float = 0.9) -> dict:
    """Create a single face detection result dict."""
    return {
        "bbox": (100.0, 100.0, 200.0, 200.0),
        "confidence": confidence,
        "landmarks": tuple(float(v) for v in range(10)),
    }


def _make_dummy_frame() -> np.ndarray:
    """Create a dummy BGR frame (100x100)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# SCRFDProcessor unit tests
# ---------------------------------------------------------------------------


class TestSCRFDProcessor:
    """Unit tests for SCRFDProcessor — no Stage involved."""

    def test_too_short_scene(self) -> None:
        """Duration < min_duration → too_short, no SCRFD calls."""
        mock_model = _make_mock_scrfd_model()
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=100,
            sampling_fps=1,
            min_duration=8,
            scene_duration=5.0,
            min_face_detect_ratio=0.95,
            min_single_face_ratio=0.95,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is False
        assert REJECT_TOO_SHORT in filter_result.reject_reasons
        assert coarse_info.sampled_frame_count == 0
        assert coarse_info.face_detect_ratio == 0.0
        # SCRFD was never called
        mock_model.detect.assert_not_called()

    def test_no_face_detected(self) -> None:
        """No faces in any frame → no_face reject."""
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[[], []]  # 2 frames, 0 faces each
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=50,  # 2 frames at 1fps from 25fps video
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.95,
            min_single_face_ratio=0.95,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is False
        assert REJECT_NO_FACE in filter_result.reject_reasons
        assert coarse_info.face_detect_ratio == 0.0
        assert coarse_info.max_face_count == 0

    def test_multi_face_detected(self) -> None:
        """Multiple faces → multi_face reject."""
        # Frame 1: 2 faces, Frame 2: 2 faces → single_face_ratio = 0.0
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[
                [_make_face_detection(0.9), _make_face_detection(0.8)],
                [_make_face_detection(0.9), _make_face_detection(0.85)],
            ]
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=50,
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.5,
            min_single_face_ratio=0.8,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is False
        assert REJECT_MULTI_FACE in filter_result.reject_reasons
        assert coarse_info.single_face_ratio == 0.0
        assert coarse_info.max_face_count == 2

    def test_low_confidence(self) -> None:
        """Low average confidence → low_confidence reject."""
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[
                [_make_face_detection(0.4)],
                [_make_face_detection(0.3)],
            ]
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=50,
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.5,
            min_single_face_ratio=0.8,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is False
        assert REJECT_LOW_CONFIDENCE in filter_result.reject_reasons
        assert coarse_info.avg_face_confidence == pytest.approx(0.35)

    def test_multiple_reject_reasons(self) -> None:
        """Scene can have multiple reject_reasons simultaneously."""
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[
                [_make_face_detection(0.3), _make_face_detection(0.2)],
                [_make_face_detection(0.3)],
            ]
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=50,
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.5,
            min_single_face_ratio=0.8,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is False
        reasons = set(filter_result.reject_reasons)
        assert REJECT_MULTI_FACE in reasons
        assert REJECT_LOW_CONFIDENCE in reasons
        assert len(filter_result.reject_reasons) == 2

    def test_coarse_face_info_fields_correct(self) -> None:
        """CoarseFaceInfo has correct calculated values."""
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[
                [_make_face_detection(0.9)],
                [],  # no face
                [_make_face_detection(0.8)],
                [_make_face_detection(0.7)],
            ]
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=100,  # 4 frames at 1fps from 25fps
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.5,
            min_single_face_ratio=0.8,
            min_face_confidence=0.6,
        )

        assert coarse_info.sampled_frame_count == 4
        assert coarse_info.face_detect_ratio == 0.75  # 3/4
        assert coarse_info.single_face_ratio == 1.0  # 3 frames with face, all single
        assert coarse_info.avg_face_confidence == pytest.approx(
            (0.9 + 0.8 + 0.7) / 3
        )
        assert coarse_info.max_face_count == 1
        assert filter_result.passed is True

    def test_all_frames_single_face_passes(self) -> None:
        """All frames have a single high-confidence face → passes."""
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[
                [_make_face_detection(0.9)],
                [_make_face_detection(0.85)],
                [_make_face_detection(0.88)],
            ]
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=75,
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.95,
            min_single_face_ratio=0.95,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is True
        assert len(filter_result.reject_reasons) == 0
        assert coarse_info.face_detect_ratio == 1.0
        assert coarse_info.single_face_ratio == 1.0

    def test_face_detect_ratio_below_threshold(self) -> None:
        """Low face_detect_ratio → no_face reject (even if some faces found)."""
        mock_model = _make_mock_scrfd_model(
            detections_per_frame=[
                [_make_face_detection(0.9)],
                [],
                [],
                [],
            ]
        )
        processor = SCRFDProcessor(
            scrfd_model=mock_model,
            video_reader=lambda idx: _make_dummy_frame(),
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(
            scene_start_frame=0,
            scene_end_frame=100,
            sampling_fps=1,
            min_duration=8,
            scene_duration=10.0,
            min_face_detect_ratio=0.8,
            min_single_face_ratio=0.8,
            min_face_confidence=0.6,
        )

        assert filter_result.passed is False
        assert REJECT_NO_FACE in filter_result.reject_reasons
        assert coarse_info.face_detect_ratio == 0.25


# ---------------------------------------------------------------------------
# Stage3CoarseFaceScreening integration tests
# ---------------------------------------------------------------------------


class TestStage3CoarseFaceScreening:
    """Tests for Stage3CoarseFaceScreening stage."""

    def test_validate_input_rejects_non_list(self) -> None:
        """validate_input returns False for non-list input."""
        stage = Stage3CoarseFaceScreening()
        assert stage.validate_input("not_a_list", _make_context()) is False

    def test_validate_input_rejects_empty_list(self) -> None:
        """validate_input returns False for empty list."""
        stage = Stage3CoarseFaceScreening()
        assert stage.validate_input([], _make_context()) is False

    def test_validate_input_rejects_mixed_types(self) -> None:
        """validate_input returns False if not all items are SceneInfo."""
        stage = Stage3CoarseFaceScreening()
        assert (
            stage.validate_input(
                [_make_scene(), "not_a_scene"], _make_context()
            )
            is False
        )

    def test_validate_input_accepts_sceneinfo_list(self) -> None:
        """validate_input returns True for List[SceneInfo]."""
        stage = Stage3CoarseFaceScreening()
        assert (
            stage.validate_input([_make_scene()], _make_context()) is True
        )

    def test_validate_output_rejects_non_list(self) -> None:
        """validate_output returns False for non-list."""
        stage = Stage3CoarseFaceScreening()
        assert stage.validate_output("not_a_list", _make_context()) is False

    def test_validate_output_accepts_sceneinfo_list(self) -> None:
        """validate_output returns True for List[SceneInfo]."""
        stage = Stage3CoarseFaceScreening()
        assert (
            stage.validate_output([_make_scene()], _make_context()) is True
        )

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    def test_stage_process_with_mock_scrfd(
        self, mock_exists, mock_video_capture
    ) -> None:
        """End-to-end Stage 3 process with mocked SCRFD and VideoCapture."""
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0  # FPS
        mock_cap.read.return_value = (True, _make_dummy_frame())
        mock_video_capture.return_value = mock_cap

        # Mock SCRFD model — always returns single high-confidence face
        mock_scrfd = _make_mock_scrfd_model()
        mock_scrfd.detect.return_value = [_make_face_detection(0.9)]

        context = _make_context()
        context.model_manager.get_model.return_value = mock_scrfd

        # Create scenes: first is too short (5s, < 8s min), second is good (8s)
        scenes = _make_scenes(5.0, 8.0)

        stage = Stage3CoarseFaceScreening()

        with patch.object(
            stage, "_resolve_video_path", return_value=Path("/fake/video.mp4")
        ):
            output = stage.run(input_data=scenes, context=context)

        assert len(output) == 2
        # Scene 0: too_short
        assert output[0].filter_result.passed is False
        assert REJECT_TOO_SHORT in output[0].filter_result.reject_reasons
        assert output[0].coarse_face_info.sampled_frame_count == 0
        # Scene 1: passed
        assert output[1].filter_result.passed is True
        assert len(output[1].filter_result.reject_reasons) == 0
        assert output[1].coarse_face_info.face_detect_ratio == 1.0

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    def test_stage_no_face_scene_rejected(
        self, mock_exists, mock_video_capture
    ) -> None:
        """Scene with no faces → no_face reject."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, _make_dummy_frame())
        mock_video_capture.return_value = mock_cap

        # All frames have no faces
        mock_scrfd = _make_mock_scrfd_model()
        mock_scrfd.detect.return_value = []  # Always no faces

        context = _make_context()
        context.model_manager.get_model.return_value = mock_scrfd

        stage = Stage3CoarseFaceScreening()

        with patch.object(
            stage, "_resolve_video_path", return_value=Path("/fake/video.mp4")
        ):
            output = stage.run(
                input_data=_make_scenes(10.0), context=context
            )

        assert output[0].filter_result.passed is False
        assert REJECT_NO_FACE in output[0].filter_result.reject_reasons

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    def test_stage_disabled_passthrough(
        self, mock_exists, mock_video_capture
    ) -> None:
        """When stage_03.enabled=False, scenes pass through unchanged."""
        context = _make_context({"stage_03": {"enabled": False}})
        stage = Stage3CoarseFaceScreening()

        output = stage.run(input_data=_make_scenes(10.0), context=context)

        assert len(output) == 1
        # No metadata attached (passthrough)
        assert output[0].coarse_face_info is None
        assert output[0].filter_result is None

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    def test_stage_output_is_immutable(
        self, mock_exists, mock_video_capture
    ) -> None:
        """Output SceneInfo objects are new instances (immutable pattern)."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, _make_dummy_frame())
        mock_video_capture.return_value = mock_cap

        mock_scrfd = _make_mock_scrfd_model(
            detections_per_frame=[[_make_face_detection(0.9)]]
        )

        context = _make_context()
        context.model_manager.get_model.return_value = mock_scrfd

        original = _make_scene(duration=10.0, end_frame=25)
        stage = Stage3CoarseFaceScreening()

        with patch.object(
            stage, "_resolve_video_path", return_value=Path("/fake/video.mp4")
        ):
            output = stage.run(input_data=[original], context=context)

        # Original is unchanged
        assert original.coarse_face_info is None
        assert original.filter_result is None
        # Output is a new object
        assert output[0] is not original
        assert output[0].coarse_face_info is not None
        assert output[0].filter_result is not None

    def test_stage_does_not_generate_statistics_report(self) -> None:
        """Stage 3 does NOT create any aggregate statistics file or report."""
        # Verify that the stage class has no report/statistics generation
        stage = Stage3CoarseFaceScreening()
        assert not hasattr(stage, "generate_report")
        assert not hasattr(stage, "save_statistics")
        assert not hasattr(stage, "aggregate_results")


# ---------------------------------------------------------------------------
# Debug Scene Export tests
# ---------------------------------------------------------------------------


class TestDebugSceneExport:
    """Tests for Stage 3 debug scene export feature."""

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("face_pipeline_v2.utils.video_utils.cut_video_segment")
    def test_debug_off_no_files_created(
        self, mock_cut, mock_exists, mock_video_capture
    ) -> None:
        """When debug.export_valid_scenes=false, no segment files are created."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, _make_dummy_frame())
        mock_video_capture.return_value = mock_cap

        mock_scrfd = _make_mock_scrfd_model()
        mock_scrfd.detect.return_value = [_make_face_detection(0.9)]

        # Debug export is OFF (default)
        context = _make_context()
        context.model_manager.get_model.return_value = mock_scrfd

        stage = Stage3CoarseFaceScreening()
        with patch.object(
            stage, "_resolve_video_path", return_value=Path("/fake/video.mp4")
        ):
            output = stage.run(
                input_data=_make_scenes(10.0), context=context
            )

        # cut_video_segment must NOT be called
        mock_cut.assert_not_called()
        assert output[0].filter_result.passed is True

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("face_pipeline_v2.utils.video_utils.cut_video_segment")
    def test_debug_on_exports_passed_only(
        self, mock_cut, mock_mkdir, mock_exists, mock_video_capture
    ) -> None:
        """When debug.export_valid_scenes=true, only passed scenes are exported."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, _make_dummy_frame())
        mock_video_capture.return_value = mock_cap

        mock_scrfd = _make_mock_scrfd_model()
        mock_scrfd.detect.return_value = [_make_face_detection(0.9)]

        # Enable debug export
        stage_03_config = {
            "enabled": True,
            "sampling_fps": 1,
            "min_duration": 8,
            "min_face_detect_ratio": 0.95,
            "min_single_face_ratio": 0.95,
            "min_face_confidence": 0.6,
            "debug": {
                "export_valid_scenes": True,
                "output_dir": "outputs/debug/stage_03_scenes",
            },
        }
        context = _make_context({"stage_03": stage_03_config})
        context.model_manager.get_model.return_value = mock_scrfd

        # Scene 0: too_short (5s) → rejected
        # Scene 1: passed (10s) → exported
        scenes = _make_scenes(5.0, 10.0)

        stage = Stage3CoarseFaceScreening()
        with patch.object(
            stage, "_resolve_video_path", return_value=Path("/fake/video.mp4")
        ):
            output = stage.run(input_data=scenes, context=context)

        # Only 1 call: the passed scene
        mock_cut.assert_called_once()
        call_args = mock_cut.call_args
        # Verify the exported scene is the passed one
        assert "test_scene_0001" in str(call_args[1]["output_path"])

        # Verify scene 0 is rejected, scene 1 is passed
        assert output[0].filter_result.passed is False
        assert output[1].filter_result.passed is True

    @patch("cv2.VideoCapture")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    @patch("face_pipeline_v2.utils.video_utils.cut_video_segment")
    def test_debug_on_rejected_scenes_not_exported(
        self, mock_cut, mock_mkdir, mock_exists, mock_video_capture
    ) -> None:
        """Failed scenes are never exported, even with debug enabled."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 25.0
        mock_cap.read.return_value = (True, _make_dummy_frame())
        mock_video_capture.return_value = mock_cap

        # All frames have no faces → all rejected
        mock_scrfd = _make_mock_scrfd_model()
        mock_scrfd.detect.return_value = []

        stage_03_config = {
            "enabled": True,
            "sampling_fps": 1,
            "min_duration": 8,
            "min_face_detect_ratio": 0.95,
            "min_single_face_ratio": 0.95,
            "min_face_confidence": 0.6,
            "debug": {
                "export_valid_scenes": True,
                "output_dir": "outputs/debug/stage_03_scenes",
            },
        }
        context = _make_context({"stage_03": stage_03_config})
        context.model_manager.get_model.return_value = mock_scrfd

        stage = Stage3CoarseFaceScreening()
        with patch.object(
            stage, "_resolve_video_path", return_value=Path("/fake/video.mp4")
        ):
            output = stage.run(
                input_data=_make_scenes(10.0), context=context
            )

        # No passed scenes → no export
        mock_cut.assert_not_called()
        assert output[0].filter_result.passed is False

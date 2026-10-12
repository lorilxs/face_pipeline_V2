"""Tests for Stage 5: Full Face Detection & Crop.

Tests cover:
    1. FaceDetector with mock SCRFD → correct FaceInfo.
    2. Face selection: largest_area strategy.
    3. Multi-face detection → all faces returned, largest first.
    4. Face cropper unit tests.
    5. Stage integration with mocked video.
    6. Input validation (wrong type, empty list).
    7. Immutability: input ClipInfo not modified.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import ClipInfo, FaceInfo, FaceSequence, FrameInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_face_detection.face_cropper import (
    crop_face,
)
from face_pipeline_v2.stages.stage_05_face_detection.face_detector import (
    FaceDetector,
    _bbox_area,
    select_face_largest_area,
)
from face_pipeline_v2.stages.stage_05_face_detection.stage import (
    Stage5FaceDetection,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_clip(
    clip_id: str = "vid_scene_0000_clip_0000",
    video_id: str = "vid",
    start_frame: int = 0,
    end_frame: int = 16,
    fps: float = 25.0,
) -> ClipInfo:
    """Create a minimal ClipInfo for testing."""
    frame_count = end_frame - start_frame
    duration = frame_count / fps
    return ClipInfo(
        clip_id=clip_id,
        scene_id="vid_scene_0000",
        video_id=video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_frame / fps,
        end_time=end_frame / fps,
        frame_count=frame_count,
        duration=duration,
    )


def _make_mock_scrfd(detections: List[Dict[str, Any]] | None = None) -> MagicMock:
    """Create a mock SCRFD model that returns the given detections."""
    mock = MagicMock()
    mock.detect.return_value = detections or []
    return mock


def _make_mock_scrfd_single_face() -> MagicMock:
    """Mock SCRFD returning one face at center of image."""
    return _make_mock_scrfd([
        {
            "bbox": (50.0, 30.0, 150.0, 170.0),
            "confidence": 0.92,
            "landmarks": (
                80.0, 60.0, 120.0, 60.0, 70.0, 90.0, 130.0, 90.0, 100.0, 120.0,
            ),
        }
    ])


def _make_mock_scrfd_multi_face() -> MagicMock:
    """Mock SCRFD returning two faces."""
    return _make_mock_scrfd([
        {
            "bbox": (200.0, 100.0, 280.0, 220.0),  # smaller
            "confidence": 0.78,
            "landmarks": (210.0, 130.0, 270.0, 130.0, 200.0, 160.0, 280.0, 160.0, 240.0, 190.0),
        },
        {
            "bbox": (50.0, 30.0, 180.0, 200.0),  # larger
            "confidence": 0.95,
            "landmarks": (80.0, 70.0, 150.0, 70.0, 60.0, 110.0, 170.0, 110.0, 115.0, 150.0),
        },
    ])


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_05": {
            "enabled": True,
            "face_selection": {"strategy": "largest_area"},
            "crop": {"enabled": True, "size": 256},
            "save_frames": False,
        },
        "stage_04": {
            "target_fps": 25,
        },
        "storage": {
            "output_dir": "/tmp/test_stage_05",
        },
        "models": {
            "scrfd": {
                "path": "models/SCRFD",
                "device": "cpu",
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
        logger=PipelineLogger(name="test_stage_05", log_dir=None),
        model_manager=object(),
    )


def _make_test_image(h: int = 360, w: int = 640) -> np.ndarray:
    """Create a synthetic BGR test image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# bbox_area / face selection unit tests
# ---------------------------------------------------------------------------


class TestBboxArea:
    def test_positive_area(self) -> None:
        assert _bbox_area((0, 0, 10, 10)) == 100.0

    def test_zero_height(self) -> None:
        assert _bbox_area((0, 0, 10, 0)) == 0.0

    def test_large_bbox(self) -> None:
        assert _bbox_area((100, 50, 300, 250)) == 40000.0


class TestFaceSelection:
    def test_empty_detections(self) -> None:
        assert select_face_largest_area([]) is None

    def test_single_face(self) -> None:
        dets = [{"bbox": (10, 10, 50, 60), "confidence": 0.9}]
        selected = select_face_largest_area(dets)
        assert selected is dets[0]

    def test_largest_area_selected(self) -> None:
        small = {"bbox": (0, 0, 10, 10), "confidence": 0.99}
        large = {"bbox": (0, 0, 100, 100), "confidence": 0.50}
        selected = select_face_largest_area([small, large])
        assert selected is large


# ---------------------------------------------------------------------------
# Face cropper unit tests
# ---------------------------------------------------------------------------


class TestFaceCropper:
    def test_crop_saves_file(self, tmp_path: Path) -> None:
        image = _make_test_image()
        out = tmp_path / "crop.jpg"
        result = crop_face(image, (10, 10, 120, 130), out, crop_size=256)
        assert result == out
        assert out.exists()

    def test_crop_respects_size(self, tmp_path: Path) -> None:
        image = _make_test_image()
        out = tmp_path / "crop.jpg"
        crop_face(image, (0, 0, 100, 100), out, crop_size=128)
        import cv2
        saved = cv2.imread(str(out))
        assert saved.shape[0] == 128
        assert saved.shape[1] == 128

    def test_invalid_bbox_raises(self, tmp_path: Path) -> None:
        image = _make_test_image()
        out = tmp_path / "crop.jpg"
        with pytest.raises(ValueError, match="Invalid crop region"):
            crop_face(image, (500, 500, 100, 100), out)


# ---------------------------------------------------------------------------
# FaceDetector unit tests
# ---------------------------------------------------------------------------


class TestFaceDetector:
    def test_single_face_detection(self, tmp_path: Path) -> None:
        """Single face → one FaceInfo with correct fields."""
        scrfd = _make_mock_scrfd_single_face()
        detector = FaceDetector(
            scrfd_model=scrfd,
            crop_enabled=True,
            crop_size=256,
            crops_base_dir=tmp_path,
        )
        image = _make_test_image()
        faces = detector.process_frame(
            frame_index=42, image=image, clip_id="clip_0", video_id="vid",
        )

        assert len(faces) == 1
        f = faces[0]
        assert f.frame_index == 42
        assert f.face_id == "vid_clip_0_face_000"
        assert f.confidence == pytest.approx(0.92)
        assert f.bbox == (50.0, 30.0, 150.0, 170.0)
        assert len(f.landmark_5) == 10
        assert str(f.face_crop_path).endswith("frame_000042_face_000.jpg")
        assert f.face_crop_path.exists()

    def test_no_faces_returns_empty(self) -> None:
        """No detections → empty list."""
        scrfd = _make_mock_scrfd([])
        detector = FaceDetector(scrfd_model=scrfd, crop_enabled=False)
        image = _make_test_image()
        faces = detector.process_frame(
            frame_index=0, image=image, clip_id="c0", video_id="v",
        )
        assert faces == []

    def test_multi_face_sorted_by_area(self, tmp_path: Path) -> None:
        """Two faces → returned sorted by bbox area descending."""
        scrfd = _make_mock_scrfd_multi_face()
        detector = FaceDetector(
            scrfd_model=scrfd,
            crop_enabled=True,
            crop_size=256,
            crops_base_dir=tmp_path,
        )
        image = _make_test_image()
        faces = detector.process_frame(
            frame_index=1, image=image, clip_id="c0", video_id="v",
        )

        assert len(faces) == 2
        # largest first
        assert faces[0].confidence == pytest.approx(0.95)  # larger bbox
        assert faces[1].confidence == pytest.approx(0.78)
        # bbox areas verify ordering
        a0 = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
        a1 = (faces[1].bbox[2] - faces[1].bbox[0]) * (faces[1].bbox[3] - faces[1].bbox[1])
        assert a0 > a1

    def test_crop_disabled(self) -> None:
        """crop_enabled=False → face_crop_path is empty Path."""
        scrfd = _make_mock_scrfd_single_face()
        detector = FaceDetector(scrfd_model=scrfd, crop_enabled=False)
        image = _make_test_image()
        faces = detector.process_frame(
            frame_index=0, image=image, clip_id="c0", video_id="v",
        )
        assert len(faces) == 1
        assert faces[0].face_crop_path == Path("")

    def test_unknown_strategy_raises(self) -> None:
        """Unknown selection strategy → DataProcessingError."""
        with pytest.raises(DataProcessingError, match="Unknown face selection strategy"):
            FaceDetector(
                scrfd_model=_make_mock_scrfd([]),
                selection_strategy="nonexistent",
            )


# ---------------------------------------------------------------------------
# Stage5FaceDetection integration tests
# ---------------------------------------------------------------------------


class TestStage5FaceDetection:
    def test_basic_processing_with_mocked_video(self, tmp_path: Path) -> None:
        """Process 4-frame clip → 1 FaceSequence with 4 FrameInfo objects."""
        clip = _make_clip(start_frame=0, end_frame=4)
        context = _make_context({"storage": {"output_dir": str(tmp_path)}})

        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.return_value = 25.0
        mock_video.read.return_value = (True, _make_test_image())

        stage = Stage5FaceDetection()

        with patch.object(stage, "_get_scrfd_model",
                          return_value=_make_mock_scrfd_single_face()), \
             patch.object(stage, "_resolve_video_path",
                          return_value=Path("/fake/video_std.mp4")), \
             patch("cv2.VideoCapture", return_value=mock_video), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=[clip], context=context)

        assert isinstance(output, list)
        assert len(output) == 1  # 1 FaceSequence
        seq = output[0]
        assert isinstance(seq, FaceSequence)
        assert seq.clip_id == "vid_scene_0000_clip_0000"
        assert len(seq.frames) == 4
        assert all(isinstance(f, FrameInfo) for f in seq.frames)
        assert seq.frames[0].frame_index == 0
        assert seq.frames[3].frame_index == 3

    def test_multiple_clips(self, tmp_path: Path) -> None:
        """Two clips → 2 FaceSequences, each with correct frames."""
        clip1 = _make_clip(clip_id="c0", start_frame=0, end_frame=2)
        clip2 = _make_clip(clip_id="c1", start_frame=10, end_frame=12)

        context = _make_context({"storage": {"output_dir": str(tmp_path)}})

        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.return_value = 25.0
        mock_video.read.return_value = (True, _make_test_image())

        stage = Stage5FaceDetection()

        with patch.object(stage, "_get_scrfd_model",
                          return_value=_make_mock_scrfd_single_face()), \
             patch.object(stage, "_resolve_video_path",
                          return_value=Path("/fake/video_std.mp4")), \
             patch("cv2.VideoCapture", return_value=mock_video), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=[clip1, clip2], context=context)

        assert len(output) == 2  # 2 FaceSequences
        assert output[0].clip_id == "c0"
        assert output[1].clip_id == "c1"
        assert len(output[0].frames) == 2
        assert len(output[1].frames) == 2

    def test_no_faces_detected(self, tmp_path: Path) -> None:
        """SCRFD finds nothing → FaceSequence with FrameInfos having empty faces."""
        clip = _make_clip(start_frame=0, end_frame=2)
        context = _make_context({"storage": {"output_dir": str(tmp_path)}})

        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.return_value = 25.0
        mock_video.read.return_value = (True, _make_test_image())

        stage = Stage5FaceDetection()

        with patch.object(stage, "_get_scrfd_model",
                          return_value=_make_mock_scrfd([])), \
             patch.object(stage, "_resolve_video_path",
                          return_value=Path("/fake/video_std.mp4")), \
             patch("cv2.VideoCapture", return_value=mock_video), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=[clip], context=context)

        assert len(output) == 1
        seq = output[0]
        assert len(seq.frames) == 2
        assert all(len(f.faces) == 0 for f in seq.frames)

    def test_input_not_mutated(self, tmp_path: Path) -> None:
        """Original ClipInfo list not modified."""
        clip = _make_clip(start_frame=0, end_frame=2)
        original_id = clip.clip_id
        original_start = clip.start_frame

        context = _make_context({"storage": {"output_dir": str(tmp_path)}})

        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.return_value = 25.0
        mock_video.read.return_value = (True, _make_test_image())

        stage = Stage5FaceDetection()

        with patch.object(stage, "_get_scrfd_model",
                          return_value=_make_mock_scrfd_single_face()), \
             patch.object(stage, "_resolve_video_path",
                          return_value=Path("/fake/video_std.mp4")), \
             patch("cv2.VideoCapture", return_value=mock_video), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True):
            stage.run(input_data=[clip], context=context)

        assert clip.clip_id == original_id
        assert clip.start_frame == original_start

    def test_disabled_stage_returns_empty(self) -> None:
        """enabled=false → empty list."""
        clip = _make_clip()
        context = _make_context({"stage_05": {"enabled": False}})

        stage = Stage5FaceDetection()
        output = stage.run(input_data=[clip], context=context)
        assert output == []

    def test_input_wrong_type_raises(self) -> None:
        """Non-list input → InputValidationError."""
        stage = Stage5FaceDetection()
        assert stage.validate_input("wrong", _make_context()) is False

        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_empty_list_raises(self) -> None:
        """Empty list → InputValidationError."""
        stage = Stage5FaceDetection()
        assert stage.validate_input([], _make_context()) is False

    def test_input_wrong_element_type(self) -> None:
        """List with non-ClipInfo → validate_input returns False."""
        stage = Stage5FaceDetection()
        assert stage.validate_input(["not_a_clip"], _make_context()) is False

    def test_validate_output_rejects_non_list(self) -> None:
        """validate_output → False for non-list."""
        stage = Stage5FaceDetection()
        assert stage.validate_output("not_a_list", _make_context()) is False

    def test_validate_output_accepts_list(self) -> None:
        """validate_output → True for List[FaceSequence]."""
        stage = Stage5FaceDetection()
        seq = FaceSequence(
            clip_id="c0", video_id="v",
            frames=(FrameInfo(
                frame_index=0, clip_id="c0", timestamp=0.0,
                image_path=Path(""),
            ),),
        )
        assert stage.validate_output([seq], _make_context()) is True

    def test_validate_output_rejects_mixed(self) -> None:
        """validate_output → False for mixed types."""
        stage = Stage5FaceDetection()
        seq = FaceSequence(
            clip_id="c0", video_id="v",
            frames=(FrameInfo(
                frame_index=0, clip_id="c0", timestamp=0.0,
                image_path=Path(""),
            ),),
        )
        assert stage.validate_output([seq, "not_sequence"], _make_context()) is False

    def test_face_id_includes_video_clip_and_face_index(self, tmp_path: Path) -> None:
        """face_id format verified through FaceSequence → frames → faces."""
        clip = _make_clip(clip_id="s0_c0", video_id="my_video", start_frame=0, end_frame=1)
        context = _make_context({"storage": {"output_dir": str(tmp_path)}})

        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.read.return_value = (True, _make_test_image())

        stage = Stage5FaceDetection()
        with patch.object(stage, "_get_scrfd_model",
                          return_value=_make_mock_scrfd_multi_face()), \
             patch.object(stage, "_resolve_video_path",
                          return_value=Path("/fake/video_std.mp4")), \
             patch("cv2.VideoCapture", return_value=mock_video), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True):
            output = stage.run(input_data=[clip], context=context)

        assert len(output) == 1
        seq = output[0]
        assert seq.clip_id == "s0_c0"
        assert seq.video_id == "my_video"
        faces = seq.frames[0].faces
        assert len(faces) == 2
        assert faces[0].face_id == "my_video_s0_c0_face_000"
        assert faces[1].face_id == "my_video_s0_c0_face_001"

    def test_landmark_none_becomes_empty_tuple(self, tmp_path: Path) -> None:
        """Detection with landmarks=None → landmark_5 is empty tuple."""
        scrfd = _make_mock_scrfd([
            {"bbox": (0, 0, 100, 100), "confidence": 0.8, "landmarks": None}
        ])
        detector = FaceDetector(scrfd_model=scrfd, crop_enabled=False)
        faces = detector.process_frame(
            frame_index=0, image=_make_test_image(), clip_id="c0", video_id="v",
        )
        assert len(faces) == 1
        assert faces[0].landmark_5 == ()

"""Tests for the Stage 5 face-resolution pre-filter.

Covers:
    1. tight-bbox validity/size helpers.
    2. per-frame classification (PASS / low-res / no-face / invalid-bbox, and
       multi-face target = largest = first).
    3. video-level evaluation (90% boundary, no-face denominator, distinct
       failure statuses, no valid face, insufficient valid frames,
       unprocessed expected frames).
    4. FaceDetector distinct InvalidBboxError / FaceCropError.
    5. Stage 5 wiring (reject raises FaceResolutionFilterError, disabled
       passthrough, downstream-block via PipelineError).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import ClipInfo, FaceInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_face_detection.face_detector import (
    FaceDetector,
)
from face_pipeline_v2.stages.stage_05_face_detection.face_resolution_filter import (
    REJECT_FACE_RESOLUTION_RATIO_BELOW,
    REJECT_INSUFFICIENT_VALID_FACE_FRAMES,
    REJECT_NO_VALID_FACE_CROP,
    REJECT_PROCESSING_ERROR,
    STATUS_FAIL_CROP,
    STATUS_FAIL_INVALID_BBOX,
    STATUS_FAIL_LOW_RESOLUTION,
    STATUS_FAIL_NO_FACE,
    STATUS_FAIL_ERROR,
    STATUS_PASS,
    FaceResolutionFilterConfig,
    FrameResolutionOutcome,
    classify_detected_faces,
    evaluate_video,
    tight_bbox_is_valid,
    tight_bbox_size,
)
from face_pipeline_v2.stages.stage_05_face_detection.stage import (
    Stage5FaceDetection,
)
from face_pipeline_v2.utils.exceptions import (
    FaceCropError,
    FaceResolutionFilterError,
    InvalidBboxError,
    PipelineError,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face(
    bbox: tuple, face_id: str = "f0", frame_index: int = 0
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index,
        face_id=face_id,
        bbox=bbox,
        landmark_5=(),
        confidence=0.9,
        face_crop_path=Path(""),
    )


def _cfg(**over) -> FaceResolutionFilterConfig:
    base = dict(
        enabled=True,
        min_face_short_side=75,
        min_face_resolution_ratio=0.9,
        min_valid_face_frames=1,
    )
    base.update(over)
    return FaceResolutionFilterConfig(**base)


def _outcome(status: str) -> FrameResolutionOutcome:
    return FrameResolutionOutcome(status=status)


def _make_clip(
    clip_id: str = "vid_scene_0_clip_0",
    video_id: str = "vid",
    start: int = 0,
    end: int = 4,
) -> ClipInfo:
    return ClipInfo(
        clip_id=clip_id,
        scene_id="vid_scene_0",
        video_id=video_id,
        start_frame=start,
        end_frame=end,
        start_time=0.0,
        end_time=end / 25.0,
        frame_count=end - start,
        duration=(end - start) / 25.0,
    )


def _make_context(overrides: dict, tmp_path: Path) -> PipelineContext:
    config = {
        "stage_05": {
            "enabled": True,
            "face_selection": {"strategy": "largest_area"},
            "crop": {"enabled": False, "size": 256},
            "save_frames": False,
        },
        "stage_04": {"target_fps": 25},
        "runtime": {"video_reader": "opencv"},
        "storage": {"output_dir": str(tmp_path)},
        "models": {"scrfd": {"path": "models/SCRFD", "device": "cpu"}},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage05_res_filter", log_dir=None),
        model_manager=object(),
    )


def _mock_scrfd(detections) -> MagicMock:
    mock = MagicMock()
    mock.detect.return_value = detections
    return mock


def _single_face(bbox: tuple) -> list:
    return [{"bbox": bbox, "confidence": 0.9, "landmarks": ()}]


# ---------------------------------------------------------------------------
# Tight bbox helpers
# ---------------------------------------------------------------------------

class TestTightBbox:
    def test_valid(self) -> None:
        assert tight_bbox_is_valid((0, 0, 100, 120)) is True

    def test_degenerate_width(self) -> None:
        assert tight_bbox_is_valid((50, 0, 30, 120)) is False

    def test_degenerate_height(self) -> None:
        assert tight_bbox_is_valid((0, 50, 100, 30)) is False

    def test_zero_area(self) -> None:
        assert tight_bbox_is_valid((10, 10, 10, 20)) is False

    def test_non_finite(self) -> None:
        assert tight_bbox_is_valid((0, 0, float("nan"), 10)) is False
        assert tight_bbox_is_valid((0, 0, float("inf"), 10)) is False

    def test_not_tuple(self) -> None:
        assert tight_bbox_is_valid(None) is False
        assert tight_bbox_is_valid("nope") is False
        assert tight_bbox_is_valid((1, 2, 3)) is False

    def test_out_of_frame_allowed(self) -> None:
        # Edge faces may be partially out of frame; only degeneracy matters.
        assert tight_bbox_is_valid((-5, -5, 40, 40)) is True

    def test_size_uses_int_truncation(self) -> None:
        assert tight_bbox_size((1.9, 2.9, 103.1, 105.1)) == (102, 103)


# ---------------------------------------------------------------------------
# Per-frame classification
# ---------------------------------------------------------------------------

class TestClassifyDetectedFaces:
    def test_no_face(self) -> None:
        assert classify_detected_faces([], _cfg()).status == STATUS_FAIL_NO_FACE

    def test_pass(self) -> None:
        o = classify_detected_faces([_make_face((0, 0, 120, 140))], _cfg())
        assert o.status == STATUS_PASS
        assert (o.face_width, o.face_height) == (120, 140)
        assert o.face_short_side == 120

    def test_exact_short_side_boundary_passes(self) -> None:
        # min(w, h) == 75 (exactly at the threshold) is accepted.
        o = classify_detected_faces([_make_face((0, 0, 75, 90))], _cfg())
        assert o.status == STATUS_PASS
        assert o.face_short_side == 75

    def test_short_side_below_threshold_fails(self) -> None:
        # One pixel below the threshold: min(w, h) == 74 -> reject.
        o = classify_detected_faces([_make_face((0, 0, 74, 90))], _cfg())
        assert o.status == STATUS_FAIL_LOW_RESOLUTION
        assert o.face_short_side == 74

    def test_short_side_uses_height_when_narrower(self) -> None:
        # Landscape bbox: height is the short side and governs the decision.
        o = classify_detected_faces([_make_face((0, 0, 200, 74))], _cfg())
        assert o.status == STATUS_FAIL_LOW_RESOLUTION
        assert o.face_short_side == 74

    def test_wide_tall_face_passes(self) -> None:
        # Both dimensions comfortably above the short-side threshold -> pass.
        o = classify_detected_faces([_make_face((0, 0, 200, 300))], _cfg())
        assert o.status == STATUS_PASS
        assert o.face_short_side == 200

    def test_invalid_bbox(self) -> None:
        o = classify_detected_faces([_make_face((50, 50, 30, 30))], _cfg())
        assert o.status == STATUS_FAIL_INVALID_BBOX

    def test_multi_face_uses_first_not_max(self) -> None:
        # Stage 5 sorts by area descending; the target is faces[0] (largest).
        # Here faces[0] is low-res but faces[1] would pass — proving the filter
        # follows the existing target-face convention rather than re-picking.
        small = _make_face((0, 0, 50, 50), face_id="small")
        big = _make_face((0, 0, 300, 300), face_id="big")
        o = classify_detected_faces([small, big], _cfg())
        assert o.status == STATUS_FAIL_LOW_RESOLUTION


# ---------------------------------------------------------------------------
# Short-side regression against the 6 real validation videos
# ---------------------------------------------------------------------------

class TestShortSideRegression:
    """Regression against measured SCRFD tight-bbox short edges (min(w, h)).

    Measured 2026-10-10 (SCRFD, confidence_threshold=0.3, largest-face target).
    Threshold ``min_face_short_side = 75`` must reject sample A (short edge
    55..71) and keep F/B/C/D/E (short edge >= 76).  Each keep-case uses that
    video's *minimum* short edge (worst case); the reject-case uses A's
    *maximum* short edge (best case), so a threshold drift in either direction
    is caught.
    """

    @pytest.mark.parametrize(
        "sample, short_side, expected",
        [
            ("A_640x360_small_face_max71", 71, STATUS_FAIL_LOW_RESOLUTION),
            ("F_640x360_big_face_min119", 119, STATUS_PASS),
            ("B_852x480_min113", 113, STATUS_PASS),
            ("C_douyin_vertical_min76", 76, STATUS_PASS),
            ("D_1920x1080_min193", 193, STATUS_PASS),
            ("E_vivihead_512x512_min207", 207, STATUS_PASS),
        ],
    )
    def test_short_side(self, sample: str, short_side: int,
                        expected: str) -> None:
        o = classify_detected_faces(
            [_make_face((0, 0, short_side, short_side + 50))], _cfg()
        )
        assert o.status == expected, (
            f"{sample}: short_side={short_side} -> {o.status}, "
            f"expected {expected}"
        )
        assert o.face_short_side == short_side


# ---------------------------------------------------------------------------
# Video-level evaluation
# ---------------------------------------------------------------------------

class TestEvaluateVideo:
    def test_ratio_exact_90_passes(self) -> None:
        out = {i: _outcome(STATUS_PASS) for i in range(9)}
        out[9] = _outcome(STATUS_FAIL_LOW_RESOLUTION)
        r = evaluate_video(out, range(10), _cfg())
        assert r.passed is True
        assert r.reject_reason is None
        assert r.qualified_ratio == pytest.approx(0.9)

    def test_ratio_below_90_fails(self) -> None:
        out = {i: _outcome(STATUS_PASS) for i in range(8)}
        out[8] = _outcome(STATUS_FAIL_LOW_RESOLUTION)
        out[9] = _outcome(STATUS_FAIL_LOW_RESOLUTION)
        r = evaluate_video(out, range(10), _cfg())
        assert r.passed is False
        assert r.reject_reason == REJECT_FACE_RESOLUTION_RATIO_BELOW
        assert r.qualified_ratio == pytest.approx(0.8)

    def test_no_face_frames_counted_in_denominator(self) -> None:
        out = {i: _outcome(STATUS_PASS) for i in range(5)}
        for i in range(5, 10):
            out[i] = _outcome(STATUS_FAIL_NO_FACE)
        r = evaluate_video(out, range(10), _cfg(min_valid_face_frames=5))
        assert r.expected_frame_count == 10
        assert r.status_counts[STATUS_FAIL_NO_FACE] == 5
        assert r.qualified_ratio == pytest.approx(0.5)
        assert r.reject_reason == REJECT_FACE_RESOLUTION_RATIO_BELOW

    def test_no_valid_face(self) -> None:
        out = {i: _outcome(STATUS_FAIL_NO_FACE) for i in range(4)}
        r = evaluate_video(out, range(4), _cfg())
        assert r.passed is False
        assert r.reject_reason == REJECT_NO_VALID_FACE_CROP

    def test_insufficient_valid_frames_takes_precedence(self) -> None:
        out = {i: _outcome(STATUS_PASS) for i in range(3)}
        for i in range(3, 10):
            out[i] = _outcome(STATUS_FAIL_NO_FACE)
        r = evaluate_video(out, range(10), _cfg(min_valid_face_frames=5))
        assert r.passed is False
        assert r.reject_reason == REJECT_INSUFFICIENT_VALID_FACE_FRAMES

    def test_unprocessed_expected_frame_counts_as_error(self) -> None:
        out = {i: _outcome(STATUS_PASS) for i in range(9)}
        # frame 9 is expected but has no recorded outcome
        r = evaluate_video(out, range(10), _cfg())
        assert r.passed is False
        assert r.reject_reason == REJECT_PROCESSING_ERROR
        assert r.status_counts[STATUS_FAIL_ERROR] == 1

    def test_error_takes_precedence_over_ratio(self) -> None:
        out = {i: _outcome(STATUS_PASS) for i in range(9)}
        out[9] = _outcome(STATUS_FAIL_ERROR)
        r = evaluate_video(out, range(10), _cfg())
        assert r.passed is False
        assert r.reject_reason == REJECT_PROCESSING_ERROR

    def test_distinct_statuses_counted(self) -> None:
        out = {
            0: _outcome(STATUS_PASS),
            1: _outcome(STATUS_FAIL_LOW_RESOLUTION),
            2: _outcome(STATUS_FAIL_NO_FACE),
            3: _outcome(STATUS_FAIL_INVALID_BBOX),
            4: _outcome(STATUS_FAIL_CROP),
        }
        r = evaluate_video(out, range(5), _cfg())
        assert r.status_counts[STATUS_PASS] == 1
        assert r.status_counts[STATUS_FAIL_LOW_RESOLUTION] == 1
        assert r.status_counts[STATUS_FAIL_NO_FACE] == 1
        assert r.status_counts[STATUS_FAIL_INVALID_BBOX] == 1
        assert r.status_counts[STATUS_FAIL_CROP] == 1
        assert r.valid_face_frame_count == 2  # PASS + LOW_RESOLUTION


# ---------------------------------------------------------------------------
# FaceDetector distinct exceptions
# ---------------------------------------------------------------------------

class TestFaceDetectorBboxCropErrors:
    def test_invalid_bbox_raises_invalid_bbox_error(self) -> None:
        detector = FaceDetector(
            scrfd_model=_mock_scrfd(_single_face((100, 100, 50, 50))),
            crop_enabled=False,
        )
        with pytest.raises(InvalidBboxError):
            detector.process_frame(0, np.zeros((360, 640, 3)), "c0", "v")

    def test_crop_failure_raises_face_crop_error(self, tmp_path: Path) -> None:
        detector = FaceDetector(
            scrfd_model=_mock_scrfd(_single_face((10, 10, 110, 130))),
            crop_enabled=True,
            crops_base_dir=tmp_path,
        )
        with patch(
            "face_pipeline_v2.stages.stage_05_face_detection.face_detector.crop_face",
            side_effect=ValueError("disk write failed"),
        ):
            with pytest.raises(FaceCropError):
                detector.process_frame(0, np.zeros((360, 640, 3)), "c0", "v")


# ---------------------------------------------------------------------------
# Stage 5 wiring
# ---------------------------------------------------------------------------

class TestStage5ResolutionFilterWiring:
    def _run(self, detections, filter_cfg: dict, tmp_path: Path, clip=None):
        clip = clip or _make_clip()
        context = _make_context(
            {"stage_05_face_resolution_filter": filter_cfg}, tmp_path
        )

        mock_video = MagicMock()
        mock_video.isOpened.return_value = True
        mock_video.get.return_value = 25.0
        mock_video.set.return_value = True
        mock_video.read.return_value = (
            True, np.zeros((360, 640, 3), dtype=np.uint8)
        )

        stage = Stage5FaceDetection()
        with patch.object(stage, "_get_scrfd_model",
                          return_value=_mock_scrfd(detections)), \
             patch.object(stage, "_resolve_video_path",
                          return_value=Path("/fake/video_std.mp4")), \
             patch("cv2.VideoCapture", return_value=mock_video):
            return stage.run([clip], context)

    def test_enabled_low_resolution_rejects(self, tmp_path: Path) -> None:
        cfg = {"enabled": True, "min_face_short_side": 75,
               "min_face_resolution_ratio": 0.9, "min_valid_face_frames": 1}
        with pytest.raises(FaceResolutionFilterError) as excinfo:
            self._run(_single_face((10, 10, 60, 80)), cfg, tmp_path)
        r = excinfo.value.result
        assert r.passed is False
        assert r.reject_reason == REJECT_FACE_RESOLUTION_RATIO_BELOW
        assert r.expected_frame_count == 4
        assert r.status_counts[STATUS_FAIL_LOW_RESOLUTION] == 4

    def test_enabled_no_face_rejects_no_valid_face(self, tmp_path: Path) -> None:
        cfg = {"enabled": True, "min_face_short_side": 75,
               "min_face_resolution_ratio": 0.9, "min_valid_face_frames": 1}
        with pytest.raises(FaceResolutionFilterError) as excinfo:
            self._run([], cfg, tmp_path)
        assert excinfo.value.result.reject_reason == REJECT_NO_VALID_FACE_CROP

    def test_enabled_pass_returns_sequences(self, tmp_path: Path) -> None:
        cfg = {"enabled": True, "min_face_short_side": 75,
               "min_face_resolution_ratio": 0.9, "min_valid_face_frames": 1}
        out = self._run(_single_face((0, 0, 200, 220)), cfg, tmp_path)
        assert isinstance(out, list) and len(out) == 1
        assert len(out[0].frames[0].faces) == 1

    def test_disabled_preserves_behavior(self, tmp_path: Path) -> None:
        out = self._run(
            _single_face((10, 10, 60, 80)),
            {"enabled": False}, tmp_path,
        )
        assert isinstance(out, list) and len(out) == 1
        assert len(out[0].frames[0].faces) == 1

    def test_reject_is_pipeline_error(self) -> None:
        assert issubclass(FaceResolutionFilterError, PipelineError)
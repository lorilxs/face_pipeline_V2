"""Tests for Stage 5.1: Face Tracking Enhancement.

Tests cover:
    1. Same bbox across frames → same track_id.
    2. Two different bbox trajectories → two different track_ids.
    3. Input FaceSequence not mutated.
    4. track_id correctly written into FaceInfo.
    5. Disabled mode → pass-through unchanged.
    6. Empty faces frame handled gracefully.
    7. Input validation.
    8. SCRFD → ByteTrack adapter format conversion.
    9. Real ByteTrackWrapper init and update.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import FaceInfo, FaceSequence, FrameInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_05_1_face_tracking.tracker_adapter import (
    faces_to_detections,
    make_track_id,
)
from face_pipeline_v2.stages.stage_05_1_face_tracking.tracker import (
    track_sequence,
)
from face_pipeline_v2.stages.stage_05_1_face_tracking.stage import (
    Stage5_1FaceTracking,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_face(
    frame_index: int = 0,
    face_id: str = "vid_c0_face_000",
    track_id: str | None = None,
    bbox: Tuple[float, float, float, float] = (50.0, 30.0, 150.0, 170.0),
    confidence: float = 0.9,
) -> FaceInfo:
    return FaceInfo(
        frame_index=frame_index,
        face_id=face_id,
        track_id=track_id,
        bbox=bbox,
        landmark_5=(),
        confidence=confidence,
        face_crop_path=Path(""),
    )


def _make_frame(
    frame_index: int,
    clip_id: str = "v_s0_c0",
    faces: Tuple[FaceInfo, ...] = (),
) -> FrameInfo:
    return FrameInfo(
        frame_index=frame_index,
        clip_id=clip_id,
        timestamp=frame_index / 25.0,
        image_path=Path(""),
        faces=faces,
    )


def _make_sequence(
    clip_id: str = "v_s0_c0",
    video_id: str = "v",
    frames: Tuple[FrameInfo, ...] = (),
) -> FaceSequence:
    return FaceSequence(
        clip_id=clip_id,
        video_id=video_id,
        frames=frames,
    )


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "enhancements": {
            "stage_05_1_face_tracking": {"enabled": True},
            "stage_05_2_single_face_filter": {"enabled": False},
            "stage_05_3_temporal_stability": {"enabled": False},
        },
        "stage_05_1": {
            "tracker": {
                "type": "bytetrack",
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
            },
        },
        "video": {
            "resolution": [640, 360],
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
        logger=PipelineLogger(name="test_stage_05_1", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# Mock tracker helpers
# ---------------------------------------------------------------------------

class MockTracker:
    """Mock ByteTrack that assigns consistent IDs to similar bboxes."""

    def __init__(self) -> None:
        self._tracks: Dict[int, int] = {}
        self._next_id: int = 0

    def update(
        self, bboxes: np.ndarray, confidences: np.ndarray,
        img_shape: tuple = (360, 640),
    ) -> List[int]:
        if len(bboxes) == 0:
            return []
        ids = []
        for i in range(len(bboxes)):
            x1, y1 = int(bboxes[i, 0] / 20), int(bboxes[i, 1] / 20)
            key = (x1, y1)
            if key not in self._tracks:
                self._tracks[key] = self._next_id
                self._next_id += 1
            ids.append(self._tracks[key])
        return ids

    def reset(self) -> None:
        """Reset per-clip tracking state (fresh tracker per FaceSequence)."""
        self._tracks = {}
        self._next_id = 0


# ---------------------------------------------------------------------------
# tracker.py unit tests
# ---------------------------------------------------------------------------


class TestTrackSequence:
    def test_same_bbox_same_track_id(self) -> None:
        """Consecutive frames with same bbox → same track_id."""
        face = _make_face(bbox=(50.0, 30.0, 150.0, 170.0))
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face,)),
            _make_frame(1, faces=(face,)),
            _make_frame(2, faces=(face,)),
        ))

        tracker = MockTracker()
        result = track_sequence(seq, tracker)

        assert len(result.frames) == 3
        ids = [result.frames[i].faces[0].track_id for i in range(3)]
        assert ids[0] == ids[1] == ids[2]
        assert ids[0] is not None
        assert ids[0].startswith("track_")

    def test_different_bbox_different_track_id(self) -> None:
        """Two different bboxes → two different track_ids."""
        face_a = _make_face(face_id="a", bbox=(50.0, 30.0, 150.0, 170.0))
        face_b = _make_face(face_id="b", bbox=(300.0, 100.0, 400.0, 250.0))
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face_a, face_b)),
        ))

        tracker = MockTracker()
        result = track_sequence(seq, tracker)

        faces = result.frames[0].faces
        assert len(faces) == 2
        assert faces[0].track_id != faces[1].track_id

    def test_multi_face_multi_frame(self) -> None:
        """Two faces across 3 frames → each keeps its track_id."""
        face_a = _make_face(face_id="a", bbox=(50.0, 30.0, 150.0, 170.0))
        face_b = _make_face(face_id="b", bbox=(300.0, 100.0, 400.0, 250.0))
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face_a, face_b)),
            _make_frame(1, faces=(face_a, face_b)),
            _make_frame(2, faces=(face_a, face_b)),
        ))

        tracker = MockTracker()
        result = track_sequence(seq, tracker)

        for fi in result.frames:
            assert fi.faces[0].track_id is not None
            assert fi.faces[1].track_id is not None
            # track_a ≠ track_b
            assert fi.faces[0].track_id != fi.faces[1].track_id

        # Same face keeps same track across frames
        ids_a = [result.frames[i].faces[0].track_id for i in range(3)]
        ids_b = [result.frames[i].faces[1].track_id for i in range(3)]
        assert len(set(ids_a)) == 1
        assert len(set(ids_b)) == 1

    def test_empty_faces_handled(self) -> None:
        """Frame with empty faces tuple → passes through unchanged."""
        seq = _make_sequence(frames=(
            _make_frame(0, faces=()),
            _make_frame(1, faces=()),
        ))

        tracker = MockTracker()
        result = track_sequence(seq, tracker)

        assert len(result.frames) == 2
        assert result.frames[0].faces == ()
        assert result.frames[1].faces == ()

    def test_original_not_mutated(self) -> None:
        """Input FaceSequence is not mutated (immutable data flow)."""
        face = _make_face()
        original_face_id = face.face_id
        original_track_id = face.track_id  # should be None
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face,)),
        ))

        tracker = MockTracker()
        result = track_sequence(seq, tracker)

        # Original untouched
        assert seq.frames[0].faces[0].track_id is None
        assert seq.frames[0].faces[0].face_id == original_face_id
        # Result has track_id
        assert result.frames[0].faces[0].track_id is not None

    def test_face_fields_preserved(self) -> None:
        """All FaceInfo fields (except track_id) preserved after tracking."""
        face = _make_face(
            bbox=(10.0, 20.0, 100.0, 150.0),
            confidence=0.88,
        )
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face,)),
        ))

        tracker = MockTracker()
        result = track_sequence(seq, tracker)

        f = result.frames[0].faces[0]
        assert f.frame_index == face.frame_index
        assert f.face_id == face.face_id
        assert f.bbox == face.bbox
        assert f.confidence == pytest.approx(face.confidence)
        assert f.landmark_5 == face.landmark_5
        assert f.face_crop_path == face.face_crop_path

    def test_tracker_none_raises(self) -> None:
        """None tracker → ValueError."""
        seq = _make_sequence()
        with pytest.raises(ValueError, match="not loaded"):
            track_sequence(seq, None)


# ---------------------------------------------------------------------------
# Stage5_1FaceTracking integration tests
# ---------------------------------------------------------------------------


class TestStage5_1FaceTracking:
    def test_enabled_mode(self) -> None:
        """enhancements.stage_05_1_face_tracking.enabled=true → assigns track_ids."""
        face = _make_face(bbox=(50.0, 30.0, 150.0, 170.0))
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face,)),
            _make_frame(1, faces=(face,)),
        ))

        stage = Stage5_1FaceTracking()
        context = _make_context({
            "enhancements": {
                "stage_05_1_face_tracking": {"enabled": True},
            },
        })

        with patch.object(stage, "_get_tracker", return_value=MockTracker()):
            output = stage.run(input_data=[seq], context=context)

        assert len(output) == 1
        assert output[0].frames[0].faces[0].track_id is not None

    # ------------------------------------------------------------------
    # Threshold binding to the shared Stage 3/5 detection floor
    # ------------------------------------------------------------------

    def _base_config(self) -> dict:
        """Config without explicit tracker thresholds / stage_03 section."""
        return {
            "enhancements": {
                "stage_05_1_face_tracking": {"enabled": True},
                "stage_05_2_single_face_filter": {"enabled": False},
                "stage_05_3_temporal_stability": {"enabled": False},
            },
            "stage_05_1": {
                "tracker": {
                    "type": "bytetrack",
                    "track_buffer": 30,
                    "match_thresh": 0.8,
                },
            },
            "video": {"resolution": [640, 360]},
        }

    def _run_stage_and_capture_thresholds(self, config: dict) -> Dict[str, float]:
        face = _make_face(bbox=(50.0, 30.0, 150.0, 170.0))
        seq = _make_sequence(frames=(_make_frame(0, faces=(face,)),))
        context = PipelineContext(
            config=config,
            logger=PipelineLogger(name="test_stage_05_1"),
            model_manager=object(),
        )
        captured: Dict[str, float] = {}

        def fake_get_tracker(
            config,
            context,
            logger,
            track_thresh,
            track_buffer,
            match_thresh,
            det_thresh,
        ):
            captured["track_thresh"] = track_thresh
            captured["det_thresh"] = det_thresh
            return MockTracker()

        stage = Stage5_1FaceTracking()
        with patch.object(stage, "_get_tracker", side_effect=fake_get_tracker):
            stage.run(input_data=[seq], context=context)
        return captured

    def test_tracker_thresholds_follow_detection_floor(self) -> None:
        """No explicit tracker thresholds → both follow stage_03 floor (0.35)."""
        cfg = self._base_config()
        cfg["stage_03"] = {"detection_confidence_threshold": 0.35}
        got = self._run_stage_and_capture_thresholds(cfg)
        assert got["track_thresh"] == 0.35
        assert got["det_thresh"] == 0.35

    def test_explicit_tracker_thresholds_override_floor(self) -> None:
        """Explicit stage_05_1.tracker values override the shared floor."""
        cfg = self._base_config()
        cfg["stage_03"] = {"detection_confidence_threshold": 0.35}
        cfg["stage_05_1"]["tracker"].update(
            {"track_thresh": 0.6, "det_thresh": 0.5}
        )
        got = self._run_stage_and_capture_thresholds(cfg)
        assert got["track_thresh"] == 0.6
        assert got["det_thresh"] == 0.5

    def test_no_stage03_defaults_thresholds_to_0_5(self) -> None:
        """No stage_03 config → shared floor falls back to 0.5."""
        got = self._run_stage_and_capture_thresholds(self._base_config())
        assert got["track_thresh"] == 0.5
        assert got["det_thresh"] == 0.5

    def test_disabled_mode_passthrough(self) -> None:
        """enhancements.stage_05_1_face_tracking.enabled=false → passthrough, no track_id."""
        face = _make_face(track_id=None)
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face,)),
        ))

        stage = Stage5_1FaceTracking()
        context = _make_context({
            "enhancements": {
                "stage_05_1_face_tracking": {"enabled": False},
            },
        })

        output = stage.run(input_data=[seq], context=context)

        assert len(output) == 1
        # Verify no track_id was added (true passthrough)
        for fi in output[0].frames:
            for f in fi.faces:
                assert f.track_id is None

    def test_multiple_sequences(self) -> None:
        """Each FaceSequence gets independent tracking."""
        face_a = _make_face(face_id="a", bbox=(50.0, 30.0, 150.0, 170.0))
        face_b = _make_face(face_id="b", bbox=(300.0, 100.0, 400.0, 250.0))
        seq1 = _make_sequence(clip_id="c0", frames=(
            _make_frame(0, clip_id="c0", faces=(face_a,)),
        ))
        seq2 = _make_sequence(clip_id="c1", frames=(
            _make_frame(10, clip_id="c1", faces=(face_b,)),
        ))

        stage = Stage5_1FaceTracking()
        context = _make_context()

        with patch.object(stage, "_get_tracker", return_value=MockTracker()):
            output = stage.run(input_data=[seq1, seq2], context=context)

        assert len(output) == 2
        assert output[0].clip_id == "c0"
        assert output[1].clip_id == "c1"
        assert output[0].frames[0].faces[0].track_id is not None
        assert output[1].frames[0].faces[0].track_id is not None

    def test_input_not_mutated(self) -> None:
        """Original sequence list not modified."""
        face = _make_face()
        seq = _make_sequence(frames=(
            _make_frame(0, faces=(face,)),
        ))
        original_clip_id = seq.clip_id

        stage = Stage5_1FaceTracking()
        context = _make_context()

        with patch.object(stage, "_get_tracker", return_value=MockTracker()):
            stage.run(input_data=[seq], context=context)

        assert seq.clip_id == original_clip_id
        assert seq.frames[0].faces[0].track_id is None  # original unchanged

    def test_input_wrong_type_raises(self) -> None:
        """Non-list → InputValidationError."""
        stage = Stage5_1FaceTracking()
        assert stage.validate_input("wrong", _make_context()) is False

        with pytest.raises(InputValidationError):
            stage.run(input_data="wrong", context=_make_context())

    def test_input_empty_list_raises(self) -> None:
        """Empty list → validate_input False."""
        stage = Stage5_1FaceTracking()
        assert stage.validate_input([], _make_context()) is False

    def test_input_wrong_element_type(self) -> None:
        """Non-FaceSequence elements → validate_input False."""
        stage = Stage5_1FaceTracking()
        assert stage.validate_input(["not_a_sequence"], _make_context()) is False

    def test_validate_output_accepts_list(self) -> None:
        """validate_output → True for List[FaceSequence]."""
        stage = Stage5_1FaceTracking()
        seq = _make_sequence()
        assert stage.validate_output([seq], _make_context()) is True

    def test_validate_output_rejects_non_list(self) -> None:
        """validate_output → False for non-list."""
        stage = Stage5_1FaceTracking()
        assert stage.validate_output("not_a_list", _make_context()) is False

    def test_validate_output_rejects_mixed(self) -> None:
        """validate_output → False for mixed types."""
        stage = Stage5_1FaceTracking()
        seq = _make_sequence()
        assert stage.validate_output([seq, "bad"], _make_context()) is False


# ---------------------------------------------------------------------------
# Tracker adapter unit tests
# ---------------------------------------------------------------------------


class TestTrackerAdapter:
    def test_faces_to_detections_single(self) -> None:
        """Single FaceInfo → (1, 5) detection array."""
        face = _make_face(bbox=(10.0, 20.0, 100.0, 120.0), confidence=0.95)
        dets = faces_to_detections([face])

        assert dets.shape == (1, 5)
        assert dets[0, 0] == 10.0  # x1
        assert dets[0, 1] == 20.0  # y1
        assert dets[0, 2] == 100.0  # x2
        assert dets[0, 3] == 120.0  # y2
        assert dets[0, 4] == pytest.approx(0.95)  # float32 score (see numpy NEP 50)

    def test_faces_to_detections_empty(self) -> None:
        """Empty face list → (0, 5) array."""
        dets = faces_to_detections([])
        assert dets.shape == (0, 5)

    def test_faces_to_detections_multiple(self) -> None:
        """Multiple FaceInfo → (N, 5) array."""
        f1 = _make_face(bbox=(0.0, 0.0, 50.0, 60.0), confidence=0.8)
        f2 = _make_face(bbox=(100.0, 200.0, 200.0, 300.0), confidence=0.6)
        dets = faces_to_detections([f1, f2])

        assert dets.shape == (2, 5)
        assert dets[1, 0] == 100.0
        assert dets[1, 4] == pytest.approx(0.6)  # float32 score (see numpy NEP 50)

    def test_make_track_id_positive(self) -> None:
        assert make_track_id(0) == "track_000"
        assert make_track_id(5) == "track_005"
        assert make_track_id(123) == "track_123"

    def test_make_track_id_negative(self) -> None:
        assert make_track_id(-1) is None


# ---------------------------------------------------------------------------
# Real ByteTrackWrapper tests
# ---------------------------------------------------------------------------


class TestByteTrackWrapper:
    def test_init_and_load(self) -> None:
        """Wrapper initializes and loads without error."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        wrapper.load()
        assert wrapper.is_loaded is True

    def test_update_with_empty_dets(self) -> None:
        """Empty detections → empty track_ids."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper()
        wrapper.load()
        ids = wrapper.update(
            bboxes=np.zeros((0, 4)),
            confidences=np.zeros((0,)),
        )
        assert ids == []

    def test_update_single_detection(self) -> None:
        """Single detection → gets a track_id >= 0."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper()
        wrapper.load()
        ids = wrapper.update(
            bboxes=np.array([[50.0, 30.0, 150.0, 170.0]]),
            confidences=np.array([0.9]),
        )
        assert len(ids) == 1
        assert ids[0] >= 0

    def test_reset_restarts_track_ids_per_clip(self) -> None:
        """reset() rebuilds tracker state so track IDs restart per clip.

        Without reset, ByteTrack's class-level ID counter (BaseTrack._count)
        keeps growing across clips, so the same bbox in clip N+1 gets a
        higher ID and previous-clip track state leaks into the next clip.
        After reset(), the same bbox must get the same fresh ID.
        """
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper()
        wrapper.load()
        bbox = np.array([[100.0, 50.0, 200.0, 150.0]])
        conf = np.array([0.9])

        wrapper.reset()  # force a clean counter regardless of prior tests
        ids_clip1 = wrapper.update(bbox, conf)
        wrapper.update(bbox, conf)  # same bbox continues the same track
        wrapper.update(bbox, conf)

        wrapper.reset()  # clip boundary → fresh tracker
        ids_clip2 = wrapper.update(bbox, conf)

        assert ids_clip1 == ids_clip2
        assert ids_clip1 == [1]

    def test_consecutive_same_detection_stable_id(self) -> None:
        """Same bbox across frames → stable track_id."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper()
        wrapper.load()
        bbox = np.array([[50.0, 30.0, 150.0, 170.0]])
        conf = np.array([0.9])

        id1 = wrapper.update(bbox, conf)[0]
        id2 = wrapper.update(bbox, conf)[0]
        id3 = wrapper.update(bbox, conf)[0]
        assert id1 == id2 == id3

    def test_two_detections_two_tracks(self) -> None:
        """Two distinct bboxes → two different track_ids."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper()
        wrapper.load()
        bboxes = np.array([
            [50.0, 30.0, 150.0, 170.0],
            [300.0, 100.0, 400.0, 250.0],
        ])
        confs = np.array([0.9, 0.8])

        ids = wrapper.update(bboxes, confs)
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_det_thresh_passed_to_bytetracker_args(self) -> None:
        """Explicit det_thresh reaches BYTETracker; absent → upstream heuristic."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper(track_thresh=0.5, det_thresh=0.35)
        wrapper.load()
        assert wrapper._tracker.args.det_thresh == 0.35
        assert wrapper._tracker.det_thresh == 0.35  # resolved gate

        wrapper_default = ByteTrackWrapper(track_thresh=0.5)
        wrapper_default.load()
        assert wrapper_default._tracker.det_thresh == 0.6  # track_thresh + 0.1

    def test_detection_above_floor_can_start_track(self) -> None:
        """Conf < track_thresh+0.1 but >= det_thresh must still start a track.

        This is the clip7 regression: with the upstream heuristic
        det_thresh = track_thresh + 0.1 = 0.6, faces scoring 0.5..0.6 at clip
        start got NO track (None) because no existing track could match them
        and the init gate rejected them.  Aligning det_thresh to the shared
        detection floor (0.35) lets every Stage 5 detected face start a track.
        """
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        bbox = np.array([[100.0, 50.0, 200.0, 150.0]])
        conf = np.array([0.55])

        wrapper = ByteTrackWrapper(track_thresh=0.5, det_thresh=0.35)
        wrapper.load()
        wrapper.reset()  # zero the shared class-level ID counter
        ids = wrapper.update(bbox, conf)
        assert ids == [1]  # fresh track initialised from frame 0

        # Old behaviour (no explicit det_thresh → 0.6): init gate refuses the
        # detection, so it gets no track. The wrapper now returns one id per
        # detection with -1 = "unassigned" (mapped to None downstream), instead
        # of silently dropping the entry with an empty list.
        wrapper_old = ByteTrackWrapper(track_thresh=0.5)
        wrapper_old.load()
        wrapper_old.reset()
        assert wrapper_old.update(bbox, conf) == [-1]

    def test_reset_preserves_det_thresh(self) -> None:
        """reset() rebuilds the tracker with the same det_thresh."""
        from face_pipeline_v2.models.wrappers.bytetrack import ByteTrackWrapper

        wrapper = ByteTrackWrapper(track_thresh=0.5, det_thresh=0.35)
        wrapper.load()
        wrapper.reset()
        assert wrapper._tracker.args.det_thresh == 0.35
        assert wrapper.det_thresh == 0.35

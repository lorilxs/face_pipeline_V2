"""Unit tests for GazeFeature datamodel."""

import numpy as np
import pytest

from face_pipeline_v2.data.models import GazeFeature


class TestGazeFeatureDatamodel:
    """Validate GazeFeature fields and types."""

    def test_basic_fields(self) -> None:
        gf = GazeFeature(
            frame_index=0,
            timestamp=0.0,
            clip_id="test_clip",
            left_eye=(1.0, 0.0, 0.0),
            right_eye=(0.0, 1.0, 0.0),
            gaze_vector=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        assert gf.frame_index == 0
        assert gf.timestamp == 0.0
        assert gf.clip_id == "test_clip"
        assert len(gf.left_eye) == 3
        assert len(gf.right_eye) == 3
        assert len(gf.gaze_vector) == 6

    def test_gaze_vector_shape(self) -> None:
        """gaze_vector must be exactly 6 floats."""
        gf = GazeFeature(
            frame_index=0,
            timestamp=0.0,
            clip_id="test",
            left_eye=(0.0, 0.0, -1.0),
            right_eye=(0.0, 0.0, -1.0),
            gaze_vector=(0.0, 0.0, -1.0, 0.0, 0.0, -1.0),
        )
        assert isinstance(gf.gaze_vector, tuple)
        assert len(gf.gaze_vector) == 6
        assert all(isinstance(v, float) for v in gf.gaze_vector)

    def test_frozen_immutable(self) -> None:
        gf = GazeFeature(
            frame_index=0,
            timestamp=0.0,
            clip_id="test",
            left_eye=(0.0, 0.0, -1.0),
            right_eye=(0.0, 0.0, -1.0),
            gaze_vector=(0.0, 0.0, -1.0, 0.0, 0.0, -1.0),
        )
        with pytest.raises(Exception):
            gf.frame_index = 1  # type: ignore[misc]

    def test_equality(self) -> None:
        a = GazeFeature(
            frame_index=0, timestamp=0.0, clip_id="c",
            left_eye=(0.0,), right_eye=(0.0,), gaze_vector=(0.0,)*6,
        )
        b = GazeFeature(
            frame_index=0, timestamp=0.0, clip_id="c",
            left_eye=(0.0,), right_eye=(0.0,), gaze_vector=(0.0,)*6,
        )
        assert a == b

    def test_timestamp_field(self) -> None:
        """Timestamp is a float, used for Stage14/17 alignment."""
        gf = GazeFeature(
            frame_index=5,
            timestamp=1.24,
            clip_id="clip_01",
            left_eye=(0.0, 0.0, -1.0),
            right_eye=(0.0, 0.0, -1.0),
            gaze_vector=(0.0, 0.0, -1.0, 0.0, 0.0, -1.0),
        )
        assert isinstance(gf.timestamp, float)
        assert gf.timestamp == 1.24

    def test_clip_id_field(self) -> None:
        gf = GazeFeature(
            frame_index=0,
            timestamp=0.0,
            clip_id="CNFACE_00001_BV18Qg96YE1W_000000_scene_0000_clip_0001",
            left_eye=(0.0, 0.0, -1.0),
            right_eye=(0.0, 0.0, -1.0),
            gaze_vector=(0.0, 0.0, -1.0, 0.0, 0.0, -1.0),
        )
        assert isinstance(gf.clip_id, str)
        assert len(gf.clip_id) > 0
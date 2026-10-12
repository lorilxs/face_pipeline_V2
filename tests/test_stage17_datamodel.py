"""Tests for the clip-level TrainingSample data model (Stage17 Phase1).

Verifies the refactored TrainingSample supports:
  - clip-level arrays: visual_feature[T,112] / audio_feature[T,768]
  - T-axis alignment / dtype normalization / validation
  - metadata dict persistence (incl. legacy tuple-pairs conversion)
  - reasonable field defaults & backward compatibility (legacy fields)
"""

from pathlib import Path

import numpy as np
import pytest

from face_pipeline_v2.data.models import TrainingSample

T = 8
VISUAL_DIM = 112
AUDIO_DIM = 768


def _make_full_sample(t: int = T) -> TrainingSample:
    rng = np.random.default_rng(0)
    return TrainingSample(
        clip_id="clip_001",
        visual_feature=rng.standard_normal((t, VISUAL_DIM)).astype(np.float32),
        audio_feature=rng.standard_normal((t, AUDIO_DIM)).astype(np.float32),
        frame_index=np.arange(t, dtype=np.int32),
        timestamps=np.arange(t, dtype=np.float32) / 25.0,
        valid_length=t,
        dropout_mask=np.zeros(t, dtype=bool),
        audio_match_idx=np.arange(t, dtype=np.int32),
        metadata={"sample_id": "clip_001", "video_id": "vid_001", "fps": 25.0},
    )


class TestTrainingSampleInstantiation:
    """TrainingSample 可以实例化。"""

    def test_instantiate_zero_arg(self) -> None:
        ts = TrainingSample()
        assert isinstance(ts, TrainingSample)

    def test_instantiate_full_clip_level(self) -> None:
        ts = _make_full_sample()
        assert isinstance(ts, TrainingSample)
        assert ts.clip_id == "clip_001"

    def test_frozen_immutable(self) -> None:
        ts = _make_full_sample()
        with pytest.raises(Exception):
            ts.clip_id = "other"  # type: ignore[misc]


class TestTrainingSampleShapes:
    """shape: visual_feature [T,112], audio_feature [T,768]。"""

    def test_visual_feature_shape(self) -> None:
        ts = _make_full_sample()
        assert ts.visual_feature.shape == (T, VISUAL_DIM)

    def test_audio_feature_shape(self) -> None:
        ts = _make_full_sample()
        assert ts.audio_feature.shape == (T, AUDIO_DIM)

    def test_alignment_arrays_shape(self) -> None:
        ts = _make_full_sample()
        assert ts.frame_index.shape == (T,)
        assert ts.timestamps.shape == (T,)
        assert ts.dropout_mask.shape == (T,)
        assert ts.audio_match_idx.shape == (T,)

    def test_dtypes(self) -> None:
        ts = _make_full_sample()
        assert ts.visual_feature.dtype == np.float32
        assert ts.audio_feature.dtype == np.float32
        assert ts.frame_index.dtype == np.int32
        assert ts.timestamps.dtype == np.float32
        assert ts.dropout_mask.dtype == bool
        assert ts.audio_match_idx.dtype == np.int32

    def test_t_axis_consistency(self) -> None:
        ts = _make_full_sample()
        assert ts.visual_feature.shape[0] == ts.audio_feature.shape[0] == ts.valid_length

    def test_t_axis_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingSample(
                clip_id="bad",
                visual_feature=np.zeros((T, VISUAL_DIM), dtype=np.float32),
                audio_feature=np.zeros((T + 1, AUDIO_DIM), dtype=np.float32),
            )

    def test_wrong_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingSample(visual_feature=np.zeros((T, 50), dtype=np.float32))


class TestTrainingSampleMetadata:
    """metadata 保存。"""

    def test_metadata_dict_preserved(self) -> None:
        meta = {"sample_id": "clip_001", "video_id": "vid_001", "fps": 25.0}
        ts = TrainingSample(clip_id="clip_001", metadata=meta)
        assert ts.metadata == meta
        assert ts.metadata["video_id"] == "vid_001"

    def test_metadata_empty_dict_by_default(self) -> None:
        ts = TrainingSample()
        assert ts.metadata == {}

    def test_legacy_tuple_metadata_converted_to_dict(self) -> None:
        ts = TrainingSample(metadata=(("source", "vid_001"), ("lang", "zh")))
        assert ts.metadata == {"source": "vid_001", "lang": "zh"}

    def test_invalid_metadata_raises(self) -> None:
        with pytest.raises(ValueError):
            TrainingSample(metadata="not-a-dict")  # type: ignore[arg-type]


class TestTrainingSampleDefaults:
    """默认字段行为。"""

    def test_empty_sample_defaults(self) -> None:
        ts = TrainingSample()
        assert ts.clip_id == ""
        assert ts.visual_feature.shape == (0, VISUAL_DIM)
        assert ts.audio_feature.shape == (0, AUDIO_DIM)
        assert ts.valid_length == 0
        assert ts.frame_index.size == 0
        assert ts.timestamps.size == 0
        assert ts.dropout_mask.size == 0
        assert ts.audio_match_idx.size == 0

    def test_valid_length_auto_filled(self) -> None:
        ts = _make_full_sample()
        assert ts.valid_length == T

    def test_axis_defaults_filled_when_t_given(self) -> None:
        ts = TrainingSample(
            clip_id="clip_002",
            visual_feature=np.zeros((T, VISUAL_DIM), dtype=np.float32),
            audio_feature=np.zeros((T, AUDIO_DIM), dtype=np.float32),
        )
        assert ts.frame_index.tolist() == list(range(T))
        assert ts.timestamps.shape == (T,)
        assert ts.dropout_mask.tolist() == [False] * T
        assert ts.audio_match_idx.tolist() == [-1] * T
        assert ts.valid_length == T

    def test_legacy_fields_default_none(self) -> None:
        ts = TrainingSample()
        assert ts.sample_id == ""  # 自动继承 clip_id
        assert ts.video_clip is None
        assert ts.face_sequence is None
        assert ts.flame_feature is None
        assert ts.gaze_feature is None

    def test_sample_id_derived_from_clip_id(self) -> None:
        ts = TrainingSample(clip_id="clip_001")
        assert ts.sample_id == "clip_001"


class TestTrainingSampleBackwardCompat:
    """向后兼容: 旧单帧关键字构造 / from_legacy 工厂。"""

    def test_legacy_keyword_construction(self) -> None:
        ts = TrainingSample(
            sample_id="smp_001",
            video_clip=Path("/out/video.mp4"),
            face_sequence=Path("/out/faces/"),
            flame_feature=(0.1,) * 56,
            gaze_feature=(0.0,) * 6,
            audio_feature=(0.5,) * AUDIO_DIM,  # 旧单帧 tuple 音频
            metadata=(("source", "vid_001"),),
        )
        assert ts.sample_id == "smp_001"
        assert len(ts.gaze_feature) == 6
        # 单帧 tuple 音频被重塑为 [1,768], 视觉特征零填充对齐为 [1,112]
        assert ts.audio_feature.shape == (1, AUDIO_DIM)
        assert ts.visual_feature.shape == (1, VISUAL_DIM)
        assert ts.valid_length == 1

    def test_from_legacy_factory(self) -> None:
        ts = TrainingSample.from_legacy(
            sample_id="smp_002",
            video_clip=Path("/out/v.mp4"),
            flame_feature=(0.1,) * 56,
            gaze_feature=(0.0,) * 6,
            audio_feature=(0.5,) * AUDIO_DIM,
        )
        assert ts.clip_id == "smp_002"
        assert ts.visual_feature.shape == (1, VISUAL_DIM)
        assert ts.audio_feature.shape == (1, AUDIO_DIM)
        assert ts.metadata.get("sample_id") == "smp_002"


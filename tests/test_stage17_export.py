"""Tests for the Stage17 Export Layer (multimodal delivery format).

覆盖 (对应任务要求):
  1. export 后文件存在: flame/*.npy, audio/*.npy, timestamp/*.npz, metadata/*.json
  2. shape 检查: flame [T,112], audio [T,768]
  3. timestamp 长度一致: len(timestamp arrays) == flame.shape[0]
  4. metadata 字段完整 (数据契约)
  5. round-trip: 读取 export 结果后 shape 一致, 数据无变化
"""

from pathlib import Path

import numpy as np
import pytest

from face_pipeline_v2.data.models import TrainingSample
from face_pipeline_v2.stages.stage_17_fusion.export import (
    AUDIO_DIR,
    AUDIO_SUFFIX,
    EXPORT_VISUAL_LAYOUT,
    FLAME_DIR,
    FLAME_SUFFIX,
    MANIFEST_FILENAME,
    METADATA_DIR,
    METADATA_SUFFIX,
    REQUIRED_METADATA_FIELDS,
    TIMESTAMP_ARRAY_FIELDS,
    TIMESTAMP_DIR,
    TIMESTAMP_SUFFIX,
    export_multimodal_sample,
    read_multimodal_export,
    validate_export_metadata,
)

T = 20
VISUAL_DIM = 112
AUDIO_DIM = 768
CLIP_ID = "clip_export_001"


def _make_full_sample(t: int = T) -> TrainingSample:
    rng = np.random.default_rng(42)
    return TrainingSample(
        clip_id=CLIP_ID,
        visual_feature=rng.standard_normal((t, VISUAL_DIM)).astype(np.float32),
        audio_feature=rng.standard_normal((t, AUDIO_DIM)).astype(np.float32),
        frame_index=np.arange(t, dtype=np.int32),
        timestamps=(np.arange(t, dtype=np.float32) / 25.0) + 4.0,
        valid_length=t,
        dropout_mask=np.zeros(t, dtype=bool),
        audio_match_idx=np.arange(t, dtype=np.int32),
        metadata={
            "sample_id": CLIP_ID,
            "video_id": "vid_001",
            "fps": 25.0,
            "audio_backend": "Chinese HuBERT",
        },
    )


def _export_paths(out: Path) -> dict:
    return {
        "flame": out / FLAME_DIR / f"{CLIP_ID}{FLAME_SUFFIX}",
        "audio": out / AUDIO_DIR / f"{CLIP_ID}{AUDIO_SUFFIX}",
        "timestamp": out / TIMESTAMP_DIR / f"{CLIP_ID}{TIMESTAMP_SUFFIX}",
        "metadata": out / METADATA_DIR / f"{CLIP_ID}{METADATA_SUFFIX}",
        "manifest": out / MANIFEST_FILENAME,
    }


class TestExportFileStructure:
    """1. export 后文件存在。"""

    def test_export_creates_all_files(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        out = export_multimodal_sample(sample, tmp_path)
        assert out == tmp_path

        paths = _export_paths(out)
        for name, p in paths.items():
            assert p.exists(), f"missing export file: {name} -> {p}"

    def test_export_dir_structure(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        out = export_multimodal_sample(sample, tmp_path)
        for sub in (FLAME_DIR, AUDIO_DIR, TIMESTAMP_DIR, METADATA_DIR):
            assert (out / sub).is_dir()

    def test_empty_clip_id_raises(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        bad = TrainingSample(
            clip_id="",
            visual_feature=sample.visual_feature,
            audio_feature=sample.audio_feature,
            frame_index=sample.frame_index,
            timestamps=sample.timestamps,
            valid_length=sample.valid_length,
            dropout_mask=sample.dropout_mask,
            audio_match_idx=sample.audio_match_idx,
            metadata=sample.metadata,
        )
        with pytest.raises(ValueError):
            export_multimodal_sample(bad, tmp_path)


class TestExportShapes:
    """2. shape 检查: flame [T,112], audio [T,768]。"""

    def test_flame_shape_and_dtype(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        flame = np.load(_export_paths(tmp_path)["flame"])
        assert flame.shape == (T, VISUAL_DIM)
        assert flame.dtype == np.float32

    def test_audio_shape_and_dtype(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        audio = np.load(_export_paths(tmp_path)["audio"])
        assert audio.shape == (T, AUDIO_DIM)
        assert audio.dtype == np.float32

    def test_timestamp_npz_keys(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        with np.load(
            _export_paths(tmp_path)["timestamp"], allow_pickle=False
        ) as data:
            assert set(data.files) == set(TIMESTAMP_ARRAY_FIELDS)


class TestTimestampAlignment:
    """3. timestamp 长度一致: len == flame.shape[0]。"""

    def test_timestamp_lengths_equal_flame(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        flame = np.load(_export_paths(tmp_path)["flame"])
        with np.load(
            _export_paths(tmp_path)["timestamp"], allow_pickle=False
        ) as data:
            for key in TIMESTAMP_ARRAY_FIELDS:
                assert data[key].shape[0] == flame.shape[0], (
                    f"{key} length {data[key].shape[0]} != flame T {flame.shape[0]}"
                )

    def test_timestamp_values_identical_to_source(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        with np.load(
            _export_paths(tmp_path)["timestamp"], allow_pickle=False
        ) as data:
            assert np.array_equal(data["frame_index"], sample.frame_index)
            assert np.array_equal(data["timestamps"], sample.timestamps)
            assert np.array_equal(data["audio_match_idx"], sample.audio_match_idx)
            assert np.array_equal(data["dropout_mask"], sample.dropout_mask)


class TestMetadataCheck:
    """4. metadata 字段完整。"""

    def test_required_fields_present(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        with open(_export_paths(tmp_path)["metadata"], encoding="utf-8") as fh:
            import json

            meta = json.load(fh)
        for field in REQUIRED_METADATA_FIELDS:
            assert field in meta, f"metadata missing field: {field}"
        assert validate_export_metadata(meta) is True

    def test_layout_and_sources(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        with open(_export_paths(tmp_path)["metadata"], encoding="utf-8") as fh:
            import json

            meta = json.load(fh)
        assert meta["visual_layout"] == {
            k: list(v) for k, v in EXPORT_VISUAL_LAYOUT.items()
        }
        assert meta["head_source"] == "emica"
        assert meta["jaw_source"] == "emica"
        assert meta["expression_source"] == "emica"
        assert meta["gaze_source"] == "mediapipe"
        assert meta["audio_source"] == "chinese_hubert"
        assert meta["visual_dim"] == VISUAL_DIM
        assert meta["audio_dim"] == AUDIO_DIM
        assert meta["num_frames"] == T
        assert meta["fps"] == 25.0



class TestSourceValidation:
    """Stage12 ablation 来源校验 — emica/spectre 合法, 未知来源拒绝。"""

    def test_spectre_sources_allowed(self, tmp_path: Path) -> None:
        """全 spectre 来源可正常导出 (Stage12 Experiment B)。"""
        rng = np.random.default_rng(7)
        sample = TrainingSample(
            clip_id=CLIP_ID,
            visual_feature=rng.standard_normal((T, VISUAL_DIM)).astype(np.float32),
            audio_feature=rng.standard_normal((T, AUDIO_DIM)).astype(np.float32),
            frame_index=np.arange(T, dtype=np.int32),
            timestamps=(np.arange(T, dtype=np.float32) / 25.0) + 4.0,
            valid_length=T,
            dropout_mask=np.zeros(T, dtype=bool),
            audio_match_idx=np.arange(T, dtype=np.int32),
            metadata={
                "fps": 25.0,
                "expression_source": "spectre",
                "head_source": "spectre",
                "jaw_source": "spectre",
                "gaze_source": "mediapipe",
                "audio_source": "chinese_hubert",
            },
        )
        export_multimodal_sample(sample, tmp_path)
        with open(_export_paths(tmp_path)["metadata"], encoding="utf-8") as fh:
            import json

            meta = json.load(fh)
        assert meta["expression_source"] == "spectre"
        assert meta["head_source"] == "spectre"
        assert meta["jaw_source"] == "spectre"
        assert validate_export_metadata(meta) is True

    def test_unknown_source_rejected(self, tmp_path: Path) -> None:
        """未知来源 (deca) 在 export 校验阶段被拒绝。"""
        sample = _make_full_sample()
        sample.metadata["expression_source"] = "deca"
        with pytest.raises(ValueError, match="Invalid expression_source"):
            export_multimodal_sample(sample, tmp_path)



class TestManifest:
    """manifest.csv 数据集级索引。"""

    def test_manifest_row(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        lines = (
            _export_paths(tmp_path)["manifest"].read_text(encoding="utf-8").splitlines()
        )
        header = lines[0].split(",")
        for col in ("clip_id", "flame_path", "audio_path", "timestamp_path",
                    "num_frames", "fps"):
            assert col in header
        row = dict(zip(header, lines[1].split(",")))
        assert row["clip_id"] == CLIP_ID
        assert row["flame_path"] == f"{FLAME_DIR}/{CLIP_ID}{FLAME_SUFFIX}"
        assert row["audio_path"] == f"{AUDIO_DIR}/{CLIP_ID}{AUDIO_SUFFIX}"
        assert row["timestamp_path"] == f"{TIMESTAMP_DIR}/{CLIP_ID}{TIMESTAMP_SUFFIX}"
        assert int(row["num_frames"]) == T
        assert float(row["fps"]) == 25.0


class TestRoundTrip:
    """5. round-trip: 读取 export 结果后 shape 一致, 数据无变化。"""

    def test_round_trip_shapes(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        loaded = read_multimodal_export(tmp_path, CLIP_ID)
        assert loaded["visual_feature"].shape == (T, VISUAL_DIM)
        assert loaded["audio_feature"].shape == (T, AUDIO_DIM)
        assert loaded["frame_index"].shape == (T,)
        assert loaded["timestamps"].shape == (T,)
        assert loaded["audio_match_idx"].shape == (T,)
        assert loaded["dropout_mask"].shape == (T,)

    def test_round_trip_data_unchanged(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        loaded = read_multimodal_export(tmp_path, CLIP_ID)

        np.testing.assert_array_equal(loaded["visual_feature"], sample.visual_feature)
        np.testing.assert_array_equal(loaded["audio_feature"], sample.audio_feature)
        np.testing.assert_array_equal(loaded["frame_index"], sample.frame_index)
        np.testing.assert_array_equal(loaded["timestamps"], sample.timestamps)
        np.testing.assert_array_equal(loaded["audio_match_idx"], sample.audio_match_idx)
        np.testing.assert_array_equal(loaded["dropout_mask"], sample.dropout_mask)

    def test_round_trip_metadata_identical(self, tmp_path: Path) -> None:
        sample = _make_full_sample()
        export_multimodal_sample(sample, tmp_path)
        loaded = read_multimodal_export(tmp_path, CLIP_ID)
        assert loaded["metadata"]["clip_id"] == CLIP_ID
        assert loaded["metadata"]["video_id"] == "vid_001"
        assert loaded["metadata"]["audio_backend"] == "Chinese HuBERT"


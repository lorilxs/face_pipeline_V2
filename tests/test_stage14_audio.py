"""Unit tests for Stage 14 Audio Feature Extraction.

Covers the processor (HuBERT extraction, resampling, per-frame mapping),
NPZ integration (save/read round-trip), and the Stage14Audio BaseStage
lifecycle with a mocked HuBERT wrapper.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from face_pipeline_v2.data.models import AVAlignmentInfo, AudioFeature
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_14_audio_feature.integration import (
    read_audio_features,
    save_audio_features,
)
from face_pipeline_v2.stages.stage_14_audio_feature.processor import (
    build_audio_features,
    extract_audio_features_for_clip,
    resample_to_fps,
)
from face_pipeline_v2.stages.stage_14_audio_feature.stage import Stage14Audio
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alignment(
    audio_path: Path,
    clip_id: str = "clip_test",
    fps: float = 25.0,
    clip_start: float = 0.0,
) -> AVAlignmentInfo:
    return AVAlignmentInfo(
        clip_id=clip_id,
        original_audio_path=Path("orig.wav"),
        offset_frames=0,
        offset_ms=0.0,
        correction_applied=False,
        fps=fps,
        aligned_audio_path=audio_path,
        alignment_status="already_aligned",
        raw_audio_path=audio_path,
        clip_start=clip_start,
        clip_duration=2.0,
    )


def _make_context(config: dict, output_dir: Path) -> PipelineContext:
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage14", log_dir=output_dir / "logs"),
        model_manager=object(),
    )


def _fake_wrapper():
    """A fake HuBERTWrapper whose extract_from_file returns fixed features."""
    wrapper = mock.Mock()
    wrapper.extract_from_file.return_value = np.random.randn(100, 768).astype(
        np.float32
    )
    return wrapper


# ---------------------------------------------------------------------------
# Processor: resample_to_fps
# ---------------------------------------------------------------------------

class TestResampleToFps:
    def test_same_rate_noop(self) -> None:
        feats = np.random.randn(50, 768).astype(np.float32)
        out = resample_to_fps(feats, src_hz=50, dst_hz=50)
        assert out.shape == (50, 768)

    def test_50_to_25_decimation(self) -> None:
        feats = np.random.randn(100, 768).astype(np.float32)
        out = resample_to_fps(feats, src_hz=50, dst_hz=25)
        assert out.shape == (50, 768)

    def test_empty_input(self) -> None:
        feats = np.zeros((0, 768), dtype=np.float32)
        out = resample_to_fps(feats, src_hz=50, dst_hz=25)
        assert out.shape == (0, 768)

    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            resample_to_fps(np.zeros(768, dtype=np.float32))

    def test_output_values_matched(self) -> None:
        # 50→25 decimation should pick every 2nd frame exactly.
        feats = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
        out = resample_to_fps(feats, src_hz=50, dst_hz=25)
        assert np.allclose(out, feats[::2])


# ---------------------------------------------------------------------------
# Processor: build_audio_features
# ---------------------------------------------------------------------------

class TestBuildAudioFeatures:
    def test_one_feature_per_frame(self) -> None:
        feats = np.random.randn(50, 768).astype(np.float32)
        audio_features = build_audio_features(feats, fps=25, start_time=1.0)
        assert len(audio_features) == 50
        assert all(isinstance(f, AudioFeature) for f in audio_features)

    def test_timestamps_and_dimension(self) -> None:
        feats = np.random.randn(4, 768).astype(np.float32)
        afs = build_audio_features(feats, fps=25, start_time=2.0)
        assert [f.timestamp for f in afs] == pytest.approx(
            [2.0, 2.04, 2.08, 2.12]
        )
        assert all(f.dimension == 768 for f in afs)
        assert len(afs[0].feature) == 768
        assert isinstance(afs[0].feature, tuple)

    def test_source_audio_recorded(self) -> None:
        feats = np.random.randn(2, 768).astype(np.float32)
        afs = build_audio_features(
            feats, fps=25, source_audio=Path("clip_audio.wav")
        )
        assert all(f.source_audio == Path("clip_audio.wav") for f in afs)


# ---------------------------------------------------------------------------
# Processor: extract_audio_features_for_clip
# ---------------------------------------------------------------------------

class TestExtractForClip:
    def test_end_to_end(self, tmp_path: Path) -> None:
        audio_path = tmp_path / "clip.wav"
        audio_path.write_bytes(b"dummy")  # not read by the fake wrapper
        wrapper = _fake_wrapper()
        afs = extract_audio_features_for_clip(
            wrapper, audio_path, fps=25, start_time=0.5
        )
        # 100 raw frames at 50 Hz → 50 frames at 25 FPS
        assert len(afs) == 50
        assert all(f.dimension == 768 for f in afs)
        assert afs[0].timestamp == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration: NPZ save / read round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_round_trip(self, tmp_path: Path) -> None:
        feats = np.random.randn(25, 768).astype(np.float32)
        afs = build_audio_features(feats, fps=25, source_audio=Path("a.wav"))
        npz = save_audio_features(afs, tmp_path, "clip_1")
        assert npz.exists()
        assert (tmp_path / "clip_1" / "audio_metadata.json").exists()

        loaded = read_audio_features(tmp_path, "clip_1")
        assert len(loaded) == 25
        assert all(f.dimension == 768 for f in loaded)
        assert np.allclose(
            [np.asarray(f.feature) for f in loaded], feats, atol=1e-6
        )
        assert loaded[0].source_audio == Path("a.wav")

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            save_audio_features([], tmp_path, "clip_1")

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_audio_features(tmp_path, "nope")


# ---------------------------------------------------------------------------
# Stage: Stage14Audio lifecycle
# ---------------------------------------------------------------------------

class TestStage14Audio:
    def _config(self, output_dir: Path) -> dict:
        return {
            "audio": {
                "backend": "hubert",
                "model_path": "models/hubert/chinese-hubert-base",
                "sample_rate": 16000,
                "feature_dim": 768,
            },
            "stage_14": {
                "enabled": True,
                "model_path": "models/hubert/chinese-hubert-base",
                "device": "cpu",
                "fps": 25,
                "output_dir": str(output_dir),
            },
        }

    def test_validate_input_ok(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"dummy")
        alignment = _make_alignment(audio)
        ctx = _make_context(self._config(tmp_path), tmp_path)
        stage = Stage14Audio()
        assert stage.validate_input(alignment, ctx) is True

    def test_validate_input_wrong_type(self, tmp_path: Path) -> None:
        ctx = _make_context(self._config(tmp_path), tmp_path)
        stage = Stage14Audio()
        assert stage.validate_input("not-an-alignment", ctx) is False

    def test_validate_input_missing_audio(self, tmp_path: Path) -> None:
        alignment = _make_alignment(tmp_path / "missing.wav")
        ctx = _make_context(self._config(tmp_path), tmp_path)
        stage = Stage14Audio()
        assert stage.validate_input(alignment, ctx) is False

    def test_process_saves_npz(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"dummy")
        alignment = _make_alignment(audio)
        ctx = _make_context(self._config(tmp_path), tmp_path)
        stage = Stage14Audio()

        with mock.patch(
            "face_pipeline_v2.stages.stage_14_audio_feature.stage.load_hubert_wrapper",
            return_value=_fake_wrapper(),
        ):
            out = stage.process(alignment, ctx)

        assert isinstance(out, list) and len(out) > 0
        assert all(isinstance(f, AudioFeature) for f in out)
        assert (tmp_path / alignment.clip_id / "audio_features.npz").exists()

    def test_process_disabled_returns_empty(self, tmp_path: Path) -> None:
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"dummy")
        alignment = _make_alignment(audio)
        config = self._config(tmp_path)
        config["stage_14"]["enabled"] = False
        ctx = _make_context(config, tmp_path)
        stage = Stage14Audio()
        out = stage.process(alignment, ctx)
        assert out == []

    def test_validate_output_ok(self, tmp_path: Path) -> None:
        feats = np.random.randn(5, 768).astype(np.float32)
        afs = build_audio_features(feats, fps=25)
        ctx = _make_context(self._config(tmp_path), tmp_path)
        stage = Stage14Audio()
        assert stage.validate_output(afs, ctx) is True

    def test_validate_output_wrong_type(self, tmp_path: Path) -> None:
        ctx = _make_context(self._config(tmp_path), tmp_path)
        stage = Stage14Audio()
        assert stage.validate_output("bad", ctx) is False

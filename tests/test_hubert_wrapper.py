"""Unit tests for HuBERTWrapper (transformers backend).

Most tests run without loading the real model (mocked transformers).
One integration test requires the model directory present in the repo.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from face_pipeline_v2.models.wrappers.hubert import HuBERTWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, samples: int = 16000, sr: int = 16000) -> Path:
    """Write a synthetic 1-second mono WAV file."""
    t = np.linspace(0, samples / sr, samples, endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), wav, sr)
    return path


def _fake_hubert_model(device: str = "cpu"):
    """Build a fake model object returning a [1, T, 768] tensor.

    The fake ignores input_values and returns a fixed-shape tensor sized
    from the input length (320x downsampling like real HuBERT).
    """
    import torch

    class FakeModel:
        def __init__(self) -> None:
            self.device = device

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def __call__(self, input_values=None, **kwargs):
            n = input_values.shape[1]
            t_h = max(1, n // 320)
            return SimpleNamespace(
                last_hidden_state=torch.zeros((1, t_h, 768))
            )

    return FakeModel()


@pytest.fixture
def fake_hf_dir(tmp_path: Path) -> Path:
    """Create a HuggingFace-format model directory (config.json + weights)."""
    model_dir = tmp_path / "hubert-base"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"fake-bytes")
    return model_dir


@pytest.fixture
def nested_hf_dir(tmp_path: Path) -> Path:
    """Create the nested layout .../chinese-hubert-base/chinese-hubert-base/."""
    outer = tmp_path / "chinese-hubert-base"
    inner = outer / "chinese-hubert-base"
    inner.mkdir(parents=True)
    (inner / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")
    (inner / "pytorch_model.bin").write_bytes(b"fake-bytes")
    return outer


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TestResolveTransformersDir:
    def test_direct_hf_dir(self, fake_hf_dir: Path) -> None:
        w = HuBERTWrapper(model_path=fake_hf_dir)
        assert w._resolve_transformers_dir(fake_hf_dir) == fake_hf_dir

    def test_nested_hf_dir(self, nested_hf_dir: Path) -> None:
        w = HuBERTWrapper(model_path=nested_hf_dir)
        resolved = w._resolve_transformers_dir(nested_hf_dir)
        assert resolved == nested_hf_dir / "chinese-hubert-base"

    def test_fairseq_checkpoint_rejected(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "hubert_base_ls960.pt"
        ckpt.write_bytes(b"x")
        w = HuBERTWrapper(model_path=ckpt)
        with pytest.raises(RuntimeError, match="fairseq"):
            w._resolve_transformers_dir(ckpt)


# ---------------------------------------------------------------------------
# Waveform normalization / resampling
# ---------------------------------------------------------------------------

class TestNormalizeWaveform:
    def test_mean_zero_std_one(self) -> None:
        wav = np.random.randn(16000).astype(np.float32) * 3.0 + 5.0
        norm = HuBERTWrapper.normalize_waveform(wav)
        assert abs(float(np.mean(norm))) < 1e-3
        assert abs(float(np.std(norm)) - 1.0) < 1e-2

    def test_returns_float32(self) -> None:
        wav = np.random.randn(100).astype(np.float64)
        norm = HuBERTWrapper.normalize_waveform(wav)
        assert norm.dtype == np.float32


class TestResample:
    def test_same_rate_unchanged(self) -> None:
        wav = np.random.randn(100).astype(np.float32)
        out = HuBERTWrapper._resample(wav, 16000, 16000)
        assert len(out) == 100
        assert out is wav  # identity return

    def test_half_rate(self) -> None:
        wav = np.random.randn(8000).astype(np.float32)
        out = HuBERTWrapper._resample(wav, 16000, 8000)
        assert len(out) == 4000

    def test_double_rate(self) -> None:
        wav = np.random.randn(100).astype(np.float32)
        out = HuBERTWrapper._resample(wav, 8000, 16000)
        assert len(out) == 200


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

class TestLoadWaveform:
    def test_load_mono_wav(self, tmp_path: Path) -> None:
        path = _write_wav(tmp_path / "mono.wav", samples=16000, sr=16000)
        w = HuBERTWrapper(model_path=tmp_path)
        wav = w.load_waveform(path)
        assert wav.ndim == 1
        assert len(wav) == 16000
        assert wav.dtype == np.float32

    def test_load_missing_file(self, tmp_path: Path) -> None:
        w = HuBERTWrapper(model_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            w.load_waveform(tmp_path / "missing.wav")

    def test_load_and_resample(self, tmp_path: Path) -> None:
        # 8 kHz source, 16 kHz target → upsampled to 2x length
        path = _write_wav(tmp_path / "low.wav", samples=8000, sr=8000)
        w = HuBERTWrapper(model_path=tmp_path, sample_rate=16000)
        wav = w.load_waveform(path)
        assert len(wav) == 16000


# ---------------------------------------------------------------------------
# Loading + feature extraction (mocked model)
# ---------------------------------------------------------------------------

class TestLoadAndExtract:
    def test_extract_before_load_raises(self, fake_hf_dir: Path) -> None:
        w = HuBERTWrapper(model_path=fake_hf_dir)
        with pytest.raises(RuntimeError, match="not loaded"):
            w.extract_features(np.zeros(16000, dtype=np.float32))

    def test_load_and_extract_shape(self, fake_hf_dir: Path) -> None:
        with mock.patch(
            "transformers.HubertModel",
        ) as m_model:
            m_model.from_pretrained.return_value = _fake_hubert_model()
            w = HuBERTWrapper(model_path=fake_hf_dir, device="cpu")
            w.load()
            assert w.is_loaded

            wav = np.random.randn(32000).astype(np.float32)  # 2 s @ 16 kHz
            feats = w.extract_features(wav)
            # 32000 / 320 = 100 frames at 50 Hz
            assert feats.shape == (100, HuBERTWrapper.EMBEDDING_DIM)
            assert feats.dtype == np.float32

    def test_load_missing_config(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        w = HuBERTWrapper(model_path=empty_dir)
        with pytest.raises(FileNotFoundError):
            w.load()

    def test_extract_2d_raises(self, fake_hf_dir: Path) -> None:
        with mock.patch(
            "transformers.HubertModel",
        ) as m_model:
            m_model.from_pretrained.return_value = _fake_hubert_model()
            w = HuBERTWrapper(model_path=fake_hf_dir, device="cpu")
            w.load()
            with pytest.raises(ValueError):
                w.extract_features(np.zeros((2, 8000), dtype=np.float32))

    def test_is_loaded_default_false(self, fake_hf_dir: Path) -> None:
        w = HuBERTWrapper(model_path=fake_hf_dir)
        assert not w.is_loaded

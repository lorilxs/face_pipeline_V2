"""HuBERT audio feature extraction wrapper (transformers backend).

Per AISDD 4.5.1, Stages MUST NOT interact with the underlying model library
directly. This wrapper encapsulates HuBERT model loading and audio feature
extraction for Stage 14.

Backend decision (2026-08-26):
    fairseq is NOT usable on Python 3.12 (mutable-default dataclass crash).
    We load the HuBERT weights via HuggingFace `transformers` `HubertModel`
    from a HuggingFace-style local directory
    (config.json + pytorch_model.bin).

    Model: `models/hubert/chinese-hubert-base/chinese-hubert-base/`
    (768-dim hidden, ~50 Hz output frame rate at 16 kHz input).

    If a fairseq checkpoint (`hubert_base_ls960.pt`) is provided instead,
    the wrapper raises a clear error (fairseq unusable on Python 3.12).

Audio loading:
    `torchaudio`'s C extension is broken in the container's Python 3.12
    (ABI mismatch with torch 2.9). WAV loading therefore uses `soundfile`.

Interface:
    wrapper = HuBERTWrapper(model_path=Path("models/hubert/chinese-hubert-base"))
    wrapper.load()
    features = wrapper.extract_features(waveform)   # np.ndarray [T, 768]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Project root (directory containing the face_pipeline_v2 package).
# Used to resolve relative model paths regardless of the process CWD,
# matching the pattern in models/wrappers/syncnet.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class HuBERTWrapper:
    """HuBERT audio feature extractor for Stage 14.

    Loads a HuggingFace-format HuBERT model and extracts per-frame
    (50 Hz) 768-dim embeddings from a 16 kHz mono waveform.
    """

    EMBEDDING_DIM = 768
    SAMPLE_RATE = 16000
    EPS = 1e-5

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        sample_rate: int = SAMPLE_RATE,
        normalize: bool = True,
    ) -> None:
        """Initialize the HuBERT wrapper.

        Args:
            model_path: Path to a HuggingFace-style HuBERT directory
                (config.json + pytorch_model.bin).
            device: "auto" (detect CUDA), "cpu", or "cuda".
            sample_rate: Target audio sample rate (16 kHz required by HuBERT).
            normalize: Apply the Wav2Vec2FeatureExtractor mean/std
                normalization before the forward pass (matches the
                model's preprocessor_config `do_normalize: true`).
        """
        self._model_path: Path = Path(model_path)
        self._device: str = device
        self._sample_rate: int = sample_rate
        self._normalize: bool = normalize
        self._model: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        """Path to the HuBERT model directory."""
        return self._model_path

    @property
    def device(self) -> str:
        """Current inference device."""
        return self._device

    @property
    def sample_rate(self) -> int:
        """Audio sample rate expected by this model."""
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the transformers HuBERT model from the local directory.

        Raises:
            FileNotFoundError: If the model directory/config is missing.
            RuntimeError: If the model fails to load or the path is a
                fairseq checkpoint (unusable on Python 3.12).
        """
        path = self._resolve_transformers_dir(self._model_path)

        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"HuBERT model config.json not found: {(path / 'config.json').resolve()}"
            )
        if not (path / "pytorch_model.bin").exists():
            raise FileNotFoundError(
                f"HuBERT model weights not found: {(path / 'pytorch_model.bin').resolve()}"
            )

        if self._device == "auto":
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from transformers import HubertModel

            self._model = HubertModel.from_pretrained(str(path))
            self._model.eval()
            self._model.to(self._device)
        except Exception as exc:  # noqa: BLE001 - surface a clear error
            raise RuntimeError(f"Failed to load HuBERT model from {path}: {exc}") from exc

        self._loaded = True
        logger.info(
            "[HuBERT] Model loaded from %s (device=%s)", path, self._device,
        )

    @staticmethod
    def _resolve_transformers_dir(path: Path) -> Path:
        """Resolve a model path to a HuggingFace-format directory.

        - If the path itself contains ``config.json`` → use it.
        - If the path is a fairseq ``.pt`` checkpoint, it cannot be loaded
          by transformers → raise RuntimeError with guidance.
        - Otherwise walk one level down looking for a subdirectory that
          contains ``config.json`` + ``pytorch_model.bin``.

        Relative paths are resolved against the project root when they do
        not exist relative to the current working directory.

        The Chinese HuBERT archive extracts to
        ``.../chinese-hubert-base/chinese-hubert-base/`` (nested), so a
        one-level search handles the common layouts.
        """
        if path.suffix in (".pt", ".pth"):
            raise RuntimeError(
                f"{path} is a fairseq checkpoint. fairseq is not usable on "
                f"Python 3.12; configure a transformers-format model directory "
                f"(config.json + pytorch_model.bin), e.g. "
                f"models/hubert/chinese-hubert-base"
            )
        if not path.is_absolute() and not path.exists():
            candidate = _PROJECT_ROOT / path
            if candidate.exists():
                path = candidate
        if (path / "config.json").exists() and (path / "pytorch_model.bin").exists():
            return path
        # Nested layout: parent/chinese-hubert-base/chinese-hubert-base/
        if path.is_dir():
            for sub in sorted(path.iterdir()):
                if sub.is_dir():
                    if (sub / "config.json").exists() and (
                        sub / "pytorch_model.bin"
                    ).exists():
                        return sub
        return path

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def load_waveform(self, audio_path: Path) -> np.ndarray:
        """Load a WAV file as a mono 16 kHz float32 waveform.

        Uses `soundfile` (torchaudio is unavailable on Python 3.12).
        Resamples to the target sample rate when the file rate differs.

        Args:
            audio_path: Path to a WAV file (16 kHz mono expected).

        Returns:
            1-D float32 numpy array of shape [N].

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If the audio cannot be decoded.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path.resolve()}")

        try:
            import soundfile as sf

            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to read audio {audio_path} with soundfile: {exc}"
            ) from exc

        # Downmix to mono (always_2d gives [N, C])
        if data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True)
        waveform = data[:, 0].astype(np.float32)

        if sr != self._sample_rate:
            waveform = self._resample(waveform, sr, self._sample_rate)

        return waveform

    @staticmethod
    def _resample(
        waveform: np.ndarray, src_rate: int, dst_rate: int
    ) -> np.ndarray:
        """Simple linear-interpolation resampler (no librosa dependency)."""

        if src_rate == dst_rate:
            return waveform
        n_out = int(round(len(waveform) * dst_rate / src_rate))
        x_old = np.linspace(0.0, 1.0, len(waveform), endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, waveform).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_waveform(waveform: np.ndarray, eps: float = EPS) -> np.ndarray:
        """Apply mean/std normalization (matches Wav2Vec2FeatureExtractor).

        Args:
            waveform: 1-D float32 waveform.
            eps: Small epsilon for numerical stability.

        Returns:
            Normalized waveform.
        """
        mean = float(np.mean(waveform))
        var = float(np.var(waveform))
        return ((waveform - mean) / np.sqrt(var + eps)).astype(np.float32)

    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract HuBERT embeddings from a 16 kHz mono waveform.

        Args:
            waveform: 1-D float32 numpy array at ``sample_rate`` Hz.

        Returns:
            Float32 numpy array of shape [T, 768], where T is the number of
            50 Hz feature frames.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("HuBERT not loaded. Call load() first.")

        import torch

        if waveform.ndim != 1:
            raise ValueError(
                f"Expected 1-D waveform, got shape {waveform.shape}"
            )

        wav = waveform.astype(np.float32)
        if self._normalize:
            wav = self.normalize_waveform(wav)

        input_values = torch.from_numpy(wav).unsqueeze(0)  # [1, T]
        input_values = input_values.to(self._device)

        with torch.no_grad():
            outputs = self._model(input_values=input_values)

        last_hidden = outputs.last_hidden_state  # [1, T_h, 768]
        features = last_hidden[0].cpu().numpy().astype(np.float32)  # [T_h, 768]
        return features

    def extract_from_file(self, audio_path: Path) -> np.ndarray:
        """Convenience: load a WAV file and extract HuBERT features.

        Args:
            audio_path: Path to a WAV file (16 kHz mono expected).

        Returns:
            Float32 numpy array of shape [T, 768].
        """
        waveform = self.load_waveform(audio_path)
        return self.extract_features(waveform)

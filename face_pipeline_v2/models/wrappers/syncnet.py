"""SyncNet audio-visual synchronization model wrapper.

Per AISDD 3.11, SyncNet detects offset and computes sync confidence
between video lip movements and audio. Stage 7 uses this wrapper for
per-clip A/V synchronization detection.

Loading order:
    1. third_party/SyncNet (primary — real SyncNet PyTorch model).
    2. FileNotFoundError raised if checkpoint is missing (no silent fallback).

Interface:
    wrapper = SyncNetWrapper(checkpoint=Path("models/syncnet/syncnet_v2.model"))
    wrapper.load()
    offset, score, conf = wrapper.predict(video_path)
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Tuple

# ── Ensure third_party/SyncNet is importable ──────────────────────────
_THIRD_PARTY_SYNCNET = Path(__file__).resolve().parents[3] / "third_party" / "SyncNet"
if str(_THIRD_PARTY_SYNCNET) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_SYNCNET))


class SyncNetWrapper:
    """SyncNet A/V sync detector wrapper.

    Encapsulates SyncNet model loading and per-clip A/V evaluation.
    Provides a unified predict() interface for Stage 7.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        batch_size: int = 32,
        vshift: int = 10,
    ) -> None:
        """Initialize SyncNet wrapper.

        Args:
            model_path: Path to syncnet_v2.model checkpoint file.
            device: "auto" (detect CUDA), "cpu", or "cuda".
            batch_size: Batch size for frame/audio feature extraction.
            vshift: Number of offset frames to search (±vshift).
        """
        self._checkpoint: Path = Path(model_path)
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = device
        self._batch_size: int = batch_size
        self._vshift: int = vshift
        self._model: Any = None
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the SyncNet model from checkpoint.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
            RuntimeError: If model loading fails.
        """
        if not self._checkpoint.exists():
            raise FileNotFoundError(
                f"SyncNet checkpoint not found: {self._checkpoint.resolve()}"
            )

        from SyncNetInstance_calc_scores import SyncNetInstance

        self._model = SyncNetInstance(device=self._device)
        self._model.loadParameters(str(self._checkpoint))
        self._loaded = True
        print(
            f"[SyncNet] SyncNet checkpoint loaded: {self._checkpoint} "
            f"(device={self._device})"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, video_path: Path) -> Tuple[int, float, float, float]:
        """Run SyncNet evaluation on a video file.

        The video must contain both video and audio streams.
        SyncNet extracts frames via ffmpeg, computes MFCC from audio,
        and compares lip movements to audio features.

        Args:
            video_path: Path to a video file with audio.

        Returns:
            Tuple of (offset_frames, sync_lse_c, sync_lse_d, median_distance).
            - offset_frames: frame offset between audio and video.
            - sync_lse_c: LSE-C = median(distance) - min(distance).
            - sync_lse_d: LSE-D = min(distance).
            - median_distance: median of frame-audio pairwise distances.

        Raises:
            RuntimeError: If the model has not been loaded.
            FileNotFoundError: If video_path does not exist.
        """
        if not self._loaded:
            raise RuntimeError("SyncNet not loaded. Call load() first.")
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        tmp_base = video_path.parent / "syncnet_tmp"
        opt = SimpleNamespace(
            reference=video_path.stem,
            tmp_dir=str(tmp_base),
            batch_size=self._batch_size,
            vshift=self._vshift,
        )

        offset, lse_c, lse_d, median_dist = self._model.evaluate(opt, str(video_path))
        return (int(offset), float(lse_c), float(lse_d), float(median_dist))

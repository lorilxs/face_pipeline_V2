"""Stage 14 Audio Processor — HuBERT audio feature extraction.

Per AISDD 3.20, Stage 14 extracts 768-dim audio embeddings from the
corrected (aligned) clip audio using HuBERT.

Pipeline:
    Corrected Audio (16 kHz WAV)
      → load_waveform()               (soundfile, torchaudio broken on py3.12)
      → HuBERT forward pass            [T_h, 768] @ 50 Hz
      → resample to 25 FPS             [N, 768]
      → build_audio_features()         List[AudioFeature] (one per video frame)

Input MUST be AVAlignmentInfo.corrected/aligned audio (Stage 7.1),
never raw audio (AISDD 3.20.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from face_pipeline_v2.data.models import AudioFeature
from face_pipeline_v2.models.wrappers.hubert import HuBERTWrapper

# HuBERT feature frame rate at 16 kHz input (conv stride 320 → 50 Hz).
HUBERT_FRAME_HZ = 50


def load_hubert_wrapper(
    model_path: Path,
    device: str = "auto",
    normalize: bool = True,
) -> HuBERTWrapper:
    """Create and load a HuBERT wrapper.

    Args:
        model_path: Path to a HuggingFace-format HuBERT model directory.
        device: "auto", "cpu", or "cuda".
        normalize: Apply Wav2Vec2FeatureExtractor mean/std normalization.

    Returns:
        A loaded HuBERTWrapper instance.
    """
    wrapper = HuBERTWrapper(
        model_path=model_path,
        device=device,
        normalize=normalize,
    )
    wrapper.load()
    return wrapper


def extract_hubert_features(
    wrapper: HuBERTWrapper,
    audio_path: Path,
) -> np.ndarray:
    """Extract raw HuBERT embeddings from a clip audio file.

    Args:
        wrapper: Loaded HuBERTWrapper.
        audio_path: Path to a 16 kHz WAV file.

    Returns:
        Float32 array of shape [T_h, 768] at 50 Hz.
    """
    return wrapper.extract_from_file(audio_path)


def resample_to_fps(
    features: np.ndarray,
    src_hz: int = HUBERT_FRAME_HZ,
    dst_hz: int = 25,
) -> np.ndarray:
    """Resample feature frames along the time axis to a target frame rate.

    Linear interpolation on the time axis (frame-rate conversion). For the
    canonical 50 Hz → 25 FPS case this selects/decimates to one feature per
    video frame, keeping temporal alignment with the 25 FPS video.

    Args:
        features: Float32 array of shape [T_src, D].
        src_hz: Source frame rate (HuBERT outputs 50 Hz).
        dst_hz: Target frame rate (video FPS, default 25).

    Returns:
        Float32 array of shape [T_dst, D], where
        T_dst = max(1, round(T_src * dst_hz / src_hz)).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features [T, D], got {features.shape}")
    if src_hz <= 0 or dst_hz <= 0:
        raise ValueError("src_hz and dst_hz must be positive")

    t_src = features.shape[0]
    if t_src == 0:
        return np.zeros((0, features.shape[1]), dtype=np.float32)

    if src_hz == dst_hz:
        return features.astype(np.float32, copy=False)

    n_out = max(1, int(round(t_src * dst_hz / src_hz)))
    # Sample positions in source-frame units (fractional for interpolation).
    # For 50→25 this maps out[i] = i * (t_src/n_out) = i*2 → exact decimation.
    positions = np.arange(n_out) * (t_src / n_out)

    result = np.zeros((n_out, features.shape[1]), dtype=np.float32)
    for d in range(features.shape[1]):
        result[:, d] = np.interp(positions, np.arange(t_src), features[:, d])
    return result


def build_audio_features(
    features: np.ndarray,
    fps: int = 25,
    source_audio: Path = Path(""),
    start_time: float = 0.0,
) -> List[AudioFeature]:
    """Convert a resampled feature array into per-frame AudioFeature objects.

    Args:
        features: Float32 array of shape [N, 768] already aligned to the
            video frame rate.
        fps: Video frame rate (default 25).
        source_audio: The corrected audio path this feature came from.
        start_time: Clip start timestamp in seconds (default 0.0).

    Returns:
        List of AudioFeature, one per frame, with
        timestamp = start_time + i / fps and dimension = feature length.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features [N, D], got {features.shape}")

    dimension = int(features.shape[1])
    audio_features: List[AudioFeature] = []
    for i in range(features.shape[0]):
        audio_features.append(
            AudioFeature(
                timestamp=float(start_time) + i / float(fps),
                feature=tuple(float(x) for x in features[i]),
                dimension=dimension,
                source_audio=Path(source_audio),
            )
        )
    return audio_features


def extract_audio_features_for_clip(
    wrapper: HuBERTWrapper,
    audio_path: Path,
    fps: int = 25,
    start_time: float = 0.0,
) -> List[AudioFeature]:
    """End-to-end Stage 14 extraction for a single clip.

    Loads the audio, runs HuBERT, resamples 50 Hz → fps, and builds
    per-frame AudioFeature objects.

    Args:
        wrapper: Loaded HuBERTWrapper.
        audio_path: Path to the corrected clip WAV (16 kHz mono).
        fps: Target video frame rate (default 25).
        start_time: Clip start timestamp in seconds.

    Returns:
        List[AudioFeature] aligned to the video frame rate.
    """
    raw = extract_hubert_features(wrapper, audio_path)
    resampled = resample_to_fps(raw, src_hz=HUBERT_FRAME_HZ, dst_hz=fps)
    return build_audio_features(
        features=resampled,
        fps=fps,
        source_audio=audio_path,
        start_time=start_time,
    )

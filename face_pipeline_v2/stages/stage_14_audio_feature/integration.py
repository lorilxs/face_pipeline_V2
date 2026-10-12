"""Stage14 pipeline integration — NPZ serialization for AudioFeature.

Provides save/read utilities for Stage 14 → Stage 17 handoff.
NPZ format: {clip_id}/audio_features.npz with arrays:
  timestamp, feature [N,768], dimension, source_audio
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from face_pipeline_v2.data.models import AudioFeature

# Fixed array names stored in the NPZ (must stay stable for Stage 17).
NPZ_FILENAME = "audio_features.npz"
META_FILENAME = "audio_metadata.json"


def save_audio_features(
    features: List[AudioFeature],
    output_dir: Path,
    clip_id: str,
) -> Path:
    """Save a list of AudioFeature to an NPZ file.

    Args:
        features: List of AudioFeature objects (one per video frame).
        output_dir: Base output directory.
        clip_id: Clip identifier (sub-directory under output_dir).

    Returns:
        Path to the saved .npz file.

    Raises:
        ValueError: If the features list is empty or dimensions are mixed.
    """
    if not features:
        raise ValueError("Cannot save empty audio feature list")

    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    dimension = features[0].dimension
    if any(f.dimension != dimension for f in features):
        raise ValueError("All AudioFeature objects must share the same dimension")

    N = len(features)
    feature_array = np.zeros((N, dimension), dtype=np.float32)
    timestamps = np.zeros(N, dtype=np.float32)
    source_audio = str(features[0].source_audio)

    for i, f in enumerate(features):
        timestamps[i] = f.timestamp
        feature_array[i] = np.asarray(f.feature, dtype=np.float32)

    npz_path = clip_dir / NPZ_FILENAME
    np.savez_compressed(
        npz_path,
        timestamp=timestamps,
        feature=feature_array,
        dimension=np.array([dimension], dtype=np.int32),
        source_audio=np.array([source_audio]),
    )

    # Also save metadata JSON for easy inspection
    meta_path = clip_dir / META_FILENAME
    meta = {
        "clip_id": clip_id,
        "num_frames": N,
        "feature_dim": dimension,
        "source_audio": source_audio,
        "fields": ["timestamp", "feature", "dimension", "source_audio"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return npz_path


def read_audio_features(
    output_dir: Path,
    clip_id: str,
) -> List[AudioFeature]:
    """Read a Stage 14 NPZ output back into AudioFeature objects.

    Args:
        output_dir: Base output directory.
        clip_id: Clip identifier.

    Returns:
        List of AudioFeature objects.

    Raises:
        FileNotFoundError: If the .npz file is not found.
    """
    npz_path = output_dir / clip_id / NPZ_FILENAME
    if not npz_path.exists():
        raise FileNotFoundError(f"Audio features not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    feature_array = data["feature"]
    timestamps = data["timestamp"]
    N = feature_array.shape[0]
    dimension = int(data["dimension"][0])
    source_audio = str(data["source_audio"][0])

    features: List[AudioFeature] = []
    for i in range(N):
        features.append(
            AudioFeature(
                timestamp=float(timestamps[i]),
                feature=tuple(float(x) for x in feature_array[i]),
                dimension=dimension,
                source_audio=Path(source_audio),
            )
        )
    return features

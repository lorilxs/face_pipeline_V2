"""Stage13 pipeline integration — NPZ serialization for GazeFeature.

Provides save/read utilities for Stage13 → Stage17 handoff.
NPZ format: {clip_id}/gaze_features.npz with arrays:
  frame_index, timestamp, clip_id, gaze_vector [T,6]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from face_pipeline_v2.data.models import GazeFeature


def save_gaze_features(
    features: List[GazeFeature],
    output_dir: Path,
    clip_id: str,
) -> Path:
    """Save GazeFeature list to NPZ file.

    Args:
        features: List of GazeFeature objects.
        output_dir: Base output directory.
        clip_id: Clip identifier.

    Returns:
        Path to the saved .npz file.
    """
    clip_dir = output_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    T = len(features)
    gaze_array = np.zeros((T, 6), dtype=np.float32)
    frame_indices = np.zeros(T, dtype=np.int32)
    timestamps = np.zeros(T, dtype=np.float32)
    clip_ids = []

    for i, f in enumerate(features):
        frame_indices[i] = f.frame_index
        timestamps[i] = f.timestamp
        clip_ids.append(f.clip_id)
        gaze_array[i] = np.array(f.gaze_vector, dtype=np.float32)

    npz_path = clip_dir / "gaze_features.npz"
    np.savez_compressed(
        npz_path,
        frame_index=frame_indices,
        timestamp=timestamps,
        clip_id=np.array(clip_ids),
        gaze_vector=gaze_array,
    )

    # Also save metadata JSON for easy inspection
    meta_path = clip_dir / "gaze_metadata.json"
    meta = {
        "clip_id": clip_id,
        "num_frames": T,
        "gaze_dim": 6,
        "fields": ["frame_index", "timestamp", "clip_id", "gaze_vector"],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return npz_path


def read_gaze_features(
    output_dir: Path,
    clip_id: str,
) -> List[GazeFeature]:
    """Read Stage13 NPZ output back to GazeFeature objects.

    Args:
        output_dir: Base output directory.
        clip_id: Clip identifier.

    Returns:
        List of GazeFeature objects.

    Raises:
        FileNotFoundError: If the .npz file is not found.
    """
    npz_path = output_dir / clip_id / "gaze_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Gaze features not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    T = data["gaze_vector"].shape[0]

    features: List[GazeFeature] = []
    for i in range(T):
        gv = tuple(data["gaze_vector"][i].tolist())
        features.append(GazeFeature(
            frame_index=int(data["frame_index"][i]),
            timestamp=float(data["timestamp"][i]),
            clip_id=str(data["clip_id"][i]),
            left_eye=(gv[0], gv[1], gv[2]),
            right_eye=(gv[3], gv[4], gv[5]),
            gaze_vector=gv,
        ))

    return features
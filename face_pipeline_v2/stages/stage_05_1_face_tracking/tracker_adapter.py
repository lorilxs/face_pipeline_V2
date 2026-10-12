"""SCRFD → ByteTrack adapter for Stage 5.1.

Converts SCRFD detection output to ByteTrack input format, and
ByteTrack track output back to per-detection track_ids.

This adapter is the ONLY module that knows about both SCRFD and
ByteTrack data formats. Stage 5.1 and tracker.py are format-agnostic.

ByteTrack input:  [x1, y1, x2, y2, score]   (N, 5) np.ndarray
SCRFD output:     {"bbox": (x1,y1,x2,y2), "confidence": float}
"""

from typing import List

import numpy as np

from face_pipeline_v2.data.models import FaceInfo


def faces_to_detections(faces: List[FaceInfo]) -> np.ndarray:
    """Convert FaceInfo list to ByteTrack detection array.

    ByteTrack expects: [x1, y1, x2, y2, score] per detection.

    Args:
        faces: SCRFD detection results as FaceInfo objects.

    Returns:
        (N, 5) np.ndarray, dtype=float32. Empty (0,5) if no faces.
    """
    if not faces:
        return np.zeros((0, 5), dtype=np.float32)

    dets = np.zeros((len(faces), 5), dtype=np.float32)
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox
        dets[i, 0] = float(x1)
        dets[i, 1] = float(y1)
        dets[i, 2] = float(x2)
        dets[i, 3] = float(y2)
        dets[i, 4] = float(face.confidence)
    return dets


def make_track_id(track_int: int) -> str | None:
    """Convert integer track ID to string format.

    Args:
        track_int: Integer track_id from ByteTrack, or -1 for unmatched.

    Returns:
        "track_001" format string, or None for -1.
    """
    if track_int < 0:
        return None
    return f"track_{track_int:03d}"

"""Data I/O for the Stage5/Stage10 temporal stability audit.

Consumes two JSON files whose canonical schemas mirror the pipeline
dataclasses ``FaceInfo`` / ``LandmarkInfo`` (see
``face_pipeline_v2/data/models.py``).

Stage5 -- ``stage5_frames.json``::

    {
      "source": {
        "video_id": str, "clip_id": str, "video_path": str,
        "fps": int, "width": int, "height": int
      },
      "frames": [
        {
          "frame_index": int, "timestamp": float,
          "faces": [
            {
              "face_id": str,
              "bbox": [x1, y1, x2, y2],            # SCRFD tight bbox, frame coords
              "confidence": float,
              "landmark_5": [.. 10 floats ..],      # SCRFD 5-point, optional
              "expanded_bbox": [ex1, ey1, ex2, ey2] # optional (Stage5 crop bbox)
            }
          ]
        }, ...
      ]
    }

Stage10 -- ``stage10_landmarks.json``::

    {
      "source": { ... same as above ... },
      "landmarks": [
        {
          "frame_index": int, "timestamp": float, "face_id": str,
          "landmarks_106": [ 212 floats in ORIGINAL frame coords ]
        }, ...
      ]
    }

``landmarks_106`` is the flattened ``[x0, y0, x1, y1, ..., x105, y105]`` array
(106 points x 2), exactly as produced by Stage 10 after ``_map_to_frame``.

The parsers also accept a few common flat variants and always return a
normalized internal structure (see :class:`Stage5Data` / :class:`Stage10Data`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalized internal structures
# ---------------------------------------------------------------------------


@dataclass
class FaceRecord:
    """One SCRFD detection (largest face per frame is the audit target)."""

    frame_index: int
    timestamp: float
    face_id: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) frame coords
    confidence: float
    landmark_5: Tuple[float, ...] = ()
    expanded_bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class LandmarkRecord:
    """One frame's 106-point landmark result (frame coordinates)."""

    frame_index: int
    timestamp: float
    face_id: str
    landmarks_106: np.ndarray  # (106, 2) float


@dataclass
class Stage5Data:
    """Normalized Stage5 per-frame SCRFD results."""

    source: Dict[str, Any]
    frames: List[FaceRecord] = field(default_factory=list)

    def by_frame(self) -> Dict[int, FaceRecord]:
        return {f.frame_index: f for f in self.frames}


@dataclass
class Stage10Data:
    """Normalized Stage10 per-frame 106-landmark results."""

    source: Dict[str, Any]
    landmarks: List[LandmarkRecord] = field(default_factory=list)

    def by_frame(self) -> Dict[int, LandmarkRecord]:
        return {l.frame_index: l for l in self.landmarks}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _as_float4(v: Any) -> Tuple[float, float, float, float]:
    return tuple(float(x) for x in v)  # type: ignore[return-value]


def _normalize_source(raw: Any) -> Dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    return {
        "video_id": str(src.get("video_id", "")),
        "clip_id": str(src.get("clip_id", "")),
        "video_path": str(src.get("video_path", "")),
        "fps": int(src.get("fps", 0) or 0),
        "width": int(src.get("width", 0) or 0),
        "height": int(src.get("height", 0) or 0),
    }


def load_stage5(path: Path) -> Stage5Data:
    """Load a Stage5 face-sequence JSON into :class:`Stage5Data`.

    Supports the canonical schema above plus a flat list variant::

        [ {"frame_index": int, "bbox": [x1,y1,x2,y2], "confidence": float, ...}, ... ]

    Raises:
        ValueError: if the file cannot be parsed into per-frame face records.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(raw, list):
        entries = raw
        source: Dict[str, Any] = {}
    elif isinstance(raw, dict) and isinstance(raw.get("frames"), list):
        entries = raw["frames"]
        source = _normalize_source(raw.get("source"))
    else:
        raise ValueError(
            f"Unrecognized Stage5 JSON layout in {path}: expected a list or "
            f'{{"source":..., "frames":[...]}}'
        )

    records: List[FaceRecord] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Stage5 entry #{i} is not a dict")
        fi = int(entry.get("frame_index", i))
        ts = float(entry.get("timestamp", 0.0))

        faces_raw = entry.get("faces")
        if isinstance(faces_raw, list) and faces_raw:
            faces = faces_raw
        else:
            # flat variant: the entry itself is a single face
            faces = [entry]

        for f in faces:
            if "bbox" not in f:
                raise ValueError(f"Stage5 frame {fi} has no 'bbox'")
            bbox = _as_float4(f["bbox"])
            if len(bbox) != 4:
                raise ValueError(f"Stage5 frame {fi} bbox must have 4 values")
            records.append(
                FaceRecord(
                    frame_index=fi,
                    timestamp=ts,
                    face_id=str(f.get("face_id", "")),
                    bbox=bbox,
                    confidence=float(f.get("confidence", 0.0)),
                    landmark_5=tuple(float(v) for v in (f.get("landmark_5") or ())),
                    expanded_bbox=(
                        _as_float4(f["expanded_bbox"])
                        if f.get("expanded_bbox")
                        else None
                    ),
                )
            )

    records.sort(key=lambda r: r.frame_index)
    return Stage5Data(source=source, frames=records)


def _extract_106(entry: Dict[str, Any], frame_index: int) -> np.ndarray:
    """Pull a (106, 2) landmark array from an entry in several accepted forms."""
    if "landmarks_106" in entry:
        vals = [float(v) for v in entry["landmarks_106"]]
    elif "landmarks" in entry and isinstance(entry["landmarks"], list):
        lms = entry["landmarks"]
        if lms and isinstance(lms[0], (list, tuple)):
            # [[x, y], ...] 106x2
            vals = [float(v) for pair in lms for v in pair]
        else:
            vals = [float(v) for v in lms]
    elif "points" in entry and isinstance(entry["points"], list):
        lms = entry["points"]
        vals = [float(v) for pair in lms for v in pair]
    else:
        raise ValueError(f"Stage10 frame {frame_index} has no 'landmarks_106'")

    if len(vals) < 212:
        raise ValueError(
            f"Stage10 frame {frame_index} has {len(vals)} values, expected 212"
        )
    arr = np.asarray(vals[:212], dtype=np.float64).reshape(106, 2)
    return arr


def load_stage10(path: Path) -> Stage10Data:
    """Load a Stage10 106-landmark JSON into :class:`Stage10Data`.

    Supports the canonical schema plus a flat list variant::

        [ {"frame_index": int, "landmarks_106": [212 floats]}, ... ]

    Raises:
        ValueError: if the file cannot be parsed into landmark records.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(raw, list):
        entries = raw
        source: Dict[str, Any] = {}
    elif isinstance(raw, dict) and isinstance(raw.get("landmarks"), list):
        entries = raw["landmarks"]
        source = _normalize_source(raw.get("source"))
    elif isinstance(raw, dict) and isinstance(raw.get("frames"), list):
        # tolerate a stage5-style "frames" container
        entries = raw["frames"]
        source = _normalize_source(raw.get("source"))
    else:
        raise ValueError(
            f"Unrecognized Stage10 JSON layout in {path}: expected a list or "
            f'{{"source":..., "landmarks":[...]}}'
        )

    records: List[LandmarkRecord] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Stage10 entry #{i} is not a dict")
        fi = int(entry.get("frame_index", i))
        ts = float(entry.get("timestamp", 0.0))
        arr = _extract_106(entry, fi)
        records.append(
            LandmarkRecord(
                frame_index=fi,
                timestamp=ts,
                face_id=str(entry.get("face_id", "")),
                landmarks_106=arr,
            )
        )

    records.sort(key=lambda r: r.frame_index)
    return Stage10Data(source=source, landmarks=records)


# ---------------------------------------------------------------------------
# Selection helper (mirrors Stage5 'largest_area' strategy)
# ---------------------------------------------------------------------------


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def select_main_face(records: List[FaceRecord]) -> Optional[FaceRecord]:
    """Return the largest-area face for a frame (Stage 5 default strategy)."""
    if not records:
        return None
    return max(records, key=lambda r: bbox_area(r.bbox))



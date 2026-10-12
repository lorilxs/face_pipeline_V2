"""Face detector for Stage 5.

Encapsulates SCRFD model calls and face selection logic.
Does NOT load models — the SCRFD model is injected by the Stage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from face_pipeline_v2.data.models import FaceInfo
from face_pipeline_v2.stages.stage_05_face_detection.face_cropper import crop_face
from face_pipeline_v2.stages.stage_05_face_detection.face_resolution_filter import (
    tight_bbox_is_valid,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    FaceCropError,
    InvalidBboxError,
)


# Type alias for a single SCRFD detection result
Detection = Dict[str, Any]
# Detection shape: {"bbox": (x1,y1,x2,y2), "confidence": float, "landmarks": tuple | None}


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate bounding box area."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def select_face_largest_area(detections: List[Detection]) -> Optional[Detection]:
    """Select the face with the largest bounding box area.

    Args:
        detections: List of SCRFD detection dicts.

    Returns:
        The detection with largest bbox area, or None if empty.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: _bbox_area(d["bbox"]))


# Registry of face selection strategies
_SELECTION_STRATEGIES = {
    "largest_area": select_face_largest_area,
}


class FaceDetector:
    """Per-frame face detector.

    Wraps the SCRFD model with face selection and crop logic.
    The SCRFD model and output dirs are injected — this class
    is pure logic.

    Usage:
        detector = FaceDetector(
            scrfd_model=model_manager.get_model("scrfd"),
            crop_enabled=True,
            crop_size=256,
            crops_base_dir=Path("outputs/faces"),
            selection_strategy="largest_area",
        )
        frame_info = detector.process_frame(frame_index, image, clip_id)
    """

    def __init__(
        self,
        scrfd_model: Any,
        crop_enabled: bool = True,
        crop_size: int = 256,
        crops_base_dir: Optional[Path] = None,
        selection_strategy: str = "largest_area",
        expand_ratio: float = 2.0,
    ) -> None:
        """Initialize the face detector.

        Args:
            scrfd_model: Loaded SCRFD model with detect(image) -> List[dict].
            crop_enabled: Whether to save face crops to disk.
            crop_size: Target square crop size in pixels (deprecated, kept for compat).
            crops_base_dir: Base directory for crop outputs.
            selection_strategy: Face selection strategy name.
            expand_ratio: Bbox expansion factor on original frame (default 2.0).
        """
        self._scrfd: Any = scrfd_model
        self._crop_enabled: bool = crop_enabled
        self._crop_size: int = crop_size
        self._crops_base_dir: Optional[Path] = crops_base_dir
        self._selection_strategy: str = selection_strategy
        self._expand_ratio: float = expand_ratio

        if selection_strategy not in _SELECTION_STRATEGIES:
            raise DataProcessingError(
                f"Unknown face selection strategy: '{selection_strategy}'. "
                f"Available: {list(_SELECTION_STRATEGIES)}"
            )
        self._selector = _SELECTION_STRATEGIES[selection_strategy]

    def process_frame(
        self,
        frame_index: int,
        image: np.ndarray,
        clip_id: str,
        video_id: str,
    ) -> List[FaceInfo]:
        """Detect faces in a single frame and produce FaceInfo objects.

        Args:
            frame_index: Global frame index in the video.
            image: BGR image array (H, W, C).
            clip_id: Parent clip identifier.
            video_id: Parent video identifier.

        Returns:
            List of FaceInfo objects, sorted by bbox area descending.
            Empty list if no faces detected.

        Raises:
            DataProcessingError: If SCRFD detection fails.
            InvalidBboxError: If a detected tight bbox is degenerate/invalid.
            FaceCropError: If crop (expansion/save) fails for a face.
        """
        # --- SCRFD detection ---
        try:
            detections: List[Detection] = self._scrfd.detect(image)
        except Exception as exc:
            raise DataProcessingError(
                f"SCRFD detection failed at frame {frame_index}: {exc}"
            ) from exc

        if not detections:
            return []

        # --- Sort by area descending (largest first) ---
        detections_sorted = sorted(
            detections, key=lambda d: _bbox_area(d["bbox"]), reverse=True
        )

        # --- Build FaceInfo list ---
        face_infos: List[FaceInfo] = []
        for i, det in enumerate(detections_sorted):
            bbox: Tuple[float, float, float, float] = det["bbox"]
            if not tight_bbox_is_valid(bbox):
                raise InvalidBboxError(
                    f"Invalid tight bbox {bbox} at frame {frame_index}, face {i}"
                )
            confidence: float = float(det.get("confidence", 0.0))
            landmarks: Tuple[float, ...] = det.get("landmarks") or ()

            face_id = f"{video_id}_{clip_id}_face_{i:03d}"

            # --- Crop with bbox expansion on original frame ---
            crop_path = Path("")  # placeholder if crop disabled
            expanded_bbox = (0.0, 0.0, 0.0, 0.0)
            if self._crop_enabled and self._crops_base_dir is not None:
                crop_dir = self._crops_base_dir / video_id / clip_id / "crops"
                crop_filename = f"frame_{frame_index:06d}_face_{i:03d}.jpg"
                crop_path = crop_dir / crop_filename
                try:
                    crop_path, expanded_bbox = crop_face(
                        image=image,
                        bbox=bbox,
                        output_path=crop_path,
                        expand_ratio=self._expand_ratio,
                        crop_size=self._crop_size,
                    )
                except (ValueError, OSError) as exc:
                    raise FaceCropError(
                        f"Face crop failed at frame {frame_index}, "
                        f"face {i}: {exc}"
                    ) from exc

            face_info = FaceInfo(
                frame_index=frame_index,
                face_id=face_id,
                bbox=bbox,
                landmark_5=landmarks,
                confidence=confidence,
                face_crop_path=crop_path,
                expanded_bbox=expanded_bbox,
            )
            face_infos.append(face_info)

        return face_infos

"""Face cropper for Stage 5.

Stage 5 responsibility (redesigned 2026-08-12):
    1. SCRFD tight bbox → expand bbox on original frame
    2. Crop the expanded region at native resolution (no resize)
    3. Save to disk as the "Base Face Crop Asset"

Expand happens at the BBOX level on the ORIGINAL FRAME, not post-crop.
This ensures the crop contains the full face (forehead, chin, cheeks),
not just the tight detection region.

Downstream stages consume the expanded raw crop and apply their own
resize as needed — no further expand required.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def expand_bbox(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    expand_ratio: float,
) -> Tuple[float, float, float, float]:
    """Expand a tight face bbox around its centre on the original frame.

    The expanded bbox includes forehead, chin, and cheeks — the full
    face region needed by downstream models (FLAME, Gaze, etc.).

    Args:
        bbox: SCRFD tight bbox (x1, y1, x2, y2).
        img_w, img_h: Original frame dimensions.
        expand_ratio: Expansion factor (2.0 = double the width/height).

    Returns:
        Expanded bbox (x1, y1, x2, y2), clamped to frame boundaries.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * expand_ratio
    bh = (y2 - y1) * expand_ratio

    ex1 = max(0.0, cx - bw / 2.0)
    ey1 = max(0.0, cy - bh / 2.0)
    ex2 = min(float(img_w), cx + bw / 2.0)
    ey2 = min(float(img_h), cy + bh / 2.0)

    return (ex1, ey1, ex2, ey2)


def crop_face(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    output_path: Path,
    expand_ratio: float = 2.0,
    crop_size: int = 256,
) -> Tuple[Path, Tuple[float, float, float, float]]:
    """Expand bbox on original frame, crop, and save raw crop.

    Args:
        image: Original BGR frame (H, W, 3).
        bbox: SCRFD tight detection bbox (x1, y1, x2, y2).
        output_path: Destination path for the raw crop image.
        expand_ratio: Bbox expansion factor (default 2.0).
        crop_size: **Ignored**. Kept for backward compatibility.

    Returns:
        Tuple of (crop_path, expanded_bbox).
        expanded_bbox is the actual bbox used for cropping (x1,y1,x2,y2).

    Raises:
        ValueError: If expanded bbox is invalid.
    """
    img_h, img_w = image.shape[:2]

    # Step 1: Expand bbox on original frame
    ex1, ey1, ex2, ey2 = expand_bbox(bbox, img_w, img_h, expand_ratio)

    # Step 2: Clamp and validate
    x1_c = max(0, int(ex1))
    y1_c = max(0, int(ey1))
    x2_c = min(img_w, int(ex2))
    y2_c = min(img_h, int(ey2))

    w = x2_c - x1_c
    h = y2_c - y1_c
    if w <= 0 or h <= 0:
        raise ValueError(
            f"Invalid expanded crop region: bbox={bbox}, "
            f"expand_ratio={expand_ratio}, "
            f"expanded=({ex1:.0f},{ey1:.0f},{ex2:.0f},{ey2:.0f}), "
            f"size={w}x{h}"
        )

    # Step 3: Crop at native resolution — NO resize
    crop = image[y1_c:y2_c, x1_c:x2_c]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)

    expanded_bbox = (ex1, ey1, ex2, ey2)
    return output_path, expanded_bbox

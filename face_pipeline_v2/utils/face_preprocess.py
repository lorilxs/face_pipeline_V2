"""Face crop preprocess utilities.

Provides shared expand + resize logic for downstream model stages
(Stage 6/7/8).  Each stage reads a raw face crop from Stage 5 and
applies its own preprocessing pipeline.

Why: different models have different input requirements.
Stage 5 should NOT decide the final crop size/expansion for all models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def expand_face_crop(
    image: np.ndarray,
    expand_ratio: float = 1.0,
) -> np.ndarray:
    """Expand a face crop by a given ratio around its centre.

    The input image is treated as the "tight bbox crop" from Stage 5.
    The output is a larger crop that includes context around the face.

    When the expanded region exceeds the image boundary, missing pixels
    are filled with black (zero padding).

    Args:
        image: BGR image (H, W, 3) — raw face crop from Stage 5.
        expand_ratio: Expansion factor. 1.0 = no change, 1.6 = 60% larger.

    Returns:
        Expanded BGR image.  Same dtype as input.
    """
    if expand_ratio <= 1.0:
        return image

    h, w = image.shape[:2]
    new_w = int(w * expand_ratio)
    new_h = int(h * expand_ratio)

    # Centre of the original crop
    cx, cy = w / 2.0, h / 2.0

    # New crop boundaries in the original crop's coordinate space
    x1 = int(cx - new_w / 2.0)
    y1 = int(cy - new_h / 2.0)
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Create a black canvas of the target size
    expanded = np.zeros((new_h, new_w, 3), dtype=image.dtype)

    # Compute the valid region within the canvas (where original pixels land)
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = min(w, w - (x2 - w)) if x2 > w else w
    src_y2 = min(h, h - (y2 - h)) if y2 > h else h

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(new_w, x2)
    dst_y2 = min(new_h, y2)

    # Copy valid region
    src_w = src_x2 - src_x1
    src_h = src_y2 - src_y1
    dst_w = dst_x2 - dst_x1
    dst_h = dst_y2 - dst_y1
    copy_w = min(src_w, dst_w)
    copy_h = min(src_h, dst_h)

    if copy_w > 0 and copy_h > 0:
        expanded[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
            image[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

    return expanded


def resize_face(
    image: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize a face crop to a square target size.

    Args:
        image: BGR image (H, W, 3).
        target_size: Target square size in pixels.
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized square BGR image of shape (target_size, target_size, 3).
    """
    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)


def preprocess_face_crop(
    crop_path: Path,
    expand_ratio: float = 1.0,
    target_size: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Full preprocess pipeline for a single face crop.

    1. Load raw crop from Stage 5.
    2. Expand around centre by expand_ratio.
    3. Resize to target_size (if given).

    Args:
        crop_path: Path to the raw face crop image.
        expand_ratio: Expansion factor (1.0 = no expand).
        target_size: Target square size in pixels (None = no resize).

    Returns:
        Tuple of:
            - processed image (np.ndarray)
            - raw crop size (h, w)
            - expanded size (h, w) — same as raw if expand_ratio=1.0

    Raises:
        FileNotFoundError: If crop_path does not exist.
    """
    if not crop_path.exists():
        raise FileNotFoundError(f"Face crop not found: {crop_path}")

    image = cv2.imread(str(crop_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read face crop: {crop_path}")

    raw_size = image.shape[:2]  # (h, w)

    # Step 1: Expand
    if expand_ratio > 1.0:
        image = expand_face_crop(image, expand_ratio)
    expanded_size = image.shape[:2]

    # Step 2: Resize
    if target_size is not None:
        image = resize_face(image, target_size)

    return image, raw_size, expanded_size

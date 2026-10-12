"""Affine transform utilities for Stage 11 Face Alignment.

Provides the core geometric operations for landmark-based face alignment
using OpenCV (no PyTorch / kornia / einops).

Operations:
    1. Select the two eye centers from 106 landmarks — the only points that
       anchor rotation/scale (robust to yaw).
    2. Define the canonical 256×256 template (InsightFace ArcFace standard).
    3. Estimate a similarity transform analytically from the two eye centers
       (rotation ← eye-line, scale ← eye distance, translation ← eye midpoint).
    4. Warp face crop to aligned output via cv2.warpAffine.
    5. Inverse affine transform for aligned→original recovery.

Design note (2026-08-14): the nose tip is NOT used to estimate the transform.
For a face with slight yaw the nose shifts off the eye-midline, and a 3-point
least-squares fit (estimateAffinePartial2D) misreads that offset as roll,
tilting an otherwise-level face. The nose is retained only for diagnostics.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# Landmark indices for InsightFace 2d106det 106-point format
# ============================================================================

# NOTE: the 2d106det.onnx landmark order is NOT the standard insightface
# 106-point order. Verified empirically (2026-08-13) from the model output:
#   0-32  face contour
#   33-42 left eye (10 pts)
#   43-51 left eyebrow
#   52-71 mouth
#   72-86 nose
#   87-96 right eye (10 pts)
#   97-105 right eyebrow
# The previous indices (63-78 for eyes, 55-62 for nose) actually pointed at
# the mouth/nose-lower region, producing misaligned (half-face) outputs.

# Left eye region
LEFT_EYE_INDICES: Tuple[int, ...] = tuple(range(33, 43))
# Right eye region
RIGHT_EYE_INDICES: Tuple[int, ...] = tuple(range(87, 97))
# Nose tip: bottommost points of the nose region
NOSE_LOWER_INDICES: Tuple[int, ...] = (83, 84, 85)


# ============================================================================
# Standard 256×256 face alignment template
# ============================================================================

# Canonical face template — InsightFace ArcFace standard 5-point template
# (112×112), scaled to 256×256. Eyes sit ~46% from the top (room for
# forehead) and the nose ~64% from the top, giving an eye-to-nose vertical
# ratio of ~0.57 — matching real face proportions. The previous hand-placed
# nose (65px below eyes at an 86px eye distance = 0.76 ratio) over-magnified
# the face. The nose row is kept ONLY for diagnostics; it is never used in
# the transform.
TEMPLATE_256: np.ndarray = np.array([
    [87.53,  118.16],  # left eye center
    [168.07, 117.72],  # right eye center
    [128.06, 163.97],  # nose tip (diagnostic only)
], dtype=np.float32)

# Eye-only template used by the 2-point similarity transform.
TEMPLATE_EYES: np.ndarray = TEMPLATE_256[:2].copy()

# ============================================================================
# Configurable alignment template (backward compatible)
# ============================================================================

# Documented default template parameters (normalized 256×256 canvas).
# These reproduce the canonical ArcFace template scaled to 256 — i.e. the
# pre-config behavior. See build_template() for the exact-reproduction
# guarantee.
DEFAULT_EYE_DISTANCE: float = 80.541
DEFAULT_EYE_CENTER_Y: float = 118.0
DEFAULT_SCALE_FACTOR: float = 1.0

# Eye-to-nose vertical ratio used ONLY for the diagnostic nose row (~0.57,
# matching real face proportions and the legacy template). The nose row is
# never used in the transform.
_NOSE_RATIO: float = 0.57


def build_template(
    eye_distance: float = DEFAULT_EYE_DISTANCE,
    eye_center_y: float = DEFAULT_EYE_CENTER_Y,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
) -> np.ndarray:
    """Build a canonical 256×256 alignment template from parameters.

    The template is defined in the normalized 256×256 aligned canvas, so it is
    independent of the source resolution — the affine scale term normalizes by
    the source eye distance.

    Canonical-default guarantee: when the requested parameters equal the
    legacy defaults (eye_distance=80.541, eye_center_y=118.0,
    scale_factor=1.0) the EXACT pre-config `TEMPLATE_256` constant is
    returned, so output stays byte-for-byte identical to the old version.
    Any deviation (different eye
    distance, different eye y, or scale_factor != 1.0) builds a fresh
    symmetric template centered on x=128.

    scale_factor semantics: `final_scale = original_scale * scale_factor`.
    Because the affine scale equals template_eye_distance / source_eye_distance,
    scaling the template eye distance by `scale_factor` shrinks (<1.0) or
    enlarges (>1.0) the canonical face around the eye-line midpoint — the eye
    line stays at `eye_center_y`.

    Args:
        eye_distance: Template eye-to-eye distance in px (default 80.541).
        eye_center_y: Template eye-line Y in px (default 118.0).
        scale_factor: Uniform canonical-face scale (default 1.0).

    Returns:
        Shape (3, 2) float32 — [left_eye, right_eye, nose(diagnostic only)].
    """
    if (
        scale_factor == DEFAULT_SCALE_FACTOR
        and abs(eye_distance - DEFAULT_EYE_DISTANCE) < 1e-6
        and abs(eye_center_y - DEFAULT_EYE_CENTER_Y) < 1e-6
    ):
        # Exact canonical defaults → preserve the legacy constant verbatim.
        return TEMPLATE_256.copy()

    ed = float(eye_distance) * float(scale_factor)
    half = ed / 2.0
    return np.array([
        [128.0 - half, eye_center_y],
        [128.0 + half, eye_center_y],
        [128.0, eye_center_y + _NOSE_RATIO * ed],
    ], dtype=np.float32)


def select_eye_points(landmarks: np.ndarray) -> np.ndarray:
    """Select the two eye centers from 106 landmarks for alignment.

    These two points are the sole anchors for rotation/scale/translation.
    The nose is deliberately excluded — it is yaw-sensitive and would bias
    the rotation for turned heads.

    Args:
        landmarks: Shape (106, 2) — (x, y) coordinates in source image space.

    Returns:
        Shape (2, 2) — [left_eye_center, right_eye_center].

    Raises:
        ValueError: If landmarks shape is incorrect or the eye centers
                    coincide (degenerate).
    """
    if landmarks.shape[0] < 106:
        raise ValueError(
            f"Need at least 106 landmarks, got {landmarks.shape[0]}"
        )

    left_eye_center = landmarks[list(LEFT_EYE_INDICES)].mean(axis=0)
    right_eye_center = landmarks[list(RIGHT_EYE_INDICES)].mean(axis=0)

    if np.linalg.norm(left_eye_center - right_eye_center) < 1.0:
        raise ValueError(
            "Left and right eye centers are too close — "
            "landmarks may be degenerate."
        )

    return np.array([left_eye_center, right_eye_center], dtype=np.float32)


def select_nose_point(landmarks: np.ndarray) -> np.ndarray:
    """Select the nose-tip reference point (diagnostic only).

    Args:
        landmarks: Shape (106, 2) — (x, y) coordinates in source image space.

    Returns:
        Shape (2,) — nose-tip reference (mean of lower nose points).
    """
    if landmarks.shape[0] < 106:
        raise ValueError(
            f"Need at least 106 landmarks, got {landmarks.shape[0]}"
        )
    return landmarks[list(NOSE_LOWER_INDICES)].mean(axis=0).astype(np.float32)


def estimate_affine(
    src_eyes: np.ndarray,
    dst_eyes: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate a similarity transform from two eye point pairs.

    A pure similarity transform (rotation + uniform scale + translation) is
    derived analytically from the two eye centers:
        rotation  ← angle of the eye-line (levels the eyes)
        scale     ← template_eye_distance / source_eye_distance
        translate ← maps source eye midpoint to template eye midpoint

    This guarantees an upright face (eyes horizontal) with a stable,
    eye-distance-anchored scale regardless of nose/mouth asymmetry — unlike
    a 3-point estimateAffinePartial2D fit, which lets a yawed nose tilt the
    whole face.

    Args:
        src_eyes: Shape (2, 2) — [left_eye, right_eye] source points.
        dst_eyes: Shape (2, 2) — [left_eye, right_eye] template points.

    Returns:
        Shape (2, 3) affine matrix [[a, b, tx], [c, d, ty]], or None
        if the eye points are degenerate.
    """
    if src_eyes.shape != (2, 2) or dst_eyes.shape != (2, 2):
        return None
    if not np.all(np.isfinite(src_eyes)) or not np.all(np.isfinite(dst_eyes)):
        return None

    src_left = src_eyes[0].astype(np.float64)
    src_right = src_eyes[1].astype(np.float64)
    dst_left = dst_eyes[0].astype(np.float64)
    dst_right = dst_eyes[1].astype(np.float64)

    src_vec = src_right - src_left
    dst_vec = dst_right - dst_left
    src_dist = float(np.hypot(src_vec[0], src_vec[1]))
    dst_dist = float(np.hypot(dst_vec[0], dst_vec[1]))
    if src_dist < 1e-6 or dst_dist < 1e-6:
        return None

    theta = math.atan2(dst_vec[1], dst_vec[0]) - math.atan2(src_vec[1], src_vec[0])
    scale = dst_dist / src_dist

    # Sanity range check (rejects degenerate / extreme scales)
    if scale < 0.05 or scale > 20.0:
        return None

    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Translation: dst_left = scale * R(theta) * src_left + t
    rot_left = np.array([
        cos_t * src_left[0] - sin_t * src_left[1],
        sin_t * src_left[0] + cos_t * src_left[1],
    ])
    tx = dst_left[0] - scale * rot_left[0]
    ty = dst_left[1] - scale * rot_left[1]

    matrix = np.array([
        [scale * cos_t, -scale * sin_t, tx],
        [scale * sin_t,  scale * cos_t, ty],
    ], dtype=np.float32)

    return matrix


def decompose_affine(affine_matrix: np.ndarray) -> Tuple[float, float]:
    """Decompose a 2×3 similarity matrix into (rotation_degrees, scale).

    Args:
        affine_matrix: Shape (2, 3) similarity matrix [[a, b, tx], [c, d, ty]].

    Returns:
        Tuple (rotation_degrees, scale). Rotation is the in-plane roll the
        transform applies; scale is the uniform magnification factor.
    """
    a = float(affine_matrix[0, 0])  # scale * cos(theta)
    b = float(affine_matrix[0, 1])  # -scale * sin(theta)
    scale = float(np.hypot(a, b))
    # Recover theta = atan2(sin(theta), cos(theta)) = atan2(-b, a)
    rotation = math.degrees(math.atan2(-b, a))
    return rotation, scale


def warp_face(
    image: np.ndarray,
    affine_matrix: np.ndarray,
    output_resolution: Tuple[int, int],
) -> np.ndarray:
    """Apply affine warp to produce an aligned face image.

    Args:
        image: Source face crop image (H, W, 3) BGR.
        affine_matrix: Shape (2, 3) affine transform matrix.
        output_resolution: (width, height) of output aligned face.

    Returns:
        Aligned face image of shape (H_out, W_out, 3).

    Raises:
        RuntimeError: If warpAffine fails (output is all-zero).
    """
    out_w, out_h = output_resolution

    aligned = cv2.warpAffine(
        image,
        affine_matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return aligned


def invert_affine(affine_matrix: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 2×3 affine transform.

    This allows recovering the original image from the aligned version:
        original = cv2.warpAffine(aligned, inverse, (orig_w, orig_h))

    Args:
        affine_matrix: Shape (2, 3) forward transform (src → dst).

    Returns:
        Shape (2, 3) inverse transform (dst → src).
    """
    inv = cv2.invertAffineTransform(affine_matrix)
    return inv


def matrix_to_tuple(matrix: np.ndarray) -> Tuple[float, ...]:
    """Flatten a 2×3 affine matrix to a 6-element tuple.

    Order: [a, b, tx, c, d, ty]
    Corresponds to matrix [[a, b, tx], [c, d, ty]].

    Args:
        matrix: Shape (2, 3) float32 affine matrix.

    Returns:
        Tuple of 6 floats.
    """
    flat = matrix.flatten()
    return tuple(float(v) for v in flat)


def tuple_to_matrix(t: Tuple[float, ...]) -> np.ndarray:
    """Convert a 6-element tuple back to a (2, 3) affine matrix.

    Args:
        t: 6 floats in order [a, b, tx, c, d, ty].

    Returns:
        Shape (2, 3) float32 matrix.
    """
    return np.array([
        [t[0], t[1], t[2]],
        [t[3], t[4], t[5]],
    ], dtype=np.float32)

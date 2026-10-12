"""Face alignment processor for Stage 11.

Encapsulates per-landmark affine alignment logic.
Does NOT call SCRFD, InsightFace, or any external model.

Processing:
    1. Build a face_crop_path lookup from FaceSequence data.
    2. For each LandmarkInfo, load the face crop image.
    3. Decompose landmarks_106 into (106, 2) numpy array.
    4. Select 3 key points (left eye, right eye, nose tip).
    5. Estimate affine transform to 256×256 template.
    6. Warp face crop → aligned 256×256 image.
    7. Save aligned image to disk.
    8. Build AlignedFaceInfo (immutable).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_pipeline_v2.data.models import (
    AlignedFaceInfo,
    FaceSequence,
    LandmarkInfo,
)
from face_pipeline_v2.stages.stage_11_alignment.affine_transform import (
    DEFAULT_EYE_CENTER_Y,
    DEFAULT_EYE_DISTANCE,
    DEFAULT_SCALE_FACTOR,
    TEMPLATE_EYES,
    build_template,
    decompose_affine,
    estimate_affine,
    matrix_to_tuple,
    select_eye_points,
    select_nose_point,
    warp_face,
)


def resolve_alignment_template(
    alignment_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the Stage 11 alignment template from config (backward compatible).

    Reads `alignment.template.{eye_distance,eye_center_y}` and
    `alignment.scale_factor`. If the `alignment.template` block is absent
    (legacy config), falls back to the documented defaults 80.541 / 118.0.

    The final canonical template is built by `build_template(...)`, which
    returns the exact legacy TEMPLATE_256 when the canonical defaults are
    requested (byte-identical output) and a freshly-built symmetric template
    otherwise.

    Args:
        alignment_cfg: The `alignment` section of the pipeline config.

    Returns:
        Dict with keys:
            eye_distance (float), eye_center_y (float), scale_factor (float),
            template ((3, 2) np.ndarray), eyes ((2, 2) np.ndarray).
    """
    scale_factor = float(
        alignment_cfg.get("scale_factor", DEFAULT_SCALE_FACTOR)
    )
    tmpl_cfg = alignment_cfg.get("template")
    if isinstance(tmpl_cfg, dict) and tmpl_cfg:
        eye_distance = float(
            tmpl_cfg.get("eye_distance", DEFAULT_EYE_DISTANCE)
        )
        eye_center_y = float(
            tmpl_cfg.get("eye_center_y", DEFAULT_EYE_CENTER_Y)
        )
    else:
        eye_distance = DEFAULT_EYE_DISTANCE
        eye_center_y = DEFAULT_EYE_CENTER_Y

    template_256 = build_template(
        eye_distance=eye_distance,
        eye_center_y=eye_center_y,
        scale_factor=scale_factor,
    )
    return {
        "eye_distance": eye_distance,
        "eye_center_y": eye_center_y,
        "scale_factor": scale_factor,
        "template": template_256,
        "eyes": template_256[:2].copy(),
    }


def align_faces(
    landmarks: List[LandmarkInfo],
    face_sequences: List[FaceSequence],
    output_dir: Path,
    output_resolution: Tuple[int, int],
    save_images: bool,
    logger: Any,
    eye_template: Optional[np.ndarray] = None,
) -> List[AlignedFaceInfo]:
    """Align all faces using their Stage 10 landmarks.

    Args:
        landmarks: List[LandmarkInfo] from Stage 10.
        face_sequences: List[FaceSequence] from Stage 5
                        (used to look up face_crop_path).
        output_dir: Directory to save aligned face images.
        output_resolution: (width, height) of aligned output, e.g. (256, 256).
        save_images: If True, write aligned images to disk.
        logger: PipelineLogger.
        eye_template: Canonical (2, 2) eye template in aligned-canvas pixels.
                      None → the exact legacy TEMPLATE_EYES (byte-identical
                      pre-config behavior). Supplied by Stage 11 from config.

    Returns:
        List[AlignedFaceInfo], one per successfully aligned face.
    """
    # Build lookup: (clip_id, frame_index, face_id) → FaceInfo
    face_lookup: Dict[Tuple[str, int, str], Any] = {}
    for seq in face_sequences:
        for frame_info in seq.frames:
            for face in frame_info.faces:
                key = (seq.clip_id, frame_info.frame_index, face.face_id)
                face_lookup[key] = face

    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Canonical template for this run — resolved from config by the caller.
    # None → legacy exact TEMPLATE_EYES (byte-identical pre-config behavior).
    if eye_template is None:
        eye_template = TEMPLATE_EYES

    aligned_list: List[AlignedFaceInfo] = []
    skipped_no_crop = 0
    skipped_bad_landmarks = 0
    skipped_affine_fail = 0
    skipped_warp_fail = 0

    # Diagnostic accumulators (nose offset is diagnostic-only, never in matrix)
    diag_rotations: List[float] = []
    diag_scales: List[float] = []
    diag_eye_before: List[float] = []
    diag_eye_after: List[float] = []
    diag_bbox_ratios: List[float] = []
    scale_warn_count = 0

    for lm in landmarks:
        clip_id = lm.clip_id
        frame_idx = lm.frame_index
        face_id = lm.face_id
        lookup_key = (clip_id, frame_idx, face_id)

        # --- Step 1: Find matching FaceInfo ---
        face_info = face_lookup.get(lookup_key)
        if face_info is None:
            skipped_no_crop += 1
            logger.debug(
                f"[Stage 11] No FaceInfo found for "
                f"clip={clip_id} frame={frame_idx} face={face_id}"
            )
            continue

        crop_path = face_info.face_crop_path
        if crop_path is None or not Path(crop_path).exists():
            skipped_no_crop += 1
            continue

        # --- Step 2: Load face crop image ---
        image = cv2.imread(str(crop_path))
        if image is None:
            skipped_no_crop += 1
            continue

        orig_h, orig_w = image.shape[:2]

        # --- Step 3: Parse landmarks ---
        # landmarks_106 is a flat tuple of 212 floats:
        # [x0, y0, x1, y1, ..., x105, y105]
        landmarks_flat = np.array(lm.landmarks_106, dtype=np.float32)
        if len(landmarks_flat) < 212:
            skipped_bad_landmarks += 1
            logger.warning(
                f"[Stage 11] Insufficient landmarks for "
                f"clip={clip_id} frame={frame_idx}: "
                f"{len(landmarks_flat)} values (need 212)"
            )
            continue

        landmarks_106 = landmarks_flat.reshape(106, 2)

        # --- Step 3b: Map landmarks from frame space → crop space ---
        # Stage 5 crops using expanded_bbox (not tight bbox), raw (no resize).
        # Stage 10 stores landmarks in frame coords:
        #   frame_x = crop_x + expanded_bbox.x1  (see _map_to_frame in Stage 10)
        # Stage 11 warps the face CROP image, so reverse the offset:
        #   crop_x = frame_x - expanded_bbox.x1
        eb = face_info.expanded_bbox  # (ex1, ey1, ex2, ey2)
        eb_w = eb[2] - eb[0]
        eb_h = eb[3] - eb[1]

        landmarks_crop = landmarks_106.copy()

        if eb_w > 0 and eb_h > 0:
            # Normal path: use expanded_bbox offset (1:1, no scale)
            landmarks_crop[:, 0] = landmarks_106[:, 0] - eb[0]
            landmarks_crop[:, 1] = landmarks_106[:, 1] - eb[1]
        else:
            # Fallback (legacy FaceInfo without expanded_bbox): tight bbox
            bbox = face_info.bbox
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            if bbox_w <= 0 or bbox_h <= 0:
                skipped_bad_landmarks += 1
                continue
            landmarks_crop[:, 0] = landmarks_106[:, 0] - bbox[0]
            landmarks_crop[:, 1] = landmarks_106[:, 1] - bbox[1]
            # Scale correction for legacy resized crops
            if orig_w != int(bbox_w) or orig_h != int(bbox_h):
                landmarks_crop[:, 0] *= (orig_w / bbox_w)
                landmarks_crop[:, 1] *= (orig_h / bbox_h)

        # Debug: log first 3 frames per clip for coordinate verification
        _debug_count = getattr(align_faces, '_debug_seen', {}).get(clip_id, 0)
        if _debug_count < 3:
            tight = face_info.bbox
            tight_w = tight[2] - tight[0]
            tight_h = tight[3] - tight[1]
            logger.info(
                f"[Stage 11] coord-check clip={clip_id} frame={frame_idx}: "
                f"tight_bbox_size={tight_w:.0f}x{tight_h:.0f} "
                f"expanded_bbox_size={eb_w:.0f}x{eb_h:.0f} "
                f"crop_size={orig_w}x{orig_h} "
                f"lm_frame0=({landmarks_106[0,0]:.1f},{landmarks_106[0,1]:.1f}) "
                f"lm_crop0=({landmarks_crop[0,0]:.1f},{landmarks_crop[0,1]:.1f})"
            )
            align_faces._debug_seen = getattr(align_faces, '_debug_seen', {})
            align_faces._debug_seen[clip_id] = _debug_count + 1

        # --- Step 4: Select eye key points (rotation/scale anchor) ---
        try:
            eye_points = select_eye_points(landmarks_crop)
            nose_point = select_nose_point(landmarks_crop)
        except ValueError as exc:
            skipped_bad_landmarks += 1
            logger.warning(
                f"[Stage 11] Key point selection failed for "
                f"clip={clip_id} frame={frame_idx}: {exc}"
            )
            continue

        # --- Step 5: Estimate similarity transform from eyes only ---
        affine_matrix = estimate_affine(eye_points, eye_template)
        if affine_matrix is None:
            skipped_affine_fail += 1
            logger.warning(
                f"[Stage 11] Affine estimation failed for "
                f"clip={clip_id} frame={frame_idx}"
            )
            continue

        # --- Step 5b: Diagnostics (nose offset is diagnostic-only) ---
        eye_left, eye_right = eye_points[0], eye_points[1]
        eye_vec = eye_right - eye_left
        eye_mid = (eye_left + eye_right) / 2.0
        eye_dist_before = float(np.hypot(eye_vec[0], eye_vec[1]))
        eye_dist_after = float(np.hypot(
            eye_template[1][0] - eye_template[0][0],
            eye_template[1][1] - eye_template[0][1],
        ))
        eye_line_roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))
        nose_offset_x = float(nose_point[0] - eye_mid[0])
        rotation_deg, scale = decompose_affine(affine_matrix)

        contour = landmarks_crop[0:33]
        cx1, cy1 = contour.min(axis=0)
        cx2, cy2 = contour.max(axis=0)
        face_area = float(max(0.0, (cx2 - cx1) * (cy2 - cy1)))
        crop_area = float(orig_w * orig_h)
        face_bbox_ratio = face_area / crop_area if crop_area > 0 else 0.0

        diag_rotations.append(rotation_deg)
        diag_scales.append(scale)
        diag_eye_before.append(eye_dist_before)
        diag_eye_after.append(eye_dist_after)
        diag_bbox_ratios.append(face_bbox_ratio)

        logger.debug(
            f"[Stage 11] align clip={clip_id} frame={frame_idx}: "
            f"eye_dist_before={eye_dist_before:.1f} "
            f"eye_dist_after={eye_dist_after:.1f} "
            f"rotation={rotation_deg:.2f}deg scale={scale:.3f} "
            f"eye_line_roll={eye_line_roll:.2f}deg "
            f"nose_offset={nose_offset_x:+.1f}px "
            f"face_bbox_ratio={face_bbox_ratio:.3f}"
        )
        if scale > 2.0:
            scale_warn_count += 1
            logger.warning(
                f"[Stage 11] HIGH SCALE clip={clip_id} frame={frame_idx}: "
                f"scale={scale:.3f}x "
                f"(eye_dist_before={eye_dist_before:.1f}px) — "
                f"face may be over-magnified"
            )

        # --- Step 6: Warp face ---
        try:
            aligned_img = warp_face(image, affine_matrix, output_resolution)
        except Exception as exc:
            skipped_warp_fail += 1
            logger.warning(
                f"[Stage 11] Warp failed for "
                f"clip={clip_id} frame={frame_idx}: {exc}"
            )
            continue

        # --- Step 7: Save aligned image ---
        aligned_path = _build_output_path(
            output_dir, clip_id, frame_idx, face_id
        )

        # Always create parent directory (required for both save and path reference)
        aligned_path.parent.mkdir(parents=True, exist_ok=True)

        if save_images:
            success = cv2.imwrite(str(aligned_path), aligned_img)
            if not success:
                raise RuntimeError(
                    f"Failed to save aligned face image: {aligned_path}"
                )

        # --- Step 8: Build AlignedFaceInfo ---
        aligned_list.append(AlignedFaceInfo(
            clip_id=clip_id,
            frame_index=frame_idx,
            face_id=face_id,
            timestamp=lm.timestamp,
            aligned_face_path=aligned_path,
            affine_matrix=matrix_to_tuple(affine_matrix),
            original_image_size=(orig_h, orig_w),
            aligned_resolution=(
                output_resolution[1], output_resolution[0]
            ),  # (height, width)
        ))

    # --- Summary ---
    total = len(landmarks)
    logger.info(
        f"[Stage 11] Alignment summary: "
        f"{len(aligned_list)}/{total} aligned, "
        f"{skipped_no_crop} no-crop, "
        f"{skipped_bad_landmarks} bad-landmarks, "
        f"{skipped_affine_fail} affine-fail, "
        f"{skipped_warp_fail} warp-fail"
    )

    if diag_scales:
        logger.info(
            f"[Stage 11] Alignment diagnostics: "
            f"rotation {min(diag_rotations):.2f}..{max(diag_rotations):.2f}° "
            f"(mean {np.mean(diag_rotations):.2f}°), "
            f"scale {min(diag_scales):.2f}..{max(diag_scales):.2f}x "
            f"(mean {np.mean(diag_scales):.2f}x), "
            f"scale>2.0 count={scale_warn_count}, "
            f"eye_dist {min(diag_eye_before):.1f}..{max(diag_eye_before):.1f}px→"
            f"{min(diag_eye_after):.1f}px, "
            f"face_bbox_ratio mean={np.mean(diag_bbox_ratios):.3f}"
        )

    return aligned_list


def _build_output_path(
    output_dir: Path,
    clip_id: str,
    frame_index: int,
    face_id: str,
) -> Path:
    """Build the output path for an aligned face image.

    Format: <output_dir>/<clip_id>/aligned_<frame_index>_<face_id>.jpg

    Args:
        output_dir: Root output directory.
        clip_id: Clip identifier.
        frame_index: Frame number.
        face_id: Face identifier.

    Returns:
        Absolute path for the aligned face image.
    """
    clip_dir = output_dir / clip_id
    filename = f"aligned_{frame_index:06d}_{face_id}.jpg"
    return clip_dir / filename

"""Landmark extraction processor for Stage 10.

Encapsulates per-FaceSequence landmark inference logic.
Does NOT load models — the model is injected by the Stage.

Processing:
    1. For each frame in a FaceSequence, select the main face.
    2. Load face crop from FaceInfo.face_crop_path.
    3. Run InsightFace 106 landmark inference → 106 (x, y) coords
       in crop pixel space.
    4. Map crop-space coordinates to original frame coordinates
       via FaceInfo.bbox.
    5. Build LandmarkInfo entries.
"""

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from face_pipeline_v2.data.models import FaceInfo, FaceSequence, LandmarkInfo


def extract_landmarks(
    sequence: FaceSequence,
    model: Any,
    logger: Any,
) -> List[LandmarkInfo]:
    """Extract 106-point landmarks for every frame in a FaceSequence.

    Only processes frames that have at least one face with a valid
    face_crop_path. Frames with no faces or missing crops are skipped.

    Args:
        sequence: Input FaceSequence with face crops.
        model: Loaded InsightFaceLandmarkWrapper with
               predict(face_crop) → np.ndarray (106, 2).
        logger: PipelineLogger for logging.

    Returns:
        List[LandmarkInfo], one per successfully processed face.
    """
    clip_id = sequence.clip_id

    # Determine main face track
    main_track_id: str | None = None
    if sequence.identity_result is not None:
        main_track_id = sequence.identity_result.main_track_id

    landmarks_list: List[LandmarkInfo] = []
    skipped_missing_crop = 0
    skipped_no_face = 0

    # Process frames in order (sorted by frame_index)
    sorted_frames = sorted(sequence.frames, key=lambda f: f.frame_index)

    for frame_info in sorted_frames:
        faces = frame_info.faces
        if not faces:
            skipped_no_face += 1
            continue

        # Select target face (main track if available, else largest = first)
        target_face = _select_target_face(faces, main_track_id)

        if target_face is None:
            skipped_no_face += 1
            continue

        # --- Load crop image ---
        crop_path = target_face.face_crop_path
        if crop_path is None or not Path(crop_path).exists():
            skipped_missing_crop += 1
            continue

        image = cv2.imread(str(crop_path))
        if image is None:
            skipped_missing_crop += 1
            continue

        # --- Run landmark inference ---
        try:
            # landmarks_crop: (106, 2) in crop pixel space
            landmarks_crop = model.predict(image)
        except Exception as exc:
            logger.warning(
                f"[Stage 10] Landmark inference failed for "
                f"clip={clip_id} frame={frame_info.frame_index} "
                f"face={target_face.face_id}: {exc}"
            )
            continue

        # --- Map crop coords → original frame coords ---
        frame_landmarks = _map_to_frame(landmarks_crop, target_face, image)

        # --- Flatten to tuple ---
        landmarks_flat: tuple = tuple(
            float(v) for v in frame_landmarks.flatten()
        )

        # --- Build LandmarkInfo ---
        landmarks_list.append(LandmarkInfo(
            clip_id=clip_id,
            frame_index=frame_info.frame_index,
            face_id=target_face.face_id,
            timestamp=frame_info.timestamp,
            landmarks_106=landmarks_flat,
            image_width=0,   # Original frame dimensions not available
            image_height=0,  # from FaceInfo alone — set to 0 as placeholder
        ))

    if skipped_missing_crop > 0 or skipped_no_face > 0:
        logger.debug(
            f"[Stage 10] clip={clip_id}: {len(landmarks_list)} landmarks, "
            f"{skipped_no_face} frames skipped (no face), "
            f"{skipped_missing_crop} frames skipped (missing crop)"
        )

    return landmarks_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_target_face(
    faces: tuple,
    main_track_id: str | None,
) -> FaceInfo | None:
    """Select the target face from a frame's face list.

    Priority:
        1. Main track face (by track_id), if available.
        2. First face (largest area, per Stage 5 convention).

    Args:
        faces: Tuple of FaceInfo for the current frame.
        main_track_id: Main track identifier from identity_result.

    Returns:
        Selected FaceInfo, or None if no faces available.
    """
    if not faces:
        return None

    if main_track_id is not None:
        for face in faces:
            if face.track_id == main_track_id:
                return face

    # Default: first face (largest area per Stage 5 sorting)
    return faces[0]


def _map_to_frame(
    landmarks_crop: np.ndarray,
    face_info: FaceInfo,
    crop_image: np.ndarray,
) -> np.ndarray:
    """Map landmarks from crop pixel space to original frame coordinates.

    Stage 5 crops the face using expanded_bbox (not tight bbox), and the
    crop is saved at native resolution (no resize).  Therefore the crop
    coordinate space is 1:1 with expanded_bbox:
        frame_x = crop_x + expanded_bbox.x1
        frame_y = crop_y + expanded_bbox.y1

    Args:
        landmarks_crop: Shape (106, 2) — (x, y) in crop pixel space.
        face_info: FaceInfo with expanded_bbox for crop mapping.
        crop_image: The crop image (H, W, 3) — used for fallback dims.

    Returns:
        Shape (106, 2) — (x, y) in original frame pixel space.
    """
    eb = face_info.expanded_bbox  # (ex1, ey1, ex2, ey2)
    eb_w = eb[2] - eb[0]
    eb_h = eb[3] - eb[1]

    frame_landmarks = np.zeros_like(landmarks_crop)

    if eb_w > 0 and eb_h > 0:
        # Normal path: expanded_bbox is valid → 1:1 mapping (no scale)
        frame_landmarks[:, 0] = landmarks_crop[:, 0] + eb[0]
        frame_landmarks[:, 1] = landmarks_crop[:, 1] + eb[1]
    else:
        # Fallback (legacy FaceInfo without expanded_bbox): tight bbox + scale
        bbox = face_info.bbox
        crop_h, crop_w = crop_image.shape[:2]
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        if bbox_w > 0 and bbox_h > 0 and crop_w > 0 and crop_h > 0:
            frame_landmarks[:, 0] = landmarks_crop[:, 0] * bbox_w / crop_w + bbox[0]
            frame_landmarks[:, 1] = landmarks_crop[:, 1] * bbox_h / crop_h + bbox[1]

    return frame_landmarks

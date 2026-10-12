"""Pose estimator for Stage 6.

Encapsulates 6DRepNet inference logic per FaceSequence.
Does NOT load models — the model is injected by the Stage.

Processing:
    1. For each frame in a FaceSequence, select the main face.
    2. Load face crop from FaceInfo.face_crop_path.
    3. Run 6DRepNet inference → yaw, pitch, roll.
    4. Build PoseInfo and PoseSequence.
"""

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from face_pipeline_v2.data.models import FaceSequence, PoseInfo, PoseSequence


def estimate_poses(
    sequence: FaceSequence,
    model: Any,
    target_size: int = 256,
) -> PoseSequence:
    """Estimate head pose for every frame in a FaceSequence.

    Reads expanded raw face crops from Stage 5, applies resize
    only (expand already done at Stage 5 bbox level).

    Args:
        sequence: Input FaceSequence with face crops.
        model: Loaded 6DRepNet wrapper with predict(images) → List[dict].
        target_size: Target square size for model input (default 256).

    Returns:
        PoseSequence with per-frame PoseInfo.
    """
    # Determine main face strategy
    main_track_id: str | None = None
    if sequence.identity_result is not None:
        main_track_id = sequence.identity_result.main_track_id

    # Collect images and face metadata
    images: List[np.ndarray] = []
    valid_frames: List[tuple] = []  # (frame_index, face_id)

    for frame_info in sequence.frames:
        faces = frame_info.faces
        if not faces:
            valid_frames.append((frame_info.frame_index, "no_face", None))
            continue

        # Select target face
        target_face = None
        if main_track_id is not None:
            for face in faces:
                if face.track_id == main_track_id:
                    target_face = face
                    break
        if target_face is None:
            target_face = faces[0]

        # Load raw crop + apply preprocess (expand → resize)
        crop_path = target_face.face_crop_path
        if crop_path and Path(crop_path).exists():
            from face_pipeline_v2.utils.face_preprocess import resize_face
            try:
                raw = cv2.imread(str(crop_path))
                if raw is not None:
                    image = resize_face(raw, target_size)
                    images.append(image)
                    valid_frames.append((
                        frame_info.frame_index, target_face.face_id, raw.shape[:2],
                    ))
                    continue
            except Exception:
                pass

        # Fallback: no valid crop
        valid_frames.append((frame_info.frame_index, target_face.face_id, None))

    # --- Batch inference ---
    if images:
        # Use batched inference for GPU efficiency (falls back to single-image
        # if predict_batch is not available, e.g. with _NoOpPoseEstimator).
        if hasattr(model, "predict_batch"):
            pose_results = model.predict_batch(images)
        else:
            pose_results = model.predict(images)
    else:
        pose_results = []

    # --- Build PoseInfo list ---
    pose_idx = 0
    pose_infos: List[PoseInfo] = []

    for frame_idx, face_id, _ in valid_frames:
        if face_id == "no_face":
            # Frame with no face: skip (no pose)
            pose_infos.append(PoseInfo(
                frame_index=frame_idx, face_id="",
                yaw=0.0, pitch=0.0, roll=0.0, confidence=0.0,
            ))
        elif pose_idx < len(pose_results):
            result = pose_results[pose_idx]
            pose_infos.append(PoseInfo(
                frame_index=frame_idx, face_id=face_id,
                yaw=float(result.get("yaw", 0.0)),
                pitch=float(result.get("pitch", 0.0)),
                roll=float(result.get("roll", 0.0)),
                confidence=float(result.get("confidence", 0.0)),
            ))
            pose_idx += 1
        else:
            # No result available
            pose_infos.append(PoseInfo(
                frame_index=frame_idx, face_id=face_id,
                yaw=0.0, pitch=0.0, roll=0.0, confidence=0.0,
            ))

    return PoseSequence(
        clip_id=sequence.clip_id,
        poses=tuple(pose_infos),
    )

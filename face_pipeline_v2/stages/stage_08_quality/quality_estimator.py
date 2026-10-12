"""Quality estimator for Stage 8.

Encapsulates HyperIQA inference per FaceSequence.
Does NOT load models — the HyperIQA wrapper is injected by the Stage.

Processing:
    1. For each frame, select main face (by track_id or first face).
    2. Load face crop from FaceInfo.face_crop_path.
    3. Run HyperIQA → quality_score.
    4. Build FaceQualityInfo and FaceQualitySequence.
"""

from pathlib import Path
from typing import Any, List

import cv2
import numpy as np

from face_pipeline_v2.data.models import FaceSequence, FaceQualityInfo, FaceQualitySequence


def estimate_quality(
    sequence: FaceSequence,
    model: Any,
    target_size: int = 224,
) -> FaceQualitySequence:
    """Estimate face quality for every frame in a FaceSequence.

    Reads expanded raw face crops from Stage 5, applies resize
    only (expand already done at Stage 5 bbox level).

    Args:
        sequence: Input FaceSequence with face crops.
        model: Loaded HyperIQA wrapper with predict(images) → List[float].
        target_size: Target square size for model input (default 224).

    Returns:
        FaceQualitySequence with per-frame quality scores.
    """
    # Determine main face strategy
    main_track_id: str | None = None
    if sequence.identity_result is not None:
        main_track_id = sequence.identity_result.main_track_id

    images: List[np.ndarray] = []
    frame_meta: List[tuple] = []  # (frame_index, face_id, has_crop)

    for frame_info in sequence.frames:
        faces = frame_info.faces
        if not faces:
            frame_meta.append((frame_info.frame_index, "", False))
            continue

        target_face = None
        if main_track_id is not None:
            for face in faces:
                if face.track_id == main_track_id:
                    target_face = face
                    break
        if target_face is None:
            target_face = faces[0]

        crop_path = target_face.face_crop_path
        if crop_path and Path(crop_path).exists():
            from face_pipeline_v2.utils.face_preprocess import resize_face
            try:
                raw = cv2.imread(str(crop_path))
                if raw is not None:
                    image = resize_face(raw, target_size)
                    images.append(image)
                    frame_meta.append((
                        frame_info.frame_index, target_face.face_id, True
                    ))
                    continue
            except Exception:
                pass

        frame_meta.append((frame_info.frame_index, target_face.face_id, False))

    # --- Batch inference ---
    if images:
        scores = model.predict(images)
    else:
        scores = []

    # --- Build FaceQualityInfo list ---
    score_idx = 0
    qualities: List[FaceQualityInfo] = []

    for frame_idx, face_id, has_crop in frame_meta:
        if has_crop and score_idx < len(scores):
            score = scores[score_idx]
            score_idx += 1
            qualities.append(FaceQualityInfo(
                frame_index=frame_idx, face_id=face_id,
                quality_score=float(score), confidence=1.0,
            ))
        else:
            qualities.append(FaceQualityInfo(
                frame_index=frame_idx, face_id=face_id,
                quality_score=0.0, confidence=0.0,
            ))

    return FaceQualitySequence(
        clip_id=sequence.clip_id,
        qualities=tuple(qualities),
    )

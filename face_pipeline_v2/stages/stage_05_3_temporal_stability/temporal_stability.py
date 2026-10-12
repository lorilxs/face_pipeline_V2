"""Temporal stability analyzer for Stage 5.3.

Pure logic — no model, no video I/O.

Analyzes frame-to-frame face motion and scale changes within a clip.
Selects the main face (by main_track_id from Stage 5.2 identity_result,
or the only single face), then computes:

    1. Center motion: Euclidean distance of bbox centers between frames.
    2. Scale stability: Relative area change between frames.
    3. BBox jumps: Transitions where motion or scale exceeds thresholds.

Uses a unified anomaly ratio: bbox_jump_count / total_transitions.
"""

from typing import List, Set

import numpy as np
from numpy.typing import NDArray

from face_pipeline_v2.data.models import FaceSequence, TemporalStabilityResult


# Reject reason constants
REASON_CENTER_MOTION = "center_motion_abnormal"
REASON_SCALE = "scale_abnormal"


def _get_bboxes(sequence: FaceSequence) -> NDArray | None:
    """Extract main-face bboxes from a FaceSequence.

    Selects faces by main_track_id (if identity_result exists and
    main_track_id is set), otherwise picks the single face from each frame.

    Returns:
        (T, 4) array of bboxes or None if no valid face bboxes found.
    """
    main_track_id: str | None = None
    if sequence.identity_result is not None:
        main_track_id = sequence.identity_result.main_track_id

    bboxes: List[List[float]] = []
    for frame_info in sequence.frames:
        faces = frame_info.faces
        if len(faces) == 0:
            continue
        if main_track_id is not None:
            # Select face matching main_track_id
            matching = [f for f in faces if f.track_id == main_track_id]
            if matching:
                bboxes.append(list(matching[0].bbox))
        else:
            # No track_id: use the single face (or first face if multi)
            bboxes.append(list(faces[0].bbox))

    if len(bboxes) < 2:
        return None

    return np.array(bboxes, dtype=np.float64)


def analyze_stability(
    sequence: FaceSequence,
    max_center_motion: float = 15.0,
    max_scale_change_ratio: float = 0.25,
    max_stability_anomaly_ratio: float = 0.05,
) -> FaceSequence:
    """Analyze temporal stability of the main face in a FaceSequence.

    Args:
        sequence: Input FaceSequence.
        max_center_motion: Maximum allowed center displacement per frame (px).
        max_scale_change_ratio: Maximum allowed relative area change per frame.
        max_stability_anomaly_ratio: Maximum allowed anomaly transition ratio.

    Returns:
        New FaceSequence with stability_result attached.
    """
    total_frames = len(sequence.frames)
    if total_frames < 2:
        return FaceSequence(
            clip_id=sequence.clip_id,
            video_id=sequence.video_id,
            frames=sequence.frames,
            identity_result=sequence.identity_result,
            stability_result=TemporalStabilityResult(
                passed=True,
                average_center_motion=0.0,
                max_center_motion=0.0,
                average_scale_change=0.0,
                max_scale_change=0.0,
                bbox_jump_count=0,
                anomaly_ratio=0.0,
                reject_reasons=(),
            ),
        )

    bboxes = _get_bboxes(sequence)
    if bboxes is None or len(bboxes) < 2:
        return FaceSequence(
            clip_id=sequence.clip_id,
            video_id=sequence.video_id,
            frames=sequence.frames,
            identity_result=sequence.identity_result,
            stability_result=TemporalStabilityResult(
                passed=False,
                average_center_motion=0.0,
                max_center_motion=0.0,
                average_scale_change=0.0,
                max_scale_change=0.0,
                bbox_jump_count=0,
                anomaly_ratio=0.0,
                reject_reasons=(REASON_CENTER_MOTION,),
            ),
        )

    T = len(bboxes)

    # --- Compute per-transition metrics ---
    # Centers: (T, 2)
    centers = np.column_stack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2.0,
        (bboxes[:, 1] + bboxes[:, 3]) / 2.0,
    ])
    # Areas: (T,)
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    center_motions: List[float] = []
    scale_changes: List[float] = []
    bbox_jump_count = 0
    reasons: Set[str] = set()
    total_transitions = T - 1

    for t in range(total_transitions):
        # Center motion
        motion = float(np.linalg.norm(centers[t + 1] - centers[t]))
        center_motions.append(motion)

        # Scale change
        if areas[t] > 0:
            scale = float(abs(areas[t + 1] - areas[t]) / areas[t])
        else:
            scale = 0.0
        scale_changes.append(scale)

        # Anomaly check
        is_anomaly = False
        if motion > max_center_motion:
            reasons.add(REASON_CENTER_MOTION)
            is_anomaly = True
        if scale > max_scale_change_ratio:
            reasons.add(REASON_SCALE)
            is_anomaly = True
        if is_anomaly:
            bbox_jump_count += 1

    avg_center_motion = float(np.mean(center_motions)) if center_motions else 0.0
    max_cm = float(np.max(center_motions)) if center_motions else 0.0
    avg_scale = float(np.mean(scale_changes)) if scale_changes else 0.0
    max_sc = float(np.max(scale_changes)) if scale_changes else 0.0
    anomaly_ratio = bbox_jump_count / total_transitions
    passed = anomaly_ratio <= max_stability_anomaly_ratio

    return FaceSequence(
        clip_id=sequence.clip_id,
        video_id=sequence.video_id,
        frames=sequence.frames,
        identity_result=sequence.identity_result,
        stability_result=TemporalStabilityResult(
            passed=passed,
            average_center_motion=avg_center_motion,
            max_center_motion=max_cm,
            average_scale_change=avg_scale,
            max_scale_change=max_sc,
            bbox_jump_count=bbox_jump_count,
            anomaly_ratio=anomaly_ratio,
            reject_reasons=tuple(sorted(reasons)),
        ),
    )

"""Face tracker for Stage 5.1.

Encapsulates ByteTrack-based cross-frame face association.
Does NOT load models — the ByteTrack wrapper is injected by the Stage.

Uses tracker_adapter for SCRFD ↔ ByteTrack format conversion.

Per-frame processing:
    1. Convert FaceInfo → ByteTrack detection format via adapter.
    2. Call tracker.update(detections, img_shape) → track_ids.
    3. Rebuild FaceInfo with track_id via adapter.make_track_id().
    4. Return new FaceSequence (immutable).
"""

from typing import Any, List

import numpy as np

from face_pipeline_v2.data.models import FaceInfo, FaceSequence, FrameInfo
from face_pipeline_v2.stages.stage_05_1_face_tracking.tracker_adapter import (
    faces_to_detections,
    make_track_id,
)


def track_sequence(
    sequence: FaceSequence,
    tracker: Any,
    img_shape: tuple = (640, 360),
) -> FaceSequence:
    """Run ByteTrack across all frames in a FaceSequence.

    Args:
        sequence: Input FaceSequence with per-frame face detections.
        tracker: Loaded ByteTrack wrapper with update(bboxes, confidences, img_shape).
        img_shape: (height, width) of video frames.

    Returns:
        New FaceSequence with track_id populated on each FaceInfo.

    Raises:
        ValueError: If tracker is None.
    """
    if tracker is None:
        raise ValueError("ByteTrack tracker is not loaded.")

    new_frames: List[FrameInfo] = []

    for frame_info in sequence.frames:
        faces = frame_info.faces

        if not faces:
            # Empty frame: update tracker with empty dets (age tracks)
            tracker.update(
                bboxes=np.zeros((0, 4)),
                confidences=np.zeros((0,)),
                img_shape=img_shape,
            )
            new_frames.append(frame_info)
            continue

        # Convert FaceInfo → ByteTrack detections
        dets = faces_to_detections(list(faces))
        bboxes = dets[:, :4]
        confidences = dets[:, 4]

        # Call tracker
        track_ids = tracker.update(bboxes, confidences, img_shape)

        # Rebuild FaceInfo with track_id
        new_faces: List[FaceInfo] = []
        for idx, face in enumerate(faces):
            tid = track_ids[idx] if idx < len(track_ids) else -1
            new_face = FaceInfo(
                frame_index=face.frame_index,
                face_id=face.face_id,
                track_id=make_track_id(tid),
                bbox=face.bbox,
                landmark_5=face.landmark_5,
                confidence=face.confidence,
                face_crop_path=face.face_crop_path,
            )
            new_faces.append(new_face)

        # Rebuild FrameInfo with new faces
        new_frame = FrameInfo(
            frame_index=frame_info.frame_index,
            clip_id=frame_info.clip_id,
            timestamp=frame_info.timestamp,
            image_path=frame_info.image_path,
            faces=tuple(new_faces),
        )
        new_frames.append(new_frame)

    return FaceSequence(
        clip_id=sequence.clip_id,
        video_id=sequence.video_id,
        frames=tuple(new_frames),
    )

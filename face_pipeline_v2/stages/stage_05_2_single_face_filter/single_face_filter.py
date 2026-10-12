"""Single-face identity anomaly filter for Stage 5.2.

Pure logic — no model, no video I/O, no detection.

Unified approach: every frame that deviates from "single face belonging
to the main track" counts as an anomaly. A single threshold controls
rejection, eliminating the blind spot where three independent thresholds
each pass but the combination is invalid.

Anomaly types (per frame):
    - empty_face:     faces count == 0
    - multi_face:     faces count > 1
    - identity_switch: 1 face but track_id != main_track_id (or no track_id)

Algorithm:
    1. Determine main_track_id (most frequent track_id across all face frames).
    2. Per-frame: classify as normal or anomaly, accumulating reasons.
    3. anomaly_ratio = anomaly_frames / total_frames.
    4. reject if anomaly_ratio > max_identity_anomaly_ratio.
"""

from collections import Counter
from typing import List, Set

from face_pipeline_v2.data.models import FaceSequence, SingleFaceIdentityResult


# Anomaly reason constants
REASON_EMPTY_FACE = "empty_face"
REASON_MULTI_FACE = "multi_face"
REASON_IDENTITY_SWITCH = "identity_switch"

# ---------------------------------------------------------------------------
# Fragment-merging thresholds (ByteTrack fragmentation recovery).
#
# ByteTrack (Stage 5.1) re-initialises a track when a face is lost for more
# than ``track_buffer`` frames (head turns / occlusions in a talking-head
# clip), splitting ONE person into 2-3 sequential track_ids.  Without a merge
# pre-pass every later fragment is misclassified as an ``identity_switch``.
# Two fragments are merged when they are:
#   - temporally adjacent: next fragment starts <= FRAGMENT_MAX_GAP_FRAMES
#     frames after the previous one ends (strictly non-overlapping), and
#   - spatially compatible: IoU of the boundary boxes >= FRAGMENT_IOU_THRESHOLD
#     OR the centre distance is <= FRAGMENT_CENTER_RATIO * face size.
# "Balanced" tuning: recovers nearly all real fragmentation while keeping the
# false-merge risk for genuinely different people (usually at a different
# screen position) low.
# ---------------------------------------------------------------------------
FRAGMENT_MAX_GAP_FRAMES = 60
FRAGMENT_IOU_THRESHOLD = 0.15
FRAGMENT_CENTER_RATIO = 0.6


def _bbox_iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    ix1, iy1, ix2, iy2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (x4 - x3) * (y4 - y3)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(bbox):
    """Centre of a (x1, y1, x2, y2) box."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_size(bbox) -> float:
    """Longer side of a (x1, y1, x2, y2) box (face size proxy)."""
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _spatially_compatible(
    prev_bbox, next_bbox, iou_threshold: float, center_ratio: float
) -> bool:
    """True if two temporally-adjacent track fragments are plausibly one person."""
    if _bbox_iou(prev_bbox, next_bbox) >= iou_threshold:
        return True
    (cx1, cy1), (cx2, cy2) = _bbox_center(prev_bbox), _bbox_center(next_bbox)
    size = max(_bbox_size(prev_bbox), _bbox_size(next_bbox))
    if size <= 0:
        return False
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    return dist <= center_ratio * size


def _track_fragment_rename(frames, gap_frames: int, iou_threshold: float,
                           center_ratio: float) -> dict:
    """Return a {child_track_id: parent_track_id} map merging track fragments.

    Tracks are processed in temporal order of first appearance.  Track B merges
    into the immediately-preceding group A when A ends before B begins, the
    frame gap is <= ``gap_frames`` and the boundary boxes are spatially
    compatible.  Merging updates A's end so chains (fragment1 -> fragment2 ->
    fragment3) collapse into a single identity.
    """
    per_track: dict = {}
    for frame_info in frames:
        for face in frame_info.faces:
            if face.track_id is None:
                continue
            per_track.setdefault(face.track_id, []).append(
                (frame_info.frame_index, tuple(face.bbox))
            )
    for recs in per_track.values():
        recs.sort(key=lambda x: x[0])

    rename: dict = {}
    groups: list = []  # [{'id', 'first_frame', 'last_frame', 'last_bbox', ...}]
    for tid in sorted(per_track, key=lambda t: per_track[t][0][0]):
        recs = per_track[tid]
        first_frame, first_bbox = recs[0]
        last_frame, last_bbox = recs[-1]
        merged = False
        for g in reversed(groups):
            if g["last_frame"] >= first_frame:
                continue  # overlapping -> different identities
            gap = first_frame - g["last_frame"]
            if gap > gap_frames:
                break  # earlier groups are even further back
            if _spatially_compatible(
                g["last_bbox"], first_bbox, iou_threshold, center_ratio
            ):
                rename[tid] = g["id"]
                g["last_frame"] = last_frame
                g["last_bbox"] = last_bbox
                merged = True
                break
        if not merged:
            groups.append({
                "id": tid,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "last_bbox": last_bbox,
            })
    return rename



def filter_sequence(
    sequence: FaceSequence,
    max_identity_anomaly_ratio: float = 0.005,
) -> FaceSequence:
    """Evaluate a FaceSequence for single-face identity anomalies.

    Args:
        sequence: Input FaceSequence from Stage 5 / 5.1.
        max_identity_anomaly_ratio: Maximum allowed anomaly frame ratio
            (default 0.005 → 1 frame in 200).

    Returns:
        New FaceSequence with identity_result attached.
    """
    total_frames = len(sequence.frames)

    if total_frames == 0:
        return FaceSequence(
            clip_id=sequence.clip_id,
            video_id=sequence.video_id,
            frames=sequence.frames,
            identity_result=SingleFaceIdentityResult(
                passed=False,
                main_track_id=None,
                total_frames=0,
                anomaly_frames=0,
                anomaly_ratio=0.0,
                reject_reasons=(REASON_EMPTY_FACE,),
            ),
        )

    # --- Pass 0: merge ByteTrack fragments (same-person track splits) ---
    # A single talking-head subject is often split into 2-3 sequential
    # track_ids by ByteTrack re-initialisation.  Merge temporally-adjacent,
    # spatially-compatible fragments so the later ones are NOT miscounted as
    # identity switches; a genuine identity switch is a different person at a
    # different position and stays split.
    rename = _track_fragment_rename(
        sequence.frames,
        gap_frames=FRAGMENT_MAX_GAP_FRAMES,
        iou_threshold=FRAGMENT_IOU_THRESHOLD,
        center_ratio=FRAGMENT_CENTER_RATIO,
    )

    def _eff_tid(tid):
        return rename.get(tid, tid) if tid is not None else None

    # --- Pass 1: determine main_track_id ---
    # Count track_ids across all face frames (single + multi)
    track_counter: Counter[str] = Counter()
    for frame_info in sequence.frames:
        for face in frame_info.faces:
            tid = _eff_tid(face.track_id)
            if tid is not None:
                track_counter[tid] += 1

    main_track_id: str | None
    if track_counter:
        main_track_id = track_counter.most_common(1)[0][0]
    else:
        # No track_ids at all (Stage 5.1 not run): every frame with a face
        # is an anomaly (can't verify identity), every empty frame is too
        main_track_id = None

    # --- Pass 2: classify each frame ---
    anomaly_frames = 0
    reasons: Set[str] = set()

    for frame_info in sequence.frames:
        face_count = len(frame_info.faces)

        if face_count == 0:
            # Empty frame → anomaly
            anomaly_frames += 1
            reasons.add(REASON_EMPTY_FACE)

        elif face_count > 1:
            # Multi-face frame → anomaly
            anomaly_frames += 1
            reasons.add(REASON_MULTI_FACE)

        elif main_track_id is None:
            # Single face but no track_id (Stage 5.1 not run) → anomaly
            anomaly_frames += 1
            reasons.add(REASON_IDENTITY_SWITCH)

        else:
            # Single face: check if it belongs to main track
            face = frame_info.faces[0]
            if _eff_tid(face.track_id) != main_track_id:
                anomaly_frames += 1
                reasons.add(REASON_IDENTITY_SWITCH)
            # else: normal frame (single face, correct identity)

    anomaly_ratio = anomaly_frames / total_frames
    passed = anomaly_ratio <= max_identity_anomaly_ratio

    return FaceSequence(
        clip_id=sequence.clip_id,
        video_id=sequence.video_id,
        frames=sequence.frames,
        identity_result=SingleFaceIdentityResult(
            passed=passed,
            main_track_id=main_track_id,
            total_frames=total_frames,
            anomaly_frames=anomaly_frames,
            anomaly_ratio=anomaly_ratio,
            reject_reasons=tuple(sorted(reasons)),
        ),
    )

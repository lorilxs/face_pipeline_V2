"""ByteTrack multi-object tracker wrapper.

Per AISDD 3.9.1, ByteTrack is used in Stage 5.1 for cross-frame face
identity association in open-domain (multi-person) videos.

This wrapper encapsulates the third_party ByteTrack implementation.
Stages MUST NOT import BYTETracker or STrack directly.

Loading order:
    1. third_party/ByteTrack (primary — real Kalman + IoU tracker).
    2. Built-in simple IoU tracker (fallback, zero dependencies).

Interface:
    wrapper = ByteTrackWrapper(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    wrapper.load()
    track_ids = wrapper.update(bboxes, confidences, img_shape)
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import numpy as np

# Ensure third_party ByteTrack is importable
_THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party" / "ByteTrack"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))


class ByteTrackWrapper:
    """ByteTrack face tracker wrapper.

    Encapsulates the real ByteTrack BYTETracker from third_party.
    Provides a minimal update() → track_ids interface for Stage 5.1.
    """

    DEFAULT_TRACK_THRESH: float = 0.5
    DEFAULT_TRACK_BUFFER: int = 30
    DEFAULT_MATCH_THRESH: float = 0.8
    DEFAULT_DET_THRESH: float | None = None

    def __init__(
        self,
        track_thresh: float = DEFAULT_TRACK_THRESH,
        track_buffer: int = DEFAULT_TRACK_BUFFER,
        match_thresh: float = DEFAULT_MATCH_THRESH,
        det_thresh: float | None = None,
    ) -> None:
        """Initialize ByteTrack wrapper.

        Args:
            track_thresh: Detection confidence threshold for the high/low
                association split.  Detections above it are "high-score" and
                may also initialise new tracks.
            track_buffer: Frames to keep lost tracks alive.
            match_thresh: IoU threshold for detection-track matching.
            det_thresh: Score gate for initialising a NEW track.  Defaults to
                None -> upstream ByteTrack heuristic (track_thresh + 0.1).  The
                pipeline aligns it with the shared Stage 3/5 detection floor so
                every face Stage 5 detected can start a track.
        """
        self._track_thresh: float = track_thresh
        self._track_buffer: int = track_buffer
        self._match_thresh: float = match_thresh
        self._det_thresh: float | None = det_thresh
        self._tracker: Any = None
        self._use_library: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def track_thresh(self) -> float:
        return self._track_thresh

    @property
    def track_buffer(self) -> int:
        return self._track_buffer

    @property
    def det_thresh(self) -> float | None:
        return self._det_thresh

    @property
    def is_loaded(self) -> bool:
        return self._tracker is not None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialize the ByteTrack tracker.

        Path 1: third_party/ByteTrack (primary).
        Path 2: Built-in simple IoU tracker (fallback).
        """
        # --- Path 1: third_party ByteTrack ---
        try:
            from tracker.byte_tracker import BYTETracker

            args = SimpleNamespace(
                track_thresh=self._track_thresh,
                track_buffer=self._track_buffer,
                match_thresh=self._match_thresh,
                mot20=False,
            )
            if self._det_thresh is not None:
                args.det_thresh = self._det_thresh
            self._tracker = BYTETracker(args, frame_rate=30)
            self._use_library = True
            return
        except ImportError:
            pass

        # --- Path 2: Built-in simple tracker ---
        self._tracker = _SimpleTracker(max_age=self._track_buffer)
        self._use_library = False

    def reset(self) -> None:
        """Reset the tracker to a pristine per-clip state.

        Rebuilds the internal tracker AND resets the third_party track-ID
        counter, so track IDs restart from 1 and no lost-track / Kalman
        state carries over from a previously processed FaceSequence.  Each
        clip must be tracked independently, not as one continuous stream.
        """
        # Rebuild instance state (frame_id, tracked/lost/removed stracks).
        self.load()
        # ByteTrack's STrack.next_id() draws from the CLASS-level counter
        # BaseTrack._count; a fresh BYTETracker instance alone keeps growing
        # IDs across clips.  Reset it here for a true per-clip restart.
        if self._use_library:
            from tracker.basetrack import BaseTrack

            BaseTrack._count = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        bboxes: np.ndarray,
        confidences: np.ndarray,
        img_shape: tuple = (640, 360),
    ) -> List[int]:
        """Update tracker and return track IDs.

        Args:
            bboxes: (N, 4) [x1, y1, x2, y2].
            confidences: (N,) scores.
            img_shape: (height, width) of the frame.

        Returns:
            List of track IDs, one per detection.
        """
        if self._tracker is None:
            raise RuntimeError("ByteTrack not loaded. Call load() first.")

        if len(bboxes) == 0:
            if self._use_library:
                dets = np.zeros((0, 5))
                self._tracker.update(dets, img_shape, img_shape)
            else:
                self._tracker.update(np.zeros((0, 5)))
            return []

        if self._use_library:
            return self._update_bytetrack(bboxes, confidences, img_shape)
        else:
            return self._update_simple(bboxes, confidences)

    def _update_bytetrack(
        self,
        bboxes: np.ndarray,
        confidences: np.ndarray,
        img_shape: tuple,
    ) -> List[int]:
        """Update using third_party BYTETracker."""
        # Build [x1, y1, x2, y2, score] format
        dets = np.column_stack([bboxes, confidences.reshape(-1, 1)])

        self._tracker.update(dets, img_shape, img_shape)
        # Use ByteTrack's own per-detection association recorded inside
        # update(); never do a post-hoc greedy IoU re-match. The re-match can
        # flip a single face between two live tracks of the same ID across
        # frames, which Stage 5.2 then mislabels as identity_switch.
        ids = getattr(self._tracker, "last_det_track_ids", None)
        if ids is None or len(ids) != len(bboxes):
            # Defensive fallback for a tracker build that does not populate
            # last_det_track_ids (e.g. a stub/mock in isolated tests).
            return _match_dets_to_tracks(
                bboxes, list(self._tracker.tracked_stracks)
            )
        return [int(t) for t in ids]

    def _update_simple(
        self, bboxes: np.ndarray, confidences: np.ndarray
    ) -> List[int]:
        """Update using built-in simple tracker."""
        dets = np.column_stack([bboxes, confidences.reshape(-1, 1)])
        return self._tracker.update(dets)


def _match_dets_to_tracks(
    bboxes: np.ndarray, tracks: List[Any]
) -> List[int]:
    """Match detection bboxes to active STrack objects by IoU.

    Each track has .tlwh (top-left x, y, width, height) and .track_id.

    Args:
        bboxes: (N, 4) [x1, y1, x2, y2].
        tracks: List of STrack objects.

    Returns:
        List of track_ids, length N. -1 if unmatched.
    """
    if not tracks:
        return [-1] * len(bboxes)

    # Build track bboxes in xyxy format
    track_boxes = np.array([_tlwh_to_xyxy(t.tlwh) for t in tracks])
    track_ids = [int(t.track_id) for t in tracks]

    result = []
    for det_box in bboxes:
        best_iou = 0.3  # minimum IoU to consider a match
        best_idx = -1
        for j, trk_box in enumerate(track_boxes):
            iou = _iou(det_box, trk_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            result.append(track_ids[best_idx])
        else:
            result.append(-1)

    return result


def _tlwh_to_xyxy(tlwh) -> np.ndarray:
    """Convert [top, left, width, height] → [x1, y1, x2, y2]."""
    x1, y1, w, h = tlwh
    return np.array([x1, y1, x1 + w, y1 + h])


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Fallback: simple IoU-based tracker (zero dependencies)
# ---------------------------------------------------------------------------


class _SimpleTracker:
    """Minimal IoU-based multi-object tracker (fallback)."""

    def __init__(self, max_age: int = 30) -> None:
        self._max_age: int = max_age
        self._next_id: int = 0
        self._tracks: List[dict] = []

    def update(self, dets: np.ndarray) -> List[int]:
        if len(dets) == 0:
            for t in self._tracks:
                t["age"] += 1
            self._tracks = [t for t in self._tracks if t["age"] <= self._max_age]
            return []

        track_ids = [-1] * len(dets)

        if not self._tracks:
            for i in range(len(dets)):
                tid = self._next_id
                self._next_id += 1
                self._tracks.append({"id": tid, "bbox": dets[i, :4].copy(), "age": 0})
                track_ids[i] = tid
            return track_ids

        active = [t for t in self._tracks if t["age"] == 0]
        if active:
            iou_mat = np.zeros((len(dets), len(active)))
            for i, det in enumerate(dets):
                for j, trk in enumerate(active):
                    iou_mat[i, j] = _iou(det[:4], trk["bbox"])

            matched = set()
            for i in range(len(dets)):
                if not active:
                    break
                best_j = int(np.argmax(iou_mat[i]))
                if iou_mat[i, best_j] > 0.3 and best_j not in matched:
                    track_ids[i] = active[best_j]["id"]
                    active[best_j]["bbox"] = dets[i, :4].copy()
                    active[best_j]["age"] = 0
                    matched.add(best_j)

        matched_ids = {t for t in track_ids if t >= 0}
        for t in self._tracks:
            if t["age"] == 0 and t["id"] not in matched_ids:
                t["age"] = 1

        for i in range(len(dets)):
            if track_ids[i] == -1:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append({"id": tid, "bbox": dets[i, :4].copy(), "age": 0})
                track_ids[i] = tid

        for t in self._tracks:
            if t["id"] not in {tr for tr in track_ids if tr >= 0}:
                t["age"] += 1
        self._tracks = [t for t in self._tracks if t["age"] <= self._max_age]

        return track_ids

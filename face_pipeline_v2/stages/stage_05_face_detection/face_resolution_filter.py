"""Configurable Stage 5 face-resolution pre-filter.

Runs AFTER Stage 5 detection/crop output and BEFORE Stage 5.1-Stage 9 as a
video-level hard gate.  It only ever inspects the SCRFD *tight* bbox retained
on each FaceInfo (``FaceInfo.bbox``) - never the expanded crop bbox and never
any resized / Stage 11 / EMICA size - so it is a coarse pre-filter: Stage 11's
eye-distance affine alignment is the authoritative canonicalization.  The gate
is driven by the tight-bbox short edge ``min(w, h)`` (see
``FaceResolutionFilterConfig``), a metric validated on 6 real videos.

Pure logic - no model, no I/O.  Kept isolated so it is independently
unit-testable while Stage 5's detection/crop/disk-layout behavior is unchanged.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from face_pipeline_v2.data.models import FaceInfo

# --- Per-frame final statuses ------------------------------------------------

STATUS_PASS: str = "PASS"
"""帧已检测到目标人脸，tight bbox 合法，裁剪成功，短边达到阈值。"""

STATUS_FAIL_LOW_RESOLUTION: str = "FAIL_LOW_RESOLUTION"
"""检测到目标人脸且 bbox/裁剪正常，但短边低于配置阈值。"""

STATUS_FAIL_NO_FACE: str = "FAIL_NO_FACE"
"""该帧未检测到任何目标人脸。"""

STATUS_FAIL_INVALID_BBOX: str = "FAIL_INVALID_BBOX"
"""目标人脸 tight bbox 缺失或非法（退化/非有限值）。"""

STATUS_FAIL_CROP: str = "FAIL_CROP"
"""检测成功但裁剪（扩展/落盘）失败。"""

STATUS_FAIL_ERROR: str = "FAIL_ERROR"
"""帧读取、解码、处理异常，或预期帧未被处理。"""

ALL_STATUSES: Tuple[str, ...] = (
    STATUS_PASS,
    STATUS_FAIL_LOW_RESOLUTION,
    STATUS_FAIL_NO_FACE,
    STATUS_FAIL_INVALID_BBOX,
    STATUS_FAIL_CROP,
    STATUS_FAIL_ERROR,
)


# --- Video-level reject reasons ----------------------------------------------

REJECT_NO_VALID_FACE_CROP: str = "no_valid_face_crop"
REJECT_INSUFFICIENT_VALID_FACE_FRAMES: str = "insufficient_valid_face_frames"
REJECT_FACE_RESOLUTION_RATIO_BELOW: str = "face_resolution_ratio_below_threshold"
REJECT_PROCESSING_ERROR: str = "frame_processing_error"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceResolutionFilterConfig:
    """Configuration for the Stage 5 face-resolution pre-filter.

    ``min_face_short_side`` is the acceptance threshold on the SCRFD *tight*
    bbox short edge ``min(w, h)``, measured in pixels on the source frame (never
    on any expanded/resized crop bbox).  The default of 75 was validated on 6
    real talking-face videos: it rejects the 640x360 small-face sample (short
    edge 55..71) while accepting every other sample (short edge >= 76).  It is
    NOT an official EMICA requirement.
    """

    enabled: bool = False
    min_face_short_side: int = 75
    min_face_resolution_ratio: float = 0.90
    min_valid_face_frames: int = 5

    @classmethod
    def from_dict(
        cls, cfg: Optional[Mapping[str, Any]]
    ) -> "FaceResolutionFilterConfig":
        """Build the config from a ``stage_05_face_resolution_filter`` dict."""
        if not cfg:
            return cls()
        short_side = cfg.get("min_face_short_side")
        if short_side is None:
            # Backward compat: derive the short edge from the legacy
            # width/height double-threshold if the new key is absent.
            legacy_w = cfg.get("min_face_width")
            legacy_h = cfg.get("min_face_height")
            if legacy_w is not None and legacy_h is not None:
                short_side = min(int(legacy_w), int(legacy_h))
        return cls(
            enabled=bool(cfg.get("enabled", cls.enabled)),
            min_face_short_side=(
                int(short_side)
                if short_side is not None
                else cls.min_face_short_side
            ),
            min_face_resolution_ratio=float(
                cfg.get(
                    "min_face_resolution_ratio", cls.min_face_resolution_ratio
                )
            ),
            min_valid_face_frames=int(
                cfg.get("min_valid_face_frames", cls.min_valid_face_frames)
            ),
        )


# ---------------------------------------------------------------------------
# Tight bbox helpers
# ---------------------------------------------------------------------------


def tight_bbox_is_valid(bbox: Any) -> bool:
    """Return True if the SCRFD tight bbox is usable.

    Invalid when not a 4-tuple/list of numbers, contains non-finite values, or
    is degenerate (``x2 <= x1`` or ``y2 <= y1``).  Out-of-frame edge
    coordinates are allowed (SCRFD may return slightly out-of-bounds boxes and
    the crop step clamps them).
    """
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return False
    try:
        floats = tuple(float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in floats):
        return False
    x1, y1, x2, y2 = floats
    return x2 > x1 and y2 > y1


def tight_bbox_size(bbox: Sequence[float]) -> Tuple[int, int]:
    """Return ``(face_width, face_height)`` from a tight bbox.

    Exactly ``int(x2) - int(x1)`` / ``int(y2) - int(y1)`` per the filter spec.
    """
    x1, y1, x2, y2 = bbox
    return (int(x2) - int(x1), int(y2) - int(y1))


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameResolutionOutcome:
    """Final per-frame status produced by the filter."""

    status: str
    face_width: int = 0
    face_height: int = 0
    face_short_side: int = 0


@dataclass(frozen=True)
class VideoResolutionResult:
    """Video-level PASS/FAIL decision plus per-status counts.

    ``face_short_side_min/median/max`` are the tight-bbox short-edge
    distribution over valid-face frames (PASS + FAIL_LOW_RESOLUTION), so the
    gate decision stays interpretable after intermediate outputs are deleted.
    """

    passed: bool
    reject_reason: Optional[str]
    expected_frame_count: int
    pass_frame_count: int
    valid_face_frame_count: int
    status_counts: Dict[str, int]
    qualified_ratio: float
    face_short_side_min: Optional[int] = None
    face_short_side_median: Optional[float] = None
    face_short_side_max: Optional[int] = None


# ---------------------------------------------------------------------------
# Per-frame classification
# ---------------------------------------------------------------------------


def classify_detected_faces(
    faces: Sequence[FaceInfo],
    cfg: FaceResolutionFilterConfig,
) -> FrameResolutionOutcome:
    """Classify one successfully-processed frame.

    Multi-face frames follow the existing Stage 5 target-face convention: the
    first face in the list (largest bbox area, as sorted by Stage 5).  Only
    that face's tight bbox is evaluated.
    """
    if not faces:
        return FrameResolutionOutcome(status=STATUS_FAIL_NO_FACE)

    target: FaceInfo = faces[0]
    if not tight_bbox_is_valid(target.bbox):
        return FrameResolutionOutcome(status=STATUS_FAIL_INVALID_BBOX)

    w, h = tight_bbox_size(target.bbox)
    short_side = min(w, h)
    if short_side >= cfg.min_face_short_side:
        return FrameResolutionOutcome(
            status=STATUS_PASS,
            face_width=w,
            face_height=h,
            face_short_side=short_side,
        )
    return FrameResolutionOutcome(
        status=STATUS_FAIL_LOW_RESOLUTION,
        face_width=w,
        face_height=h,
        face_short_side=short_side,
    )


# ---------------------------------------------------------------------------
# Video-level evaluation
# ---------------------------------------------------------------------------


def _short_side_stats(
    frame_outcomes: Mapping[int, FrameResolutionOutcome],
) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    """Min / median / max of the tight-bbox short edge over valid-face frames.

    Only PASS / FAIL_LOW_RESOLUTION frames carry a meaningful
    ``face_short_side`` (a target face was detected with a valid tight bbox);
    every other status is excluded so failed frames never pollute the range.
    """
    sides = [
        int(o.face_short_side)
        for o in frame_outcomes.values()
        if o.status in (STATUS_PASS, STATUS_FAIL_LOW_RESOLUTION)
        and o.face_short_side > 0
    ]
    if not sides:
        return (None, None, None)
    return (min(sides), statistics.median(sides), max(sides))


def evaluate_video(
    frame_outcomes: Mapping[int, FrameResolutionOutcome],
    expected_frames: Sequence[int],
    cfg: FaceResolutionFilterConfig,
) -> VideoResolutionResult:
    """Aggregate per-frame outcomes into a video-level decision.

    Denominator is the set of expected frames (each counted once).  A frame
    that is expected but has no recorded outcome counts as a processing error
    and is never silently ignored.  The pass conditions are:

        1. every expected frame is fully accounted (no error frames);
        2. valid target-face frames >= ``min_valid_face_frames``;
        3. ``pass / expected >= min_face_resolution_ratio``.
    """
    expected = sorted(set(expected_frames))
    expected_count = len(expected)

    counts: Dict[str, int] = {s: 0 for s in ALL_STATUSES}

    for idx in expected:
        outcome = frame_outcomes.get(idx)
        if outcome is None:
            counts[STATUS_FAIL_ERROR] += 1
            continue
        if outcome.status in counts:
            counts[outcome.status] += 1
        else:
            # Unknown/corrupt status is treated conservatively as an error.
            counts[STATUS_FAIL_ERROR] += 1

    pass_count = counts[STATUS_PASS]
    # A "valid target face" frame = face found + tight bbox valid + crop ok,
    # i.e. PASS or FAIL_LOW_RESOLUTION.
    valid_face_count = pass_count + counts[STATUS_FAIL_LOW_RESOLUTION]
    error_count = counts[STATUS_FAIL_ERROR]

    qualified_ratio = (pass_count / expected_count) if expected_count > 0 else 0.0

    if error_count > 0:
        passed, reason = False, REJECT_PROCESSING_ERROR
    elif valid_face_count == 0:
        passed, reason = False, REJECT_NO_VALID_FACE_CROP
    elif valid_face_count < cfg.min_valid_face_frames:
        passed, reason = False, REJECT_INSUFFICIENT_VALID_FACE_FRAMES
    elif qualified_ratio < cfg.min_face_resolution_ratio:
        passed, reason = False, REJECT_FACE_RESOLUTION_RATIO_BELOW
    else:
        passed, reason = True, None

    short_min, short_median, short_max = _short_side_stats(frame_outcomes)

    return VideoResolutionResult(
        passed=passed,
        reject_reason=reason,
        expected_frame_count=expected_count,
        pass_frame_count=pass_count,
        valid_face_frame_count=valid_face_count,
        status_counts=counts,
        qualified_ratio=qualified_ratio,
        face_short_side_min=short_min,
        face_short_side_median=short_median,
        face_short_side_max=short_max,
    )
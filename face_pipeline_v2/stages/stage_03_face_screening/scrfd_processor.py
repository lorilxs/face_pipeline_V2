"""SCRFD coarse face screening processor for Stage 3.

Per AISDD 3.6, this processor:
    - Samples frames at 1 FPS from a Scene.
    - Runs SCRFD detection on each sampled frame.
    - Calculates per-scene CoarseFaceInfo metadata.
    - Generates SceneFilterResult based on configurable thresholds.

This is a pure processor — it does NOT load models or read config files.
The model is injected by the Stage, and config is passed as parameters.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from face_pipeline_v2.data.models import CoarseFaceInfo, SceneFilterResult
from face_pipeline_v2.utils.exceptions import DataProcessingError


# Possible reject_reasons (not hard-coded in filtering logic; used as constants)
REJECT_TOO_SHORT = "too_short"
REJECT_NO_FACE = "no_face"
REJECT_MULTI_FACE = "multi_face"
REJECT_LOW_CONFIDENCE = "low_confidence"


# Type alias for a single SCRFD detection result
Detection = Dict[str, Any]
# Detection shape: {"bbox": (x1,y1,x2,y2), "confidence": float, "landmarks": tuple | None}


class SCRFDProcessor:
    """Coarse face screening processor.

    Processes a single Scene's sampled frames through SCRFD and
    produces CoarseFaceInfo + SceneFilterResult metadata.

    Usage:
        processor = SCRFDProcessor(
            scrfd_model=model_manager.get_model("scrfd"),
            video_reader=video_reader_callable,
            video_fps=25,
        )
        coarse_info, filter_result = processor.process_scene(scene_info, config)
    """

    def __init__(
        self,
        scrfd_model: Any,
        video_reader: Any,
        video_fps: float,
    ) -> None:
        """Initialize the SCRFD processor.

        Args:
            scrfd_model: Loaded SCRFD model with a detect(image) -> List[dict] method.
            video_reader: Callable that returns a frame (np.ndarray) for a given
                          frame_index. Signature: reader(frame_index: int) -> np.ndarray.
            video_fps: Frames per second of the standardized video (e.g. 25).
        """
        self._scrfd: Any = scrfd_model
        self._video_reader: Any = video_reader
        self._video_fps: float = float(video_fps)
        # Per-frame summaries from the most recent process_scene() call.
        # Exposed for diagnostics so Stage 3 can persist per-frame detection
        # stats without changing the (coarse_info, filter_result) return shape.
        self.frame_summaries: List[Dict[str, Any]] = []

    @property
    def video_fps(self) -> float:
        """Return the video FPS used for frame sampling."""
        return self._video_fps

    def process_scene(
        self,
        scene_start_frame: int,
        scene_end_frame: int,
        sampling_fps: int,
        min_duration: float,
        scene_duration: float,
        min_face_detect_ratio: float,
        min_single_face_ratio: float,
        min_face_confidence: float,
    ) -> Tuple[CoarseFaceInfo, SceneFilterResult]:
        """Process a single Scene: sample frames, run SCRFD, generate metadata.

        Args:
            scene_start_frame: First frame index of the scene (inclusive).
            scene_end_frame: Last frame index of the scene (exclusive).
            sampling_fps: Target sampling rate in FPS (e.g. 1).
            min_duration: Minimum scene duration in seconds.
            scene_duration: Actual scene duration in seconds.
            min_face_detect_ratio: Min ratio of frames with face detection.
            min_single_face_ratio: Min ratio of single-face frames.
            min_face_confidence: Min average face confidence.

        Returns:
            Tuple of (CoarseFaceInfo, SceneFilterResult).
        Raises:
            DataProcessingError: If frame reading or detection fails.
        """
        self.frame_summaries = []

        # --- Duration check ---
        if scene_duration < min_duration:
            coarse_info = CoarseFaceInfo(
                sampled_frame_count=0,
                face_detect_ratio=0.0,
                single_face_ratio=0.0,
                avg_face_confidence=0.0,
                max_face_count=0,
            )
            filter_result = SceneFilterResult(
                passed=False,
                reject_reasons=(REJECT_TOO_SHORT,),
            )
            return coarse_info, filter_result

        # --- Frame sampling ---
        step = max(1, int(self._video_fps / sampling_fps))
        frame_indices = list(range(scene_start_frame, scene_end_frame, step))

        if not frame_indices:
            coarse_info = CoarseFaceInfo(
                sampled_frame_count=0,
                face_detect_ratio=0.0,
                single_face_ratio=0.0,
                avg_face_confidence=0.0,
                max_face_count=0,
            )
            filter_result = SceneFilterResult(
                passed=False,
                reject_reasons=(REJECT_TOO_SHORT,),
            )
            return coarse_info, filter_result

        # --- SCRFD detection ---
        all_detections: List[List[Detection]] = []
        for frame_idx in frame_indices:
            try:
                frame = self._video_reader(frame_idx)
                detections = self._scrfd.detect(frame)
                all_detections.append(detections)
                confs = [
                    float(d.get("confidence", 0.0)) for d in detections
                ]
                self.frame_summaries.append(
                    {
                        "frame_index": frame_idx,
                        "face_count": len(detections),
                        "top_confidence": (
                            round(max(confs), 6) if confs else None
                        ),
                    }
                )
            except Exception as exc:
                raise DataProcessingError(
                    f"SCRFD detection failed at frame {frame_idx}: {exc}"
                ) from exc

        # --- Calculate metadata ---
        sampled_frame_count = len(frame_indices)
        frames_with_face = sum(
            1 for dets in all_detections if len(dets) > 0
        )
        frames_with_single_face = sum(
            1 for dets in all_detections if len(dets) == 1
        )
        all_confidences: List[float] = []
        max_face_count = 0
        for dets in all_detections:
            face_count = len(dets)
            if face_count > max_face_count:
                max_face_count = face_count
            for d in dets:
                conf = d.get("confidence", 0.0)
                all_confidences.append(float(conf))

        face_detect_ratio = (
            frames_with_face / sampled_frame_count
            if sampled_frame_count > 0
            else 0.0
        )
        single_face_ratio = (
            frames_with_single_face / frames_with_face
            if frames_with_face > 0
            else 0.0
        )
        avg_face_confidence = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else 0.0
        )

        coarse_info = CoarseFaceInfo(
            sampled_frame_count=sampled_frame_count,
            face_detect_ratio=face_detect_ratio,
            single_face_ratio=single_face_ratio,
            avg_face_confidence=avg_face_confidence,
            max_face_count=max_face_count,
        )

        # --- Filtering decision ---
        reject_reasons: List[str] = []

        if face_detect_ratio == 0.0:
            reject_reasons.append(REJECT_NO_FACE)
        else:
            if face_detect_ratio < min_face_detect_ratio:
                reject_reasons.append(REJECT_NO_FACE)
            if single_face_ratio < min_single_face_ratio:
                reject_reasons.append(REJECT_MULTI_FACE)
            if avg_face_confidence < min_face_confidence:
                reject_reasons.append(REJECT_LOW_CONFIDENCE)

        passed = len(reject_reasons) == 0
        filter_result = SceneFilterResult(
            passed=passed,
            reject_reasons=tuple(reject_reasons),
        )

        return coarse_info, filter_result

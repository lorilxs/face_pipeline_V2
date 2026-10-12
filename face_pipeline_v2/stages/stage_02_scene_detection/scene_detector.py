"""PySceneDetect wrapper for video scene segmentation.

Per AISDD 3.5, Stage 2 uses PySceneDetect (ContentDetector) to:
    - Detect shot transitions in standardized video.
    - Generate scene boundaries (start/end frames + timestamps).
    - Return a list of scene metadata dicts.

Uses PySceneDetect Python API — never CLI subprocess.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SceneDetector:
    """PySceneDetect封装 — detects scene boundaries via ContentDetector.

    Usage:
        detector = SceneDetector(threshold=27.0, min_scene_len=15)
        scenes = detector.detect(Path("standardized.mp4"))
        # Returns list of dicts with start_frame, end_frame, etc.
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
        adaptive_factor: Optional[float] = None,
    ) -> None:
        """Initialize SceneDetector with ContentDetector parameters.

        Args:
            threshold: ContentDetector threshold (higher = fewer cuts). When
                ``adaptive_factor`` is set, this acts as the lower-bound floor.
            min_scene_len: Minimum scene length in frames.
            adaptive_factor: If set, the effective threshold becomes
                ``max(threshold, adaptive_factor * p99_content_delta)`` where
                ``p99_content_delta`` is the clip's own 99th-percentile
                consecutive-frame luma delta (its motion-noise ceiling). This
                suppresses speaker-motion over-splitting in high-motion clips
                while preserving subtle transitions in low-motion clips.
        """
        self._threshold: float = threshold
        self._min_scene_len: int = min_scene_len
        self._adaptive_factor: Optional[float] = adaptive_factor
        # Diagnostics from the most recent detect() run (for logging only).
        self._last_noise_ceiling: Optional[float] = None
        self._last_effective_threshold: float = threshold

    @property
    def threshold(self) -> float:
        """ContentDetector threshold value."""
        return self._threshold

    @property
    def min_scene_len(self) -> int:
        """Minimum scene length in frames."""
        return self._min_scene_len

    @property
    def adaptive_factor(self) -> Optional[float]:
        """Adaptive noise-ceiling multiplier, or None when disabled."""
        return self._adaptive_factor

    @property
    def last_noise_ceiling(self) -> Optional[float]:
        """Noise ceiling (p99 luma delta) from the most recent detect() run."""
        return self._last_noise_ceiling

    @property
    def last_effective_threshold(self) -> float:
        """Effective ContentDetector threshold from the most recent run."""
        return self._last_effective_threshold

    def detect(
        self,
        video_path: Path,
        face_subtitle_cfg: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect scene boundaries in a video using PySceneDetect ContentDetector.

        Optionally cuts large face-covering subtitles (e.g. opening captions)
        into their own scenes using ``face_subtitle_cfg`` so downstream
        face-quality rejection can drop them independently.

        Args:
            video_path: Path to the standardized video file.
            face_subtitle_cfg: Optional config dict for face-subtitle detection
                (see ``_analyze_video``). Disabled when None.

        Returns:
            List of scene dicts, each containing:
                - scene_index: int (0-based scene number)
                - start_frame: int (inclusive)
                - end_frame: int (exclusive, i.e. last frame + 1)
                - start_time: float (seconds)
                - end_time: float (seconds)
                - duration: float (seconds)

        Raises:
            MissingDependencyError: If PySceneDetect is not installed.
            VideoReadError: If the video cannot be opened/read.
            DataProcessingError: If scene detection fails.
        """
        from face_pipeline_v2.utils.exceptions import (
            DataProcessingError,
            MissingDependencyError,
            VideoReadError,
        )

        # Lazy import: PySceneDetect is required only at runtime
        try:
            import scenedetect
        except ImportError as exc:
            raise MissingDependencyError(
                "PySceneDetect is not installed. "
                "Run: pip install scenedetect[opencv]"
            ) from exc

        if not video_path.exists():
            raise VideoReadError(
                f"Video file not found for scene detection: {video_path}"
            )

        try:
            # Open video
            video = scenedetect.open_video(str(video_path))

            # Single optional decode pass: the adaptive motion-noise ceiling and
            # the face-cover subtitle intervals are computed together so the
            # video is decoded at most once for both.
            face_enabled = bool(
                face_subtitle_cfg and face_subtitle_cfg.get("enabled", False)
            )
            intervals: List[Tuple[int, int]] = []
            noise_ceiling: Optional[float] = None
            if face_enabled or self._adaptive_factor:
                intervals, noise_ceiling = self._analyze_video(
                    video_path, face_subtitle_cfg
                )

            # Adaptive threshold: scale the ContentDetector threshold to the
            # clip's own motion-noise ceiling so high-motion single-shot clips
            # are not over-split by speaker motion while low-motion multi-shot
            # clips keep their subtle transitions.
            threshold: float = self._threshold
            if self._adaptive_factor and noise_ceiling is not None:
                threshold = max(threshold, self._adaptive_factor * noise_ceiling)

            # Record diagnostics for downstream debug logging.
            self._last_noise_ceiling = noise_ceiling
            self._last_effective_threshold = threshold

            # Create SceneManager and add ContentDetector
            scene_manager = scenedetect.SceneManager()
            scene_manager.add_detector(
                scenedetect.ContentDetector(
                    threshold=threshold,
                    min_scene_len=self._min_scene_len,
                )
            )

            # Detect scenes
            scene_manager.detect_scenes(video=video)

            # Retrieve scene list
            scene_list = scene_manager.get_scene_list()

            fps: float = float(video.frame_rate) if video.frame_rate else 1.0
            total_frames: int = int(video.duration.frame_num)

            # Convert PySceneDetect FrameTimecode pairs to plain dicts
            scenes: List[Dict[str, Any]] = []
            for start_tc, end_tc in scene_list:
                scenes.append({
                    "start_frame": int(start_tc.frame_num),
                    "end_frame": int(end_tc.frame_num),
                })

            # No scenes found → single scene covering entire video
            if not scenes:
                scenes = [{"start_frame": 0, "end_frame": total_frames}]

            # Cut large face-covering subtitles into their own scenes so that
            # downstream face-quality rejection can drop them independently.
            if intervals:
                scenes = self._split_scenes(
                    scenes=scenes,
                    intervals=intervals,
                    min_scene_len=self._min_scene_len,
                )

            # Finalize scene_index + timestamps
            result: List[Dict[str, Any]] = []
            for idx, sc in enumerate(scenes):
                start_frame = int(sc["start_frame"])
                end_frame = int(sc["end_frame"])
                start_time = start_frame / fps
                end_time = end_frame / fps
                result.append({
                    "scene_index": idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                })

            return result

        except MissingDependencyError:
            raise
        except Exception as exc:
            raise DataProcessingError(
                f"Scene detection failed for '{video_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Single-pass frame analysis (adaptive noise ceiling + face subtitles)
    # ------------------------------------------------------------------

    def _analyze_video(
        self,
        video_path: Path,
        face_subtitle_cfg: Optional[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], Optional[float]]:
        """Single decode pass returning (face_subtitle_intervals, noise_ceiling).

        Reads the video once and, depending on what is enabled, computes:
          * the mid/upper-face band Sobel edge-energy series for locating
            large face-covering subtitles (when ``face_subtitle_cfg`` enabled);
          * the consecutive-frame luma-thumbnail delta series for the adaptive
            motion-noise ceiling (when ``self._adaptive_factor`` is set).

        Returns:
            A ``(intervals, noise_ceiling)`` tuple. ``intervals`` is a list of
            inclusive ``[start, end]`` frame ranges (empty when disabled or no
            subtitle run found); ``noise_ceiling`` is the 99th percentile of the
            luma deltas, or ``None`` when adaptive thresholding is disabled or
            the clip has too few frames.
        """
        from face_pipeline_v2.utils.exceptions import (
            MissingDependencyError,
            VideoReadError,
        )

        try:
            import cv2
        except ImportError as exc:
            raise MissingDependencyError(
                "opencv-python is required for scene analysis."
            ) from exc

        cfg: Dict[str, Any] = face_subtitle_cfg or {}
        face_enabled = bool(cfg.get("enabled", False))
        want_deltas = bool(self._adaptive_factor)

        band_top = float(cfg.get("band_top", 0.33))
        band_bottom = float(cfg.get("band_bottom", 0.67))
        sensitivity = float(cfg.get("sensitivity", 4.0))
        bright_abs_min = float(cfg.get("bright_abs_min", 0.002))
        bright_luma = float(cfg.get("bright_luma", 235.0))
        min_frames = int(cfg.get("min_subtitle_frames", 5))
        settle = int(cfg.get("settle_frames", 2))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoReadError(
                f"Cannot open video for scene analysis: {video_path}"
            )

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        y0 = max(0, int(round(height * band_top)))
        y1 = min(height, int(round(height * band_bottom)))
        band_ok = face_enabled and (y1 - y0) >= 4

        downscale = 16
        edge_scores: List[float] = []
        bright_scores: List[float] = []
        luma_deltas: List[float] = []
        prev_small: Optional[np.ndarray] = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if band_ok:
                band = gray_u8.astype(np.float32)[y0:y1, :]
                gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
                edge_scores.append(float(cv2.magnitude(gx, gy).mean()))
                bright_scores.append(float(np.mean(band >= bright_luma)))
            if want_deltas:
                small = cv2.resize(
                    gray_u8, (downscale, downscale), interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)
                if prev_small is not None:
                    luma_deltas.append(float(np.mean(np.abs(small - prev_small))))
                prev_small = small
        cap.release()

        # Face-covering subtitle intervals from the band texture series.
        #
        # A face-covering subtitle adds *extra* texture and/or near-white pixels
        # to the mid/upper-face band relative to the clip's own clean baseline.
        # Detection is a two-state hysteresis over two per-frame channels:
        #   * near-white text  -> a spike in the band's bright-pixel fraction
        #     (reliable on dark talking-head backgrounds);
        #   * text/graphic edges -> a higher mean Sobel gradient (reliable on
        #     bright backgrounds).
        # A frame ENTERS the "subtitle active" state on a strong deviation
        # (``sensitivity`` robust sigmas above the median/MAD baseline) and STAYS
        # active until the signal falls back near the baseline (0.5 sigma margin),
        # so gradual dissolve / fade-out tails are included rather than cut
        # through.  A small ``settle_frames`` buffer then guarantees the boundary
        # lands on a fully clean frame.
        #
        # Every threshold is expressed relative to the clip's own baseline, so a
        # single set of defaults generalizes without per-video tuning.
        intervals: List[Tuple[int, int]] = []
        if edge_scores:
            n = len(edge_scores)
            edges = np.asarray(edge_scores, dtype=np.float64)
            brights = np.asarray(bright_scores, dtype=np.float64)

            med_edge = float(np.median(edges))
            sig_edge = 1.4826 * float(np.median(np.abs(edges - med_edge)))
            sig_edge = max(sig_edge, max(abs(med_edge) * 0.02, 1e-3))

            med_bright = float(np.median(brights))
            sig_bright = 1.4826 * float(np.median(np.abs(brights - med_bright)))
            sig_bright = max(sig_bright, max(abs(med_bright) * 0.02, 1e-3))

            enter_edge = edges >= med_edge + sensitivity * sig_edge
            enter_bright = (brights >= bright_abs_min) & (
                brights >= med_bright + sensitivity * sig_bright
            )
            enter = enter_edge | enter_bright

            exit_edge = edges >= med_edge + 0.5 * sig_edge
            exit_bright = (brights >= bright_abs_min * 0.1) & (
                brights >= med_bright + 0.5 * sig_bright
            )
            still = exit_edge | exit_bright

            run_start: Optional[int] = None
            for i in range(n):
                if run_start is None:
                    if enter[i]:
                        run_start = i
                elif not still[i]:
                    if i - run_start >= min_frames:
                        intervals.append((run_start, i - 1))
                    run_start = None
            if run_start is not None and n - run_start >= min_frames:
                intervals.append((run_start, n - 1))

            last = (num_frames - 1) if num_frames else (n - 1)
            padded: List[Tuple[int, int]] = []
            for start, end in intervals:
                padded.append((max(0, start - settle), min(last, end + settle)))
            intervals = padded

        # Adaptive motion-noise ceiling from the luma delta series.
        noise_ceiling: Optional[float] = None
        if want_deltas and len(luma_deltas) >= 2:
            noise_ceiling = float(np.percentile(luma_deltas, 99))

        return intervals, noise_ceiling

    @staticmethod
    def _split_scenes(
        scenes: List[Dict[str, Any]],
        intervals: List[Tuple[int, int]],
        min_scene_len: int,
    ) -> List[Dict[str, Any]]:
        """Split ``scenes`` at the given intervals, merging short slivers."""
        if not intervals:
            return scenes

        total = max((int(sc["end_frame"]) for sc in scenes), default=0)
        cuts = {0, total}
        for sc in scenes:
            cuts.add(int(sc["start_frame"]))
            cuts.add(int(sc["end_frame"]))
        for start, end in intervals:
            if 0 < start <= total:
                cuts.add(int(start))
            if 0 <= end < total:
                cuts.add(int(end) + 1)

        bounds = sorted(b for b in cuts if 0 <= b <= total)
        segments: List[List[int]] = []
        for a, b in zip(bounds, bounds[1:]):
            if b > a:
                segments.append([int(a), int(b)])

        # Merge segments shorter than min_scene_len into the previous one.
        merged: List[List[int]] = []
        for seg in segments:
            if merged and (seg[1] - seg[0]) < min_scene_len:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        # Merge a still-too-short leading segment forward.
        if merged and (merged[0][1] - merged[0][0]) < min_scene_len and len(merged) > 1:
            merged[1][0] = merged[0][0]
            merged = merged[1:]

        return [
            {"start_frame": a, "end_frame": b}
            for a, b in merged
        ]

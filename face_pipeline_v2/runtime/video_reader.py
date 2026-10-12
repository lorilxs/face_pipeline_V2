"""Unified video frame reader abstraction.

Provides a pluggable reader backend for Stage 5 (and future stages).
Supports two backends:
    - ffmpeg: rawvideo pipe, sequential decode, no GOP re-decode overhead
    - opencv: cv2.VideoCapture, seek-based (legacy, for fallback)

Usage:
    reader = create_video_reader("ffmpeg", video_path)
    for frame_idx in target_frames:
        frame = reader.read_frame(frame_idx)
    reader.close()
"""

import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════

class VideoFrameReader(ABC):
    """Abstract base for video frame readers."""

    @abstractmethod
    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Read a single frame by index. Returns None if out of range."""
        ...

    @abstractmethod
    def read_next(self) -> Optional[Tuple[int, np.ndarray]]:
        """Read the next sequential frame. Returns (frame_index, frame) or None."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...


# ══════════════════════════════════════════════════════════════════════
# OpenCV backend
# ══════════════════════════════════════════════════════════════════════

class OpenCVVideoReader(VideoFrameReader):
    """OpenCV VideoCapture wrapper — seek-based, legacy compatibility."""

    def __init__(self, video_path: Path) -> None:
        self._path = Path(video_path)
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_idx = 0

    @property
    def width(self) -> int: return self._w

    @property
    def height(self) -> int: return self._h

    @property
    def backend_name(self) -> str: return "opencv"

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        self._next_idx = frame_index + 1
        return frame if ret else None

    def read_next(self) -> Optional[Tuple[int, np.ndarray]]:
        idx = self._next_idx
        ret, frame = self._cap.read()
        if ret:
            self._next_idx = idx + 1
            return (idx, frame)
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ══════════════════════════════════════════════════════════════════════
# FFmpeg backend
# ══════════════════════════════════════════════════════════════════════

class FFmpegVideoReader(VideoFrameReader):
    """FFmpeg rawvideo pipe reader — sequential decode, no seek overhead.

    Starts an ffmpeg subprocess that decodes the entire video to raw
    BGR frames on stdout.  Read sequentially via read_next().

    Supports forward-only access: read_frame(n) will seek by reading
    and discarding frames until the target index, OR start a new
    ffmpeg process at the target position.
    """

    def __init__(self, video_path: Path) -> None:
        self._path = Path(video_path)
        self._proc: Optional[subprocess.Popen] = None
        self._frame_size: int = 0
        self._next_idx: int = 0
        self._w: int = 0
        self._h: int = 0
        self._total_frames_read: int = 0
        self._decode_start: float = 0.0
        self._decode_time: float = 0.0

    @property
    def width(self) -> int: return self._w

    @property
    def height(self) -> int: return self._h

    @property
    def backend_name(self) -> str: return "ffmpeg"

    def total_frames_read(self) -> int: return self._total_frames_read

    def decode_time(self) -> float: return self._decode_time

    def start(self, frame_count: Optional[int] = None) -> None:
        """Start the ffmpeg pipe from frame 0.

        Must be called before read_next().

        Args:
            frame_count: Total frames to read (None = until EOF).
        """
        # Probe dimensions first
        self._probe_dimensions()

        vf = f"-vframes {frame_count}" if frame_count else ""
        cmd = (
            f"ffmpeg -loglevel error "
            f"-i {self._path} "
            f"{vf} "
            f"-f rawvideo -pix_fmt bgr24 -"
        )
        self._proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._frame_size = self._w * self._h * 3
        self._next_idx = 0
        self._total_frames_read = 0
        self._decode_start = time.monotonic()

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Read frame at index — reads and discards forward if needed."""
        if self._proc is None:
            self.start()
        # Fast path: we're at the right position
        if frame_index == self._next_idx:
            result = self.read_next()
            return result[1] if result else None
        # Forward seek: read + discard
        while self._next_idx < frame_index:
            result = self.read_next()
            if result is None:
                return None
        result = self.read_next()
        return result[1] if result else None

    def read_next(self) -> Optional[Tuple[int, np.ndarray]]:
        """Read the next frame from the pipe."""
        if self._proc is None or self._frame_size == 0:
            return None

        raw = self._proc.stdout.read(self._frame_size)
        if not raw or len(raw) < self._frame_size:
            self._decode_time = time.monotonic() - self._decode_start
            self._close_proc()
            return None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._h, self._w, 3)
        )
        idx = self._next_idx
        self._next_idx = idx + 1
        self._total_frames_read += 1
        return (idx, frame)

    def close(self) -> None:
        self._decode_time = time.monotonic() - self._decode_start
        self._close_proc()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _probe_dimensions(self) -> None:
        """Probe video dimensions via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(self._path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                w_str, h_str = result.stdout.strip().split(",")
                self._w, self._h = int(w_str), int(h_str)
                return
        except Exception:
            pass
        # Fallback: OpenCV probe
        cap = cv2.VideoCapture(str(self._path))
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def _close_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdout.close()
                self._proc.stderr.close()  # type: ignore[union-attr]
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def create_video_reader(backend: str, video_path: Path) -> VideoFrameReader:
    """Create a video frame reader by backend name.

    Args:
        backend: "ffmpeg" or "opencv".
        video_path: Path to the video file.

    Returns:
        VideoFrameReader instance.

    Raises:
        ValueError: If backend is unknown.
    """
    if backend == "ffmpeg":
        return FFmpegVideoReader(video_path)
    elif backend == "opencv":
        return OpenCVVideoReader(video_path)
    else:
        raise ValueError(
            f"Unknown video reader backend: '{backend}'. "
            f"Use 'ffmpeg' or 'opencv'."
        )

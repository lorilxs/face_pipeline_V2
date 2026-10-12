"""Sync estimator for Stage 7.

Encapsulates SyncNet inference per clip using face crops as visual input.

Per AISDD 3.11 (revised), SyncNet consumes face crops from Stage 5,
NOT full-frame video.  This eliminates background noise and strengthens
lip-motion features.

For each clip:
    1. Collect face crop images from FaceSequence, sorted by frame_index.
    2. Generate a temporary video from face crops at the clip's FPS.
    3. Mux with the standardized audio.
    4. Run SyncNet on the face-crop video.
    5. Build AVSyncInfo with offset, sync_lse_c, sync_lse_d, median_distance.

Does NOT load models — the SyncNet wrapper is injected by the Stage.
Does NOT re-detect faces — consumes Stage 5 output exclusively.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from face_pipeline_v2.data.models import AVSyncInfo, ClipInfo, FaceSequence
from face_pipeline_v2.runtime.video_reader import FFmpegVideoReader


# ---------------------------------------------------------------------------
# Crop geometry constants — reproduce the colleague S3FD tight crop exactly.
# Reference: third_party/SyncNet/syncnet_hzy/run_pipeline.py::_crop_track
#   cs  = crop_scale (0.40)   → horizontal halo 1 + cs, bottom 1 + 2cs
#   bs  = max(w, h) / 2       → half of the longer tight-bbox side
#   pad = int(bs * (1 + 2cs)) → symmetric gray pad before slicing
#   y slice  [cy - bs,          cy + bs * (1 + 2cs)]   (asymmetric: shave top)
#   x slice  [cx - bs*(1 + cs), cx + bs*(1 + cs)]
# ---------------------------------------------------------------------------
SYNCNET_MEDFILT_KERNEL: int = 13   # whole-track temporal smoothing kernel
SYNCNET_PAD_VALUE: int = 110       # gray fill, matches joonson np.pad(..., 110)
SYNCNET_INPUT_SIZE: int = 256      # resize target for the SyncNet video


class FaceCrop(NamedTuple):
    """One per-frame SCRFD tight detection used as SyncNet visual input.

    The S3FD tight crop is reconstructed directly from the standardized
    full-frame video (not from the Stage-5 2x expanded crop), so FaceCrop only
    needs the frame index and the raw tight bbox in full-frame coordinates.

    Attributes:
        frame_index: Global frame index in the standardized video.
        bbox: SCRFD tight detection bbox in full-frame coords (x1, y1, x2, y2).
    """

    frame_index: int
    bbox: Tuple[float, float, float, float]


def estimate_sync(
    sequence: FaceSequence,
    clip: ClipInfo,
    audio_path: Path,
    model: Any,
    std_video_path: Path,
    temp_base: Optional[Path] = None,
    logger: Any = None,
    crop_scale: float = 0.40,
    input_size: int = SYNCNET_INPUT_SIZE,
) -> AVSyncInfo:
    """Estimate A/V sync for a single clip using face crops + clip audio.

    Reproduces the colleague S3FD tight crop from the *raw* Stage 5 tight
    bbox (``FaceInfo.bbox``), not from the expanded bbox/crop.  For each
    frame we compute ``cx``, ``cy``, ``bs = max(w, h) / 2``, smooth the
    whole track with a median filter, then crop a square of side
    ``2.8 * bs`` with x symmetric ``[cx - 1.4bs, cx + 1.4bs]`` and y
    asymmetric ``[cy - 1.0bs, cy + 1.8bs]`` (shaving the forehead, keeping
    the chin/neck — making the mouth prominent and centered).  Crops are
    resized to ``input_size``×``input_size``, clip audio extracted, and
    SyncNet run.

    Args:
        sequence: FaceSequence from Stage 5 with per-frame face crops.  Must
            carry the per-frame frame_index and raw tight ``bbox`` (full-frame
            coordinates).
        clip: ClipInfo with clip_id, start_time, end_time, duration.
        audio_path: Path to the standardized audio file (Stage 1).
        model: Loaded SyncNet wrapper.
        std_video_path: Path to the standardized full-frame video (Stage 1).
        temp_base: Optional base directory for temporary files.
        logger: Optional PipelineLogger for input-size diagnostics.
        crop_scale: S3FD ``crop_scale`` (cs).  Horizontal halo = 1 + cs,
            bottom halo = 1 + 2cs.  0.40 reproduces the colleague reference;
            sweep 0.20–0.40 to calibrate the SCRFD-vs-S3FD box difference.
        input_size: Resize target for the SyncNet video (square).

    Returns:
        AVSyncInfo with sync results.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="syncnet_", dir=temp_base))
    offset, lse_c, lse_d, median_dist = 0, 0.0, 0.0, 0.0

    try:
        # ---- 1. Collect face crops (raw tight bbox + expanded bbox) ----
        face_crops = _collect_face_crops(sequence)

        if not face_crops:
            if logger:
                logger.warning(
                    f"[Stage 7] {clip.clip_id}: no face crops found"
                )
            return AVSyncInfo(
                clip_id=clip.clip_id,
                offset=0, sync_lse_c=0.0, sync_lse_d=0.0, median_distance=0.0,
                start_frame=clip.start_frame,
                end_frame=clip.end_frame,
                fps=clip.frame_count / clip.duration if clip.duration > 0 else None,
                duration=clip.duration,
                timestamp=clip.start_time,
            )

        # Log raw crop size stats
        if logger:
            _log_crop_sizes(logger, clip.clip_id, face_crops)

        # ---- 2. Extract clip-level audio segment ----
        clip_audio_path = tmp_dir / f"{clip.clip_id}_audio.wav"
        _extract_clip_audio(
            audio_path, clip_audio_path, clip.start_time, clip.duration,
            logger=logger, clip_id=clip.clip_id,
        )

        # ---- 3. Generate face-crop video + mux with clip audio ----
        fps = clip.frame_count / clip.duration if clip.duration > 0 else 25.0
        segment_path = _build_face_crop_video(
            face_crops=face_crops,
            audio_path=clip_audio_path,
            output_path=tmp_dir / f"{clip.clip_id}.mp4",
            fps=fps,
            clip_duration=clip.duration,
            std_video_path=std_video_path,
            logger=logger,
            clip_id=clip.clip_id,
            crop_scale=crop_scale,
            input_size=input_size,
        )

        # ---- 4. Run SyncNet (no silent fallback — fail loudly) ----
        if not segment_path.exists():
            raise RuntimeError(
                f"[Stage 7] {clip.clip_id}: face-crop video was not generated "
                f"at {segment_path}"
            )

        offset, lse_c, lse_d, median_dist = model.predict(segment_path)

        # ---- 5. Per-clip debug ----
        if logger:
            logger.info(
                f"[Stage 7] sync result clip={clip.clip_id} "
                f"video={segment_path.name} audio={clip_audio_path.name} "
                f"duration={clip.duration:.2f}s "
                f"offset={offset} lse_c={lse_c:.4f} lse_d={lse_d:.4f} "
                f"median_dist={median_dist:.4f}"
            )

    except Exception as exc:
        # No silent fallback: log the real error and propagate it so that the
        # PPU/Docker environment never silently reports all-zero output.
        if logger:
            logger.error(
                f"[Stage 7] {clip.clip_id}: SyncNet inference failed: "
                f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return AVSyncInfo(
        clip_id=clip.clip_id,
        offset=offset,
        sync_lse_c=lse_c,
        sync_lse_d=lse_d,
        median_distance=median_dist,
        start_frame=clip.start_frame,
        end_frame=clip.end_frame,
        fps=clip.frame_count / clip.duration if clip.duration > 0 else None,
        duration=clip.duration,
        timestamp=clip.start_time,
    )


def _log_crop_sizes(logger: Any, clip_id: str, face_crops: List[FaceCrop]) -> None:
    """Log raw tight-bbox size statistics for SyncNet input diagnostics."""
    ws, hs = [], []
    for crop in face_crops[:50]:  # sample first 50 for stats
        x1, y1, x2, y2 = crop.bbox
        ws.append(x2 - x1)
        hs.append(y2 - y1)
    if ws and hs:
        logger.info(
            f"[Stage 7] SyncNet input: clip={clip_id} "
            f"tight_bbox_size={min(ws):.0f}x{min(hs):.0f}~{max(ws):.0f}x{max(hs):.0f} "
            f"after_resize={SYNCNET_INPUT_SIZE}x{SYNCNET_INPUT_SIZE} "
            f"frames={len(face_crops)}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_face_crops(
    sequence: FaceSequence,
    fill_forward: bool = True,
) -> List[FaceCrop]:
    """Collect face crop records from a FaceSequence.

    Each record carries the frame index together with the raw SCRFD tight
    ``bbox`` in full-frame coordinates.  Stage 7 reconstructs the S3FD-style
    tight crop directly from the standardized full-frame video using these
    raw tight boxes (never the Stage-5 expanded bbox, which is clamped to the
    frame edges and would corrupt the tight-crop geometry).

    ``fill_forward`` (default True) emits a crop record for EVERY frame from
    the first detected face onward, reusing the most recent tight bbox for
    frames without a detection.  This matches the reference probe's
    fill-forward construction so the SyncNet video always covers the full
    clip duration.  Without it, frames where SCRFD finds no face (head turns,
    occlusions) are silently dropped and the video ends up shorter than the
    audio; ``-shortest`` then truncates the tail of the audio, corrupting the
    A/V alignment estimate and dragging ``sync_lse_c`` down (the reported
    pipeline-vs-probe gap).  Stage 5's FaceSequence already contains every
    frame of the clip (empty frames carry ``faces=()``), so filling them keeps
    the crop track frame-complete without re-detection.

    Args:
        sequence: FaceSequence from Stage 5.
        fill_forward: Whether to fill frames without a detection with the
            most recent bbox (True, default) or skip them (False).

    Returns:
        List of FaceCrop records, ordered by frame_index.
    """
    crops: List[FaceCrop] = []
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    for frame_info in sequence.frames:
        if frame_info.faces:
            main_face = frame_info.faces[0]
            last_bbox = tuple(main_face.bbox)
            crops.append(FaceCrop(
                frame_index=frame_info.frame_index,
                bbox=last_bbox,
            ))
        elif fill_forward and last_bbox is not None:
            # No detection this frame (head turn / occlusion): reuse the most
            # recent tight bbox so the crop track stays frame-complete.
            crops.append(FaceCrop(
                frame_index=frame_info.frame_index,
                bbox=last_bbox,
            ))
    return crops


def _extract_clip_audio(
    full_audio_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    logger: Any = None,
    clip_id: str = "",
) -> Path:
    """Extract a clip-level audio segment from the full standardized audio.

    Uses ffmpeg to trim audio to [start_time, start_time + duration].

    Args:
        full_audio_path: Path to the full standardized audio (Stage 1).
        output_path: Destination path for the clip audio segment.
        start_time: Clip start time in seconds.
        duration: Clip duration in seconds.
        logger: Optional PipelineLogger for diagnostics.
        clip_id: Clip id for log context.

    Returns:
        Path to the extracted clip audio file.

    Raises:
        RuntimeError: If ffmpeg fails or the output audio is not produced.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(full_audio_path),
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or not output_path.exists():
        stderr = result.stderr.decode(errors="ignore") if isinstance(result.stderr, bytes) else result.stderr
        raise RuntimeError(
            f"[Stage 7] {clip_id}: ffmpeg clip-audio extraction failed: {stderr}"
        )
    if logger:
        logger.info(
            f"[Stage 7] {clip_id}: clip audio extracted "
            f"{output_path.name} ({output_path.stat().st_size}B)"
        )
    return output_path


def _temporal_smooth(values: List[float], kernel_size: int) -> List[float]:
    """Smooth a whole-track sequence with ``scipy.signal.medfilt``.

    ``cx``, ``cy`` and ``bs`` are smoothed across the full clip (track) before
    cropping, exactly as the colleague S3FD reference does (kernel_size=13).
    Short tracks (`len < kernel_size`) are returned unchanged — medfilt
    zero-pads tiny series and would otherwise collapse them toward zero.
    """
    if len(values) <= 1:
        return list(values)
    kernel = int(kernel_size)
    if kernel <= 1 or len(values) < kernel:
        return list(values)
    from scipy import signal

    return [
        float(v)
        for v in signal.medfilt(np.asarray(values, dtype=float), kernel_size=kernel)
    ]


def _s3fd_style_crop(
    img: Any,
    cx: float,
    cy: float,
    bs: float,
    crop_scale: float = 0.40,
    pad_value: int = SYNCNET_PAD_VALUE,
) -> Any:
    """Apply the colleague S3FD tight crop (asymmetric vertical shave).

    Mirrors ``run_pipeline.py::_crop_track``:
        cs   = crop_scale
        bsi  = int(bs * (1 + 2*cs))    # symmetric gray pad before slicing
        my   = cy + bsi;  mx = cx + bsi
        y    = [my - bs, my + bs*(1 + 2*cs)]   # shave forehead, keep chin
        x    = [mx - bs*(1 + cs), mx + bs*(1 + cs)]

    Returns a square-ish BGR array of side ≈ 2.8 * bs (then resized by the
    caller), with gray (``pad_value``) filling any out-of-frame regions.
    """
    if img is None or bs <= 0:
        return img
    cs = float(crop_scale)
    bsi = int(bs * (1 + 2 * cs))
    padded = np.pad(
        img,
        ((bsi, bsi), (bsi, bsi), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )
    my = cy + bsi
    mx = cx + bsi
    y0 = int(my - bs)
    y1 = int(my + bs * (1 + 2 * cs))
    x0 = int(mx - bs * (1 + cs))
    x1 = int(mx + bs * (1 + cs))
    return padded[y0:y1, x0:x1]


def _s3fd_gray_fill_ratio(
    img_w: float,
    img_h: float,
    cx: float,
    cy: float,
    bs: float,
    crop_scale: float = 0.40,
) -> float:
    """Fraction of the S3FD-style crop filled by gray pad (0.0–1.0).

    The colleague recipe pads *outside the source image* with gray, so any
    part of the shave window that falls outside ``[0, w] x [0, h]`` becomes
    pad.  This mirrors the actual crop geometry exactly (it only counts the
    area outside the image), making it sensitive to how much of the tight-bbox
    halo sticks out past the Stage-5 2× crop — i.e. the part that would be
    real background pixels in the colleague full-frame crop but gray here.

    Pure geometry, so it does NOT confound genuine gray-ish pixels with the
    pad value.

    Args:
        img_w, img_h: Source crop image width/height (the Stage-5 2× crop).
        cx, cy, bs: Smoothed tight-bbox center and half-long-side.
        crop_scale: S3FD ``crop_scale`` (cs).

    Returns:
        Area fraction of the crop that is gray pad.  0.0 = fully inside the
        image (identical source to the colleague full-frame crop); 1.0 = fully
        padded.
    """
    if bs <= 0.0 or img_w <= 0.0 or img_h <= 0.0:
        return 0.0
    cs = float(crop_scale)
    x0 = cx - bs * (1.0 + cs)
    x1 = cx + bs * (1.0 + cs)
    y0 = cy - bs
    y1 = cy + bs * (1.0 + 2.0 * cs)
    total_w = x1 - x0
    total_h = y1 - y0
    if total_w <= 0.0 or total_h <= 0.0:
        return 1.0
    inter_w = max(0.0, min(float(img_w), x1) - max(0.0, x0))
    inter_h = max(0.0, min(float(img_h), y1) - max(0.0, y0))
    valid_area = inter_w * inter_h
    total_area = total_w * total_h
    return float(min(1.0, max(0.0, 1.0 - valid_area / total_area)))


def _build_face_crop_video(
    face_crops: List[FaceCrop],
    audio_path: Path,
    output_path: Path,
    fps: float,
    clip_duration: float,
    std_video_path: Path,
    logger: Any = None,
    clip_id: str = "",
    crop_scale: float = 0.40,
    input_size: int = SYNCNET_INPUT_SIZE,
) -> Path:
    """Build a face-crop video by streaming S3FD tight crops from the full
    standardized video straight into an ffmpeg image2pipe encoder (no
    intermediate JPEG files).

    Steps:
        1. Smooth the whole track's ``cx``, ``cy``, ``bs`` (full-frame coords)
           with medfilt(13).
        2. Stream-decode std-video frames for every collected frame index.
        3. Apply the asymmetric S3FD shave (x ± 1.4bs, y [−bs, +1.8bs]) and
           resize each frame to ``input_size``².
        4. Write raw BGR frames to the image2pipe encoder, muxed with the clip
           audio (-shortest).

    Args:
        face_crops: FaceCrop records (frame index + tight bbox in full-frame
            coordinates).
        audio_path: Path to the clip-level audio file.
        output_path: Destination path for the output .mp4 file.
        fps: Frame rate for the output video.
        clip_duration: Expected clip duration in seconds (used for -t).
        std_video_path: Path to the standardized full-frame video.
        logger: Optional PipelineLogger for diagnostics.
        clip_id: Clip id for log context.
        crop_scale: S3FD ``crop_scale`` (see _s3fd_style_crop).
        input_size: Square resize target for the SyncNet video.

    Returns:
        Path to the generated video file.

    Raises:
        RuntimeError: If ffmpeg fails or the output video is not produced.
    """
    if not face_crops:
        raise RuntimeError(f"[Stage 7] {clip_id}: no face crops to build video")

    # ---- Whole-track tight bboxes in FULL-FRAME coords, medfilt-smooth ---- #
    boxes = [crop.bbox for crop in face_crops]
    cxs = _temporal_smooth(
        [(b[0] + b[2]) / 2.0 for b in boxes], SYNCNET_MEDFILT_KERNEL
    )
    cys = _temporal_smooth(
        [(b[1] + b[3]) / 2.0 for b in boxes], SYNCNET_MEDFILT_KERNEL
    )
    bss = _temporal_smooth(
        [max(b[2] - b[0], b[3] - b[1]) / 2.0 for b in boxes],
        SYNCNET_MEDFILT_KERNEL,
    )

    # Frame indices are strictly ascending (collected in sequence order), so
    # the reader only ever scans forward, never backward.
    reader = FFmpegVideoReader(std_video_path)
    reader.start()

    # ---- ffmpeg image2pipe encoder (zero intermediate files) ------------- #
    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{input_size}x{input_size}",
            "-framerate", str(fps),
            "-i", "-",
            "-i", str(audio_path),
            "-t", str(clip_duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    fill_ratios: List[float] = []
    try:
        for i, crop in enumerate(face_crops):
            img = reader.read_frame(crop.frame_index)
            if img is None:
                continue
            h, w = img.shape[:2]
            face = _s3fd_style_crop(
                img, cxs[i], cys[i], bss[i], crop_scale=crop_scale
            )
            fill_ratios.append(
                _s3fd_gray_fill_ratio(
                    w, h, cxs[i], cys[i], bss[i], crop_scale=crop_scale
                )
            )
            resized = cv2.resize(
                face, (input_size, input_size), interpolation=cv2.INTER_LINEAR
            )
            enc.stdin.write(resized.tobytes())
    finally:
        reader.close()

    enc.stdin.close()
    stderr = enc.stderr.read() if enc.stderr is not None else b""
    enc.wait(timeout=120)

    if enc.returncode != 0 or not output_path.exists():
        err = stderr.decode(errors="ignore") if isinstance(stderr, bytes) else str(stderr)
        raise RuntimeError(
            f"[Stage 7] {clip_id}: ffmpeg image2pipe video generation failed: "
            f"{err}"
        )

    if logger and fill_ratios:
        mean_fill = sum(fill_ratios) / len(fill_ratios)
        max_fill = max(fill_ratios)
        logger.info(
            f"[Stage 7] {clip_id}: S3FD crop gray-fill ratio "
            f"mean={mean_fill:.4f} max={max_fill:.4f} over {len(fill_ratios)} frames "
            f"(fraction of the shave window outside the std-video frame; "
            f"now reconstructed from full frames so it stays near 0)"
        )

    if logger:
        _log_video_info(logger, clip_id, output_path)

    return output_path


def _log_video_info(logger: Any, clip_id: str, path: Path) -> None:
    """Log the generated temp sync video's size / fps / duration."""
    size_bytes = path.stat().st_size
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = nframes / fps if fps > 0 else 0.0
    logger.info(
        f"[Stage 7] {clip_id}: temp sync video {path.name} "
        f"size={size_bytes}B fps={fps:.2f} frames={nframes} "
        f"duration={duration:.2f}s {w}x{h}"
    )


def _check_syncnet_dependencies(logger: Any) -> None:
    """Log SyncNet python dependencies (torch, cv2, scipy, speech_features)."""
    import importlib

    for dep in ("torch", "cv2", "scipy", "python_speech_features"):
        try:
            mod = importlib.import_module(dep)
            version = getattr(mod, "__version__", "?")
            logger.info(f"[Stage 7] SyncNet dependency: {dep} = {version}")
        except ImportError:
            logger.error(f"[Stage 7] SyncNet dependency MISSING: {dep}")

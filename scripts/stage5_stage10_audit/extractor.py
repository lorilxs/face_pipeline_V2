"""Read-only data extraction for the Stage5/Stage10 audit.

The pipeline keeps FaceSequence / LandmarkInfo in memory and never persists
them, so this module *regenerates* them read-only:

* Stage5: runs SCRFD on the clip video frame-by-frame (same model + same
  pre/post-processing the pipeline uses) and records the tight bbox,
  confidence, 5-point landmark and the expanded crop bbox per frame.
* Stage10: runs InsightFace 2d106det on the Stage5 crops that are already on
  disk (``faces/{video_id}/{clip_id}/crops/frame_XXXXXX_face_000.jpg``) and
  maps crop-space landmarks back to frame coordinates via the Stage5
  ``expanded_bbox`` origin -- exactly mirroring
  ``stage_10_landmark/processor.py::_map_to_frame``.

Nothing here writes into the pipeline's own output directories and no model
or pipeline code is modified. Output JSONs follow the canonical schemas in
``data_io.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .data_io import Stage5Data, load_stage5


# ---------------------------------------------------------------------------
# Locating inputs
# ---------------------------------------------------------------------------

FRAME_PAT = re.compile(r"frame_(\d{6})_face_(\d{3})\.jpg")


def load_clip_manifest(run_dir: Path, clip_id: str) -> Dict[str, Any]:
    """Find the manifest entry for ``clip_id`` under ``run_dir``."""
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in data:
        if entry.get("clip_id") == clip_id:
            return entry
    return {}


def locate_clip_video(run_dir: Path, clip_id: str) -> Path:
    """Resolve the on-disk clip video for ``clip_id``."""
    run_dir = Path(run_dir)
    entry = load_clip_manifest(run_dir, clip_id)
    if entry.get("video_path"):
        cand = Path(entry["video_path"])
        if not cand.is_absolute():
            cand = run_dir / cand
        if cand.exists():
            return cand
    # fallbacks
    for cand in (
        run_dir / "clips" / f"{clip_id}.mp4",
        run_dir / "clips" / f"{clip_id}.avi",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Clip video for {clip_id!r} not found under {run_dir}")


def locate_crops_dir(run_dir: Path, clip_id: str) -> Path:
    """Locate the Stage5 crop directory for a clip (search across video_ids)."""
    run_dir = Path(run_dir)
    matches = sorted(run_dir.glob(f"faces/*/{clip_id}/crops"))
    if not matches:
        raise FileNotFoundError(
            f"Stage5 crops for {clip_id!r} not found under {run_dir}/faces/*/"
        )
    return matches[0]


def crop_frame_mapping(crops_dir: Path) -> Tuple[int, Dict[int, Tuple[int, int, Path]]]:
    """Map clip-local frame indices to on-disk crops.

    Crop filenames carry *global* frame numbers (e.g. ``frame_000100``).
    Clips are contiguous temporal segments, so the global offset is constant
    within a clip and derived from the smallest global frame number.

    Returns:
        ``(g0, mapping)`` where ``mapping[local] = (global_frame, face_idx, path)``.
    """
    crops_dir = Path(crops_dir)
    items: List[Tuple[int, int, Path]] = []
    for p in sorted(crops_dir.glob("frame_*_face_*.jpg")):
        m = FRAME_PAT.match(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), int(m.group(2)), p))
    if not items:
        raise FileNotFoundError(f"No crops found in {crops_dir}")

    g0 = min(g for g, _, _ in items)
    mapping: Dict[int, Tuple[int, int, Path]] = {}
    for g, face_idx, p in items:
        mapping[g - g0] = (g, face_idx, p)
    return g0, mapping


def _video_props(path: Path) -> Tuple[float, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return float(fps), w, h


def find_crop_origin(
    frame: np.ndarray,
    crop: np.ndarray,
    fx: float = 0.5,
    min_score: float = 0.9,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """Locate the crop's top-left corner within its source video frame.

    The Stage5 crops were cut from the (original) pipeline's expanded bbox,
    which is not persisted. Re-running SCRFD only reproduces it to within a
    few px, so to keep the audit faithful to the real Stage 10 output we find
    the *exact* crop origin by normalised cross-correlation template matching.

    Returns:
        ``((ox, oy), score)`` on success, or ``(None, score)`` if the match is
        below ``min_score`` (caller should fall back to the expanded bbox).
    """
    fh, fw = frame.shape[:2]
    ch, cw = crop.shape[:2]
    if fw < cw or fh < ch:
        return None, -1.0

    fs = cv2.resize(frame, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
    cs = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(fs, cs, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    ox, oy = int(round(loc[0] / fx)), int(round(loc[1] / fx))

    if score < min_score:
        return None, float(score)
    # sanity: crop must fit inside the frame at this origin
    if ox < 0 or oy < 0 or ox + cw > fw or oy + ch > fh:
        return None, float(score)
    return (ox, oy), float(score)



# ---------------------------------------------------------------------------
# Stage 5 extraction (SCRFD on clip video)
# ---------------------------------------------------------------------------


def extract_stage5(
    video_path: Path,
    out_path: Path,
    scrfd_model_dir: Path,
    expand_ratio: float = 2.0,
    device: str = "cpu",
    execution_provider: str = "auto",
    max_frames: Optional[int] = None,
    clip_id: str = "",
    video_id: str = "",
) -> Dict[str, Any]:
    """Re-run SCRFD over the clip video and export per-frame face records.

    Returns the payload dict (also written to ``out_path``).
    """
    from face_pipeline_v2.models.wrappers.scrfd import SCRFDWrapper
    from face_pipeline_v2.stages.stage_05_face_detection.face_cropper import (
        expand_bbox,
    )

    video_path = Path(video_path)
    fps, w, h = _video_props(video_path)

    wrapper = SCRFDWrapper(
        model_path=Path(scrfd_model_dir), device=device,
        execution_provider=execution_provider,
    )
    wrapper.load()
    print(f"[audit] SCRFD loaded: {wrapper.model_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames: List[Dict[str, Any]] = []
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and fi >= max_frames:
                break

            dets = wrapper.detect(frame)
            faces: List[Dict[str, Any]] = []
            for d in dets:
                bbox = tuple(float(v) for v in d["bbox"])
                conf = float(d.get("confidence", 0.0))
                lm5 = tuple(float(v) for v in (d.get("landmarks") or ()))
                eb = expand_bbox(bbox, w, h, expand_ratio)
                faces.append(
                    {
                        "face_id": f"face_{len(faces):03d}",
                        "bbox": bbox,
                        "confidence": conf,
                        "landmark_5": lm5,
                        "expanded_bbox": eb,
                    }
                )
            frames.append(
                {
                    "frame_index": fi,
                    "timestamp": round(fi / fps, 6),
                    "faces": faces,
                }
            )
            fi += 1
    finally:
        cap.release()

    payload = {
        "source": {
            "video_id": video_id,
            "clip_id": clip_id,
            "video_path": str(video_path),
            "fps": int(round(fps)),
            "width": w,
            "height": h,
        },
        "frames": frames,
    }
    _write_json(payload, out_path)
    print(
        f"[audit] Stage5: {len(frames)} frames, "
        f"{sum(len(f['faces']) for f in frames)} detections "
        f"-> {out_path}"
    )
    return payload


# ---------------------------------------------------------------------------
# Stage 10 extraction (InsightFace 2d106det on existing crops)
# ---------------------------------------------------------------------------


def extract_stage10(
    crops_dir: Path,
    stage5_json_path: Path,
    out_path: Path,
    landmark_model_path: Path,
    device: str = "cpu",
    video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run InsightFace 2d106det on the Stage5 crops and map to frame coords.

    ``stage5_json_path`` must be the Stage5 export produced by
    :func:`extract_stage5`. When ``video_path`` is provided the crop->frame
    offset is recovered *exactly* by template matching the crop against its
    source frame (:func:`find_crop_origin`); otherwise it falls back to the
    reproduced ``expanded_bbox`` origin.

    Returns the payload dict (also written to ``out_path``).
    """
    from face_pipeline_v2.models.wrappers.insightface_landmark import (
        InsightFaceLandmarkWrapper,
    )

    stage5: Stage5Data = load_stage5(stage5_json_path)
    by_frame = stage5.by_frame()

    g0, mapping = crop_frame_mapping(crops_dir)
    print(f"[audit] crops: {len(mapping)} files, global offset g0={g0}")

    wrapper = InsightFaceLandmarkWrapper(
        model_path=Path(landmark_model_path), device=device
    )
    wrapper.load()
    print(f"[audit] 2d106det loaded: {wrapper.model_path}")

    fps = stage5.source.get("fps") or 25
    landmarks: List[Dict[str, Any]] = []
    skipped = 0
    matched = 0
    cap = None
    cap_frame = -1
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
    try:
        for local in sorted(mapping):
            g, face_idx, crop_path = mapping[local]
            rec = by_frame.get(local)
            img = cv2.imread(str(crop_path))
            if img is None:
                skipped += 1
                continue
            if rec is None:
                # No Stage5 record for this frame (no face detected on re-run).
                # Skip: landmark would have no offset / frame context.
                skipped += 1
                continue

            crop_lm = wrapper.predict(img)  # (106, 2) crop pixel space

            origin: Optional[Tuple[int, int]] = None
            if cap is not None:
                # advance the video sequentially; the frame consumed at
                # ``local`` becomes the template-matching source.
                frame_for_match = None
                while cap_frame < local:
                    ok_f, f = cap.read()
                    if not ok_f:
                        break
                    cap_frame += 1
                    if cap_frame == local:
                        frame_for_match = f
                if frame_for_match is not None:
                    origin, score = find_crop_origin(frame_for_match, img)
                    if origin is not None:
                        matched += 1
                    elif score >= 0:
                        print(
                            f"[audit] WARN frame {local}: template score "
                            f"{score:.3f} below threshold; using expanded_bbox"
                        )

            if origin is not None:
                frame_lm = np.zeros_like(crop_lm, dtype=np.float64)
                frame_lm[:, 0] = crop_lm[:, 0] + origin[0]
                frame_lm[:, 1] = crop_lm[:, 1] + origin[1]
                crop_origin = origin
            else:
                frame_lm = _map_crop_to_frame(crop_lm, rec, img)
                crop_origin = None

            flat = [round(float(v), 4) for v in frame_lm.flatten()]
            landmarks.append(
                {
                    "frame_index": local,
                    "global_frame_index": g,
                    "timestamp": round(local / fps, 6),
                    "face_id": rec.face_id,
                    "crop_origin": crop_origin,
                    "landmarks_106": flat,
                }
            )
    finally:
        if cap is not None:
            cap.release()

    payload = {
        "source": dict(stage5.source),
        "landmarks": landmarks,
    }
    _write_json(payload, out_path)
    print(
        f"[audit] Stage10: {len(landmarks)} landmarks ({skipped} skipped, "
        f"{matched} template-matched) -> {out_path}"
    )
    return payload



def _map_crop_to_frame(crop_lm: np.ndarray, rec, crop_image: np.ndarray) -> np.ndarray:
    """Map (106,2) crop-space landmarks to frame coords (mirrors pipeline)."""
    eb = rec.expanded_bbox
    out = np.zeros_like(crop_lm, dtype=np.float64)
    if eb and (eb[2] - eb[0]) > 0 and (eb[3] - eb[1]) > 0:
        # 1:1 mapping via expanded_bbox origin (pipeline normal path)
        out[:, 0] = crop_lm[:, 0] + eb[0]
        out[:, 1] = crop_lm[:, 1] + eb[1]
    else:
        # fallback: tight bbox + scale (mirrors pipeline legacy path)
        bbox = rec.bbox
        crop_h, crop_w = crop_image.shape[:2]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw > 0 and bh > 0 and crop_w > 0 and crop_h > 0:
            out[:, 0] = crop_lm[:, 0] * bw / crop_w + bbox[0]
            out[:, 1] = crop_lm[:, 1] * bh / crop_h + bbox[1]
    return out


def _write_json(payload: Dict[str, Any], path: Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )



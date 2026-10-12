"""Regenerate Stage 11 aligned faces on REAL data with a configurable template.

Reuses the existing Stage 5 crops from a previous full Stage 0-11 run
(NO re-detection / NO re-crop). Pipeline used:

  Stage 10 : InsightFace 2d106det landmark inference on the existing crops.
  Stage 11 : align_faces() with the configurable eye template
             (resolve_alignment_template → build_template).

Variants generated per clip:
  default  : no alignment.template / scale_factor in config → legacy template
             (byte-identical output). Verified against the existing
             aligned_faces from the previous run (pixel comparison).
  sf_0.85  : alignment.scale_factor = 0.85  → face shrinks in the canvas.
  sf_1.15  : alignment.scale_factor = 1.15  → face enlarges in the canvas.
  ey_122   : alignment.template.eye_center_y = 122 → face shifts down
             (adds top margin for head-top cropping fixes).
  sf_0.85_ey_122 : both sf_0.85 AND eye_center_y=122 combined.

Also writes contact sheets and side-by-side comparisons for visual judgment.

Usage:
    python3 scripts/regen_aligned_visual_check.py \
        --run outputs/stage0_11_batch/CNFACE_00001_BV18Qg96YE1W_000000 \
        --clip CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000 \
        --out outputs/alignment_visual_check
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from face_pipeline_v2.data.models import (  # noqa: E402
    FaceInfo,
    FaceSequence,
    FrameInfo,
)
from face_pipeline_v2.models.wrappers.insightface_landmark import (  # noqa: E402
    InsightFaceLandmarkWrapper,
)
from face_pipeline_v2.stages.stage_10_landmark.processor import (  # noqa: E402
    extract_landmarks,
)
from face_pipeline_v2.stages.stage_11_alignment.processor import (  # noqa: E402
    align_faces,
    resolve_alignment_template,
)

_FRAME_RE = re.compile(r"^frame_(\d+)_(.+)\.jpg$")
_ALIGNED_RE = re.compile(r"^aligned_(\d+)_.*_(.+)\.jpg$")

logger = logging.getLogger("visual_check")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)


def _build_face_sequence(clip_id: str, crops_dir: Path, fps: float) -> FaceSequence:
    """Rebuild a FaceSequence from existing Stage 5 crop files.

    The crop coordinate space is 1:1 with expanded_bbox (Stage 5 saves raw
    expanded crops). We set bbox == expanded_bbox == (0,0,W,H) so the
    Stage 10 frame-mapping (+ offset) and Stage 11 crop-mapping (- offset)
    cancel exactly — landmarks stay in crop pixel space, identical to what
    the original pipeline used.
    """
    frame_infos = []
    for crop in sorted(crops_dir.glob("frame_*_*.jpg")):
        m = _FRAME_RE.match(crop.name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        face_id = m.group(2)
        img = cv2.imread(str(crop))
        if img is None:
            logger.warning(f"unreadable crop: {crop.name} — skipping")
            continue
        h, w = img.shape[:2]
        face_info = FaceInfo(
            frame_index=frame_idx,
            face_id=face_id,
            bbox=(0.0, 0.0, float(w), float(h)),
            landmark_5=(),
            confidence=1.0,
            face_crop_path=crop,
            track_id="0",
            expanded_bbox=(0.0, 0.0, float(w), float(h)),
        )
        frame_infos.append(
            FrameInfo(
                frame_index=frame_idx,
                clip_id=clip_id,
                timestamp=frame_idx / fps,
                image_path=crop,
                faces=(face_info,),
            )
        )
    frame_infos.sort(key=lambda f: f.frame_index)
    return FaceSequence(
        clip_id=clip_id,
        video_id=clip_id,
        frames=tuple(frame_infos),
    )


def _parse_frame_from_aligned(name: str) -> int | None:
    m = _ALIGNED_RE.match(name)
    return int(m.group(1)) if m else None


def _verify_against_existing(
    regenerated_dir: Path,
    existing_dir: Path,
    label: str,
) -> dict:
    """Compare regenerated default aligned faces to the previous run's output."""
    stats = {"frames": 0, "byte_identical_pixels": 0, "max_abs_diff": 0.0}
    for reg in sorted(regenerated_dir.glob("aligned_*.jpg")):
        frame_idx = _parse_frame_from_aligned(reg.name)
        if frame_idx is None:
            continue
        candidates = [
            p for p in existing_dir.glob(f"aligned_{frame_idx:06d}_*.jpg")
        ]
        if not candidates:
            continue
        a = cv2.imread(str(reg))
        b = cv2.imread(str(candidates[0]))
        if a is None or b is None or a.shape != b.shape:
            continue
        stats["frames"] += 1
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
        stats["max_abs_diff"] = max(stats["max_abs_diff"], float(diff.max()))
        if diff.max() == 0:
            stats["byte_identical_pixels"] += 1
    logger.info(
        f"[verify/{label}] matched={stats['frames']} "
        f"pixel-identical={stats['byte_identical_pixels']} "
        f"max_abs_diff={stats['max_abs_diff']:.0f}"
    )
    return stats


def _make_contact_sheet(aligned_dir: Path, out_path: Path, cols: int = 5) -> None:
    files = sorted(aligned_dir.glob("aligned_*.jpg"))
    if not files:
        return
    step = max(1, len(files) // 20)
    picked = files[::step][:20]
    cells = [cv2.imread(str(p)) for p in picked]
    cells = [c for c in cells if c is not None]
    rows = (len(cells) + cols - 1) // cols
    cell_h, cell_w = cells[0].shape[:2]
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
        cv2.putText(
            canvas, picked[i].name[8:8 + 6], (c * cell_w + 4, r * cell_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    logger.info(f"contact sheet → {out_path}")


def _make_comparison(
    variants: dict[str, Path],
    out_path: Path,
    sample_frames: tuple[int, ...],
    eyes_template: dict[str, np.ndarray],
) -> None:
    """Side-by-side variant panels with eye-line markers."""
    var_names = list(variants.keys())
    for fidx in sample_frames:
        panels = []
        labels = []
        for v in var_names:
            d = variants[v]
            matches = sorted(d.glob(f"aligned_{fidx:06d}_*.jpg"))
            if not matches:
                continue
            img = cv2.imread(str(matches[0]))
            if img is None:
                continue
            eyes = eyes_template.get(v)
            if eyes is not None:
                for (ex, ey) in eyes:
                    cv2.circle(img, (int(ex), int(ey)), 5, (0, 0, 255), -1)
            panels.append(img)
            labels.append(v)
        if len(panels) < 2:
            continue
        gap = 10
        W = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
        H = max(p.shape[0] for p in panels) + 30
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x = 0
        for p, lab in zip(panels, labels):
            canvas[30:30 + p.shape[0], x:x + p.shape[1]] = p
            cv2.putText(canvas, lab, (x + 4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            x += p.shape[1] + gap
        out_path.mkdir(parents=True, exist_ok=True)
        fp = out_path / f"compare_frame_{fidx:06d}.png"
        cv2.imwrite(str(fp), canvas)
        logger.info(f"comparison frame {fidx} → {fp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="previous run output dir")
    ap.add_argument("--clip", required=True, help="clip id to regenerate")
    ap.add_argument("--out", required=True, help="output root dir")
    ap.add_argument("--video-id", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    clip_id = args.clip
    video_id = args.video_id or clip_id.split("_scene_")[0]
    out_root = Path(args.out)

    crops_dir = run_dir / "faces" / video_id / clip_id / "crops"
    existing_dir = run_dir / "aligned_faces" / clip_id
    if not crops_dir.is_dir():
        raise SystemExit(f"crops dir not found: {crops_dir}")

    # --- Stage 10: landmarks on the existing crops ---
    lm_model = InsightFaceLandmarkWrapper(
        model_path=_PROJ / "models" / "insightface" / "2d106det.onnx"
    )
    lm_model.load()
    seq = _build_face_sequence(clip_id, crops_dir, fps=25.0)
    landmarks = extract_landmarks(seq, lm_model, logger)
    logger.info(f"Stage 10: {len(landmarks)} landmarks for {clip_id}")
    if not landmarks:
        raise SystemExit("no landmarks produced — aborting")

    # --- Variants ---
    variants_cfg = {
        "default": {},
        "sf_0.85": {"scale_factor": 0.85},
        "sf_1.15": {"scale_factor": 1.15},
        "ey_122": {"template": {"eye_center_y": 122.0}},
        "sf_0.85_ey_122": {
            "template": {"eye_center_y": 122.0},
            "scale_factor": 0.85,
        },
    }
    variants_out: dict[str, Path] = {}
    eyes_templates: dict[str, np.ndarray] = {}

    report: list[str] = []
    report.append(f"clip={clip_id}")
    report.append(f"landmarks={len(landmarks)}")

    for label, cfg in variants_cfg.items():
        tp = resolve_alignment_template(cfg)
        eyes = tp["eyes"]
        eyes_templates[label] = eyes
        out_dir = out_root / label
        aligned = align_faces(
            landmarks=landmarks,
            face_sequences=[seq],
            output_dir=out_dir,
            output_resolution=(256, 256),
            save_images=True,
            logger=logger,
            eye_template=eyes,
        )
        variants_out[label] = out_dir / clip_id
        report.append(
            f"{label}: eye_distance={tp['eye_distance']:.2f} "
            f"eye_center_y={tp['eye_center_y']:.2f} "
            f"scale_factor={tp['scale_factor']:.3f} "
            f"template_eyes=[({eyes[0,0]:.3f},{eyes[0,1]:.3f}),"
            f"({eyes[1,0]:.3f},{eyes[1,1]:.3f})] "
            f"aligned={len(aligned)}"
        )
        logger.info(report[-1])

    # --- Verification: default vs previous run ---
    if existing_dir.is_dir():
        stats = _verify_against_existing(
            variants_out["default"], existing_dir, "default"
        )
        report.append(
            f"verify: frames={stats['frames']} "
            f"pixel-identical={stats['byte_identical_pixels']} "
            f"max_abs_diff={stats['max_abs_diff']:.0f}"
        )

    # --- Visual artifacts ---
    for label, d in variants_out.items():
        _make_contact_sheet(d, out_root / f"montage_{label}.png")

    sample_frames = tuple(
        int(p.name[8:14]) for p in
        sorted(variants_out["default"].glob("aligned_*.jpg"))[::25][:4]
    )
    _make_comparison(variants_out, out_root, sample_frames, eyes_templates)

    (out_root / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Build human-verification videos for the *face-only* view (video + audio merged).

Fork of ``build_verification_videos.py``: the picture source is the no-audio
face-only mp4 produced by ``export_faces_only_video.py`` (only the frames in
``training_sample.npz``'s ``frame_index``) instead of the full clip / std mp4.
This pairs every frame the mesh was fitted on with the same audio the mesh was
driven by.  Muxes, for every clip that reached Stage12-14 (i.e. every clip with
``stage_inputs/<clip>_av_alignment.json``):

  open_domain mode (config.yaml, Stage 6-9.5 ENABLED):
    <clip>_faces_only.mp4           = face-only video + ACTUAL audio Stage14 consumed
                                       (= stage14_audio/<clip>/audio_metadata.json
                                       source_audio, i.e. stage0_11/audio/<clip>.wav
                                       after the av_fix step; this is the
                                       AV-aligned audio).
    <clip>_faces_only_original.mp4  = face-only video + ORIGINAL (un-aligned) clip
                                       audio, cut from av_alignment.json
                                       original_audio_path at the clip boundary.
                                       Lets reviewers hear the alignment delta.

  dataset mode (config_dataset_mode.yaml, Stage 6-9.5 DISABLED):
    <clip>_faces_only.mp4           = face-only video + original clip audio only.
                                       Stage 7.1 never runs, so aligned == raw
                                       (offset 0, already_aligned). A single
                                       version is produced and the manifest flags
                                       original_equals_aligned=True.

Video source: the face-only frames from the Stage 17 frame index (``frame_index``),
resolved from ``stage0_11/stage17_fusion/<clip>/training_sample.npz`` or, when
absent, ``<sample_root>/stage17_multimodal_export/timestamp/<clip>.npz``.  Frames
are read from the std video and re-encoded to a transient no-audio stream; the
temporary file is deleted immediately after muxing, so no ``*_faces_only.mp4``
intermediate is ever persisted.  Samples without a Stage17 export are skipped
(not reported as errors), since only they carry the face ``frame_index``.

Output (per sample root):
    <sample_root>/verification_face_only/<clip>_faces_only.mp4
    <sample_root>/verification_face_only/<clip>_faces_only_original.mp4   (open_domain only)
    <sample_root>/verification_face_only/face_only_csv/verify_manifest.csv

Usage:
    python3 scripts/build_verification_face_only_videos.py \
        --sample-root outputs/.../video_<id>_open_domain
    # batch over many samples:
    python3 scripts/build_verification_face_only_videos.py \
        --glob 'outputs/chinese_talking_videos_bilibili/video_*_open_domain' \
        --glob 'outputs/chinese_002_24_F_YS/video_*_FACE'
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_verification_face_only_videos")

AUDIO_SAMPLE_RATE = 16000          # Stage 1 extracts 16 kHz mono pcm_s16le
MUX_AUDIO_AR = 48000               # aac sample rate for player compatibility
MUX_AUDIO_BITRATE = "192k"
VIDEO_FPS = 25                     # standardization target (both modes)


def _run(cmd: List[str]) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    logger.debug("  cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"stdout={proc.stdout.decode(errors='replace')[-800:]}\n"
            f"stderr={proc.stderr.decode(errors='replace')[-800:]}"
        )


# --------------------------------------------------------------------------- #
# mode / input discovery
# --------------------------------------------------------------------------- #

def _detect_mode(sample_root: Path) -> str:
    """Detect pipeline mode from results json, else from clips/ presence."""
    for name, mode in (
        ("stage0_11_results_open_domain.json", "open_domain"),
        ("stage0_11_results_dataset.json", "dataset"),
    ):
        p = sample_root / name
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                m = d.get("pipeline_mode")
                if m in ("open_domain", "dataset"):
                    return m
            except Exception:
                pass
            return mode
    # Fallback: open_domain produces clips/ (Stage 9.5), dataset does not.
    clips_dir = sample_root / "stage0_11" / "clips"
    if clips_dir.is_dir() and any(clips_dir.glob("*.mp4")):
        return "open_domain"
    return "dataset"


def _read_av_alignment(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _has_stage17_export(sample_root: Path) -> bool:
    """True when the sample produced a Stage17 multimodal export.

    The face-only view needs the per-frame ``frame_index`` that Stage17 writes
    (``stage17_multimodal_export/timestamp/<clip>.npz``, with the
    ``stage0_11/stage17_fusion/<clip>/training_sample.npz`` fallback).  Only
    samples with a Stage17 export are "passed"; everything else is skipped here
    instead of being reported as a mux error.
    """
    return (sample_root / "stage17_multimodal_export").is_dir()


def _read_stage14_metadata(sample_root: Path, clip_id: str) -> Optional[Dict[str, Any]]:
    p = sample_root / "stage0_11" / "stage14_audio" / clip_id / "audio_metadata.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_frame_index_npz(sample_root: Path, clip_id: str) -> Path:
    """Return the npz that carries ``frame_index`` for a clip.

    ``frame_index`` lists the face frames as 0-based frame numbers.  Depending on
    which stage produced it, it may be clip-relative (``training_sample.npz``) or
    std-video absolute (``stage17_multimodal_export/timestamp``).
    ``_build_faces_only_video`` disambiguates the two using the clip start frame.
    """
    candidates = [
        sample_root / "stage0_11" / "stage17_fusion" / clip_id / "training_sample.npz",
        sample_root / "stage17_multimodal_export" / "timestamp" / f"{clip_id}.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"no frame_index npz for {clip_id} under "
        f"{sample_root}/stage0_11/stage17_fusion/ or "
        f"{sample_root}/stage17_multimodal_export/timestamp/"
    )


def _find_std_video(sample_root: Path) -> Path:
    videos_dir = sample_root / "stage0_11" / "videos"
    for cand in sorted(videos_dir.glob("*/*_std.mp4")):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no *_std.mp4 under {videos_dir}")


def _build_faces_only_video(
    sample_root: Path,
    clip_id: str,
    fps: float,
    clip_start_frame: int,
    tmp_dir: Path,
    dry_run: bool,
) -> Tuple[Path, str, float]:
    """Generate the no-audio face-only mp4 into ``tmp_dir``; return (path, desc, duration_s).

    The mp4 is a transient intermediate deleted by the caller right after muxing,
    so nothing is persisted to disk.
    """
    npz_path = _resolve_frame_index_npz(sample_root, clip_id)
    std_video = _find_std_video(sample_root)
    desc = f"frame_index={npz_path.name}, video={std_video.name}"
    dest = tmp_dir / f"{clip_id}_faces_only.mp4"
    if dry_run:
        return dest, desc, 0.0

    with np.load(npz_path, allow_pickle=False) as data:
        frame_index = data["frame_index"].astype(np.int64)
    fi = np.sort(frame_index)
    if fi.size == 0:
        raise RuntimeError(f"empty frame_index in {npz_path}")

    # frame_index may be clip-relative (starts near 0) or std-video absolute.
    absolute = fi if int(fi.min()) >= clip_start_frame else fi + clip_start_frame
    keep = sorted({int(k) for k in absolute if int(k) >= 0})

    cap = cv2.VideoCapture(str(std_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {std_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", f"{fps}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-r", f"{fps}",
        str(dest),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    keep_ptr = 0
    idx = 0
    written = 0
    while keep_ptr < len(keep):
        ret, frame = cap.read()
        if not ret:
            break
        target = keep[keep_ptr]
        if idx == target:
            proc.stdin.write(frame.tobytes())
            written += 1
            keep_ptr += 1
        elif idx > target:
            keep_ptr += 1
            continue
        idx += 1

    cap.release()
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg failed while generating the face-only video")
    if written == 0:
        raise RuntimeError(
            f"no face frames extracted for {clip_id} "
            f"(frame_index does not match {std_video.name})"
        )
    if keep_ptr < len(keep):
        logger.warning(
            "  [warn] %s: %d of %d face frames found in %s",
            clip_id, written, len(keep), std_video.name,
        )
    return dest, desc, written / fps



def _mux(
    video: Path,
    audio: Path,
    dest: Path,
    cut_audio: bool,
    cut_start: float,
    cut_duration: float,
    video_duration: float,
    force: bool,
) -> None:
    """Mux one video with one audio track into an mp4.

    The audio is silence-padded (``apad``) and the output is cut at
    ``video_duration`` (``-t``) — *not* at the audio end — so the muxed mp4
    keeps every video frame.  That makes ``face_only.mp4``'s frame count
    strictly equal to ``frame_index`` (== flame rows) instead of losing the
    tail frame when the audio ends a fraction of a frame earlier than the
    video.  Note: ``-shortest`` must NOT be combined with ``apad`` here —
    in this ffmpeg build that combination hangs forever (audio never EOFs).
    """
    if dest.exists() and not force:
        logger.info("  [skip] %s exists (use --force to overwrite)", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    if cut_audio:
        cmd += ["-ss", f"{cut_start:.6f}", "-t", f"{cut_duration:.6f}"]
    cmd += ["-i", str(audio), "-i", str(video)]
    cmd += [
        "-map", "0:a:0", "-map", "1:v:0",
        "-c:v", "copy",
        "-c:a", "aac", "-ar", str(MUX_AUDIO_AR), "-b:a", MUX_AUDIO_BITRATE,
        "-af", "apad",
        "-t", f"{video_duration:.6f}",
        str(dest),
    ]
    _run(cmd)


# --------------------------------------------------------------------------- #
# per-sample build
# --------------------------------------------------------------------------- #

MANIFEST_FIELDS = [
    "clip_id", "pipeline_mode", "alignment_status", "offset_ms",
    "correction_applied", "original_equals_aligned",
    "video_source", "audio_used", "audio_original",
    "muxed_actual", "muxed_original",
]


def build_sample(
    sample_root: Path, out_dir: Path, force: bool, dry_run: bool
) -> Dict[str, Any]:
    mode = _detect_mode(sample_root)
    logger.info("== sample: %s  mode=%s", sample_root, mode)

    if not _has_stage17_export(sample_root):
        logger.info("  no stage17_multimodal_export — skipping non-passed sample")
        return {"mode": mode, "clips": 0, "muxed": 0, "errors": 0}

    stage0_11 = sample_root / "stage0_11"
    av_dir = stage0_11 / "stage_inputs"
    av_files = sorted(av_dir.glob("*_av_alignment.json"))
    if not av_files:
        logger.warning("  no stage_inputs/*_av_alignment.json — nothing to build")
        return {"mode": mode, "clips": 0, "muxed": 0, "errors": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="verification_face_only_"))
    rows: List[Dict[str, Any]] = []
    muxed_total = 0
    errors = 0

    for av_path in av_files:
        clip_id = av_path.name[: -len("_av_alignment.json")]
        logger.info("[%s] %s", mode, clip_id)
        try:
            av = _read_av_alignment(av_path)
            clip_start = float(av.get("clip_start") or 0.0)
            clip_duration = float(av.get("clip_duration") or 0.0)
            original_audio = Path(av["original_audio_path"])
            if not original_audio.exists():
                # The metadata may reference a stale/moved root (e.g. the
                # chinese_talking_videos_bilibili_0904 dir was renamed to
                # chinese_talking_videos_bilibili).  Fall back to the
                # full-length Stage 1 audio under the current sample.
                cands = sorted((sample_root / "stage0_11" / "videos").glob("*/*_audio.wav"))
                if cands:
                    original_audio = cands[0]
            status = av.get("alignment_status", "unknown")
            offset_ms = av.get("offset_ms")
            correction = bool(av.get("correction_applied", False))

            st14 = _read_stage14_metadata(sample_root, clip_id)
            used_audio: Optional[Path] = None
            if st14 and st14.get("source_audio"):
                cand = Path(st14["source_audio"])
                if cand.exists():
                    used_audio = cand
            if used_audio is None:
                # fallback: av json aligned/corrected path
                for key in ("corrected_audio_path", "aligned_audio_path", "raw_audio_path"):
                    v = av.get(key)
                    if v and Path(v).exists():
                        used_audio = Path(v)
                        break
            if used_audio is None:
                # Stale metadata path (e.g. *_bilibili_0904 moved to
                # *_bilibili).  Fall back to the aligned clip audio inside the
                # current sample.
                cand = sample_root / "stage0_11" / "audio" / f"{clip_id}.wav"
                if cand.exists():
                    used_audio = cand
            if used_audio is None:
                raise FileNotFoundError(
                    f"no usable audio for Stage14 input of {clip_id} "
                    "(source_audio missing)"
                )

            fps = float(av.get("fps") or VIDEO_FPS)
            clip_start_frame = int(round(clip_start * fps))
            video, video_desc, video_duration = _build_faces_only_video(
                sample_root, clip_id, fps, clip_start_frame, tmp_dir, dry_run=dry_run
            )

            row: Dict[str, Any] = {
                "clip_id": clip_id,
                "pipeline_mode": mode,
                "alignment_status": status,
                "offset_ms": offset_ms,
                "correction_applied": correction,
                "original_equals_aligned": False,
                "video_source": video_desc,
                "audio_used": str(used_audio),
                "audio_original": str(original_audio),
                "muxed_actual": "",
                "muxed_original": "",
            }

            actual_mp4 = out_dir / f"{clip_id}_faces_only.mp4"
            if not dry_run:
                _mux(video, used_audio, actual_mp4, cut_audio=False,
                     cut_start=0.0, cut_duration=0.0, video_duration=video_duration,
                     force=force)
            row["muxed_actual"] = f"verification_face_only/{clip_id}_faces_only.mp4"
            muxed_total += 1

            if mode == "open_domain":
                # Original (un-aligned) version — cut from the full-length
                # Stage 1 audio at the clip boundary.  Skipped when the
                # clip is already aligned (audio identical by construction).
                if status == "already_aligned" and not correction:
                    row["original_equals_aligned"] = True
                else:
                    orig_mp4 = out_dir / f"{clip_id}_faces_only_original.mp4"
                    if not dry_run:
                        _mux(video, original_audio, orig_mp4, cut_audio=True,
                             cut_start=clip_start, cut_duration=clip_duration,
                             video_duration=video_duration, force=force)
                    row["muxed_original"] = f"verification_face_only/{clip_id}_faces_only_original.mp4"
                    muxed_total += 1
            else:
                # dataset mode: Stage 7.1 never ran -> aligned == raw.
                row["original_equals_aligned"] = True

            video.unlink(missing_ok=True)
            rows.append(row)
        except Exception as e:  # noqa: BLE001 — per-clip isolation
            errors += 1
            logger.error("  [FAIL] %s: %s", clip_id, e)
            rows.append({
                "clip_id": clip_id,
                "pipeline_mode": mode,
                "alignment_status": "",
                "offset_ms": "",
                "correction_applied": "",
                "original_equals_aligned": "",
                "video_source": "",
                "audio_used": "",
                "audio_original": "",
                "muxed_actual": "",
                "muxed_original": "",
            })

    manifest_path = out_dir / "face_only_csv" / "verify_manifest.csv"
    if rows:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("  manifest -> %s", manifest_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("  done: clips=%d muxed=%d errors=%d", len(rows), muxed_total, errors)
    return {"mode": mode, "clips": len(rows), "muxed": muxed_total, "errors": errors}


def _expand_globs(patterns: List[str]) -> List[Path]:
    from glob import glob

    roots: List[Path] = []
    for pat in patterns:
        for hit in sorted(glob(str(PROJECT_ROOT / pat))):
            p = Path(hit)
            if p.is_dir() and (p / "stage0_11").is_dir() and _has_stage17_export(p):
                roots.append(p)
    return roots


def _process_root(
    root: Path, out_dir_override: Optional[str], force: bool, dry_run: bool
) -> Dict[str, Any]:
    """Top-level per-root worker (picklable for ProcessPoolExecutor)."""
    sample_root = root / "stage0_11" if (root / "stage0_11").is_dir() else root
    sample_root = sample_root.parent
    out_dir = Path(out_dir_override) if out_dir_override else (sample_root / "verification_face_only")
    return build_sample(sample_root, out_dir, force, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", action="append", default=[],
                        help="sample output root (repeatable)")
    parser.add_argument("--glob", action="append", default=[],
                        help="glob (relative to project root) matching sample roots (repeatable)")
    parser.add_argument("--out-dir",
                        help="override output dir (default: <sample_root>/verification_face_only)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing muxed files")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only; do not run ffmpeg")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel sample workers (default 1)")
    args = parser.parse_args()

    roots = [Path(r) for r in args.sample_root] + _expand_globs(args.glob)
    if not roots:
        parser.error("at least one --sample-root or --glob is required")
    roots = [r if r.is_absolute() else PROJECT_ROOT / r for r in roots]
    logger.info("resolved %d sample roots", len(roots))

    results: List[Dict[str, Any]]
    if args.jobs > 1:
        import concurrent.futures
        from functools import partial

        worker = partial(
            _process_root,
            out_dir_override=args.out_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(worker, roots))
    else:
        results = [_process_root(r, args.out_dir, args.force, args.dry_run) for r in roots]

    totals = {"muxed": 0, "errors": 0}
    for res in results:
        totals["muxed"] += res["muxed"]
        totals["errors"] += res["errors"]

    print("\n===== BUILD VERIFICATION FACE-ONLY VIDEOS: %d muxed, %d errors =====" % (
        totals["muxed"], totals["errors"]))
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())





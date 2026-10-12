#!/usr/bin/env python3
"""Build human-verification videos (video + audio merged) for Stage0-17 samples.

Colleagues reviewing meshed faces need a playable mp4 that pairs the clip
picture with the SAME audio the mesh was driven by.  This script muxes, for
every clip that reached Stage12-14 (i.e. every clip with
``stage_inputs/<clip>_av_alignment.json``):

  open_domain mode (config.yaml, Stage 6-9.5 ENABLED):
    <clip>.mp4           = video + ACTUAL audio Stage14 consumed
                           (= stage14_audio/<clip>/audio_metadata.json source_audio,
                           i.e. stage0_11/audio/<clip>.wav after the av_fix step;
                           this is the AV-aligned audio).
    <clip>_original.mp4  = same video + ORIGINAL (un-aligned) clip audio,
                           cut from av_alignment.json original_audio_path
                           (= stage0_11/videos/<id>/<id>_audio.wav) at the clip
                           boundary.  Lets reviewers hear the alignment delta.

  dataset mode (config_dataset_mode.yaml, Stage 6-9.5 DISABLED):
    <clip>.mp4           = video + original clip audio only.  Stage 7.1 never
                           runs, so aligned == raw (offset 0, already_aligned).
                           A single version is produced and the manifest flags
                           original_equals_aligned=True.

Video source: ``stage0_11/clips/<clip>.mp4`` when present (open_domain),
otherwise (dataset) cut from ``stage0_11/videos/<id>/<id>_std.mp4`` at the
clip boundary (clip_start/clip_duration from av_alignment.json).

Output (per sample root):
    <sample_root>/verification/<clip>.mp4
    <sample_root>/verification/<clip>_original.mp4     (open_domain only)
    <sample_root>/verification/verify_manifest.csv

Usage:
    python3 scripts/build_verification_videos.py \
        --sample-root outputs/.../video_<id>_open_domain
    # batch over many samples:
    python3 scripts/build_verification_videos.py \
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_verification_videos")

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


def _read_stage14_metadata(sample_root: Path, clip_id: str) -> Optional[Dict[str, Any]]:
    p = sample_root / "stage0_11" / "stage14_audio" / clip_id / "audio_metadata.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _find_std_video(sample_root: Path) -> Optional[Path]:
    videos_root = sample_root / "stage0_11" / "videos"
    if not videos_root.is_dir():
        return None
    candidates = list(videos_root.glob("*/*_std.mp4"))
    return candidates[0] if candidates else None


def _resolve_video_source(
    sample_root: Path,
    clip_id: str,
    clip_start: float,
    clip_duration: float,
    tmp_dir: Path,
    force: bool,
    dry_run: bool,
) -> Tuple[Path, str, bool]:
    """Return (video_path, description, was_cut).

    Prefer the Stage 9.5 clip mp4 (open_domain); otherwise cut the Stage 1
    std.mp4 at the clip boundary (dataset mode).
    """
    clips_path = sample_root / "stage0_11" / "clips" / f"{clip_id}.mp4"
    if clips_path.exists():
        return clips_path, f"clips/{clip_id}.mp4", False

    std_video = _find_std_video(sample_root)
    if std_video is None:
        raise FileNotFoundError(
            f"no clips/{clip_id}.mp4 and no videos/*/*_std.mp4 under {sample_root}"
        )
    cut_path = tmp_dir / f"{clip_id}.mp4"
    if dry_run:
        return cut_path, f"std.mp4@{clip_start:.2f}+{clip_duration:.2f}s (cut)", True
    if force or not cut_path.exists():
        logger.info("  [video] cutting std.mp4 -> %s", cut_path.name)
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{clip_start:.6f}", "-t", f"{clip_duration:.6f}",
            "-i", str(std_video),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an",
            str(cut_path),
        ])
    return cut_path, f"std.mp4@{clip_start:.2f}+{clip_duration:.2f}s (cut)", True



def _mux(
    video: Path,
    audio: Path,
    dest: Path,
    cut_audio: bool,
    cut_start: float,
    cut_duration: float,
    force: bool,
) -> None:
    """Mux one video with one audio track into an mp4."""
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
        "-shortest",
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

    stage0_11 = sample_root / "stage0_11"
    av_dir = stage0_11 / "stage_inputs"
    av_files = sorted(av_dir.glob("*_av_alignment.json"))
    if not av_files:
        logger.warning("  no stage_inputs/*_av_alignment.json — nothing to build")
        return {"mode": mode, "clips": 0, "muxed": 0, "errors": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="verify_", dir=str(out_dir)))
    rows: List[Dict[str, Any]] = []
    muxed_total = 0
    errors = 0

    try:
        for av_path in av_files:
            clip_id = av_path.name[: -len("_av_alignment.json")]
            logger.info("[%s] %s", mode, clip_id)
            try:
                av = _read_av_alignment(av_path)
                clip_start = float(av.get("clip_start") or 0.0)
                clip_duration = float(av.get("clip_duration") or 0.0)
                original_audio = Path(av["original_audio_path"])
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
                    raise FileNotFoundError(
                        f"no usable audio for Stage14 input of {clip_id} "
                        "(source_audio missing)"
                    )

                video, video_desc, _was_cut = _resolve_video_source(
                    sample_root, clip_id, clip_start, clip_duration, tmp_dir,
                    force, dry_run
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

                actual_mp4 = out_dir / f"{clip_id}.mp4"
                if not dry_run:
                    _mux(video, used_audio, actual_mp4, cut_audio=False,
                         cut_start=0.0, cut_duration=0.0, force=force)
                row["muxed_actual"] = f"verification/{clip_id}.mp4"
                muxed_total += 1

                if mode == "open_domain":
                    # Original (un-aligned) version — cut from the full-length
                    # Stage 1 audio at the clip boundary.  Skipped when the
                    # clip is already aligned (audio identical by construction).
                    if status == "already_aligned" and not correction:
                        row["original_equals_aligned"] = True
                    else:
                        orig_mp4 = out_dir / f"{clip_id}_original.mp4"
                        if not dry_run:
                            _mux(video, original_audio, orig_mp4, cut_audio=True,
                                 cut_start=clip_start, cut_duration=clip_duration,
                                 force=force)
                        row["muxed_original"] = f"verification/{clip_id}_original.mp4"
                        muxed_total += 1
                else:
                    # dataset mode: Stage 7.1 never ran -> aligned == raw.
                    row["original_equals_aligned"] = True

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest_path = out_dir / "verify_manifest.csv"
    if rows:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("  manifest -> %s", manifest_path)

    logger.info("  done: clips=%d muxed=%d errors=%d", len(rows), muxed_total, errors)
    return {"mode": mode, "clips": len(rows), "muxed": muxed_total, "errors": errors}


def _expand_globs(patterns: List[str]) -> List[Path]:
    from glob import glob

    roots: List[Path] = []
    for pat in patterns:
        for hit in sorted(glob(str(PROJECT_ROOT / pat))):
            p = Path(hit)
            if p.is_dir() and (p / "stage0_11").is_dir():
                roots.append(p)
    return roots


def _process_root(
    root: Path, out_dir_override: Optional[str], force: bool, dry_run: bool
) -> Dict[str, Any]:
    """Top-level per-root worker (picklable for ProcessPoolExecutor)."""
    sample_root = root / "stage0_11" if (root / "stage0_11").is_dir() else root
    sample_root = sample_root.parent
    out_dir = Path(out_dir_override) if out_dir_override else (sample_root / "verification")
    return build_sample(sample_root, out_dir, force, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", action="append", default=[],
                        help="sample output root (repeatable)")
    parser.add_argument("--glob", action="append", default=[],
                        help="glob (relative to project root) matching sample roots (repeatable)")
    parser.add_argument("--out-dir",
                        help="override output dir (default: <sample_root>/verification)")
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

    print("\n===== BUILD VERIFICATION VIDEOS: %d muxed, %d errors =====" % (
        totals["muxed"], totals["errors"]))
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())





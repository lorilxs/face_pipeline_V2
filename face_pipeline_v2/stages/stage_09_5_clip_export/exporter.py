"""Clip export logic for Stage 9.5.

Exports Stage-9-accepted clips as final training data assets:
    - clips/{clip_id}.mp4            — video clip (cut from Stage 1 video)
    - audio/{clip_id}.wav            — canonical aligned clip audio
                                       (Stage 7.1 aligned_audio_path; the raw
                                       clip audio is NOT exported as a final asset)
    - manifest entry + Audio Alignment Validation Report (auto)
    - (optional) removes the Stage 7.1 intermediate audio dirs
      (stage_outputs/av_alignment/<clip_id>/) after a successful export
      (config: audio_alignment.cleanup_after_export=true).

FFmpeg-only, no models, no GPU.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from face_pipeline_v2.data.models import (
    AVAlignmentInfo,
    AVSyncInfo,
    ClipInfo,
    QualityMetrics,
)


def export_accepted_clips(
    accepted_metrics: List[QualityMetrics],
    clip_by_id: Dict[str, ClipInfo],
    align_by_id: Dict[str, AVAlignmentInfo],
    std_video_path: Path,
    clips_dir: Path,
    audio_dir: Path,
    manifest_path: Path,
    sync_by_id: Optional[Dict[str, AVSyncInfo]] = None,
    logger: Any = None,
    validate: bool = True,
    tolerance_ms: float = 50.0,
    validation_report_path: Optional[Path] = None,
    cleanup_intermediate: bool = False,
) -> Dict[str, Any]:
    """Export accepted clips with their clip-level audio assets.

    Only exports clips where QualityMetrics.is_accepted=True.
    Per accepted clip two files are produced, keyed by clip_id:
        clips/{clip_id}.mp4            — video cut from Stage 1 video
        audio/{clip_id}.wav            — canonical aligned clip audio
    The aligned audio is the Stage 7.1 offset-corrected clip audio
    (AVAlignmentInfo.aligned_audio_path); the raw clip audio is intentionally
    NOT exported as a final asset (it stays only as the Stage 7.1 intermediate
    under stage_outputs/av_alignment/<clip_id>/ for debugging).  The aligned
    audio has the same duration as the clip video (validated automatically).

    Args:
        accepted_metrics: Accepted QualityMetrics from Stage 9.
        clip_by_id: clip_id → ClipInfo lookup.
        align_by_id: clip_id → AVAlignmentInfo lookup (Stage 7.1).
        std_video_path: Path to Stage 1 standardized video.
        clips_dir: Output directory for video clips.
        audio_dir: Output directory for the final canonical (aligned) clip
            audio files.
        manifest_path: Path for manifest.json output.
        sync_by_id: Optional clip_id → AVSyncInfo lookup (for LSE-C in manifest).
        logger: Optional PipelineLogger.
        validate: If True, run the Audio Alignment Validation Report after export.
        tolerance_ms: Allowed video/audio duration mismatch (ms).
        validation_report_path: Optional JSON path for the validation report.
        cleanup_intermediate: If True, after export remove each Stage 7.1
            intermediate clip dir (stage_outputs/av_alignment/<clip_id>/) whose
            video + aligned audio were both copied successfully.  Final asset
            dirs (clips/audio) and the output root are protected and are never
            deleted; any clip that was not fully exported keeps its
            intermediate audio.

    Returns:
        Dict with export summary: accepted_count, exported_videos,
        exported_audio, audio_unavailable, failures, validation, and cleanup
        (intermediate-cleanup stats when cleanup_intermediate=True).
    """
    clips_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    exported_videos = 0
    exported_audio = 0
    audio_unavailable = 0
    failures: List[str] = []
    manifest_entries: List[Dict[str, Any]] = []
    # (clip_id, Stage 7.1 intermediate dir) for clips exported with all three
    # assets (video + raw audio + aligned audio) — candidates for cleanup.
    cleanup_targets: List[Tuple[str, Path]] = []

    for qm in accepted_metrics:
        clip_id = qm.clip_id
        clip = clip_by_id.get(clip_id)
        if clip is None:
            failures.append(f"{clip_id}: no ClipInfo")
            continue

        # --- 1. Export video clip ---
        video_path = clips_dir / f"{clip_id}.mp4"
        try:
            _cut_video_segment(std_video_path, video_path,
                               clip.start_time, clip.duration)
            exported_videos += 1
        except Exception as e:
            failures.append(f"{clip_id}: video export failed - {e}")
            continue

        # --- 2. Export clip-level audio asset (canonical = aligned only) ---
        # audio/{clip_id}.wav = aligned clip audio (Stage 7.1 aligned_audio_path).
        # The raw clip audio is NOT exported as a final asset — it stays only as
        # the Stage 7.1 intermediate under stage_outputs/av_alignment/<clip_id>/.
        audio_path = audio_dir / f"{clip_id}.wav"
        al = align_by_id.get(clip_id)
        audio_ok = False
        if al and al.aligned_audio_path and al.aligned_audio_path.exists():
            try:
                shutil.copy2(str(al.aligned_audio_path), str(audio_path))
                audio_ok = True
            except Exception as e:
                failures.append(f"{clip_id}: aligned audio copy failed - {e}")
        if not audio_ok:
            # aligned_audio_path missing → severe offset / missing clip boundaries →
            # audio unavailable (expected for severe_offset, not an error).
            audio_unavailable += 1
        else:
            exported_audio += 1

        # Only clips where video + aligned audio both copied successfully are
        # eligible for Stage 7.1 intermediate cleanup (done after export).
        if (
            cleanup_intermediate
            and al
            and audio_ok
            and al.aligned_audio_path is not None
        ):
            cleanup_targets.append((clip_id, al.aligned_audio_path.parent))

        # --- 3. Build manifest entry ---
        entry: Dict[str, Any] = {
            "clip_id": clip_id,
            "video_path": str(video_path),
            # Single canonical audio asset — always the aligned clip audio.
            "audio_path": str(audio_path) if audio_ok else None,
            "duration": round(clip.duration, 2),
            "fps": 25,
            "quality_score": round(qm.fusion_score, 6),
            "pose_score": qm.pose_score,
            "face_quality": qm.quality_score,
        }
        # Add sync info if available
        if sync_by_id:
            s = sync_by_id.get(clip_id)
            if s:
                entry["sync_offset"] = s.offset
                entry["sync_lse_c"] = round(s.sync_lse_c, 4)
        # Add AV alignment info if available
        if al:
            entry["audio_offset_frames"] = al.offset_frames
            entry["audio_correction_applied"] = al.correction_applied

        manifest_entries.append(entry)

        if logger:
            if audio_ok:
                logger.info(
                    f"[Stage 9.5] {clip_id}: exported "
                    f"video={video_path.name} ({clip.duration:.1f}s) "
                    f"audio={audio_path.name}"
                )
            else:
                logger.info(
                    f"[Stage 9.5] {clip_id}: exported video only "
                    f"(audio_unavailable)"
                )

    # --- Write manifest ---
    if manifest_entries:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2, ensure_ascii=False)

    # --- Audio Alignment Validation Report (auto, after export) ---
    validation = None
    if validate:
        accepted_ids = {qm.clip_id for qm in accepted_metrics}
        validation = validate_exported_audio(
            clips_dir=clips_dir,
            audio_dir=audio_dir,
            clip_by_id=clip_by_id,
            accepted_ids=accepted_ids,
            tolerance_ms=tolerance_ms,
            logger=logger,
        )
        report_path = validation_report_path or (
            manifest_path.parent / "audio_alignment_validation_report.json"
        )
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(validation, f, indent=2, ensure_ascii=False)
        except Exception as e:  # report write must not fail the export
            failures.append(f"validation report write failed - {e}")
        if logger:
            logger.info(
                f"[Stage 9.5] Audio Alignment Validation Report: "
                f"{validation['passed']}/{validation['total_clips']} clips passed "
                f"(tolerance={tolerance_ms:.0f}ms) → {report_path}"
            )
            for row in validation["clips"]:
                if row["status"] != "pass":
                    logger.warning(
                        f"[Stage 9.5] validation {row['clip_id']}: "
                        f"video={row['video_duration']:.3f}s "
                        f"audio={row['audio_duration']} "
                        f"max_diff_ms={row['max_diff_ms']} → {row['status']}"
                    )

    # --- Stage 7.1 intermediate cleanup (after successful export) ---
    cleanup: Dict[str, Any] = {
        "cleanup_after_export": cleanup_intermediate,
        "intermediate_found": 0,
        "fully_exported": 0,
        "removed": 0,
        "removed_paths": [],
        "kept": [],
    }
    if cleanup_intermediate:
        protected_dirs: List[Path] = [clips_dir, audio_dir]
        if manifest_path is not None:
            protected_dirs.append(manifest_path.parent)
        cleanup = _cleanup_stage71_intermediates(
            align_by_id=align_by_id,
            cleanup_targets=cleanup_targets,
            protected_dirs=protected_dirs,
            logger=logger,
        )

    return {
        "accepted_count": len(accepted_metrics),
        "exported_videos": exported_videos,
        "exported_audio": exported_audio,
        "audio_unavailable": audio_unavailable,
        "manifest_entries": len(manifest_entries),
        "failures": failures,
        "validation": validation,
        "cleanup": cleanup,
    }


def _cleanup_stage71_intermediates(
    align_by_id: Dict[str, AVAlignmentInfo],
    cleanup_targets: List[Tuple[str, Path]],
    protected_dirs: List[Path],
    logger: Any = None,
) -> Dict[str, Any]:
    """Remove Stage 7.1 per-clip intermediate audio after successful export.

    Only clip dirs that were fully exported by Stage 9.5 (video + aligned
    audio all copied) are removed.  Final asset dirs (clips/audio) and the
    output root are always protected and never deleted.

    Args:
        align_by_id: clip_id → AVAlignmentInfo (for the "before" inventory of
            Stage 7.1 intermediate dirs present on disk).
        cleanup_targets: (clip_id, intermediate_dir) of fully exported clips.
        protected_dirs: Final asset dirs / output root that must never be
            removed.
        logger: Optional PipelineLogger.

    Returns:
        Dict with cleanup stats: cleanup_after_export, intermediate_found,
        fully_exported, removed, removed_paths, kept.
    """
    # Inventory of distinct Stage 7.1 intermediate dirs still on disk (all
    # aligned clips, not only accepted/exported ones) for the "before" log.
    seen: set[Path] = set()
    for al in align_by_id.values():
        for p in (al.raw_audio_path, al.aligned_audio_path):
            if p is None:
                continue
            try:
                d = p.parent
            except (OSError, ValueError):
                continue
            if d not in seen and d.is_dir():
                seen.add(d)
    intermediate_found = len(seen)

    removed_paths: List[str] = []
    kept: List[Dict[str, str]] = []
    for clip_id, clip_dir in cleanup_targets:
        if _safe_remove_intermediate_dir(clip_dir, protected_dirs, logger):
            removed_paths.append(str(clip_dir))
            if logger:
                logger.info(f"[Stage 9.5] cleanup removed: {clip_dir}")
        else:
            kept.append({
                "clip_id": clip_id,
                "path": str(clip_dir),
                "reason": "removal blocked (protected / missing / not a dir)",
            })
            if logger:
                logger.warning(
                    f"[Stage 9.5] cleanup kept {clip_dir} ({clip_id})"
                )

    if logger:
        logger.info(
            f"[Stage 9.5] Intermediate cleanup (cleanup_after_export=True): "
            f"Stage7.1 temporary audio before={intermediate_found} clip(s), "
            f"fully exported={len(cleanup_targets)}, "
            f"removed={len(removed_paths)}, kept={len(kept)}, "
            f"left={intermediate_found - len(removed_paths)}"
        )

    return {
        "cleanup_after_export": True,
        "intermediate_found": intermediate_found,
        "fully_exported": len(cleanup_targets),
        "removed": len(removed_paths),
        "removed_paths": removed_paths,
        "kept": kept,
        "left": intermediate_found - len(removed_paths),
    }


def _safe_remove_intermediate_dir(
    clip_dir: Path,
    protected_dirs: List[Path],
    logger: Any = None,
) -> bool:
    """Remove one Stage 7.1 intermediate clip dir, guarding final assets.

    Refuses to delete (returns False):
      - a path that does not exist or is not a directory;
      - any protected dir (final clips/audio dirs, output root);
      - filesystem anchors or their direct children (e.g. /, /tmp);
      - the current working directory or the user home directory.

    Returns True only if the directory was actually removed.
    """
    if clip_dir is None or not clip_dir.exists() or not clip_dir.is_dir():
        return False
    resolved = clip_dir.resolve()
    anchor = Path(resolved.anchor)

    # 1) Never remove a protected final-asset / output-root dir.
    for p in protected_dirs:
        if p is None:
            continue
        try:
            rp = p.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == rp:
            if logger:
                logger.warning(
                    f"[Stage 9.5] cleanup protection: refusing to delete "
                    f"protected dir {clip_dir}"
                )
            return False

    # 2) Never remove anchors, their direct children, cwd or home.
    if resolved == anchor:
        return False
    if resolved.parent == anchor:
        return False
    if resolved == Path.cwd().resolve() or resolved == Path.home().resolve():
        return False

    # 3) Basename sanity — must look like a per-clip dir.
    if not clip_dir.name or clip_dir.name in {".", ".."}:
        return False

    try:
        shutil.rmtree(str(clip_dir))
    except Exception as e:
        if logger:
            logger.warning(f"[Stage 9.5] cleanup failed for {clip_dir}: {e}")
        return False
    return True


def _cut_video_segment(
    source: Path,
    dest: Path,
    start_time: float,
    duration: float,
) -> None:
    """Cut a video segment using ffmpeg.

    Args:
        source: Source video path.
        dest: Destination path.
        start_time: Start time in seconds.
        duration: Duration in seconds.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_time:.6f}",
        "-i", str(source),
        "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(dest),
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)


def _probe_duration(path: Path) -> Optional[float]:
    """Return media duration in seconds by parsing ``ffmpeg -i`` stderr.

    Standard ``ffprobe -show_entries`` may be unavailable in the container,
    so the portable ``ffmpeg -i`` Duration: line is parsed instead.

    Args:
        path: Path to a media file (video or audio).

    Returns:
        Duration in seconds, or None if it could not be determined.
    """
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(path)],
            capture_output=True, timeout=30,
        )
    except Exception:
        return None
    text = result.stderr.decode(errors="ignore")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", text)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    s = float(m.group(3))
    return h * 3600 + mi * 60 + s


def validate_exported_audio(
    clips_dir: Path,
    audio_dir: Path,
    clip_by_id: Dict[str, ClipInfo],
    accepted_ids: set,
    tolerance_ms: float = 50.0,
    logger: Any = None,
) -> Dict[str, Any]:
    """Audio Alignment Validation Report.

    For every accepted clip, probe the durations of the two final assets and
    verify they match the expected clip duration within ``tolerance_ms``:

        clips/{clip_id}.mp4            → video_duration
        audio/{clip_id}.wav            → audio_duration (aligned clip audio)

    Args:
        clips_dir: Final video clip directory (validation_final/clips).
        audio_dir: Final canonical audio directory holding the aligned clip
                   audio (validation_final/audio).
        clip_by_id: clip_id → ClipInfo lookup (for expected duration).
        accepted_ids: Set of accepted clip ids to validate.
        tolerance_ms: Max allowed deviation of any duration from the expected
                      clip duration (also used as clip-to-clip consistency).
        logger: Optional PipelineLogger.

    Returns:
        Report dict:
            {
              "report": "Audio Alignment Validation Report",
              "tolerance_ms": float,
              "total_clips": int,
              "passed": int,
              "failed": int,
              "skipped": int,
              "clips": [ {clip_id, expected_duration, video_duration,
                          audio_duration, max_diff_ms, status} ... ],
            }
    """
    rows: List[Dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped = 0

    for clip_id in sorted(accepted_ids):
        clip = clip_by_id.get(clip_id)
        expected = clip.duration if clip is not None else None
        video = _probe_duration(clips_dir / f"{clip_id}.mp4")
        audio = _probe_duration(audio_dir / f"{clip_id}.wav")

        durations = [d for d in (video, audio) if d is not None]
        if not durations:
            skipped += 1
            rows.append({
                "clip_id": clip_id,
                "expected_duration": expected,
                "video_duration": video,
                "audio_duration": audio,
                "max_diff_ms": None,
                "status": "skipped_no_audio",
            })
            continue

        max_diff_ms = max(
            abs(d - expected) * 1000.0 for d in durations
            if expected is not None and d is not None
        ) if expected is not None else None

        # All probed assets within tolerance AND mutually consistent.
        ok = True
        if expected is not None and max_diff_ms is not None:
            ok = max_diff_ms <= tolerance_ms
        if len(durations) > 1:
            spread = max(durations) - min(durations)
            ok = ok and spread * 1000.0 <= tolerance_ms

        status = "pass" if ok else "fail"
        if ok:
            passed += 1
        else:
            failed += 1

        rows.append({
            "clip_id": clip_id,
            "expected_duration": expected,
            "video_duration": video,
            "audio_duration": audio,
            "max_diff_ms": round(max_diff_ms, 3) if max_diff_ms is not None else None,
            "status": status,
        })

    report: Dict[str, Any] = {
        "report": "Audio Alignment Validation Report",
        "tolerance_ms": tolerance_ms,
        "total_clips": len(rows),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "clips": rows,
    }
    if logger:
        logger.info(
            f"[Stage 9.5] Validation: {passed} passed, {failed} failed, "
            f"{skipped} skipped (tolerance={tolerance_ms:.0f}ms)"
        )
    return report

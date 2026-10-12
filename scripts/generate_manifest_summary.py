#!/usr/bin/env python3
"""Build a flat CSV summary from every ``pipeline_manifest.json`` under outputs/.

Row granularity (one row == one *clip*, which for the production
``stage_04.mode == segment`` config is exactly one *scene* per row):

  * scene produced 1 clip  -> 1 row (segment mode: 1 clip == 1 scene).
  * scene produced N>1 clips (sliding_window) -> N rows (one per clip).
  * scene rejected at Stage 3 (no clip) -> still 1 row, ``clip_id`` empty,
    every Stage-4+ field empty (null).
  * video failed before Stage 2 (no scenes) -> 1 placeholder row, so
    per-video counting is not lost.

The failure cascade is recorded verbatim: per-clip stage dicts
(``clips[].stage_XX``) are already null downstream of a rejection, so
"Stage 3 rejected -> Stage 4+ null" / "Stage 9 rejected -> Stage 10/11 null"
happens naturally. Stage 9.5 / 10 / 11 are intentionally NOT exported.
Stage 12/13/14/17 fields exist only for phase-2-merged manifests.

Usage:
    python3 scripts/generate_manifest_summary.py                        # scan outputs/
    python3 scripts/generate_manifest_summary.py --out-root outputs/Talkvid_Chinese_videos_clips002
    python3 scripts/generate_manifest_summary.py --output logs/summary.csv --limit 100

Standard library only, so it runs on the host (no docker) and never touches
the pipeline itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = "outputs"
MANIFEST_FILENAME = "pipeline_manifest.json"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _g(d: Any, path: str, default: Any = None) -> Any:
    """Safe dotted-path getter. Returns ``default`` on any missing/None hop."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part, default)
        if cur is None:
            return default
    return cur


def _fmt(v: Any) -> str:
    """CSV cell formatter: empty string for null, JSON for list/dict, 1/0 for bool."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# ---------------------------------------------------------------------------
# column spec: (name, source, path)
#   source: video  -> path into the assembled video-level info dict
#           scene  -> path into the merged scene dict: stage_02 geometry +
#                     stage_03 coarse screen (stage_details.stage_02/03.scenes[])
#           clip   -> path into the clip record (clips[])
# ---------------------------------------------------------------------------

COLUMNS: List[Tuple[str, str, str]] = [
    # --- A. video-level info (repeated on every clip/scene row) --------------
    ("batch_dir", "video", "batch_dir"),
    ("run_dir", "video", "run_dir"),
    ("manifest_rel_path", "video", "manifest_rel_path"),
    ("video_id", "video", "video_id"),
    ("pipeline_mode", "video", "pipeline_mode"),
    ("input_video", "video", "input_video"),
    ("duration_s", "video", "duration"),
    ("width", "video", "width"),
    ("height", "video", "height"),
    ("fps", "video", "fps"),
    ("total_frames", "video", "total_frames"),
    ("final_status", "video", "final_status"),
    ("phase1_complete", "video", "phase1_complete"),
    ("phase2_complete", "video", "phase2_complete"),
    ("video_window_seconds", "video", "window_seconds"),
    ("manifest_version", "video", "manifest_version"),
    ("generated_at", "video", "generated_at"),
    ("updated_at", "video", "updated_at"),

    # --- B. scene identity / geometry ---------------------------------------
    ("scene_id", "scene", "scene_id"),
    ("clip_id", "clip", "clip_id"),
    ("clip_count_in_scene", "video", "_clip_count_in_scene"),
    ("scene_start_time", "scene", "start_time"),
    ("scene_end_time", "scene", "end_time"),
    ("scene_duration_s", "scene", "duration"),
    ("clip_start_time", "clip", "start_time"),
    ("clip_end_time", "clip", "end_time"),
    ("clip_duration_s", "clip", "duration"),
    ("clip_frame_count", "clip", "frame_count"),
    ("clip_window_seconds", "clip", "window_seconds"),

    # --- C. Stage 3 coarse screen (scene level) -----------------------------
    ("stage_03__status", "scene", "_status"),
    ("stage_03__passed", "scene", "passed"),
    ("stage_03__reject_reasons", "scene", "reject_reasons"),
    ("stage_03__sampled_frame_count", "scene", "coarse_face_info.sampled_frame_count"),
    ("stage_03__face_detect_ratio", "scene", "coarse_face_info.face_detect_ratio"),
    ("stage_03__single_face_ratio", "scene", "coarse_face_info.single_face_ratio"),
    ("stage_03__avg_face_confidence", "scene", "coarse_face_info.avg_face_confidence"),
    ("stage_03__max_face_count", "scene", "coarse_face_info.max_face_count"),
# --- Stage 5 face-resolution gate (video level; repeated on every row) ----
    ("stage_05_face_res__status", "video", "stage_05_face_res__status"),
    ("stage_05_face_res__passed", "video", "stage_05_face_res__passed"),
    ("stage_05_face_res__reject_reason", "video", "stage_05_face_res__reject_reason"),
    ("stage_05_face_res__short_side_min", "video", "stage_05_face_res__short_side_min"),
    ("stage_05_face_res__short_side_median", "video", "stage_05_face_res__short_side_median"),
    ("stage_05_face_res__short_side_max", "video", "stage_05_face_res__short_side_max"),
    ("stage_05_face_res__valid_face_frame_count", "video", "stage_05_face_res__valid_face_frame_count"),
    ("stage_05_face_res__qualified_ratio", "video", "stage_05_face_res__qualified_ratio"),

# --- D. Stage 5 detection (clip level) -----------------------------------
    ("stage_05__status", "clip", "stage_05.status"),
    ("stage_05__frames", "clip", "stage_05.frames"),
    ("stage_05__face_detections", "clip", "stage_05.face_detections"),
    ("stage_05__max_faces_in_frame", "clip", "stage_05.max_faces_in_frame"),
    ("stage_05__avg_confidence", "clip", "stage_05.avg_confidence"),

    # --- Stage 5.1 tracking --------------------------------------------------
    ("stage_05_1__status", "clip", "stage_05_1.status"),
    ("stage_05_1__track_count", "clip", "stage_05_1.track_count"),
    ("stage_05_1__track_ids", "clip", "stage_05_1.track_ids"),

    # --- Stage 5.2 single-face filter ---------------------------------------
    ("stage_05_2__status", "clip", "stage_05_2.status"),
    ("stage_05_2__reasons", "clip", "stage_05_2.reasons"),
    ("stage_05_2__main_track_id", "clip", "stage_05_2.identity_result.main_track_id"),
    ("stage_05_2__total_frames", "clip", "stage_05_2.identity_result.total_frames"),
    ("stage_05_2__anomaly_frames", "clip", "stage_05_2.identity_result.anomaly_frames"),
    ("stage_05_2__anomaly_ratio", "clip", "stage_05_2.identity_result.anomaly_ratio"),

    # --- Stage 5.3 temporal stability ---------------------------------------
    ("stage_05_3__status", "clip", "stage_05_3.status"),
    ("stage_05_3__reasons", "clip", "stage_05_3.reasons"),
    ("stage_05_3__average_center_motion", "clip", "stage_05_3.stability_result.average_center_motion"),
    ("stage_05_3__max_center_motion", "clip", "stage_05_3.stability_result.max_center_motion"),
    ("stage_05_3__average_scale_change", "clip", "stage_05_3.stability_result.average_scale_change"),
    ("stage_05_3__max_scale_change", "clip", "stage_05_3.stability_result.max_scale_change"),
    ("stage_05_3__bbox_jump_count", "clip", "stage_05_3.stability_result.bbox_jump_count"),
    ("stage_05_3__anomaly_ratio", "clip", "stage_05_3.stability_result.anomaly_ratio"),

    # --- Stage 6 pose --------------------------------------------------------
    ("stage_06__status", "clip", "stage_06.status"),
    ("stage_06__pose_frames", "clip", "stage_06.pose_frames"),
    ("stage_06__mean_yaw", "clip", "stage_06.mean_yaw"),
    ("stage_06__mean_pitch", "clip", "stage_06.mean_pitch"),
    ("stage_06__mean_roll", "clip", "stage_06.mean_roll"),
    ("stage_06__pose_ratio", "clip", "stage_06.pose_ratio"),

    # --- Stage 7 A/V sync ----------------------------------------------------
    ("stage_07__status", "clip", "stage_07.status"),
    ("stage_07__offset_frames", "clip", "stage_07.offset_frames"),
    ("stage_07__sync_lse_c", "clip", "stage_07.sync_lse_c"),
    ("stage_07__sync_lse_d", "clip", "stage_07.sync_lse_d"),
    ("stage_07__median_distance", "clip", "stage_07.median_distance"),

    # --- Stage 7.1 A/V alignment --------------------------------------------
    ("stage_07_1__status", "clip", "stage_07_1.status"),
    ("stage_07_1__offset_frames", "clip", "stage_07_1.offset_frames"),
    ("stage_07_1__offset_ms", "clip", "stage_07_1.offset_ms"),
    ("stage_07_1__correction_applied", "clip", "stage_07_1.correction_applied"),
    ("stage_07_1__alignment_status", "clip", "stage_07_1.alignment_status"),
    ("stage_07_1__fps", "clip", "stage_07_1.fps"),
    ("stage_07_1__clip_start", "clip", "stage_07_1.clip_start"),
    ("stage_07_1__clip_duration", "clip", "stage_07_1.clip_duration"),
    ("stage_07_1__audio_available", "clip", "stage_07_1.audio_available"),

    # --- Stage 8 quality -----------------------------------------------------
    ("stage_08__status", "clip", "stage_08.status"),
    ("stage_08__quality_frames", "clip", "stage_08.quality_frames"),
    ("stage_08__quality_mean", "clip", "stage_08.quality_mean"),
    ("stage_08__quality_min", "clip", "stage_08.quality_min"),
    ("stage_08__quality_max", "clip", "stage_08.quality_max"),
# --- Stage 9 fusion filter (Stage 9.5/10/11 intentionally omitted) ------
    ("stage_09__status", "clip", "stage_09.status"),
    ("stage_09__pose_score", "clip", "stage_09.pose_score"),
    ("stage_09__sync_score", "clip", "stage_09.sync_score"),
    ("stage_09__quality_score", "clip", "stage_09.quality_score"),
    ("stage_09__fusion_score", "clip", "stage_09.fusion_score"),
    ("stage_09__is_accepted", "clip", "stage_09.is_accepted"),
    ("stage_09__reject_reasons", "clip", "stage_09.reject_reasons"),

    # --- Stage 12 FLAME/EMICA (phase 2) -------------------------------------
    ("stage_12__status", "clip", "stage_12.status"),
    ("stage_12__num_frames", "clip", "stage_12.num_frames"),
    ("stage_12__backend", "clip", "stage_12.backend"),
    ("stage_12__jaw_source", "clip", "stage_12.jaw_source"),
    ("stage_12__head_source", "clip", "stage_12.head_source"),
    ("stage_12__expression_source", "clip", "stage_12.expression_source"),
    ("stage_12__expression_dim", "clip", "stage_12.expression_dim"),
    ("stage_12__jaw_pose_dim", "clip", "stage_12.jaw_pose_dim"),
    ("stage_12__head_pose_dim", "clip", "stage_12.head_pose_dim"),
    ("stage_12__shapecode_dim", "clip", "stage_12.shapecode_dim"),
    ("stage_12__reason", "clip", "stage_12.reason"),

    # --- Stage 13 gaze (phase 2) --------------------------------------------
    ("stage_13__status", "clip", "stage_13.status"),
    ("stage_13__num_frames", "clip", "stage_13.num_frames"),
    ("stage_13__gaze_dim", "clip", "stage_13.gaze_dim"),
    ("stage_13__reason", "clip", "stage_13.reason"),

    # --- Stage 14 audio feature (phase 2) -----------------------------------
    ("stage_14__status", "clip", "stage_14.status"),
    ("stage_14__num_frames", "clip", "stage_14.num_frames"),
    ("stage_14__feature_dim", "clip", "stage_14.feature_dim"),
    ("stage_14__source_audio", "clip", "stage_14.source_audio"),
    ("stage_14__reason", "clip", "stage_14.reason"),

    # --- Stage 17 fusion export (phase 2) -----------------------------------
    ("stage_17__status", "clip", "stage_17.status"),
    ("stage_17__n_flame", "clip", "stage_17.n_flame"),
    ("stage_17__n_gaze", "clip", "stage_17.n_gaze"),
    ("stage_17__n_audio", "clip", "stage_17.n_audio"),
    ("stage_17__num_frames", "clip", "stage_17.num_frames"),
    ("stage_17__valid_length", "clip", "stage_17.valid_length"),
    ("stage_17__gaze_missing_frames", "clip", "stage_17.gaze_missing_frames"),
    ("stage_17__audio_missing_frames", "clip", "stage_17.audio_missing_frames"),
    ("stage_17__failed_reason", "clip", "stage_17.failed_reason"),
]
def _find_manifests(out_root: Path, limit: Optional[int]) -> List[Path]:
    """Collect all manifest paths under ``out_root`` using os.walk (fast)."""
    found: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(out_root):
        if MANIFEST_FILENAME in filenames:
            found.append(Path(dirpath) / MANIFEST_FILENAME)
            if limit is not None and len(found) >= limit:
                return found
    return found


def _build_video_info(manifest: Dict[str, Any], rel_path: str, run_dir: str,
                      batch_dir: str) -> Dict[str, Any]:
    """Assemble the video-level info dict (source == 'video' in COLUMNS)."""
    src = manifest.get("source_video") or {}
    run = manifest.get("run") or {}
    stage5_res = _g(manifest, "stage_details.stage_05_face_resolution_filter") or \
        _g(manifest, "audit.stage_05_face_resolution_filter") or {}
    s5_res_passed = stage5_res.get("passed")
    if s5_res_passed is True:
        s5_res_status = "passed"
    elif s5_res_passed is False:
        s5_res_status = "rejected"
    else:
        s5_res_status = None
    return {
        "batch_dir": batch_dir,
        "run_dir": run_dir,
        "manifest_rel_path": rel_path,
        "video_id": src.get("video_id"),
        "pipeline_mode": run.get("pipeline_mode"),
        "input_video": run.get("input_video"),
        "duration": src.get("duration"),
        "width": src.get("width"),
        "height": src.get("height"),
        "fps": src.get("fps"),
        "total_frames": src.get("total_frames"),
        "final_status": run.get("final_status"),
        "phase1_complete": manifest.get("_phase1_complete"),
        "phase2_complete": manifest.get("_phase2_complete"),
        "window_seconds": run.get("window_seconds"),
        "manifest_version": manifest.get("manifest_version"),
        "generated_at": manifest.get("generated_at"),
        "updated_at": manifest.get("updated_at"),
        "stage_05_face_res__status": s5_res_status,
        "stage_05_face_res__passed": s5_res_passed,
        "stage_05_face_res__reject_reason": stage5_res.get("reject_reason"),
        "stage_05_face_res__short_side_min": stage5_res.get("face_short_side_min"),
        "stage_05_face_res__short_side_median": stage5_res.get("face_short_side_median"),
        "stage_05_face_res__short_side_max": stage5_res.get("face_short_side_max"),
        "stage_05_face_res__valid_face_frame_count": stage5_res.get("valid_face_frame_count"),
        "stage_05_face_res__qualified_ratio": stage5_res.get("qualified_ratio"),
    }


def _stage03_status(scene3: Dict[str, Any]) -> Optional[str]:
    passed = scene3.get("passed")
    if passed is True:
        return "passed"
    if passed is False:
        return "rejected"
    return None


def _iter_rows(manifest: Dict[str, Any]):
    """Yield (scene, clip, clip_count) tuples.

    Scenes come from ``stage_details.stage_02/03`` (merged by ``scene_id``:
    geometry + coarse screen); clips are joined by ``scene_id``. A scene
    without clips (Stage 3 rejected) still yields a row with ``clip = {}``.
    A video with no scenes yields one placeholder row.
    """
    sd = manifest.get("stage_details") or {}
    clips = manifest.get("clips") or []

    # scene geometry + coarse-screen, keyed by scene_id (insertion-ordered).
    scenes: Dict[str, Dict[str, Any]] = {}
    for s in (sd.get("stage_02") or {}).get("scenes") or []:
        sid = s.get("scene_id")
        if sid is None:
            continue
        scenes.setdefault(sid, {}).update(s)
    for s in (sd.get("stage_03") or {}).get("scenes") or []:
        sid = s.get("scene_id")
        if sid is None:
            continue
        scenes.setdefault(sid, {}).update(s)

    clips_by_scene: Dict[Optional[str], List[Dict[str, Any]]] = defaultdict(list)
    for c in clips:
        clips_by_scene[c.get("scene_id")].append(c)

    if not scenes and not clips:
        # video failed before Stage 2: single placeholder row.
        yield ({}, {}, 0)
        return

    yielded_clip_ids = set()
    for sid, scene in scenes.items():
        scene["_status"] = _stage03_status(scene)
        scene_clips = clips_by_scene.get(sid, [])
        if not scene_clips:
            yield (scene, {}, 0)
            continue
        for clip in scene_clips:
            yielded_clip_ids.add(clip.get("clip_id"))
            yield (scene, clip, len(scene_clips))

    # defensive: clips whose scene_id was not present in stage_02/03 scenes.
    for sid, scene_clips in clips_by_scene.items():
        for clip in scene_clips:
            if clip.get("clip_id") in yielded_clip_ids:
                continue
            scene = scenes.get(sid, {"scene_id": sid})
            scene["_status"] = _stage03_status(scene)
            yielded_clip_ids.add(clip.get("clip_id"))
            yield (scene, clip, len(scene_clips))


def _resolve(spec: Tuple[str, str, str], video: Dict[str, Any],
             scene: Dict[str, Any], clip: Dict[str, Any]) -> Any:
    name, source, path = spec
    if source == "video":
        return video.get(path)
    if source == "scene":
        return _g(scene, path)
    if source == "clip":
        return _g(clip, path)
    return None


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Summarize all pipeline_manifest.json files into one flat CSV."
    )
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help="Directory to scan for pipeline_manifest.json (default: outputs).")
    ap.add_argument("--output", "-o", default=None,
                    help="Output CSV path (default: <out-root>/pipeline_manifest_summary_<ts>.csv).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N manifests (for quick testing).")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    out_root = Path(args.out_root)
    if not out_root.is_dir():
        print(f"ERROR: scan root does not exist: {out_root}", file=sys.stderr)
        return 2

    manifests = _find_manifests(out_root, args.limit)
    print(f"found {len(manifests)} manifest(s) under {out_root}", file=sys.stderr)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_csv = Path(args.output)
    else:
        out_csv = out_root / f"pipeline_manifest_summary_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [name for name, _, _ in COLUMNS]
    rows_written = 0
    parse_errors = 0

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for mf in manifests:
            try:
                with mf.open("r", encoding="utf-8") as fh_in:
                    manifest = json.load(fh_in)
            except Exception as exc:  # noqa: BLE001 - keep going, report at end
                parse_errors += 1
                print(f"  !! skip malformed {mf}: {exc}", file=sys.stderr)
                continue

            rel_path = os.path.relpath(mf, PROJECT_ROOT)
            run_dir = mf.parent.name
            # first directory component under the scan root (the batch/group root);
            # a manifest sitting directly under the scan root falls back to run_dir.
            rel_to_outputs = os.path.relpath(mf, out_root)
            batch_dir = rel_to_outputs.split(os.sep)[0] if os.sep in rel_to_outputs else run_dir

            video = _build_video_info(manifest, rel_path, run_dir, batch_dir)

            for scene, clip, clip_count in _iter_rows(manifest):
                video["_clip_count_in_scene"] = clip_count
                row = [_fmt(_resolve(spec, video, scene, clip)) for spec in COLUMNS]
                writer.writerow(row)
                rows_written += 1

    print(f"wrote {rows_written} row(s) -> {out_csv}", file=sys.stderr)
    if parse_errors:
        print(f"WARNING: {parse_errors} manifest(s) could not be parsed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Per-run production statistics manifest (survives intermediate-output deletion).

Two-phase lifecycle (open-domain pipeline):

  Phase 1 — written by ``scripts/stage0_17_e2e_validate.py::_finalize`` right
  after ``stage0_11_results.json``. Built entirely from the in-memory
  ``env``/``context`` objects of the e2e driver, so no intermediate on-disk
  output is required. Captures: run / source_video / config_snapshot,
  per-stage status + details (Stage 0 -> 11), the per-clip lineage
  (scenes -> clips -> Stage5/5.1/5.2/5.3 -> Stage6/7/7.1/8/9/9.5 -> Stage10/11),
  and the Stage 5.2 / 5.3 reject audit from ``context.stage_audit``.
  Stage 4's sliding-window regime is recorded verbatim
  (``mode``/``window_size``/``stride``) so downstream stats stay
  interpretable: when ``stage_04.mode == sliding_window`` every downstream
  stage operates on fixed-length (e.g. 8 s) windows.

  Phase 2 — merged by ``scripts/export_production_multimodal.py::main`` after
  the Stage17 multimodal export. Reads the Stage12/13/14/17 metadata JSONs
  produced by the official runners and adds per-clip feature statistics plus
  the ``final_assets`` inventory (the surviving deliverable tree).

The manifest is the ONLY record of the per-run production statistics that
survives deletion of the intermediate directories. All writes are atomic
(temp file + ``os.replace``) and strictly non-fatal: every public function
catches its own errors and returns ``None``/``False``/partial results instead
of raising into the integration points. This module never deletes anything.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

MANIFEST_VERSION: str = "1.1"
MANIFEST_FILENAME: str = "pipeline_manifest.json"

# stage_id -> exact e2e results entry name (scripts/stage0_17_e2e_validate.py).
STAGE_RESULT_NAMES: Dict[str, str] = {
    "stage_00": "Stage 0 (FFprobe integrity check)",
    "stage_00_5": "Stage 0.5 (Resolution filter)",
    "stage_01": "Stage 1 (FFmpeg standardization)",
    "stage_02": "Stage 2 (PySceneDetect)",
    "stage_03": "Stage 3 (SCRFD coarse screening)",
    "stage_04": "Stage 4 (Sliding-window clips)",
    "stage_05": "Stage 5 (SCRFD detect/crop)",
    "stage_05_1": "Stage 5.1 (ByteTrack face tracking)",
    "stage_05_2": "Stage 5.2 (Single-face filter)",
    "stage_05_3": "Stage 5.3 (Temporal stability)",
    "stage_06": "Stage 6 (6DRepNet pose)",
    "stage_07": "Stage 7 (SyncNet A/V sync)",
    "stage_07_1": "Stage 7.1 (A/V alignment)",
    "stage_08": "Stage 8 (HyperIQA quality)",
    "stage_09": "Stage 9 (Quality fusion/filter)",
    "stage_09_5": "Stage 9.5 (Clip export)",
    "stage_10": "Stage 10 (InsightFace landmark)",
    "stage_11": "Stage 11 (Affine alignment)",
}

STAGE_ORDER: Tuple[str, ...] = (
    "stage_00", "stage_00_5", "stage_01", "stage_02", "stage_03", "stage_04",
    "stage_05", "stage_05_1", "stage_05_2", "stage_05_3",
    "stage_06", "stage_07", "stage_07_1", "stage_08", "stage_09", "stage_09_5",
    "stage_10", "stage_11", "stage_12", "stage_13", "stage_14", "stage_17",
)

PHASE2_STAGES: Tuple[str, ...] = ("stage_12", "stage_13", "stage_14", "stage_17")

# Stage12/13/14 official-runner output layout under the Stage0-11 test base.
FEATURE_SUBDIRS: Dict[str, Tuple[str, str]] = {
    "stage_12": ("stage12_flame", "flame_metadata.json"),
    "stage_13": ("stage13_gaze", "gaze_metadata.json"),
    "stage_14": ("stage14_audio", "audio_metadata.json"),
}

# The per-clip feature payload each stage must produce. ``merge_phase2`` uses
# the *absence* of these files (for a clip that reached Stage11) as the signal
# that the stage failed for that clip — the official runners do not persist any
# exit status, they simply exit non-zero on failure.
FEATURE_NPZ: Dict[str, str] = {
    "stage_12": "flame_features.npz",
    "stage_13": "gaze_features.npz",
    "stage_14": "audio_features.npz",
}

# Per-clip Stage17 fusion/export report written by
# ``scripts/export_production_multimodal.py``. This is the authoritative record
# of per-clip modality availability + matching statistics + export checks, and
# is what lets failures (e.g. Stage13 gaze producing no frames -> final
# flame/gaze/audio parameters unmatchable) survive intermediate-output deletion.
EXPORT_REPORT_FILENAME: str = "export_report.json"


def _utc_now() -> str:
    """Current UTC time in the same ISO format used by the e2e driver."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _p(x: Any) -> Optional[str]:
    """Path -> str with None preserved."""
    return str(x) if x is not None else None


def _f(x: Any, nd: int = 6) -> Optional[float]:
    """float -> rounded float with None preserved."""
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def manifest_path(run_root: Any) -> Path:
    """Manifest location: ``<run_root>/pipeline_manifest.json``.

    ``run_root`` is the parent of the Stage0-11 test base, e.g.
    ``outputs/video_<id>_open_domain/``.
    """
    return Path(run_root).resolve() / MANIFEST_FILENAME


def load(path: Any, logger: Any = None) -> Optional[dict]:
    """Non-fatal JSON load — returns None (and warns) on any failure."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 - non-fatal by design
        _warn(logger, f"pipeline_manifest load failed ({path}): {exc}")
        return None


def save(doc: dict, path: Any, logger: Any = None) -> bool:
    """Atomic, non-fatal JSON write (temp file + os.replace)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, p)
        return True
    except Exception as exc:  # noqa: BLE001 - non-fatal by design
        _warn(logger, f"pipeline_manifest save failed ({path}): {exc}")
        return False


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------


def config_snapshot(config: dict) -> dict:
    """Serialize the config knobs that the recorded statistics depend on.

    Mirrors exactly the reads performed by the actual stages (keys verified
    against ``stage_04``, ``stage_05_2/5.3`` and ``_build_thresholds`` in
    stage_09) so the manifest stays interpretable after intermediate outputs
    are deleted.
    """
    stage4 = config.get("stage_04") or {}
    stage5 = config.get("stage_05") or {}
    face_cfg = config.get("face") or {}
    face_crop = config.get("stage_05_face_crop") or {}
    res_filter = config.get("stage_05_face_resolution_filter") or {}
    enh = config.get("enhancements") or {}
    s51 = enh.get("stage_05_1_face_tracking") or {}
    s51_cfg = config.get("stage_05_1") or {}
    tracker = s51_cfg.get("tracker") or {}
    s52 = enh.get("stage_05_2_single_face_filter") or {}
    s53 = enh.get("stage_05_3_temporal_stability") or {}
    pose_cfg = config.get("pose") or {}
    quality_cfg = config.get("quality") or {}
    sync_cfg = quality_cfg.get("sync") or {}
    s09 = config.get("stage_09") or {}
    s09_5 = config.get("stage_09_5") or {}
    fw = s09.get("fusion_weights") or {}
    return {
        "stage_04": {
            "enabled": bool(stage4.get("enabled", True)),
            "mode": stage4.get("mode", "sliding_window"),
            "window_size": stage4.get("window_size", 8),
            "stride": stage4.get("stride", 4),
            "target_fps": stage4.get("target_fps", 25),
            "note": (
                "sliding_window: fixed-length windows of window_size s advanced "
                "by stride s; segment: one full-length clip per scene (1:1). "
                "When mode=sliding_window all downstream stats operate on "
                f"{stage4.get('window_size', 8)} s windows."
            ),
        },
        "stage_05": {
            "enabled": bool(stage5.get("enabled", True)),
            "confidence_threshold": face_cfg.get("confidence_threshold", 0.6),
        },
        "stage_05_face_crop": {
            "expand_ratio": face_crop.get("expand_ratio", 2.0),
            "save_raw_crop": face_crop.get("save_raw_crop", True),
            "save_frames": face_crop.get("save_frames", False),
        },
        "stage_05_face_resolution_filter": {
            "enabled": bool(res_filter.get("enabled", False)),
            "min_face_short_side": int(res_filter.get("min_face_short_side", 75)),
            "min_face_resolution_ratio": float(res_filter.get("min_face_resolution_ratio", 0.90)),
            "min_valid_face_frames": int(res_filter.get("min_valid_face_frames", 5)),
        },
        "stage_05_1_face_tracking": {
            "enabled": bool(s51.get("enabled", True)),
            "tracker_type": tracker.get("type", "bytetrack"),
            "track_thresh": tracker.get("track_thresh", 0.5),
            "track_buffer": tracker.get("track_buffer", 30),
            "match_thresh": tracker.get("match_thresh", 0.8),
        },
        "stage_05_2_single_face_filter": {
            "enabled": bool(s52.get("enabled", True)),
            "max_identity_anomaly_ratio": s52.get("max_identity_anomaly_ratio", 0.005),
        },
        "stage_05_3_temporal_stability": {
            "enabled": bool(s53.get("enabled", True)),
            "max_center_motion": s53.get("max_center_motion", 15.0),
            "max_scale_change_ratio": s53.get("max_scale_change_ratio", 0.25),
            "max_stability_anomaly_ratio": s53.get("max_stability_anomaly_ratio", 0.05),
        },
        "stage_09": {
            "enabled": bool(s09.get("enabled", True)),
            "yaw_threshold": float(pose_cfg.get("yaw_threshold", 30.0)),
            "pitch_threshold": float(pose_cfg.get("pitch_threshold", 20.0)),
            "roll_threshold": float(pose_cfg.get("roll_threshold", 30.0)),
            "min_pose_ratio": float(s09.get("min_pose_ratio", 0.9)),
            "min_pose_confidence": float(s09.get("min_pose_confidence", 0.0)),
            "max_offset_frames": int(sync_cfg.get("max_offset_frames", 8)),
            "min_lse_c": float(sync_cfg.get("min_lse_c", 5.0)),
            "quality_threshold": float(quality_cfg.get("quality_threshold", 0.5)),
            "fusion_threshold": s09.get("fusion_threshold"),
            "fusion_weights": {
                "pose": fw.get("pose", 0.4),
                "sync": fw.get("sync", 0.3),
                "quality": fw.get("quality", 0.3),
            },
        },
        "stage_09_5": {
            "clips_dir": s09_5.get("clips_dir", "clips"),
            "audio_dir": s09_5.get("audio_dir", "audio"),
            "manifest": s09_5.get("manifest", "manifest.json"),
        },
    }



def _warn(logger: Any, message: str) -> None:
    if logger is None:
        return
    try:
        logger.warning(message)
    except Exception:  # noqa: BLE001 - logging must never break the manifest
        pass


# ---------------------------------------------------------------------------
# Small record serializers (models -> JSON-safe dicts)
# ---------------------------------------------------------------------------


def _scene_to_dict(scene: Any) -> dict:
    return {
        "scene_id": scene.scene_id,
        "video_id": scene.video_id,
        "start_frame": scene.start_frame,
        "end_frame": scene.end_frame,
        "start_time": _f(scene.start_time, 3),
        "end_time": _f(scene.end_time, 3),
        "duration": _f(scene.duration, 3),
    }


def _coarse_face_to_dict(cfi: Any) -> Optional[dict]:
    if cfi is None:
        return None
    return {
        "sampled_frame_count": cfi.sampled_frame_count,
        "face_detect_ratio": _f(cfi.face_detect_ratio),
        "single_face_ratio": _f(cfi.single_face_ratio),
        "avg_face_confidence": _f(cfi.avg_face_confidence),
        "max_face_count": cfi.max_face_count,
    }


def _scene_filter_to_dict(fr: Any) -> Optional[dict]:
    if fr is None:
        return None
    return {
        "passed": bool(fr.passed),
        "reject_reasons": list(fr.reject_reasons),
    }


def _identity_to_dict(idr: Any) -> Optional[dict]:
    if idr is None:
        return None
    return {
        "main_track_id": idr.main_track_id,
        "total_frames": idr.total_frames,
        "anomaly_frames": idr.anomaly_frames,
        "anomaly_ratio": _f(idr.anomaly_ratio),
        "reject_reasons": list(idr.reject_reasons),
    }


def _stability_to_dict(sr: Any) -> Optional[dict]:
    if sr is None:
        return None
    return {
        "average_center_motion": _f(sr.average_center_motion, 3),
        "max_center_motion": _f(sr.max_center_motion, 3),
        "average_scale_change": _f(sr.average_scale_change, 3),
        "max_scale_change": _f(sr.max_scale_change, 3),
        "bbox_jump_count": sr.bbox_jump_count,
        "anomaly_ratio": _f(sr.anomaly_ratio),
        "reject_reasons": list(sr.reject_reasons),
    }


def _pose_ratio(poses: Sequence[Any], yaw_th: float, pitch_th: float,
                roll_th: float, min_conf: float) -> Optional[float]:
    """Replicates Stage9 ``compute_pose_score`` acceptance math."""
    if not poses:
        return None
    ok = 0
    for p in poses:
        try:
            conf = float(getattr(p, "confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        if conf <= min_conf:
            continue
        if abs(float(p.yaw)) <= yaw_th and abs(float(p.pitch)) <= pitch_th \
                and abs(float(p.roll)) <= roll_th:
            ok += 1
    return round(ok / len(poses), 6)



# ---------------------------------------------------------------------------
# Run context: bundles the in-memory env/context lookups for the builders
# ---------------------------------------------------------------------------


def _build_run_ctx(config: dict, env: dict, context: Any,
                   test_base: Path, run_root: Path) -> dict:
    """Assemble all per-run lookups into one plain dict.

    ``env`` is the e2e driver's shared env dict (model instances from the
    pipeline stages); ``context`` is the ``PipelineContext`` (used for the
    Stage 5.2 / 5.3 reject audit).
    """
    pipeline_mode = env.get("pipeline_mode") or context.runtime_info.get(
        "pipeline_mode", "open_domain")
    scenes = env.get("scenes") or []
    clips = env.get("clips") or []
    face_sequences = env.get("face_sequences") or []
    tracked = env.get("tracked_sequences") or []
    filtered = env.get("filtered_sequences") or []
    stabilized = env.get("stabilized_sequences") or []

    # The e2e driver stores these under the *_sequences / *_infos keys;
    # accept both spellings (defensive, and the smoke test uses the real names).
    def _first(*keys: str) -> List[Any]:
        for k in keys:
            v = env.get(k)
            if v:
                return v
        return []

    poses = _first("pose_results", "pose_sequences")
    av = _first("av_results", "av_sync_infos")
    aligns = _first("alignment_results", "av_alignments")
    qualities = _first("quality_results", "quality_sequences")
    metrics = env.get("quality_metrics") or []
    landmarks = env.get("landmarks") or []
    aligned = env.get("aligned") or []

    def _by_id(seqs: Sequence[Any]) -> dict:
        out: dict = {}
        for s in seqs:
            try:
                out[s.clip_id] = s
            except (AttributeError, TypeError):
                continue
        return out

    stage_audit = getattr(context, "stage_audit", None) or {}
    audit_52 = stage_audit.get("stage_05_2_single_face_filter") or {}
    audit_53 = stage_audit.get("stage_05_3_temporal_stability") or {}
    resolution_audit = stage_audit.get("stage_05_face_resolution_filter") or {}
    audit_rejected_52 = {a.get("clip_id"): a for a in (audit_52.get("rejected") or [])}
    audit_rejected_53 = {a.get("clip_id"): a for a in (audit_53.get("rejected") or [])}

    if pipeline_mode == "open_domain":
        accepted_ids: Set[str] = {m.clip_id for m in metrics if getattr(m, "is_accepted", False)}
    else:  # dataset mode: Stage9 skipped, all stabilized clips are accepted
        accepted_ids = {s.clip_id for s in stabilized}

    stage4_cfg = config.get("stage_04") or {}
    s09_5_cfg = config.get("stage_09_5") or {}
    window_seconds = None
    if str(stage4_cfg.get("mode", "sliding_window")) == "sliding_window":
        window_seconds = stage4_cfg.get("window_size", 8)

    clips_dir = str(s09_5_cfg.get("clips_dir", "clips"))
    audio_dir = str(s09_5_cfg.get("audio_dir", "audio"))

    s09_cfg = config.get("stage_09") or {}
    pose_cfg = config.get("pose") or {}
    yaw_th = float(pose_cfg.get("yaw_threshold", 30.0))
    pitch_th = float(pose_cfg.get("pitch_threshold", 20.0))
    roll_th = float(pose_cfg.get("roll_threshold", 30.0))
    min_conf = float(s09_cfg.get("min_pose_confidence", 0.0))

    def _rel(p: Any) -> str:
        """Path relative to run_root (portable), else absolute string."""
        try:
            pp = Path(p).resolve()
            return PurePosixPath(pp.relative_to(run_root)).as_posix()
        except (ValueError, OSError):
            return str(p)

    return {
        "pipeline_mode": pipeline_mode,
        "test_base": test_base,
        "run_root": run_root,
        "video_info": env.get("video_info"),
        "resolution_info": env.get("resolution_info"),
        "stage1_out": env.get("stage1_out") or {},
        "scenes": scenes,
        "clips": clips,
        "face_sequences": face_sequences,
        "face_by_id": _by_id(face_sequences),
        "tracked_sequences": tracked,
        "tracked_by_id": _by_id(tracked),
        "filtered_sequences": filtered,
        "filtered_by_id": _by_id(filtered),
        "stabilized_sequences": stabilized,
        "stabilized_by_id": _by_id(stabilized),
        "pose_results": poses,
        "pose_by_id": _by_id(poses),
        "av_results": av,
        "av_by_id": _by_id(av),
        "alignment_results": aligns,
        "align_by_id": _by_id(aligns),
        "quality_results": qualities,
        "qual_by_id": _by_id(qualities),
        "quality_metrics": metrics,
        "metrics_by_id": _by_id(metrics),
        "landmarks": landmarks,
        "aligned": aligned,
        "accepted_ids": accepted_ids,
        "export_result": env.get("export_result"),
        "audit_52": audit_52,
        "audit_53": audit_53,
        "resolution_audit": resolution_audit,
        "audit_rejected_52": audit_rejected_52,
        "audit_rejected_53": audit_rejected_53,
        "window_seconds": window_seconds,
        "yaw_th": yaw_th,
        "pitch_th": pitch_th,
        "roll_th": roll_th,
        "min_conf": min_conf,
        "clips_rel_dir": clips_dir,
        "audio_rel_dir": audio_dir,
        "rel": _rel,
    }


def _result_entry(results: Sequence[dict], stage_id: str) -> dict:
    """Stage-level status block from the e2e results list.

    Status vocabulary matches the e2e driver: ``PASS`` / ``FAIL`` /
    ``SKIPPED`` at stage level.  Stages 12-17 are skipped inside the e2e
    driver and merged in phase2.
    """
    name = STAGE_RESULT_NAMES.get(stage_id)
    if name is not None:
        for r in results:
            if r.get("stage") == name:
                return {
                    "status": r.get("status", "SKIPPED"),
                    "started": r.get("started"),
                    "finished": r.get("finished"),
                    "elapsed_s": _f(r.get("elapsed_s"), 3),
                    "output_summary": r.get("output_summary", ""),
                    "error": r.get("error"),
                    "phase": 1,
                }
        return {
            "status": "SKIPPED",
            "note": "stage did not run (no e2e results entry)",
            "phase": 1,
        }
    return {
        "status": "SKIPPED",
        "note": "run by official runner in a separate environment; merged in phase2",
        "phase": 1,
    }




# ---------------------------------------------------------------------------
# Per-clip stage records
# ---------------------------------------------------------------------------
# Clip-level status vocabulary:
#   "passed"   — the stage produced output for this clip
#   "rejected" — the stage hard-filtered this clip (reasons recorded)
#   "null"     — the stage was not reached for this clip (no record)
# Stage6-9 are `null` for clips rejected at Stage5.2/5.3 — they are never
# `"FAIL"` at clip level (FAIL is reserved for stage-level e2e status).


def _s52_entry(clip_id: str, ctx: dict) -> dict:
    filtered = ctx["filtered_by_id"].get(clip_id)
    tracked = ctx["tracked_by_id"].get(clip_id)
    if filtered is not None:
        if filtered.identity_result is not None:
            return {"status": "passed",
                    "identity_result": _identity_to_dict(filtered.identity_result)}
        # Enhancement disabled (or no identity result): the stage passes the
        # clip through unchanged — this is a pass, not a rejection.
        return {"status": "passed",
                "note": "no identity_result (enhancement disabled or pass-through)"}
    if tracked is not None:
        audit_rec = ctx["audit_rejected_52"].get(clip_id)
        if audit_rec:
            rest = {k: v for k, v in audit_rec.items() if k != "reasons"}
            return {"status": "rejected", "reasons": list(audit_rec.get("reasons", [])), "audit": rest}
        return {"status": "rejected", "reasons": [],
                "note": "rejected at Stage5.2 but audit record unavailable (pre-manifest run)"}
    if clip_id in ctx["face_by_id"]:
        return {"status": "null", "note": "no track assigned by Stage5.1"}
    return {"status": "null", "note": "no Stage5 face sequence for this clip"}


def _s53_entry(clip_id: str, ctx: dict) -> dict:
    stab = ctx["stabilized_by_id"].get(clip_id)
    filt = ctx["filtered_by_id"].get(clip_id)
    if stab is not None:
        if stab.stability_result is not None:
            return {"status": "passed",
                    "stability_result": _stability_to_dict(stab.stability_result)}
        # Enhancement disabled (or no stability result): pass-through.
        return {"status": "passed",
                "note": "no stability_result (enhancement disabled or pass-through)"}
    if filt is not None:
        audit_rec = ctx["audit_rejected_53"].get(clip_id)
        if audit_rec:
            rest = {k: v for k, v in audit_rec.items() if k != "reasons"}
            return {"status": "rejected", "reasons": list(audit_rec.get("reasons", [])), "audit": rest}
        return {"status": "rejected", "reasons": [],
                "note": "rejected at Stage5.3 but audit record unavailable (pre-manifest run)"}
    if clip_id in ctx["tracked_by_id"]:
        return {"status": "null", "note": "dropped at Stage5.2 (single-face filter)"}
    return {"status": "null", "note": "no Stage5.1 track for this clip"}


def _s06_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    ps = ctx["pose_by_id"].get(clip_id)
    if ps is None:
        return None  # null: not reached (rejected at 5.2/5.3) or dataset mode
    poses = ps.poses
    return {
        "status": "passed",
        "pose_frames": len(poses),
        "mean_yaw": _f(_mean([float(p.yaw) for p in poses])),
        "mean_pitch": _f(_mean([float(p.pitch) for p in poses])),
        "mean_roll": _f(_mean([float(p.roll) for p in poses])),
        "pose_ratio": _pose_ratio(poses, ctx["yaw_th"], ctx["pitch_th"],
                                  ctx["roll_th"], ctx["min_conf"]),
    }


def _s07_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    av = ctx["av_by_id"].get(clip_id)
    if av is None:
        return None
    return {
        "status": "passed",
        "offset_frames": int(av.offset),
        "sync_lse_c": _f(av.sync_lse_c),
        "sync_lse_d": _f(av.sync_lse_d),
        "median_distance": _f(av.median_distance),
    }


def _s07_1_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    al = ctx["align_by_id"].get(clip_id)
    if al is None:
        return None
    return {
        "status": "passed",
        "offset_frames": int(al.offset_frames),
        "offset_ms": _f(al.offset_ms, 1),
        "correction_applied": bool(al.correction_applied),
        "alignment_status": al.alignment_status,
        "fps": _f(al.fps),
        "clip_start": _f(al.clip_start),
        "clip_duration": _f(al.clip_duration),
        "audio_available": bool(getattr(al, "aligned_audio_path", None)),
    }

def _build_stages(results: Sequence[dict]) -> dict:
    return {sid: _result_entry(results, sid) for sid in STAGE_ORDER}




def _s08_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    qs = ctx["qual_by_id"].get(clip_id)
    if qs is None:
        return None
    q = [float(x.quality_score) for x in qs.qualities]
    return {
        "status": "passed",
        "quality_frames": len(q),
        "quality_mean": _f(_mean(q)),
        "quality_min": _f(min(q)) if q else None,
        "quality_max": _f(max(q)) if q else None,
    }


def _s09_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    m = ctx["metrics_by_id"].get(clip_id)
    if m is None:
        return None
    return {
        "status": "passed" if m.is_accepted else "rejected",
        "pose_score": _f(m.pose_score),
        "sync_score": _f(m.sync_score),
        "quality_score": _f(m.quality_score),
        "fusion_score": _f(m.fusion_score),
        "is_accepted": bool(m.is_accepted),
        "reject_reasons": list(m.reject_reasons),
    }


def _s09_5_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    er = ctx["export_result"]
    if er is None:
        return None  # dataset mode (Stage9.5 skipped)
    exported = clip_id in ctx["accepted_ids"]
    base = ctx["test_base"]
    return {
        "status": "passed" if exported else "rejected",
        "exported": exported,
        "video_path": ctx["rel"](base / ctx["clips_rel_dir"] / f"{clip_id}.mp4"),
        "audio_path": ctx["rel"](base / ctx["audio_rel_dir"] / f"{clip_id}.wav"),
    }


def _s10_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    if clip_id not in ctx["accepted_ids"]:
        return None  # null: not reached
    count = sum(1 for lm in ctx["landmarks"] if getattr(lm, "clip_id", None) == clip_id)
    return {"status": "passed", "landmark_frames": count}


def _s11_entry(clip_id: str, ctx: dict) -> Optional[dict]:
    if clip_id not in ctx["accepted_ids"]:
        return None
    entries = [a for a in ctx["aligned"] if getattr(a, "clip_id", None) == clip_id]
    return {
        "status": "passed",
        "aligned_faces": len(entries),
        "frame_indices": sorted(int(a.frame_index) for a in entries),
    }


def _clip_record(clip: Any, ctx: dict) -> dict:
    """One JSON record per clip, capturing the full stage lineage."""
    clip_id = clip.clip_id
    scene_id = getattr(clip, "scene_id", None)
    start_frame = int(clip.start_frame)
    end_frame = int(clip.end_frame)
    duration = float(clip.duration)
    frame_count = int(clip.frame_count)

    rec = {
        "clip_id": clip_id,
        "scene_id": scene_id,
        "video_id": getattr(clip, "video_id", None),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time": _f(clip.start_time, 3),
        "end_time": _f(clip.end_time, 3),
        "frame_count": frame_count,
        "duration": _f(duration, 3),
        # Stage4 sliding-window semantics: 8.0 for fixed windows, None for 1:1
        # segment clips (where clip duration IS the window length).
        "window_seconds": ctx["window_seconds"],
        "stage_03": _clip_stage03(clip, ctx),
        "stage_05": _clip_stage05(clip_id, ctx),
        "stage_05_1": _clip_stage05_1(clip_id, ctx),
        "stage_05_2": _s52_entry(clip_id, ctx),
        "stage_05_3": _s53_entry(clip_id, ctx),
        "stage_06": _s06_entry(clip_id, ctx),
        "stage_07": _s07_entry(clip_id, ctx),
        "stage_07_1": _s07_1_entry(clip_id, ctx),
        "stage_08": _s08_entry(clip_id, ctx),
        "stage_09": _s09_entry(clip_id, ctx),
        "stage_09_5": _s09_5_entry(clip_id, ctx),
        "stage_10": _s10_entry(clip_id, ctx),
        "stage_11": _s11_entry(clip_id, ctx),
        # Phase2 fields (filled by merge_phase2 after Stage17 export).
        "stage_12": None,
        "stage_13": None,
        "stage_14": None,
        "stage_17": None,
    }
    return rec


def _clip_stage03(clip: Any, ctx: dict) -> Optional[dict]:
    scene = next((s for s in ctx["scenes"] if s.scene_id == clip.scene_id), None)
    if scene is None:
        return None
    fr = scene.filter_result
    return {
        "status": "passed" if (fr is not None and fr.passed) else "rejected" if fr is not None else None,
        "passed": bool(fr.passed) if fr is not None else None,
        "reject_reasons": list(fr.reject_reasons) if fr is not None else [],
        "coarse_face_info": _coarse_face_to_dict(scene.coarse_face_info),
    }


def _clip_stage05(clip_id: str, ctx: dict) -> Optional[dict]:
    fs = ctx["face_by_id"].get(clip_id)
    if fs is None:
        return None
    detections = [fa for f in fs.frames for fa in f.faces]
    return {
        "status": "passed",
        "frames": len(fs.frames),
        "face_detections": len(detections),
        "max_faces_in_frame": max((len(f.faces) for f in fs.frames), default=0),
        "avg_confidence": _f(_mean([float(fa.confidence) for fa in detections])),
    }


def _clip_stage05_1(clip_id: str, ctx: dict) -> Optional[dict]:
    ts = ctx["tracked_by_id"].get(clip_id)
    if ts is None:
        return None
    track_ids = sorted({fa.track_id for f in ts.frames for fa in f.faces
                        if getattr(fa, "track_id", None) is not None})
    return {
        "status": "passed",
        "track_count": len(track_ids),
        "track_ids": track_ids,
    }


def _build_clip_records(ctx: dict) -> List[dict]:
    """Clip records in stage_04 chronological order, plus any clip ids that
    appear in downstream stage outputs but not in the clips list (defensive)."""
    seen: Set[str] = set()
    records: List[dict] = []
    for clip in ctx["clips"]:
        seen.add(clip.clip_id)
        records.append(_clip_record(clip, ctx))
    extra: Set[str] = set()
    for bucket in ("face_by_id", "tracked_by_id", "filtered_by_id",
                   "stabilized_by_id", "pose_by_id", "av_by_id", "align_by_id",
                   "qual_by_id", "metrics_by_id"):
        extra |= set(ctx[bucket].keys())
    for cid in sorted(extra - seen):
        records.append(_clip_record(_PlaceholderClip(cid), ctx))
    return records


class _PlaceholderClip:
    """Minimal ClipInfo stand-in for clip ids found in downstream stage
    outputs but missing from env['clips'] (should not happen in practice)."""

    def __init__(self, clip_id: str) -> None:
        self.clip_id = clip_id
        self.scene_id = None
        self.video_id = None
        self.start_frame = 0
        self.end_frame = 0
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        self.duration = 0.0




# ---------------------------------------------------------------------------
# Stage details (aggregate per-stage statistics)
# ---------------------------------------------------------------------------


def _build_stage_details(ctx: dict, config: dict) -> dict:
    """Aggregate statistics per stage; complements the ``stages`` status map.

    Each entry mirrors what the corresponding stage's ``output_summary`` /
    on-disk artifacts would have said — but is derived from the in-memory
    env so it survives intermediate-output deletion.
    """
    details: dict = {}
    stage4_cfg = config.get("stage_04") or {}

    # --- Stage 0 / 0.5 / 1: source video facts --------------------------------
    vi = ctx["video_info"]
    if vi is not None:
        details["stage_00"] = {
            "video_id": vi.video_id,
            "video_path": _p(vi.video_path),
            "width": vi.width,
            "height": vi.height,
            "fps": _f(getattr(vi, "original_fps", None) or getattr(vi, "fps", None)),
            "total_frames": int(vi.total_frames),
            "duration": _f(vi.duration, 3),
            "codec": vi.codec,
            "has_audio": bool(vi.has_audio),
        }
    ri = ctx["resolution_info"]
    if ri is not None:
        details["stage_00_5"] = {
            "width": ri.width,
            "height": ri.height,
            "min_width": ri.min_width,
            "min_height": ri.min_height,
            "is_valid": bool(ri.is_valid),
            "reject_reason": ri.reject_reason,
        }
    s1 = ctx["stage1_out"]
    if s1:
        v1 = s1.get("video_info")
        a1 = s1.get("audio_info")
        details["stage_01"] = {
            "video_path": _p(s1.get("video_path") or (v1.video_path if v1 else None)),
            "target_fps": _f(s1.get("target_fps") or (v1.target_fps if v1 else None)),
            "resolution": list(v1.resolution) if v1 and v1.resolution is not None else None,
            "codec": v1.codec if v1 else None,
            "audio_path": _p(s1.get("audio_path") or (a1.audio_path if a1 else None)),
            "audio": {
                "sample_rate": a1.sample_rate if a1 else None,
                "channels": a1.channels if a1 else None,
                "duration": _f(a1.duration, 3) if a1 else None,
            } if a1 else None,
        }

    # --- Stage 2 / 3: scene detection + coarse screening ----------------------
    scenes = ctx["scenes"]
    if scenes:
        details["stage_02"] = {
            "scene_count": len(scenes),
            "total_duration_s": round(sum(float(s.duration) for s in scenes), 3),
            "min_duration_s": round(min(float(s.duration) for s in scenes), 3),
            "max_duration_s": round(max(float(s.duration) for s in scenes), 3),
            "scenes": [_scene_to_dict(s) for s in scenes],
        }
    if scenes:
        reason_counts: dict = {}
        passed_count = 0
        rejected_count = 0
        scene_rows = []
        for s in scenes:
            fr = s.filter_result
            passed = bool(fr.passed) if fr is not None else None
            if passed is True:
                passed_count += 1
            elif passed is False:
                rejected_count += 1
                for r in (fr.reject_reasons if fr else []):
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            scene_rows.append({
                "scene_id": s.scene_id,
                "passed": passed,
                "reject_reasons": list(fr.reject_reasons) if fr else [],
                "coarse_face_info": _coarse_face_to_dict(s.coarse_face_info),
            })
        details["stage_03"] = {
            "scene_count": len(scenes),
            "passed_count": passed_count,
            "rejected_count": rejected_count,
            "reason_counts": reason_counts,
            "scenes": scene_rows,
        }

    # --- Stage 4: sliding-window clip generation ------------------------------
    clips = ctx["clips"]
    details["stage_04"] = {
        "config": {
            "mode": stage4_cfg.get("mode", "sliding_window"),
            "window_size": stage4_cfg.get("window_size", 8),
            "stride": stage4_cfg.get("stride", 4),
            "target_fps": stage4_cfg.get("target_fps", 25),
        },
        "window_seconds": ctx["window_seconds"],
        "clip_count": len(clips),
        "source_scene_count": len({c.scene_id for c in clips if c.scene_id}),
        "clips": [
            {
                "clip_id": c.clip_id,
                "scene_id": c.scene_id,
                "start_time": _f(c.start_time, 3),
                "end_time": _f(c.end_time, 3),
                "duration": _f(c.duration, 3),
                "frame_count": int(c.frame_count),
            }
            for c in clips
        ],
    }

    # --- Stage 5 face-resolution pre-filter (video-level hard gate) ------------
    res_audit = ctx["resolution_audit"]
    if res_audit:
        details["stage_05_face_resolution_filter"] = {
            "enabled": bool((config.get("stage_05_face_resolution_filter") or {}).get("enabled", False)),
            "passed": bool(res_audit.get("passed")),
            "reject_reason": res_audit.get("reject_reason"),
            "expected_frame_count": res_audit.get("expected_frame_count"),
            "pass_frame_count": res_audit.get("pass_frame_count"),
            "valid_face_frame_count": res_audit.get("valid_face_frame_count"),
            "qualified_ratio": _f(res_audit.get("qualified_ratio")),
            "face_short_side_min": res_audit.get("face_short_side_min"),
            "face_short_side_median": _f(res_audit.get("face_short_side_median")),
            "face_short_side_max": res_audit.get("face_short_side_max"),
            "status_counts": res_audit.get("status_counts"),
        }

    # --- Stage 5 / 5.1: detection + tracking ----------------------------------
    fseq = ctx["face_sequences"]
    if fseq:
        detections = [fa for fs in fseq for f in fs.frames for fa in f.faces]
        details["stage_05"] = {
            "sequence_count": len(fseq),
            "total_frames": sum(len(fs.frames) for fs in fseq),
            "total_face_detections": len(detections),
            "max_faces_in_frame": max((len(f.faces) for fs in fseq for f in fs.frames), default=0),
            "avg_confidence": _f(_mean([float(fa.confidence) for fa in detections])),
        }
    tseq = ctx["tracked_sequences"]
    if tseq:
        details["stage_05_1"] = {
            "sequence_count": len(tseq),
            "unique_track_count": len({
                fa.track_id for fs in tseq for f in fs.frames for fa in f.faces
                if getattr(fa, "track_id", None) is not None
            }),
        }

    # --- Stage 5.2 / 5.3: hard filters (audit-backed) -------------------------
    details["stage_05_2"] = _filter_stage_details(ctx, "filtered_sequences", "tracked_sequences", "audit_52")
    details["stage_05_3"] = _filter_stage_details(ctx, "stabilized_sequences", "filtered_sequences", "audit_53")

    # --- Stage 6 / 7 / 7.1 / 8: quality-signal extraction ---------------------
    pose_all = [p for ps in ctx["pose_results"] for p in ps.poses]
    if pose_all:
        ratios = [
            _pose_ratio(ps.poses, ctx["yaw_th"], ctx["pitch_th"], ctx["roll_th"], ctx["min_conf"])
            for ps in ctx["pose_results"]
        ]
        ratios = [r for r in ratios if r is not None]
        details["stage_06"] = {
            "clip_count": len(ctx["pose_results"]),
            "pose_count": len(pose_all),
            "mean_yaw": _f(_mean([float(p.yaw) for p in pose_all])),
            "mean_pitch": _f(_mean([float(p.pitch) for p in pose_all])),
            "mean_roll": _f(_mean([float(p.roll) for p in pose_all])),
            "pose_ratio_mean": _mean(ratios),
        }
    av_all = ctx["av_results"]
    if av_all:
        details["stage_07"] = {
            "clip_count": len(av_all),
            "mean_offset_frames": _f(_mean([float(a.offset) for a in av_all]), 3),
            "mean_lse_c": _f(_mean([float(a.sync_lse_c) for a in av_all])),
            "mean_lse_d": _f(_mean([float(a.sync_lse_d) for a in av_all])),
        }
    al_all = ctx["alignment_results"]
    if al_all:
        details["stage_07_1"] = {
            "clip_count": len(al_all),
            "corrected_count": sum(1 for a in al_all if a.correction_applied),
            "already_aligned_count": sum(1 for a in al_all if not a.correction_applied),
            "severe_offset_unavailable_count": sum(
                1 for a in al_all if a.alignment_status == "severe_offset_unavailable"),
        }
    q_all = [float(x.quality_score) for qs in ctx["quality_results"] for x in qs.qualities]
    if q_all:
        details["stage_08"] = {
            "clip_count": len(ctx["quality_results"]),
            "score_count": len(q_all),
            "quality_mean": _f(_mean(q_all)),
            "quality_min": _f(min(q_all)),
            "quality_max": _f(max(q_all)),
        }

    # --- Stage 9 / 9.5: fusion + export ---------------------------------------
    metrics = ctx["quality_metrics"]
    if metrics:
        reason_counts = {}
        for m in metrics:
            for r in m.reject_reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        details["stage_09"] = {
            "clip_count": len(metrics),
            "accepted_count": sum(1 for m in metrics if m.is_accepted),
            "rejected_count": sum(1 for m in metrics if not m.is_accepted),
            "mean_pose_score": _f(_mean([m.pose_score for m in metrics])),
            "mean_sync_score": _f(_mean([m.sync_score for m in metrics])),
            "mean_quality_score": _f(_mean([m.quality_score for m in metrics])),
            "mean_fusion_score": _f(_mean([m.fusion_score for m in metrics])),
            "reason_counts": reason_counts,
        }
    er = ctx["export_result"]
    if er is not None:
        details["stage_09_5"] = {
            "accepted_count": er.get("exported_videos") if isinstance(er.get("exported_videos"), int) else len(ctx["accepted_ids"]),
            "exported_videos": er.get("exported_videos"),
            "exported_audio": er.get("exported_audio"),
            "manifest_entries": er.get("manifest_entries"),
            "failures": er.get("failures"),
        }

    # --- Stage 10 / 11: landmark + alignment ----------------------------------
    landmarks = ctx["landmarks"]
    if landmarks:
        details["stage_10"] = {"landmark_count": len(landmarks),
                               "clip_count": len({lm.clip_id for lm in landmarks})}
    aligned = ctx["aligned"]
    if aligned:
        details["stage_11"] = {
            "aligned_count": len(aligned),
            "clip_count": len({a.clip_id for a in aligned}),
            "total_aligned_frames": len(aligned),
        }

    return details


def _filter_stage_details(ctx: dict, output_key: str, input_key: str,
                          audit_key: str) -> dict:
    """Shared stats builder for the Stage5.2 / 5.3 hard filters."""
    outputs = ctx[output_key]
    inputs = ctx[input_key]
    reason_counts: dict = {}
    for rec in (ctx[audit_key].get("rejected") or []):
        for r in (rec.get("reasons") or []):
            reason_counts[r] = reason_counts.get(r, 0) + 1
    return {
        "input_count": len(inputs),
        "accepted_count": len(outputs),
        "rejected_count": len(inputs) - len(outputs),
        "reason_counts": reason_counts,
    }




# ---------------------------------------------------------------------------
# Phase 1 document assembly
# ---------------------------------------------------------------------------


def _source_video(ctx: dict) -> dict:
    vi = ctx["video_info"]
    return {
        "video_id": vi.video_id if vi else None,
        "path": _p(vi.video_path) if vi else None,
        "width": vi.width if vi else None,
        "height": vi.height if vi else None,
        "fps": _f(getattr(vi, "original_fps", None) or getattr(vi, "fps", None)) if vi else None,
        "total_frames": int(vi.total_frames) if vi else None,
        "duration": _f(vi.duration, 3) if vi else None,
    }


def _build_audit(ctx: dict) -> dict:
    """Stage5.2 / 5.3 reject audit — per-clip reject reasons recorded while
    the stages were running (survives intermediate-output deletion). Also
    carries the Stage 5 face-resolution pre-filter's video-level decision."""
    return {
        "stage_05_face_resolution_filter": dict(ctx["resolution_audit"])
        if ctx["resolution_audit"] else {},
        "stage_05_2_single_face_filter": {
            "input_count": len(ctx["tracked_sequences"]),
            "accepted_count": len(ctx["filtered_sequences"]),
            "rejected_count": len(ctx["tracked_sequences"]) - len(ctx["filtered_sequences"]),
            "rejected": list(ctx["audit_52"].get("rejected") or []),
        },
        "stage_05_3_temporal_stability": {
            "input_count": len(ctx["filtered_sequences"]),
            "accepted_count": len(ctx["stabilized_sequences"]),
            "rejected_count": len(ctx["filtered_sequences"]) - len(ctx["stabilized_sequences"]),
            "rejected": list(ctx["audit_53"].get("rejected") or []),
        },
    }


def _build_final_assets_phase1(ctx: dict) -> dict:
    """Phase1 skeleton: the export-time asset inventory is completed in
    phase2, but the accepted-clip counts are preserved here now."""
    er = ctx["export_result"]
    return {
        "partial": True,
        "note": "final_assets completed in phase2 (Stage17 multimodal export)",
        "accepted_clip_count": len(ctx["accepted_ids"]),
        "exported_videos": er.get("exported_videos") if er else None,
        "exported_audio": er.get("exported_audio") if er else None,
        "assets": [],
    }


def build_phase1(config: dict, args: Any, results: Sequence[dict],
                 env: dict, final_status: str, context: Any,
                 test_base: Any) -> dict:
    """Assemble the complete Phase-1 manifest document (pure, no I/O).

    Called by ``scripts/stage0_17_e2e_validate.py::_finalize`` right after
    ``stage0_11_results.json`` is written.  ``args``/``final_status`` mirror
    the e2e driver's own values so the manifest documents the run exactly as
    the driver saw it.
    """
    test_base = Path(test_base).resolve()
    run_root = test_base.parent
    ctx = _build_run_ctx(config, env, context, test_base, run_root)

    input_video = getattr(args, "input_video", None) or getattr(args, "video", None)
    config_path = getattr(args, "config", None)

    doc = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": _utc_now(),
        "updated_at": None,
        "run": {
            "run_root": str(run_root),
            "test_base": str(test_base),
            "input_video": _p(input_video),
            "config_path": _p(config_path),
            "pipeline_mode": ctx["pipeline_mode"],
            "final_status": final_status,
            "window_seconds": ctx["window_seconds"],
            "note": (
                "Phase-1 production statistics are captured in-memory so they "
                "survive deletion of intermediate outputs. Stage12-17 are "
                "merged by the Stage17 exporter (phase2)."
            ),
        },
        "source_video": _source_video(ctx),
        "config_snapshot": config_snapshot(config),
        "stages": _build_stages(results),
        "stage_details": _build_stage_details(ctx, config),
        "clips": _build_clip_records(ctx),
        "final_assets": _build_final_assets_phase1(ctx),
        "audit": _build_audit(ctx),
        "retention": {
            "required": [
                "pipeline_manifest.json",
                "stage17_multimodal_export/",
            ],
            "note": (
                "Intermediate stage outputs under the test base may be deleted; "
                "only the manifest (this file) and the Stage17 multimodal "
                "export are required to reconstruct per-run production "
                "statistics."
            ),
        },
        "_phase1_complete": True,
        "_phase2_complete": False,
    }
    return doc


# ---------------------------------------------------------------------------
# Phase 2 merge (Stage12/13/14/17 metadata + final asset inventory)
# ---------------------------------------------------------------------------


def _feature_meta_to_stage(sid: str, meta: dict) -> dict:
    """Per-clip Stage12/13/14 record from the official-runner metadata JSON."""
    if sid == "stage_12":
        return {
            "status": "passed",
            "num_frames": meta.get("num_frames"),
            "backend": meta.get("backend"),
            "jaw_source": meta.get("jaw_source"),
            "head_source": meta.get("head_source"),
            "expression_source": meta.get("expression_source"),
            "expression_dim": meta.get("expression_dim"),
            "jaw_pose_dim": meta.get("jaw_pose_dim"),
            "head_pose_dim": meta.get("head_pose_dim"),
            "shapecode_dim": meta.get("shapecode_dim"),
        }
    if sid == "stage_13":
        return {
            "status": "passed",
            "num_frames": meta.get("num_frames"),
            "gaze_dim": meta.get("gaze_dim"),
        }
    if sid == "stage_14":
        return {
            "status": "passed",
            "num_frames": meta.get("num_frames"),
            "feature_dim": meta.get("feature_dim"),
            "source_audio": meta.get("source_audio"),
        }
    return {"status": "passed", "num_frames": meta.get("num_frames")}


def _feature_stage_details(sid: str, per_clip: dict) -> dict:
    rows = []
    total_frames = 0
    for cid in sorted(per_clip):
        meta = per_clip[cid]
        nf = meta.get("num_frames")
        if isinstance(nf, (int, float)):
            total_frames += int(nf)
        dim = meta.get("gaze_dim") or meta.get("feature_dim") or meta.get("expression_dim")
        rows.append({"clip_id": cid, "num_frames": nf, "feature_dim": dim})
    return {"clip_count": len(rows), "num_frames_total": total_frames, "clips": rows}


def _s17_meta_to_stage(meta: dict) -> dict:
    return {
        "status": "passed",
        "fps": meta.get("fps"),
        "visual_dim": meta.get("visual_dim"),
        "audio_dim": meta.get("audio_dim"),
        "visual_layout": meta.get("visual_layout"),
        "head_source": meta.get("head_source"),
        "jaw_source": meta.get("jaw_source"),
        "expression_source": meta.get("expression_source"),
        "gaze_source": meta.get("gaze_source"),
        "audio_source": meta.get("audio_source"),
        "num_frames": meta.get("num_frames"),
        "valid_length": meta.get("valid_length"),
        "source_stage": meta.get("source_stage"),
        "n_input": meta.get("n_input"),
    }


def _expected_clip_ids(doc: dict) -> Set[str]:
    """Clip ids that reached Stage11 (open-domain accepted) and are therefore
    expected to have Stage12/13/14 feature outputs in phase2."""
    ids: Set[str] = set()
    for rec in doc.get("clips", []):
        cid = rec.get("clip_id")
        s11 = rec.get("stage_11")
        if cid and isinstance(s11, dict) and s11.get("status") == "passed":
            ids.add(cid)
    # fallback: stage_09 accepted flag when stage_11 was never populated
    if not ids:
        for rec in doc.get("clips", []):
            cid = rec.get("clip_id")
            s09 = rec.get("stage_09") or {}
            if cid and s09.get("is_accepted"):
                ids.add(cid)
    return ids


def _load_export_report(export_dir: Path, logger: Any) -> Optional[dict]:
    """Non-fatal load of the Stage17 export report (None on legacy runs)."""
    path = export_dir / EXPORT_REPORT_FILENAME
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - non-fatal
        _warn(logger, f"pipeline_manifest: skipping malformed {path}: {exc}")
        return None


def _s17_report_to_stage(entry: dict) -> dict:
    """Per-clip Stage17 record from an ``export_report.json`` clips entry."""
    rec: dict = {
        "status": "passed" if entry.get("status") == "PASS" else "failed",
        "n_flame": entry.get("n_flame"),
        "n_gaze": entry.get("n_gaze"),
        "n_audio": entry.get("n_audio"),
        "num_frames": entry.get("num_frames"),
        "valid_length": entry.get("valid_length"),
        "gaze_missing_frames": entry.get("gaze_missing_frames"),
        "audio_missing_frames": entry.get("audio_missing_frames"),
    }
    if entry.get("failed_reason"):
        rec["failed_reason"] = entry["failed_reason"]
    if entry.get("checks"):
        rec["checks"] = entry["checks"]
    return rec


def _merge_feature_stages(doc: dict, base: Path, logger: Any) -> None:
    """Merge Stage12/13/14 official-runner metadata into the manifest."""
    clips_by_id = {c.get("clip_id"): c for c in doc.get("clips", [])}
    expected = _expected_clip_ids(doc)
    for sid, (subdir, fname) in FEATURE_SUBDIRS.items():
        root = base / subdir
        npz_name = FEATURE_NPZ[sid]
        per_clip: dict = {}   # clip_id -> metadata (successful)
        failed: dict = {}     # clip_id -> failure reason
        if root.is_dir():
            for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                mf = clip_dir / fname
                meta = None
                if mf.is_file():
                    try:
                        with mf.open("r", encoding="utf-8") as fh:
                            meta = json.load(fh)
                    except Exception as exc:  # noqa: BLE001 - keep going
                        _warn(logger, f"pipeline_manifest: skipping malformed {mf}: {exc}")
                if meta is None:
                    continue
                clip_id = meta.get("clip_id") or clip_dir.name
                if (root / clip_id / npz_name).is_file():
                    per_clip[clip_id] = meta
                else:
                    failed[clip_id] = f"{npz_name} missing"
            for clip_id in sorted(expected - set(per_clip) - set(failed)):
                feature_npz = root / clip_id / npz_name
                failed[clip_id] = (
                    f"{npz_name} missing" if not feature_npz.is_file()
                    else "metadata missing"
                )
        for clip_id, meta in per_clip.items():
            rec = clips_by_id.get(clip_id)
            if rec is None:
                rec = {"clip_id": clip_id}
                doc["clips"].append(rec)
                clips_by_id[clip_id] = rec
            rec[sid] = _feature_meta_to_stage(sid, meta)
        for clip_id, reason in failed.items():
            rec = clips_by_id.get(clip_id)
            if rec is None:
                rec = {"clip_id": clip_id}
                doc["clips"].append(rec)
                clips_by_id[clip_id] = rec
            rec[sid] = {"status": "failed", "reason": reason, "num_frames": None}

        n_expected = len(expected)
        n_ok = len(per_clip)
        n_fail = len(failed)
        if not root.is_dir():
            status = "SKIPPED"
            note = "no stage output directory (stage not run for this base)"
        elif n_expected == 0:
            status = "PASS" if (n_ok or n_fail) else "SKIPPED"
            note = ("no accepted clips for this stage"
                    if not (n_ok or n_fail) else "merged from metadata")
        elif n_fail == 0:
            status = "PASS"
            note = "merged from official-runner metadata (phase2)"
        elif n_ok == 0:
            status = "FAIL"
            note = f"{n_fail}/{n_expected} expected clips missing feature payload"
        else:
            status = "PARTIAL"
            note = f"{n_ok}/{n_expected} clips produced features; {n_fail} missing"
        doc.setdefault("stages", {})[sid] = {
            "status": status,
            "phase": 2,
            "expected_clip_count": n_expected,
            "passed_clip_count": n_ok,
            "failed_clip_count": n_fail,
            "failed_clips": {cid: reason for cid, reason in sorted(failed.items())},
            "note": note,
        }
        doc.setdefault("stage_details", {})[sid] = _feature_stage_details(sid, per_clip)


def _merge_stage17(doc: dict, export_dir: Path, logger: Any) -> None:
    """Merge the Stage17 export report (or metadata fallback) into the manifest.

    When ``export_report.json`` exists it is authoritative: it carries the
    per-clip modality availability + matching statistics + export checks, and is
    what turns a Stage13 gaze failure (missing frames -> final flame/gaze/audio
    unmatchable) into an explicit ``failed`` clip/status. Only when the report is
    absent do we fall back to the legacy metadata-directory scan (success-only).
    """
    clips_by_id = {c.get("clip_id"): c for c in doc.get("clips", [])}
    per_clip: dict = {}
    failed_reasons: List[str] = []
    report = _load_export_report(export_dir, logger)
    if report is not None:
        for clip_id, entry in report.get("clips", {}).items():
            per_clip[clip_id] = entry
            rec = clips_by_id.get(clip_id)
            if rec is None:
                rec = {"clip_id": clip_id}
                doc["clips"].append(rec)
                clips_by_id[clip_id] = rec
            rec["stage_17"] = _s17_report_to_stage(entry)
            if entry.get("status") != "PASS":
                failed_reasons.append(
                    f"{clip_id}: {entry.get('failed_reason') or 'export failed'}"
                )
    else:
        meta_dir = export_dir / "metadata"
        if meta_dir.is_dir():
            for mf in sorted(meta_dir.glob("*.json")):
                try:
                    with mf.open("r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception as exc:  # noqa: BLE001 - keep going
                    _warn(logger, f"pipeline_manifest: skipping malformed {mf}: {exc}")
                    continue
                cid = meta.get("clip_id") or mf.stem
                per_clip[cid] = meta
                rec = clips_by_id.get(cid)
                if rec is None:
                    rec = {"clip_id": cid}
                    doc["clips"].append(rec)
                    clips_by_id[cid] = rec
                rec["stage_17"] = _s17_meta_to_stage(meta)
    count = len(per_clip)
    n_failed = len(failed_reasons)
    if count == 0:
        status = "SKIPPED"
        note = "no Stage17 export report/metadata found"
    elif n_failed == 0:
        status = "PASS"
        note = "merged from Stage17 export report (phase2)"
    elif n_failed == count:
        status = "FAIL"
        note = f"{n_failed}/{count} clips failed Stage17 export"
    else:
        status = "PARTIAL"
        note = f"{n_failed}/{count} clips failed Stage17 export"
    doc.setdefault("stages", {})["stage_17"] = {
        "status": status,
        "phase": 2,
        "clip_count": count,
        "failed_clip_count": n_failed,
        "failed_reasons": failed_reasons,
        "note": note,
    }
    doc.setdefault("stage_details", {})["stage_17"] = {
        "clip_count": count,
        "clips": sorted(per_clip),
    }


def _finalize_assets(doc: dict, base: Path, export_dir: Path,
                     run_root: Path, logger: Any) -> None:
    """Build the final asset inventory from the merged Stage17 records."""

    def _rel(p: Any) -> str:
        try:
            pp = Path(p).resolve()
            return PurePosixPath(pp.relative_to(run_root)).as_posix()
        except (ValueError, OSError):
            return str(p)

    assets: List[dict] = []
    failed_clips: List[dict] = []
    for rec in doc.get("clips", []):
        clip_id = rec.get("clip_id")
        s17 = rec.get("stage_17") or {}
        if not clip_id or not s17:
            continue
        status = s17.get("status")
        if status == "failed":
            failed_clips.append({
                "clip_id": clip_id,
                "failed_reason": s17.get("failed_reason"),
            })
            continue
        if status != "passed":
            continue  # unknown/clipped legacy record — not a usable asset
        assets.append({
            "clip_id": clip_id,
            "flame": _rel(export_dir / "flame" / f"{clip_id}.npy"),
            "audio": _rel(export_dir / "audio" / f"{clip_id}.npy"),
            "timestamp": _rel(export_dir / "timestamp" / f"{clip_id}.npz"),
            "metadata": _rel(export_dir / "metadata" / f"{clip_id}.json"),
            "num_frames": s17.get("num_frames"),
            "valid_length": s17.get("valid_length"),
            "visual_dim": s17.get("visual_dim"),
            "audio_dim": s17.get("audio_dim"),
            "duration": rec.get("duration"),
        })
    doc["final_assets"] = {
        "partial": bool(failed_clips),
        "asset_count": len(assets),
        "accepted_clip_count": len(assets),
        "failed_clip_count": len(failed_clips),
        "failed_clips": sorted(failed_clips, key=lambda c: c["clip_id"]),
        "export_dir": _rel(export_dir),
        "manifest_csv": _rel(export_dir / "manifest.csv"),
        "assets": sorted(assets, key=lambda a: a["clip_id"]),
        "note": "final multimodal assets produced by Stage17 export (phase2)",
    }


def merge_phase2(base: Any, export_dir: Any, logger: Any = None) -> dict:
    """Load the Phase-1 manifest, merge Stage12/13/14/17 metadata + final
    asset inventory, and atomically re-save.

    Non-fatal: a manifest failure never fails the export.  If no Phase-1
    manifest exists a minimal Phase-2 document is created instead.
    """
    base = Path(base).resolve()
    export_dir = Path(export_dir).resolve()
    run_root = base.parent
    path = manifest_path(run_root)
    doc = load(path, logger)
    if doc is None:
        _warn(logger, f"pipeline_manifest not found at {path}; creating phase2-only doc")
        doc = {
            "manifest_version": MANIFEST_VERSION,
            "generated_at": _utc_now(),
            "run": {"run_root": str(run_root), "test_base": str(base)},
            "stages": {},
            "stage_details": {},
            "clips": [],
            "final_assets": {"partial": True, "assets": []},
        }
    doc.setdefault("clips", [])
    doc.setdefault("stages", {})
    doc.setdefault("stage_details", {})

    _merge_feature_stages(doc, base, logger)
    _merge_stage17(doc, export_dir, logger)
    _finalize_assets(doc, base, export_dir, run_root, logger)

    doc["updated_at"] = _utc_now()
    doc["_phase1_complete"] = doc.get("_phase1_complete", True)
    doc["_phase2_complete"] = True
    save(doc, path, logger)
    return doc


#!/usr/bin/env python3
"""Stage 0-11 Full Pipeline Integration Test Driver.

NEW standalone test harness (NOT part of the pipeline architecture).
Verifies that the already-implemented stages 0 through 11 can run
end-to-end from a single input video, using ONLY the existing stage
classes / interfaces / config (no refactoring).

Usage (inside the face_pipeline_lxs container, where the project is
mounted at /workspace/Face_Pipeline_V2):

    python tests/stage0_11_integration_test.py \\
        --video tests/006_Bv7wdHRhifM_1min.mp4 \\
        --config configs/config.yaml

Notes
-----
* Config is loaded from configs/config.yaml (ConfigLoader) WITHOUT editing
  the file.  Only ``storage.output_dir`` and ``stage_01.output_dir`` are
  pointed at a fresh test-base under ``outputs/stage0_11_test`` so the run
  is self-contained and does not clobber earlier validation outputs.
* Each stage is executed via its standard ``BaseStage.run()`` lifecycle
  (validate_input -> process -> validate_output).
* If an upstream stage produces no data that downstream stages need
  (e.g. zero accepted scenes -> zero clips), downstream stages are marked
  SKIPPED rather than FAILED — that is a data-flow consequence, not a
  stage bug.
* A report is written to tests/reports/stage0_11_integration_test_report.md
  plus a machine-readable tests/reports/stage0_11_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------- path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------- pipeline deps
from face_pipeline_v2.config.config_loader import ConfigLoader
from face_pipeline_v2.models.model_manager import ModelManager
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.utils.logger import PipelineLogger

# ---------------------------------------------------------------- stage imports
from face_pipeline_v2.stages.stage_00_video_check.stage import Stage0VideoIntegrityCheck
from face_pipeline_v2.stages.stage_00_5_resolution_filter.stage import Stage0_5ResolutionFilter
from face_pipeline_v2.stages.stage_01_standardization.stage import Stage1AVStandardization
from face_pipeline_v2.stages.stage_02_scene_detection.stage import Stage2SceneSegmentation
from face_pipeline_v2.stages.stage_03_face_screening.stage import Stage3CoarseFaceScreening
from face_pipeline_v2.stages.stage_04_temporal_clip_generation.stage import Stage4TemporalClipGeneration
from face_pipeline_v2.stages.stage_05_face_detection.stage import Stage5FaceDetection
from face_pipeline_v2.stages.stage_06_pose.stage import Stage6PoseEstimation
from face_pipeline_v2.stages.stage_07_sync.stage import Stage7SyncDetection
from face_pipeline_v2.stages.stage_07_1_av_alignment.stage import Stage7_1AVAlignment
from face_pipeline_v2.stages.stage_08_quality.stage import Stage8QualityAssessment
from face_pipeline_v2.stages.stage_09_quality_fusion.stage import Stage9QualityFusion
from face_pipeline_v2.stages.stage_09_5_clip_export.stage import Stage9_5ClipExport
from face_pipeline_v2.stages.stage_10_landmark.stage import Stage10Landmark
from face_pipeline_v2.stages.stage_11_alignment.stage import Stage11Alignment
from face_pipeline_v2.stages.stage_05_1_face_tracking.stage import Stage5_1FaceTracking
from face_pipeline_v2.stages.stage_05_2_single_face_filter.stage import Stage5_2SingleFaceFilter
from face_pipeline_v2.stages.stage_05_3_temporal_stability.stage import Stage5_3TemporalStability
from face_pipeline_v2.stages.stage_12_flame.stage import Stage12FLAME

# ---------------------------------------------------------------- recording
STATUS_OK = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIPPED"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_stage(
    stage_name: str,
    fn: Callable[[], Any],
    results: List[Dict[str, Any]],
) -> Any:
    """Execute one stage's run() and record PASS/FAIL + timing.

    Returns the stage output on success, else None (error recorded).
    """
    entry: Dict[str, Any] = {
        "stage": stage_name,
        "status": STATUS_OK,
        "started": _now_utc(),
        "elapsed_s": None,
        "output_summary": "",
        "error": None,
    }
    t0 = time.monotonic()
    try:
        out = fn()
        entry["elapsed_s"] = round(time.monotonic() - t0, 2)
        entry["finished"] = _now_utc()
        results.append(entry)
        return out
    except Exception as exc:  # noqa: BLE001 - test harness must capture all
        entry["status"] = STATUS_FAIL
        entry["elapsed_s"] = round(time.monotonic() - t0, 2)
        entry["finished"] = _now_utc()
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        results.append(entry)
        return None


def mark_skipped(stage_name: str, reason: str, results: List[Dict[str, Any]]) -> None:
    results.append(
        {
            "stage": stage_name,
            "status": STATUS_SKIP,
            "started": _now_utc(),
            "elapsed_s": 0.0,
            "output_summary": "",
            "error": None,
            "skip_reason": reason,
        }
    )


def record(stage_name: str, summary: str, results: List[Dict[str, Any]]) -> None:
    """Attach an output summary to the most recently recorded stage."""
    if results and results[-1]["stage"] == stage_name:
        results[-1]["output_summary"] = summary

def _empty_skip_check(value: Any) -> bool:
    """True if value is an empty list/dict (downstream stages cannot run)."""
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--test-base", default="outputs/stage0_11_test",
                        help="Relative test output base under the project root")
    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)
    test_base = Path(args.test_base)

    # ------------------------------------------------------------------ config
    loader = ConfigLoader(config_path=config_path)
    config = loader.load()
    loader.validate(config)

    # Point output dirs at a fresh, self-contained test base (driver-only choice;
    # keeps the run isolated and avoids clobbering earlier validation outputs).
    config["storage"]["output_dir"] = str(test_base)
    config["stage_01"]["output_dir"] = str(test_base / "videos")

    # ------------------------------------------------------------------ context
    reports_dir = PROJECT_ROOT / "tests" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = PipelineLogger(name="stage0_11_integration_test", log_dir=reports_dir)
    model_manager = ModelManager(device="auto")
    context = PipelineContext(
        config=config,
        logger=logger,
        model_manager=model_manager,
        runtime_info={"video_id": video_path.stem},
    )

    results: List[Dict[str, Any]] = []
    env: Dict[str, Any] = {}
    logger.info("=" * 72)
    logger.info("Stage 0-11 integration test starting")
    logger.info(f"video={video_path}  config={config_path.resolve()}  test_base={test_base}")

    # ================================================================ STAGE 0
    env["video_info"] = run_stage(
        "Stage 0 (FFprobe integrity check)",
        lambda: Stage0VideoIntegrityCheck().run(video_path, context),
        results,
    )
    if env["video_info"] is None:
        logger.error("Stage 0 failed - aborting (no metadata).")
        return _finalize(args, results, env, "FAIL")
    vi = env["video_info"]
    record("Stage 0 (FFprobe integrity check)",
           f"id={vi.video_id} | {vi.width}x{vi.height} | {vi.original_fps:.2f}fps | "
           f"{vi.duration:.1f}s | {vi.total_frames} frames | audio={vi.has_audio}",
           results)

    # ============================================================ STAGE 0.5
    env["resolution_info"] = run_stage(
        "Stage 0.5 (Resolution filter)",
        lambda: Stage0_5ResolutionFilter().run(env["video_info"], context),
        results,
    )
    if env["resolution_info"] is None:
        return _finalize(args, results, env, "FAIL")
    ri = env["resolution_info"]
    record("Stage 0.5 (Resolution filter)",
           f"{ri.width}x{ri.height} vs min {ri.min_width}x{ri.min_height} "
           f"| is_valid={ri.is_valid} | reject={ri.reject_reason}",
           results)
    if not ri.is_valid:
        logger.error("Stage 0.5 rejected the video - aborting.")
        return _finalize(args, results, env, "FAIL")

    # ================================================================ STAGE 1
    env["stage1_out"] = run_stage(
        "Stage 1 (FFmpeg standardization)",
        lambda: Stage1AVStandardization().run(env["video_info"], context),
        results,
    )
    if env["stage1_out"] is None:
        return _finalize(args, results, env, "FAIL")
    std_info = env["stage1_out"]["video_info"]
    audio_info = env["stage1_out"]["audio_info"]
    record("Stage 1 (FFmpeg standardization)",
           f"std_video={std_info.video_path} | audio={audio_info.audio_path}",
           results)

    # ================================================================ STAGE 2
    env["scenes"] = run_stage(
        "Stage 2 (PySceneDetect)",
        lambda: Stage2SceneSegmentation().run(std_info, context),
        results,
    )
    if env["scenes"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 2 (PySceneDetect)", f"{len(env['scenes'])} scene(s)", results)

    # ================================================================ STAGE 3
    env["scenes"] = run_stage(
        "Stage 3 (SCRFD coarse screening)",
        lambda: Stage3CoarseFaceScreening().run(env["scenes"], context),
        results,
    )
    if env["scenes"] is None:
        return _finalize(args, results, env, "FAIL")
    passed = [s for s in env["scenes"] if s.filter_result and s.filter_result.passed]
    rejected = [s for s in env["scenes"] if not (s.filter_result and s.filter_result.passed)]
    record("Stage 3 (SCRFD coarse screening)",
           f"{len(env['scenes'])} scene(s): {len(passed)} passed, {len(rejected)} rejected",
           results)
    if not passed:
        logger.error("Stage 3 rejected all scenes - no clips possible downstream.")
        return _finalize(args, results, env, "FAIL")

    # ================================================================ STAGE 4
    env["clips"] = run_stage(
        "Stage 4 (Sliding-window clips)",
        lambda: Stage4TemporalClipGeneration().run(env["scenes"], context),
        results,
    )
    if env["clips"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 4 (Sliding-window clips)", f"{len(env['clips'])} clip(s)", results)
    if _empty_skip_check(env["clips"]):
        _skip_from("Stage 5 (SCRFD detect/crop)", "Stage 4 produced 0 clips", results)
        return _finalize(args, results, env, "PARTIAL")

    # ================================================================ STAGE 5
    env["face_sequences"] = run_stage(
        "Stage 5 (SCRFD detect/crop)",
        lambda: Stage5FaceDetection().run(env["clips"], context),
        results,
    )
    if env["face_sequences"] is None:
        return _finalize(args, results, env, "FAIL")
    total_faces = sum(len(f.faces) for seq in env["face_sequences"] for f in seq.frames)
    record("Stage 5 (SCRFD detect/crop)",
           f"{len(env['face_sequences'])} sequence(s), {total_faces} face detections",
           results)
    if _empty_skip_check(env["face_sequences"]):
        _skip_from("Stage 5.1 (ByteTrack face tracking)", "Stage 5 produced no face sequences", results)
        return _finalize(args, results, env, "PARTIAL")

    face_sequences = env["face_sequences"]
    clips = env["clips"]
    video_id = vi.video_id

    # ====================================== STAGE 5.1 / 5.2 / 5.3 (enhancements)
    env["tracked_sequences"] = run_stage(
        "Stage 5.1 (ByteTrack face tracking)",
        lambda: Stage5_1FaceTracking().run(face_sequences, context),
        results,
    )
    if env["tracked_sequences"] is None:
        return _finalize(args, results, env, "FAIL")
    uniq_tracks = {
        fa.track_id
        for seq in env["tracked_sequences"]
        for fi in seq.frames
        for fa in fi.faces
        if fa.track_id is not None
    }
    record("Stage 5.1 (ByteTrack face tracking)",
           f"{len(env['tracked_sequences'])} sequence(s), {len(uniq_tracks)} unique track(s)",
           results)
    if _empty_skip_check(env["tracked_sequences"]):
        _skip_from("Stage 5.2 (Single-face filter)", "Stage 5.1 produced no sequences", results)
        return _finalize(args, results, env, "PARTIAL")

    env["filtered_sequences"] = run_stage(
        "Stage 5.2 (Single-face filter)",
        lambda: Stage5_2SingleFaceFilter().run(env["tracked_sequences"], context),
        results,
    )
    if env["filtered_sequences"] is None:
        return _finalize(args, results, env, "FAIL")
    s52_pass = len(env["filtered_sequences"])  # hard filter: output == passed only
    s52_input = len(env["tracked_sequences"])
    record("Stage 5.2 (Single-face filter)",
           f"{s52_input} input: "
           f"{s52_pass} passed, {s52_input - s52_pass} rejected (hard filter, dropped)",
           results)
    if _empty_skip_check(env["filtered_sequences"]):
        _skip_from("Stage 5.3 (Temporal stability)", "Stage 5.2 produced no sequences", results)
        return _finalize(args, results, env, "PARTIAL")

    env["stabilized_sequences"] = run_stage(
        "Stage 5.3 (Temporal stability)",
        lambda: Stage5_3TemporalStability().run(env["filtered_sequences"], context),
        results,
    )
    if env["stabilized_sequences"] is None:
        return _finalize(args, results, env, "FAIL")
    s53_pass = len(env["stabilized_sequences"])  # hard filter: output == passed only
    s53_input = len(env["filtered_sequences"])
    record("Stage 5.3 (Temporal stability)",
           f"{s53_input} input: "
           f"{s53_pass} passed, {s53_input - s53_pass} rejected (hard filter, dropped)",
           results)
    if _empty_skip_check(env["stabilized_sequences"]):
        _skip_from("Stage 6 (6DRepNet pose)", "Stage 5.3 produced no sequences", results)
        return _finalize(args, results, env, "PARTIAL")

    # Enhanced sequences flow downstream (Stage 6/7/8 consume these)
    face_sequences = env["stabilized_sequences"]

    # ================================================================ STAGE 6
    env["pose_sequences"] = run_stage(
        "Stage 6 (6DRepNet pose)",
        lambda: Stage6PoseEstimation().run(face_sequences, context),
        results,
    )
    if env["pose_sequences"] is None:
        return _finalize(args, results, env, "FAIL")
    pose_frames = sum(len(p.poses) for p in env["pose_sequences"])
    record("Stage 6 (6DRepNet pose)",
           f"{len(env['pose_sequences'])} sequence(s), {pose_frames} pose(s)",
           results)

    # ================================================================ STAGE 7
    env["av_sync_infos"] = run_stage(
        "Stage 7 (SyncNet A/V sync)",
        lambda: Stage7SyncDetection().run(
            {
                "sequences": face_sequences,
                "audio_info": audio_info,
                "clip_infos": clips,
                "std_video_path": std_info.video_path,
            },
            context,
        ),
        results,
    )
    if env["av_sync_infos"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 7 (SyncNet A/V sync)", f"{len(env['av_sync_infos'])} clip(s)", results)

    # ============================================================ STAGE 7.1
    env["av_alignments"] = run_stage(
        "Stage 7.1 (A/V alignment)",
        lambda: Stage7_1AVAlignment().run(
            {
                "av_sync_infos": env["av_sync_infos"],
                "std_video_info": std_info,
                # clip_infos 提供 clip 边界回退（AVSyncInfo 已携带边界时被忽略）
                "clip_infos": clips,
            },
            context,
        ),
        results,
    )
    if env["av_alignments"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 7.1 (A/V alignment)", f"{len(env['av_alignments'])} clip(s)", results)

    # ================================================================ STAGE 8
    env["quality_sequences"] = run_stage(
        "Stage 8 (HyperIQA quality)",
        lambda: Stage8QualityAssessment().run(face_sequences, context),
        results,
    )
    if env["quality_sequences"] is None:
        return _finalize(args, results, env, "FAIL")
    qual_frames = sum(len(q.qualities) for q in env["quality_sequences"])
    record("Stage 8 (HyperIQA quality)",
           f"{len(env['quality_sequences'])} sequence(s), {qual_frames} score(s)",
           results)

    # ================================================================ STAGE 9
    env["quality_metrics"] = run_stage(
        "Stage 9 (Quality fusion/filter)",
        lambda: Stage9QualityFusion().run(
            {
                "pose_sequences": env["pose_sequences"],
                "av_sync_infos": env["av_sync_infos"],
                "quality_sequences": env["quality_sequences"],
                "video_id": video_id,
            },
            context,
        ),
        results,
    )
    if env["quality_metrics"] is None:
        return _finalize(args, results, env, "FAIL")
    accepted = [m for m in env["quality_metrics"] if m.is_accepted]
    record("Stage 9 (Quality fusion/filter)",
           f"{len(env['quality_metrics'])} clip(s), {len(accepted)} accepted",
           results)
    accepted_ids = {m.clip_id for m in accepted}

    # ============================================================ STAGE 9.5
    env["export_result"] = run_stage(
        "Stage 9.5 (Clip export)",
        lambda: Stage9_5ClipExport().run(
            {
                "quality_metrics": env["quality_metrics"],
                "clip_infos": clips,
                "std_video_info": std_info,
                "av_alignments": env["av_alignments"],
                "av_sync_infos": env["av_sync_infos"],
            },
            context,
        ),
        results,
    )
    if env["export_result"] is None:
        return _finalize(args, results, env, "FAIL")
    er = env["export_result"]
    record("Stage 9.5 (Clip export)",
           f"accepted={er.get('accepted_count')} exported_videos={er.get('exported_videos')} "
           f"exported_audio={er.get('exported_audio')} manifest={er.get('manifest_entries')} "
           f"failures={len(er.get('failures', []))}",
           results)

    # =============================================================== STAGE 10
    if not accepted_ids:
        _skip_from("Stage 10 (InsightFace landmark)", "Stage 9 accepted 0 clips", results)
        return _finalize(args, results, env, "PARTIAL")
    env["landmarks"] = run_stage(
        "Stage 10 (InsightFace landmark)",
        lambda: Stage10Landmark().run(
            {"face_sequences": face_sequences, "accepted_clip_ids": accepted_ids},
            context,
        ),
        results,
    )
    if env["landmarks"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 10 (InsightFace landmark)", f"{len(env['landmarks'])} landmark(s)", results)

    # =============================================================== STAGE 11
    env["aligned"] = run_stage(
        "Stage 11 (Affine alignment)",
        lambda: Stage11Alignment().run(
            {"landmarks": env["landmarks"], "face_sequences": face_sequences},
            context,
        ),
        results,
    )
    if env["aligned"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 11 (Affine alignment)", f"{len(env['aligned'])} aligned face(s)", results)

    # =============================================================== STAGE 12
    if not env["aligned"]:
        _skip_from("Stage 12 (FLAME extraction)", "Stage 11 produced 0 aligned faces", results)
        return _finalize(args, results, env, "PARTIAL")
    env["flame_features"] = run_stage(
        "Stage 12 (FLAME extraction)",
        lambda: Stage12FLAME().run(env["aligned"], context),
        results,
    )
    if env["flame_features"] is None:
        return _finalize(args, results, env, "FAIL")
    record("Stage 12 (FLAME extraction)",
           f"{len(env['flame_features'])} FLAME feature(s) | "
           f"expcode[100] jawpose[3] globalpose[3] eye_pose[6]",
           results)

    return _finalize(args, results, env, "PASS")

def _skip_from(stage_name: str, reason: str, results: List[Dict[str, Any]]) -> None:
    """Mark ``stage_name`` and every downstream stage as SKIPPED with a reason."""
    mark_skipped(stage_name, reason, results)
    for name in _DOWNSTREAM_NAMES.get(stage_name, []):
        mark_skipped(name, reason, results)
    _DOWNSTREAM_NAMES.clear()  # avoid double-marking on multiple call sites


# Stages that follow a given stage in the DAG (used to mark downstream SKIPPED).
_DOWNSTREAM_NAMES: Dict[str, List[str]] = {
    "Stage 5 (SCRFD detect/crop)": [
        "Stage 5.1 (ByteTrack face tracking)",
        "Stage 5.2 (Single-face filter)",
        "Stage 5.3 (Temporal stability)",
        "Stage 6 (6DRepNet pose)",
        "Stage 7 (SyncNet A/V sync)",
        "Stage 7.1 (A/V alignment)",
        "Stage 8 (HyperIQA quality)",
        "Stage 9 (Quality fusion/filter)",
        "Stage 9.5 (Clip export)",
        "Stage 10 (InsightFace landmark)",
        "Stage 11 (Affine alignment)",
    ],
    "Stage 5.1 (ByteTrack face tracking)": [
        "Stage 5.2 (Single-face filter)",
        "Stage 5.3 (Temporal stability)",
        "Stage 6 (6DRepNet pose)",
        "Stage 7 (SyncNet A/V sync)",
        "Stage 7.1 (A/V alignment)",
        "Stage 8 (HyperIQA quality)",
        "Stage 9 (Quality fusion/filter)",
        "Stage 9.5 (Clip export)",
        "Stage 10 (InsightFace landmark)",
        "Stage 11 (Affine alignment)",
    ],
    "Stage 5.2 (Single-face filter)": [
        "Stage 5.3 (Temporal stability)",
        "Stage 6 (6DRepNet pose)",
        "Stage 7 (SyncNet A/V sync)",
        "Stage 7.1 (A/V alignment)",
        "Stage 8 (HyperIQA quality)",
        "Stage 9 (Quality fusion/filter)",
        "Stage 9.5 (Clip export)",
        "Stage 10 (InsightFace landmark)",
        "Stage 11 (Affine alignment)",
    ],
    "Stage 5.3 (Temporal stability)": [
        "Stage 6 (6DRepNet pose)",
        "Stage 7 (SyncNet A/V sync)",
        "Stage 7.1 (A/V alignment)",
        "Stage 8 (HyperIQA quality)",
        "Stage 9 (Quality fusion/filter)",
        "Stage 9.5 (Clip export)",
        "Stage 10 (InsightFace landmark)",
        "Stage 11 (Affine alignment)",
    ],
    "Stage 6 (6DRepNet pose)": [
        "Stage 7 (SyncNet A/V sync)",
        "Stage 7.1 (A/V alignment)",
        "Stage 8 (HyperIQA quality)",
        "Stage 9 (Quality fusion/filter)",
        "Stage 9.5 (Clip export)",
        "Stage 10 (InsightFace landmark)",
        "Stage 11 (Affine alignment)",
    ],
    "Stage 10 (InsightFace landmark)": ["Stage 11 (Affine alignment)"],
    "Stage 11 (Affine alignment)": ["Stage 12 (FLAME extraction)"],
}


def _finalize(args: Any, results: List[Dict[str, Any]], env: Dict[str, Any], final_status: str) -> int:
    """Write the markdown report + JSON results, print a summary, return exit code."""
    reports_dir = PROJECT_ROOT / "tests" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / "stage0_11_integration_test_report.md"
    json_path = reports_dir / "stage0_11_results.json"

    payload = {
        "final_status": final_status,
        "generated_at": _now_utc(),
        "input_video": str(args.video),
        "config": str(Path(args.config).resolve()),
        "test_base": str(args.test_base),
        "stages": results,
    }
    # Carry the real probed video facts + resolution filter thresholds into the
    # payload so the markdown report reflects the actual input (no hardcoding).
    vi = env.get("video_info")
    if vi is not None:
        payload["video_info"] = {
            "video_id": vi.video_id,
            "width": vi.width,
            "height": vi.height,
            "fps": round(vi.original_fps, 2),
            "total_frames": vi.total_frames,
            "duration": round(vi.duration, 1),
            "has_audio": bool(vi.has_audio),
        }
    ri = env.get("resolution_info")
    if ri is not None:
        payload["resolution_min"] = {"width": ri.min_width, "height": ri.min_height}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _render_markdown(payload)
    md_path.write_text(md, encoding="utf-8")
    print(f"\n===== STAGE 0-11 INTEGRATION TEST: {final_status} =====")
    for r in results:
        print(f"  [{r['status']:8s}] {r['stage']:34s} {r['elapsed_s']}s  {r.get('output_summary', '')}")
        if r.get("error"):
            print(f"           ERROR: {r['error']}")
    print(f"Report written to {md_path}")
    return 0 if final_status == "PASS" else 1

def _render_markdown(payload: Dict[str, Any]) -> str:
    """Render the human-readable Stage 0-11 integration test report."""
    lines: List[str] = []
    add = lines.append
    status = payload["final_status"]
    add("# Face Pipeline V2 — Stage 0-11 Integration Test Report")
    add("")
    add(f"**Final Result**: `{status}`")
    add("")
    add("## 1. Environment")
    add("")
    add("| Item | Value |")
    add("| --- | --- |")
    add("| Container | `face_pipeline_lxs` (host `/mnt/home/lxs/face_pipeline/Face_Pipeline_V2` → `/workspace/Face_Pipeline_V2`) |")
    add("| Python | 3.12.3 |")
    add("| ONNX Runtime | 1.21.1+ppu1.7.0.oe (`CUDAExecutionProvider`, `CPUExecutionProvider`) |")
    add("| PyTorch | 2.9.0 (`torch.cuda.is_available()=True`, device `PPU-ZW810E`) |")
    add("| OpenCV / PySceneDetect | 4.13.0 / 0.7.1 |")
    add("| face_pipeline_v2 | 0.1.0 (editable) |")
    add("| Config used | `configs/config.yaml` (loaded via `ConfigLoader`; file unchanged) |")
    add("| Test output base | `" + payload["test_base"] + "` (driver-only override of `storage.output_dir`/`stage_01.output_dir`) |")
    add("| Test generated at (UTC) | " + payload["generated_at"] + " |")
    add("")
    add("## 2. Input Video")
    add("")
    add(f"- **Path**: `{payload['input_video']}`")
    vi = payload.get("video_info")
    if vi:
        add(f"- **Resolution**: {vi['width']}×{vi['height']}")
        rm = payload.get("resolution_min")
        if rm:
            meets = vi["width"] >= rm["width"] and vi["height"] >= rm["height"]
            add(f"- **Stage 0.5 resolution check**: {vi['width']}×{vi['height']} vs min "
                f"{rm['width']}×{rm['height']} → "
                f"{'accepted' if meets else 'BELOW MINIMUM (rejected)'}")
        add(f"- **Frame rate / frames / duration**: {vi['fps']:.2f} fps / "
            f"{vi['total_frames']} frames / {vi['duration']:.1f} s")
        add(f"- **Audio**: {'yes' if vi['has_audio'] else 'no'}")
    else:
        add("- **Resolution**: unknown (Stage 0 did not complete)")
        add("- **Frame rate / frames / duration**: unknown")
    add("")
    add("## 3. Stage Results")
    add("")
    add("| Stage | Status | Elapsed (s) | Output summary |")
    add("| --- | --- | ---: | --- |")
    for r in payload["stages"]:
        err = (r.get("error") or "").replace("|", "\\|").replace("\n", " ")
        skip = r.get("skip_reason") or ""
        summ = r.get("output_summary") or ""
        cell = summ + (f" | _error:_ `{err}`" if err else "") + (f" | _skipped:_ {skip}" if skip else "")
        add(f"| {r['stage']} | {r['status']} | {r['elapsed_s']} | {cell} |")
    add("")
    add("## 4. Errors / Diagnostics")
    add("")
    has_err = False
    for r in payload["stages"]:
        if r.get("traceback"):
            has_err = True
            add(f"### {r['stage']}")
            add("")
            add("```text")
            add(r["traceback"].rstrip())
            add("```")
            add("")
    if not has_err:
        add("No stage raised an exception.")
    add("")
    add("## 5. Remaining Issues / Notes")
    add("")
    add("- `python_speech_features` (Stage 7 dependency) was installed from PyPI directly "
        "(`--index-url https://pypi.org/simple/`) because the private pip mirror returns 401.")
    add("- The test harness lives at `tests/stage0_11_integration_test.py`; it drives the existing "
        "stage classes via their standard `run()` lifecycle without modifying pipeline code.")
    add("- All output artifacts are under `" + payload["test_base"] + "/`; machine-readable results in "
        "`tests/reports/stage0_11_results.json`.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())

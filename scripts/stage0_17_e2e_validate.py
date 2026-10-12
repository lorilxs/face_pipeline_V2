#!/usr/bin/env python3
"""Stage0-17 End-to-End real-data validation driver.

Runs Stage 0 -> Stage 11 in the docker environment (Environment A) using the
existing stage classes (same chain as tests/stage0_11_integration_test.py),
records per-stage results, and exports per-clip input metadata for the
Stage 12 / 13 / 14 official runners:

    <test_base>/stage_inputs/<clip_id>_aligned_faces.json   (Stage12/13 input)
    <test_base>/stage_inputs/<clip_id>_av_alignment.json    (Stage14 input)

Stage 12 is intentionally NOT run here -- it must run separately in the
emica_env (Environment B) via the official runner.  Stage 13/14/17 also run
separately via their official runners.

Usage (inside the face_pipeline_lxs container):
    python scripts/stage0_17_e2e_validate.py \\
        --video test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4 \\
        --config configs/config.yaml \\
        --test-base outputs/stage0_17_e2e/stage0_11
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.config.config_loader import ConfigLoader  # noqa: E402
from face_pipeline_v2.models.model_manager import ModelManager  # noqa: E402
from face_pipeline_v2.pipeline.context import PipelineContext  # noqa: E402
from face_pipeline_v2.utils.logger import PipelineLogger  # noqa: E402
from face_pipeline_v2.utils.pipeline_manifest import (  # noqa: E402
    build_phase1,
    manifest_path,
    save,
)

from face_pipeline_v2.stages.stage_00_video_check.stage import Stage0VideoIntegrityCheck  # noqa: E402
from face_pipeline_v2.stages.stage_00_5_resolution_filter.stage import Stage0_5ResolutionFilter  # noqa: E402
from face_pipeline_v2.stages.stage_01_standardization.stage import Stage1AVStandardization  # noqa: E402
from face_pipeline_v2.stages.stage_02_scene_detection.stage import Stage2SceneSegmentation  # noqa: E402
from face_pipeline_v2.stages.stage_03_face_screening.stage import Stage3CoarseFaceScreening  # noqa: E402
from face_pipeline_v2.stages.stage_04_temporal_clip_generation.stage import Stage4TemporalClipGeneration  # noqa: E402
from face_pipeline_v2.stages.stage_05_face_detection.stage import Stage5FaceDetection  # noqa: E402
from face_pipeline_v2.stages.stage_05_1_face_tracking.stage import Stage5_1FaceTracking  # noqa: E402
from face_pipeline_v2.stages.stage_05_2_single_face_filter.stage import Stage5_2SingleFaceFilter  # noqa: E402
from face_pipeline_v2.stages.stage_05_3_temporal_stability.stage import Stage5_3TemporalStability  # noqa: E402
from face_pipeline_v2.stages.stage_06_pose.stage import Stage6PoseEstimation  # noqa: E402
from face_pipeline_v2.stages.stage_07_sync.stage import Stage7SyncDetection  # noqa: E402
from face_pipeline_v2.stages.stage_07_1_av_alignment.stage import Stage7_1AVAlignment  # noqa: E402
from face_pipeline_v2.stages.stage_08_quality.stage import Stage8QualityAssessment  # noqa: E402
from face_pipeline_v2.stages.stage_09_quality_fusion.stage import Stage9QualityFusion  # noqa: E402
from face_pipeline_v2.stages.stage_09_5_clip_export.stage import Stage9_5ClipExport  # noqa: E402
from face_pipeline_v2.stages.stage_10_landmark.stage import Stage10Landmark  # noqa: E402
from face_pipeline_v2.stages.stage_11_alignment.stage import Stage11Alignment  # noqa: E402
from face_pipeline_v2.pipeline.dataset_adapter import (  # noqa: E402
    DATASET_MODE,
    OPEN_DOMAIN_MODE,
    build_dataset_av_alignments,
    collect_accepted_clip_ids,
    resolve_pipeline_mode,
)

STATUS_OK = "PASS"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIPPED"

# Quality-filtering stages skipped in dataset mode (Stage0-5 -> Stage10-17).
SKIPPED_QUALITY_STAGES: List[str] = [
    "Stage 6 (6DRepNet pose)",
    "Stage 7 (SyncNet A/V sync)",
    "Stage 7.1 (A/V alignment)",
    "Stage 8 (HyperIQA quality)",
    "Stage 9 (Quality fusion/filter)",
    "Stage 9.5 (Clip export)",
]


def skip_stage(stage_name: str, results: List[Dict[str, Any]],
               reason: str = "") -> None:
    """Record a SKIPPED stage entry (used for dataset mode)."""
    results.append({
        "stage": stage_name,
        "status": STATUS_SKIP,
        "started": _now_utc(),
        "elapsed_s": None,
        "finished": _now_utc(),
        "output_summary": reason or "Skipped in dataset mode",
        "error": None,
    })


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_stage(stage_name: str, fn: Callable[[], Any],
              results: List[Dict[str, Any]]) -> Any:
    """Execute one stage's run() and record PASS/FAIL + timing."""
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
    except Exception as exc:  # noqa: BLE001 - driver must capture everything
        entry["status"] = STATUS_FAIL
        entry["elapsed_s"] = round(time.monotonic() - t0, 2)
        entry["finished"] = _now_utc()
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        results.append(entry)
        return None


def record(stage_name: str, summary: str, results: List[Dict[str, Any]]) -> None:
    if results and results[-1]["stage"] == stage_name:
        results[-1]["output_summary"] = summary


def _av_alignment_to_dict(av) -> Dict[str, Any]:
    """Serialize AVAlignmentInfo fields consumed by the Stage14 runner."""
    def p(x):
        return str(x) if x is not None else None
    return {
        "clip_id": av.clip_id,
        "original_audio_path": p(av.original_audio_path),
        "offset_frames": int(av.offset_frames),
        "offset_ms": float(av.offset_ms),
        "correction_applied": bool(av.correction_applied),
        "fps": float(av.fps),
        "corrected_audio_path": p(av.corrected_audio_path),
        "aligned_audio_path": p(av.aligned_audio_path),
        "alignment_status": av.alignment_status,
        "raw_audio_path": p(av.raw_audio_path),
        "clip_start": float(av.clip_start) if av.clip_start is not None else None,
        "clip_duration": float(av.clip_duration) if av.clip_duration is not None else None,
    }


def _aligned_face_to_dict(af) -> Dict[str, Any]:
    """Serialize AlignedFaceInfo fields consumed by the Stage12/13 runners."""
    return {
        "clip_id": af.clip_id,
        "frame_index": int(af.frame_index),
        "face_id": af.face_id,
        "timestamp": float(af.timestamp),
        "aligned_face_path": str(af.aligned_face_path),
        "affine_matrix": list(af.affine_matrix),
        "original_image_size": list(af.original_image_size),
        "aligned_resolution": list(af.aligned_resolution),
    }

def _run_open_domain_quality_stages(
    args: Any,
    results: List[Dict[str, Any]],
    env: Dict[str, Any],
    context: PipelineContext,
    face_sequences: Any,
    audio_info: Any,
    clips: Any,
    std_info: Any,
    video_id: str,
) -> Any:
    """Run Stage 6 / 7 / 7.1 / 8 / 9 / 9.5 (open-domain quality filtering).

    Verbatim open-domain block — behavior unchanged from the pre-dataset-mode
    driver.  Returns the accepted clip id set on success, or the exit code
    returned by ``_finalize`` (an int) when a quality stage failed and the
    driver was already finalized (the caller must return it immediately).
    """
    # ================================================================ STAGE 6
    env["pose_sequences"] = run_stage(
        "Stage 6 (6DRepNet pose)",
        lambda: Stage6PoseEstimation().run(face_sequences, context),
        results,
    )
    if env["pose_sequences"] is None:
        return _finalize(args, results, env, "FAIL", context)
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
        return _finalize(args, results, env, "FAIL", context)
    record("Stage 7 (SyncNet A/V sync)", f"{len(env['av_sync_infos'])} clip(s)", results)

    # ============================================================ STAGE 7.1
    env["av_alignments"] = run_stage(
        "Stage 7.1 (A/V alignment)",
        lambda: Stage7_1AVAlignment().run(
            {
                "av_sync_infos": env["av_sync_infos"],
                "std_video_info": std_info,
                "clip_infos": clips,
            },
            context,
        ),
        results,
    )
    if env["av_alignments"] is None:
        return _finalize(args, results, env, "FAIL", context)
    record("Stage 7.1 (A/V alignment)", f"{len(env['av_alignments'])} clip(s)", results)


    # ================================================================ STAGE 8
    env["quality_sequences"] = run_stage(
        "Stage 8 (HyperIQA quality)",
        lambda: Stage8QualityAssessment().run(face_sequences, context),
        results,
    )
    if env["quality_sequences"] is None:
        return _finalize(args, results, env, "FAIL", context)
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
        return _finalize(args, results, env, "FAIL", context)
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
        return _finalize(args, results, env, "FAIL", context)
    er = env["export_result"]
    record("Stage 9.5 (Clip export)",
           f"accepted={er.get('accepted_count')} exported_videos={er.get('exported_videos')} "
           f"exported_audio={er.get('exported_audio')} manifest={er.get('manifest_entries')} "
           f"failures={len(er.get('failures', []))}",
           results)
    return accepted_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--test-base", default="outputs/stage0_17_e2e/stage0_11",
                        help="Stage0-11 output base (relative to project root)")
    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)
    test_base = Path(args.test_base)

    # ------------------------------------------------------------------ config
    loader = ConfigLoader(config_path=config_path)
    config = loader.load()
    loader.validate(config)
    pipeline_mode = resolve_pipeline_mode(config)
    config["storage"]["output_dir"] = str(test_base)
    config["stage_01"]["output_dir"] = str(test_base / "videos")

    # ------------------------------------------------------------------ context
    reports_dir = PROJECT_ROOT / "outputs" / "stage0_17_e2e"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = PipelineLogger(name="stage0_17_e2e", log_dir=reports_dir)
    model_manager = ModelManager(device="auto")
    context = PipelineContext(
        config=config,
        logger=logger,
        model_manager=model_manager,
        runtime_info={"video_id": video_path.stem},
    )

    results: List[Dict[str, Any]] = []
    env: Dict[str, Any] = {}
    env["pipeline_mode"] = pipeline_mode
    logger.info("Stage0-17 E2E: Stage 0-11 starting (docker env)")
    logger.info(f"video={video_path} config={config_path} test_base={test_base}")
    logger.info(f"Pipeline mode: {pipeline_mode}")

    # ================================================================ STAGE 0
    env["video_info"] = run_stage(
        "Stage 0 (FFprobe integrity check)",
        lambda: Stage0VideoIntegrityCheck().run(video_path, context),
        results,
    )
    if env["video_info"] is None:
        logger.error("Stage 0 failed - aborting.")
        return _finalize(args, results, env, "FAIL", context)
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
        return _finalize(args, results, env, "FAIL", context)
    ri = env["resolution_info"]
    record("Stage 0.5 (Resolution filter)",
           f"{ri.width}x{ri.height} vs min {ri.min_width}x{ri.min_height} "
           f"| is_valid={ri.is_valid} | reject={ri.reject_reason}",
           results)
    if not ri.is_valid:
        logger.error("Stage 0.5 rejected the video - aborting.")
        return _finalize(args, results, env, "FAIL", context)

    # ================================================================ STAGE 1
    env["stage1_out"] = run_stage(
        "Stage 1 (FFmpeg standardization)",
        lambda: Stage1AVStandardization().run(env["video_info"], context),
        results,
    )
    if env["stage1_out"] is None:
        return _finalize(args, results, env, "FAIL", context)
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
        return _finalize(args, results, env, "FAIL", context)
    record("Stage 2 (PySceneDetect)", f"{len(env['scenes'])} scene(s)", results)

    # ================================================================ STAGE 3
    env["scenes"] = run_stage(
        "Stage 3 (SCRFD coarse screening)",
        lambda: Stage3CoarseFaceScreening().run(env["scenes"], context),
        results,
    )
    if env["scenes"] is None:
        return _finalize(args, results, env, "FAIL", context)
    passed = [s for s in env["scenes"] if s.filter_result and s.filter_result.passed]
    rejected = [s for s in env["scenes"] if not (s.filter_result and s.filter_result.passed)]
    record("Stage 3 (SCRFD coarse screening)",
           f"{len(env['scenes'])} scene(s): {len(passed)} passed, {len(rejected)} rejected",
           results)
    if not passed:
        logger.error("Stage 3 rejected all scenes - aborting.")
        return _finalize(args, results, env, "FAIL", context)

    # ================================================================ STAGE 4
    env["clips"] = run_stage(
        "Stage 4 (Sliding-window clips)",
        lambda: Stage4TemporalClipGeneration().run(env["scenes"], context),
        results,
    )
    if env["clips"] is None:
        return _finalize(args, results, env, "FAIL", context)
    record("Stage 4 (Sliding-window clips)", f"{len(env['clips'])} clip(s)", results)
    if not env["clips"]:
        return _finalize(args, results, env, "PARTIAL", context)

    # ================================================================ STAGE 5
    env["face_sequences"] = run_stage(
        "Stage 5 (SCRFD detect/crop)",
        lambda: Stage5FaceDetection().run(env["clips"], context),
        results,
    )
    if env["face_sequences"] is None:
        return _finalize(args, results, env, "FAIL", context)
    total_faces = sum(len(f.faces) for seq in env["face_sequences"] for f in seq.frames)
    record("Stage 5 (SCRFD detect/crop)",
           f"{len(env['face_sequences'])} sequence(s), {total_faces} face detections",
           results)
    if not env["face_sequences"]:
        return _finalize(args, results, env, "PARTIAL", context)


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
        return _finalize(args, results, env, "FAIL", context)
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
    if not env["tracked_sequences"]:
        return _finalize(args, results, env, "PARTIAL", context)

    env["filtered_sequences"] = run_stage(
        "Stage 5.2 (Single-face filter)",
        lambda: Stage5_2SingleFaceFilter().run(env["tracked_sequences"], context),
        results,
    )
    if env["filtered_sequences"] is None:
        return _finalize(args, results, env, "FAIL", context)
    s52_pass = len(env["filtered_sequences"])
    s52_input = len(env["tracked_sequences"])
    record("Stage 5.2 (Single-face filter)",
           f"{s52_input} input: {s52_pass} passed, {s52_input - s52_pass} rejected "
           f"(hard filter, dropped)",
           results)
    if not env["filtered_sequences"]:
        return _finalize(args, results, env, "PARTIAL", context)

    env["stabilized_sequences"] = run_stage(
        "Stage 5.3 (Temporal stability)",
        lambda: Stage5_3TemporalStability().run(env["filtered_sequences"], context),
        results,
    )
    if env["stabilized_sequences"] is None:
        return _finalize(args, results, env, "FAIL", context)
    s53_pass = len(env["stabilized_sequences"])
    s53_input = len(env["filtered_sequences"])
    record("Stage 5.3 (Temporal stability)",
           f"{s53_input} input: {s53_pass} passed, {s53_input - s53_pass} rejected "
           f"(hard filter, dropped)",
           results)
    if not env["stabilized_sequences"]:
        return _finalize(args, results, env, "PARTIAL", context)

    face_sequences = env["stabilized_sequences"]

    if pipeline_mode == OPEN_DOMAIN_MODE:
        # Open-domain: full quality filtering (Stage 6/7/7.1/8/9/9.5).
        _quality_result = _run_open_domain_quality_stages(
            args, results, env, context, face_sequences,
            audio_info, clips, std_info, video_id,
        )
        if isinstance(_quality_result, int):
            # A quality stage failed — the driver was already finalized.
            return _quality_result
        accepted_ids = _quality_result
    else:
        # ============= DATASET MODE: skip Stage6/7/7.1/8/9/9.5 ==============
        # Existing high-quality dataset input requires no quality filtering.
        # Record the skipped quality stages, then synthesize the Stage9 /
        # Stage7.1 outputs that downstream stages (10 / 11 / 14) still expect.
        for _stage_name in SKIPPED_QUALITY_STAGES:
            skip_stage(_stage_name, results, f"pipeline_mode={DATASET_MODE}")
        # Stage10 adapter: every valid Stage5 clip is accepted (no filtering).
        accepted_ids = collect_accepted_clip_ids(face_sequences)
        env["pose_sequences"] = []
        env["av_sync_infos"] = []
        env["quality_sequences"] = []
        env["quality_metrics"] = []
        # Stage14 adapter: aligned audio = original clip wav (no SyncNet).
        env["av_alignments"] = build_dataset_av_alignments(
            clips=clips,
            face_sequences=face_sequences,
            std_info=std_info,
            output_dir=Path(config["storage"]["output_dir"])
            / "stage_outputs" / "av_alignment",
        )
        env["export_result"] = None
        record(
            SKIPPED_QUALITY_STAGES[-1],
            f"pipeline_mode={DATASET_MODE}: {len(accepted_ids)} accepted clip(s), "
            f"{len(env['av_alignments'])} AV alignment(s) from original clip wav",
            results,
        )


    # =============================================================== STAGE 10
    if not accepted_ids:
        return _finalize(args, results, env, "PARTIAL", context)
    env["landmarks"] = run_stage(
        "Stage 10 (InsightFace landmark)",
        lambda: Stage10Landmark().run(
            {"face_sequences": face_sequences, "accepted_clip_ids": accepted_ids},
            context,
        ),
        results,
    )
    if env["landmarks"] is None:
        return _finalize(args, results, env, "FAIL", context)
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
        return _finalize(args, results, env, "FAIL", context)
    aligned = env["aligned"]
    record("Stage 11 (Affine alignment)", f"{len(aligned)} aligned face(s)", results)

    # ====================================================== METADATA EXPORT
    # 为 Stage12/13/14 官方 runner 导出每 clip 输入 metadata
    env["stage_inputs_dir"] = test_base / "stage_inputs"
    env["stage_inputs_dir"].mkdir(parents=True, exist_ok=True)

    aligned_by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for af in aligned:
        aligned_by_clip.setdefault(af.clip_id, []).append(_aligned_face_to_dict(af))

    av_by_clip: Dict[str, Dict[str, Any]] = {}
    for av in env["av_alignments"]:
        av_by_clip[av.clip_id] = _av_alignment_to_dict(av)

    exported_meta: Dict[str, Dict[str, str]] = {}
    for clip_id, records in sorted(aligned_by_clip.items()):
        # 只导出被 Stage9 接受的 clip 的 metadata
        if clip_id not in accepted_ids:
            continue
        recs = sorted(records, key=lambda r: r["frame_index"])
        af_path = env["stage_inputs_dir"] / f"{clip_id}_aligned_faces.json"
        af_path.write_text(json.dumps(recs, indent=2), encoding="utf-8")
        av_path = env["stage_inputs_dir"] / f"{clip_id}_av_alignment.json"
        av_path.write_text(json.dumps(av_by_clip.get(clip_id, {}), indent=2),
                           encoding="utf-8")
        exported_meta[clip_id] = {
            "aligned_faces": str(af_path),
            "n_frames": str(len(recs)),
            "av_alignment": str(av_path),
        }
    env["exported_meta"] = exported_meta
    record("Stage 11 (Affine alignment)",
           f"{len(aligned)} aligned face(s) | exported {len(exported_meta)} clip metadata",
           results)

    return _finalize(args, results, env, "PASS", context)


def _finalize(args: Any, results: List[Dict[str, Any]],
              env: Dict[str, Any], final_status: str,
              context: PipelineContext) -> int:
    """Write machine-readable results JSON and print a summary.

    Also writes the phase-1 ``pipeline_manifest.json`` (production statistics
    captured from in-memory ``env``/``context``) so per-run statistics survive
    deletion of the intermediate outputs.  The manifest write is strictly
    best-effort: a failure only logs a warning and never changes the exit code.
    """
    out_dir = PROJECT_ROOT / "outputs" / "stage0_17_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stage0_11_results.json"

    payload: Dict[str, Any] = {
        "final_status": final_status,
        "generated_at": _now_utc(),
        "input_video": str(args.video),
        "config": str(Path(args.config).resolve()),
        "test_base": str(args.test_base),
        "env": "docker face_pipeline_lxs (python3.12)",
        "pipeline_mode": env.get("pipeline_mode", OPEN_DOMAIN_MODE),
        "stages": results,
    }
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
    if "exported_meta" in env:
        payload["accepted_clips"] = env["exported_meta"]
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # ------------------------------------------------------------------
    # Phase-1 pipeline manifest (best-effort; must never change exit code)
    # ------------------------------------------------------------------
    # Captures run/config/source/per-clip lineage/5.2+5.3 reject audit from
    # the in-memory env + context, so statistics survive deletion of the
    # intermediate directories under test_base.
    try:
        manifest_doc = build_phase1(
            config=context.config,
            args=args,
            results=results,
            env=env,
            final_status=final_status,
            context=context,
            test_base=Path(args.test_base),
        )
        saved = save(
            manifest_doc,
            manifest_path(Path(args.test_base).parent),
            logger=context.logger,
        )
        context.logger.info(
            f"[pipeline_manifest] phase1 {'saved' if saved else 'save FAILED'} -> "
            f"{manifest_path(Path(args.test_base).parent)}"
        )
    except Exception as exc:  # noqa: BLE001 - manifest must never break the run
        context.logger.warning(
            f"[pipeline_manifest] phase1 write skipped: {exc}"
        )

    print("\n===== STAGE 0-11 E2E: %s =====" % final_status)
    for r in results:
        print(f"  [{r['status']:8s}] {r['stage']:34s} {r['elapsed_s']}s  "
              f"{r.get('output_summary', '')}")
        if r.get("error"):
            print(f"           ERROR: {r['error']}")
    print(f"Results JSON -> {json_path}")
    return 0 if final_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


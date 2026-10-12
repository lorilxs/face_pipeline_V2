"""Smoke test: build_phase1 + merge_phase2 against REAL model dataclasses.

Validates every field-access path in pipeline_manifest.py against the actual
AISDD data models (models.py), exercising accepted / 5.2-rejected / 5.3-rejected
clips, full stage_details (Stage0-11), audit capture, atomic save/load, and the
phase-2 merge (Stage12/13/14/17 metadata + final asset inventory).
"""
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from face_pipeline_v2.data.models import (  # noqa: E402
    AVAlignmentInfo,
    AVSyncInfo,
    AlignedFaceInfo,
    AudioInfo,
    ClipInfo,
    CoarseFaceInfo,
    FaceInfo,
    FaceQualityInfo,
    FaceQualitySequence,
    FaceSequence,
    FrameInfo,
    LandmarkInfo,
    PoseInfo,
    PoseSequence,
    QualityMetrics,
    ResolutionValidationInfo,
    SceneFilterResult,
    SceneInfo,
    SingleFaceIdentityResult,
    StandardizedVideoInfo,
    TemporalStabilityResult,
    VideoInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext  # noqa: E402
from face_pipeline_v2.utils.pipeline_manifest import (  # noqa: E402
    build_phase1,
    load,
    manifest_path,
    merge_phase2,
    save,
    STAGE_RESULT_NAMES,
)

VID = "smoke_001"
CLIP_OK = VID + "_scene_0000_clip_0000"
CLIP_R53 = VID + "_scene_0000_clip_0001"
CLIP_R52 = VID + "_scene_0000_clip_0002"
SCENE_OK = VID + "_scene_0000"
SCENE_BAD = VID + "_scene_0001"


def _frames(clip_id, n=20, tid="t1", conf=0.95):
    out = []
    for i in range(n):
        out.append(FrameInfo(
            frame_index=i,
            clip_id=clip_id,
            timestamp=i / 25.0,
            image_path=Path("/x/frame_%d.jpg" % i),
            faces=(FaceInfo(
                frame_index=i,
                face_id="f_%d" % i,
                bbox=(10.0, 10.0, 110.0, 110.0),
                landmark_5=(0.0,) * 10,
                confidence=conf,
                face_crop_path=Path("/x/crop_%d.jpg" % i),
                track_id=tid,
            ),),
        ))
    return tuple(out)


def _fseq(clip_id, identity_result, stability_result):
    return FaceSequence(
        clip_id=clip_id,
        video_id=VID,
        frames=_frames(clip_id),
        identity_result=identity_result,
        stability_result=stability_result,
    )

def main():
    tmp = Path(tempfile.mkdtemp(prefix="manifest_smoke_"))
    run_root = tmp / "outputs" / (VID + "_open_domain")
    test_base = run_root / "stage0_11"
    export_dir = run_root / "stage17_multimodal_export"
    (test_base / "clips").mkdir(parents=True)
    (export_dir / "metadata").mkdir(parents=True)
    (test_base / "stage12_flame" / CLIP_OK).mkdir(parents=True)
    (test_base / "stage13_gaze" / CLIP_OK).mkdir(parents=True)
    (test_base / "stage14_audio" / CLIP_OK).mkdir(parents=True)

    # --- config: real configs/config.yaml (stage_04.mode == segment) ---
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))

    # --- REAL model objects ---------------------------------------------------
    vi = VideoInfo(
        video_id=VID,
        video_path=Path("/videos/src.mp4"),
        original_fps=25.0,
        width=1920,
        height=1080,
        duration=60.2,
        total_frames=1505,
        codec="h264",
        has_audio=True,
    )
    ri = ResolutionValidationInfo(
        width=1920, height=1080, min_width=512, min_height=512,
        is_valid=True, reject_reason=None,
    )
    std_v = StandardizedVideoInfo(
        video_path=Path("/x/std.mp4"), audio_path=Path("/x/std.wav"),
        target_fps=25, resolution=(1280, 720), codec="h264",
    )
    aud = AudioInfo(audio_path=Path("/x/std.wav"), sample_rate=16000,
                    duration=60.2, channels=1)
    stage1_out = {"video_info": std_v, "audio_info": aud}

    scene_ok = SceneInfo(
        scene_id=SCENE_OK, video_id=VID, start_frame=0, end_frame=1505,
        start_time=0.0, end_time=60.2, duration=60.2,
        coarse_face_info=CoarseFaceInfo(
            sampled_frame_count=60, face_detect_ratio=1.0, single_face_ratio=0.98,
            avg_face_confidence=0.92, max_face_count=1,
        ),
        filter_result=SceneFilterResult(passed=True, reject_reasons=()),
    )
    scene_bad = SceneInfo(
        scene_id=SCENE_BAD, video_id=VID, start_frame=1505, end_frame=1600,
        start_time=60.2, end_time=64.0, duration=3.8,
        coarse_face_info=None,
        filter_result=SceneFilterResult(passed=False, reject_reasons=("no_face",)),
    )

    def _clip(cid, scene_id):
        return ClipInfo(clip_id=cid, scene_id=scene_id, video_id=VID,
                        start_frame=0, end_frame=20, start_time=0.0, end_time=0.8,
                        frame_count=20, duration=0.8)

    clip_ok = _clip(CLIP_OK, SCENE_OK)
    clip_r53 = _clip(CLIP_R53, SCENE_OK)
    clip_r52 = _clip(CLIP_R52, SCENE_OK)

    idr_ok = SingleFaceIdentityResult(passed=True, main_track_id="t1",
                                      total_frames=20, anomaly_frames=0,
                                      anomaly_ratio=0.0, reject_reasons=())
    sr_ok = TemporalStabilityResult(passed=True, average_center_motion=3.0,
                                    max_center_motion=9.0, average_scale_change=0.05,
                                    max_scale_change=0.12, bbox_jump_count=0,
                                    anomaly_ratio=0.0, reject_reasons=())
    idr_r52 = SingleFaceIdentityResult(passed=False, main_track_id="t2",
                                       total_frames=20, anomaly_frames=6,
                                       anomaly_ratio=0.30, reject_reasons=("multi_face",))
    sr_r53 = TemporalStabilityResult(passed=False, average_center_motion=20.0,
                                     max_center_motion=52.0, average_scale_change=0.2,
                                     max_scale_change=0.45, bbox_jump_count=5,
                                     anomaly_ratio=0.25,
                                     reject_reasons=("center_motion_abnormal",))

    fs_ok = _fseq(CLIP_OK, idr_ok, sr_ok)
    fs_r53 = _fseq(CLIP_R53, idr_ok, sr_r53)
    fs_r52 = _fseq(CLIP_R52, idr_r52, None)

    pose_ok = PoseSequence(clip_id=CLIP_OK, poses=tuple(
        PoseInfo(frame_index=i, face_id="f_%d" % i, yaw=5.0, pitch=-3.0,
                 roll=2.0, confidence=0.9)
        for i in range(20)
    ))
    av_ok = AVSyncInfo(clip_id=CLIP_OK, offset=2, sync_lse_c=7.5, sync_lse_d=-0.8,
                       median_distance=5.2, start_frame=0, end_frame=20, fps=25.0,
                       duration=0.8, timestamp=0.0)
    align_ok = AVAlignmentInfo(
        clip_id=CLIP_OK, original_audio_path=Path("/x/std.wav"),
        offset_frames=2, offset_ms=80.0, correction_applied=True, fps=25.0,
        corrected_audio_path=Path("/x/corr.wav"), aligned_audio_path=Path("/x/corr.wav"),
        alignment_status="corrected", raw_audio_path=Path("/x/raw.wav"),
        clip_start=0.0, clip_duration=0.8,
    )
    qs_ok = FaceQualitySequence(clip_id=CLIP_OK, qualities=tuple(
        FaceQualityInfo(frame_index=i, face_id="f_%d" % i, quality_score=0.85,
                        confidence=0.9)
        for i in range(20)
    ))
    qm_ok = QualityMetrics(clip_id=CLIP_OK, video_id=VID, pose_score=0.92,
                           sync_score=0.88, quality_score=0.9, fusion_score=0.90,
                           is_accepted=True, reject_reasons=())
    lms = [LandmarkInfo(clip_id=CLIP_OK, frame_index=i, face_id="f_%d" % i,
                        timestamp=i / 25.0, landmarks_106=(0.0,) * 212,
                        image_width=1920, image_height=1080)
           for i in range(3)]
    afs = [AlignedFaceInfo(clip_id=CLIP_OK, frame_index=i, face_id="f_%d" % i,
                           timestamp=i / 25.0,
                           aligned_face_path=Path("/x/al_%d.jpg" % i),
                           affine_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                           original_image_size=(1080, 1920),
                           aligned_resolution=(256, 256))
           for i in range(3)]


    env = {
        "pipeline_mode": "open_domain",
        "video_info": vi,
        "resolution_info": ri,
        "stage1_out": stage1_out,
        "scenes": [scene_ok, scene_bad],
        "clips": [clip_ok, clip_r53, clip_r52],
        "face_sequences": [fs_ok, fs_r53, fs_r52],
        "tracked_sequences": [fs_ok, fs_r53, fs_r52],
        "filtered_sequences": [fs_ok, fs_r53],
        "stabilized_sequences": [fs_ok],
        "pose_sequences": [pose_ok],
        "av_sync_infos": [av_ok],
        "av_alignments": [align_ok],
        "quality_sequences": [qs_ok],
        "quality_metrics": [qm_ok],
        "landmarks": lms,
        "aligned": afs,
        "export_result": {"exported_videos": 1, "exported_audio": 1,
                          "manifest_entries": 1, "failures": []},
    }

    # --- audit (simulates context.add_stage_audit() during the run) ----------
    audit = {
        "stage_05_2_single_face_filter": {
            "rejected": [{
                "clip_id": CLIP_R52, "reasons": ["multi_face"], "main_track_id": "t2",
                "total_frames": 20, "anomaly_frames": 6, "anomaly_ratio": 0.3,
            }],
        },
        "stage_05_3_temporal_stability": {
            "rejected": [{
                "clip_id": CLIP_R53, "reasons": ["center_motion_abnormal"],
                "bbox_jump_count": 5, "average_center_motion": 20.0,
                "max_center_motion": 52.0, "average_scale_change": 0.2,
                "max_scale_change": 0.45, "anomaly_ratio": 0.25,
            }],
        },
    }
    context = PipelineContext(
        config=config, logger=None, model_manager=None,
        runtime_info={"pipeline_mode": "open_domain", "video_id": VID},
        stage_audit=audit,
    )

    # --- e2e results list ----------------------------------------------------
    results = []
    for name in STAGE_RESULT_NAMES.values():
        results.append({
            "stage": name, "status": "PASS", "started": "2026-09-12T00:00:00Z",
            "finished": "2026-09-12T00:00:01Z", "elapsed_s": 1.0,
            "output_summary": "ok", "error": None,
        })

    args = Namespace(input_video="/videos/src.mp4",
                     config=str(ROOT / "configs/config.yaml"))

    # --- Phase 1 -------------------------------------------------------------
    doc = build_phase1(config=config, args=args, results=results, env=env,
                       final_status="PASS", context=context,
                       test_base=test_base)
    assert doc["_phase1_complete"] is True
    assert doc["run"]["pipeline_mode"] == "open_domain"
    assert doc["run"]["window_seconds"] is None  # stage_04.mode == segment
    assert doc["source_video"]["fps"] == 25.0  # original_fps path
    assert doc["stage_details"]["stage_00"]["fps"] == 25.0
    assert doc["stage_details"]["stage_01"]["target_fps"] == 25
    assert doc["stage_details"]["stage_06"]["clip_count"] == 1  # dead-code fix
    assert doc["stage_details"]["stage_11"]["total_aligned_frames"] == 3
    assert doc["stage_details"]["stage_03"]["rejected_count"] == 1
    assert doc["stage_details"]["stage_08"]["score_count"] == 20  # quality_score fix

    clips_by_id = {c["clip_id"]: c for c in doc["clips"]}
    assert clips_by_id[CLIP_OK]["stage_05_2"]["status"] == "passed"
    assert clips_by_id[CLIP_OK]["stage_05_3"]["status"] == "passed"
    assert clips_by_id[CLIP_OK]["stage_06"]["status"] == "passed"
    assert clips_by_id[CLIP_OK]["stage_07"]["offset_frames"] == 2  # offset_frames fix
    assert clips_by_id[CLIP_OK]["stage_07_1"]["offset_frames"] == 2
    assert clips_by_id[CLIP_OK]["stage_09"]["status"] == "passed"
    assert clips_by_id[CLIP_OK]["stage_11"]["aligned_faces"] == 3  # frame_index fix
    assert clips_by_id[CLIP_R52]["stage_05_2"]["status"] == "rejected"
    assert clips_by_id[CLIP_R53]["stage_05_3"]["status"] == "rejected"
    assert doc["audit"]["stage_05_2_single_face_filter"]["rejected_count"] == 1
    assert doc["audit"]["stage_05_3_temporal_stability"]["rejected_count"] == 1
    assert doc["final_assets"]["partial"] is True

    saved = save(doc, manifest_path(run_root))
    assert saved is True
    loaded = load(manifest_path(run_root))
    assert loaded is not None and loaded["source_video"]["fps"] == 25.0

    print("[phase1] OK: clips=%d stage_details=%d audit_rejected=%d" % (
        len(doc["clips"]), len(doc["stage_details"]),
        len(doc["audit"]["stage_05_2_single_face_filter"]["rejected"])
        + len(doc["audit"]["stage_05_3_temporal_stability"]["rejected"])))

    # --- pass-through (enhancement-disabled) unit check ----------------------
    from face_pipeline_v2.utils import pipeline_manifest as _pm
    pt = _fseq("clip_pass_through", None, None)
    pt_ctx = {
        "filtered_by_id": {pt.clip_id: pt},
        "stabilized_by_id": {pt.clip_id: pt},
        "tracked_by_id": {pt.clip_id: pt},
        "audit_rejected_52": {}, "audit_rejected_53": {},
        "face_by_id": {pt.clip_id: pt},
    }
    s52 = _pm._s52_entry(pt.clip_id, pt_ctx)
    s53 = _pm._s53_entry(pt.clip_id, pt_ctx)
    assert s52["status"] == "passed", s52  # not a false "rejected"
    assert s53["status"] == "passed", s53
    print("[pass-through] OK: 5.2/5.3 report passed for enhancement-disabled clip")



    # --- Phase 2 metadata fixtures -------------------------------------------
    (test_base / "stage12_flame" / CLIP_OK / "flame_metadata.json").write_text(
        json.dumps({"num_frames": 20, "backend": "emica", "jaw_source": "emica",
                    "head_source": "deca", "expression_source": "emica",
                    "expression_dim": 100, "jaw_pose_dim": 3, "head_pose_dim": 3,
                    "shapecode_dim": 300}),
        encoding="utf-8")
    (test_base / "stage13_gaze" / CLIP_OK / "gaze_metadata.json").write_text(
        json.dumps({"num_frames": 20, "gaze_dim": 6}), encoding="utf-8")
    (test_base / "stage14_audio" / CLIP_OK / "audio_metadata.json").write_text(
        json.dumps({"num_frames": 20, "feature_dim": 1024, "source_audio": "std.wav"}),
        encoding="utf-8")
    # per-clip feature payloads: a successful feature stage writes both the
    # npz and its metadata; the absence of the npz is the durable failure signal.
    (test_base / "stage12_flame" / CLIP_OK / "flame_features.npz").write_bytes(b"npz")
    (test_base / "stage13_gaze" / CLIP_OK / "gaze_features.npz").write_bytes(b"npz")
    (test_base / "stage14_audio" / CLIP_OK / "audio_features.npz").write_bytes(b"npz")
    (export_dir / "metadata" / (CLIP_OK + ".json")).write_text(
        json.dumps({"clip_id": CLIP_OK, "fps": 25.0, "visual_dim": 300,
                    "audio_dim": 1024, "visual_layout": "hifi3d",
                    "head_source": "deca", "jaw_source": "emica",
                    "expression_source": "emica", "gaze_source": "gaze",
                    "audio_source": "hubert", "num_frames": 20, "valid_length": 19,
                    "source_stage": "stage17", "n_input": 20}),
        encoding="utf-8")

    # --- Phase 2 merge -------------------------------------------------------
    doc2 = merge_phase2(base=test_base, export_dir=export_dir)
    assert doc2["_phase2_complete"] is True
    assert doc2["final_assets"]["partial"] is False
    assert doc2["final_assets"]["asset_count"] == 1
    assert doc2["stages"]["stage_17"]["status"] == "PASS"
    assert doc2["stages"]["stage_12"]["status"] == "PASS"
    rec = {c["clip_id"]: c for c in doc2["clips"]}[CLIP_OK]
    assert rec["stage_12"]["num_frames"] == 20
    assert rec["stage_13"]["gaze_dim"] == 6
    assert rec["stage_14"]["feature_dim"] == 1024
    assert rec["stage_17"]["fps"] == 25.0
    # phase1 content survived the phase2 merge untouched
    assert {c["clip_id"] for c in doc2["clips"]} == {CLIP_OK, CLIP_R52, CLIP_R53}
    assert doc2["audit"]["stage_05_2_single_face_filter"]["rejected_count"] == 1
    assert doc2["stage_details"]["stage_07"]["mean_offset_frames"] == 2

    # --- retention survival: manifest survives deletion of the test base -----
    shutil.rmtree(test_base)
    survived = load(manifest_path(run_root))
    assert survived is not None
    assert survived["clips"] and survived["clips"][0]["clip_id"] == CLIP_OK
    assert survived["final_assets"]["partial"] is False

    print("[phase2] OK: assets=%d stages12/13/14/17 merged, "
          "manifest survives test-base deletion" % survived["final_assets"]["asset_count"])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()



def test_manifest_real_smoke():
    """pytest entry point: exercise build_phase1 + merge_phase2 against real models."""
    main()

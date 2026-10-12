"""Unit tests for Stage12/13/14 failure inference + authoritative Stage17 report.

Covers ``merge_phase2`` failure recording:
* feature stage with missing per-clip npz -> failed (Stage12/13/14);
* Stage17 uses ``export_report.json`` when present, carrying the failed reason;
* failed Stage17 clips excluded from ``final_assets`` -> ``partial`` True;
* missing feature output dir -> ``SKIPPED`` (never ``FAIL``).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face_pipeline_v2.utils.pipeline_manifest import (  # noqa: E402
    load,
    manifest_path,
    merge_phase2,
    save,
)

VID = "failure_probe"
CLIP_OK = VID + "_scene_0000_clip_0000"
CLIP_BAD = VID + "_scene_0000_clip_0001"

FEATURES = [
    ("stage12_flame", "flame_features.npz", "flame_metadata.json"),
    ("stage13_gaze", "gaze_features.npz", "gaze_metadata.json"),
    ("stage14_audio", "audio_features.npz", "audio_metadata.json"),
]


def _phase1_doc(run_root, base, clip_ids):
    return {
        "manifest_version": 1,
        "generated_at": "2026-09-15T00:00:00Z",
        "run": {"run_root": str(run_root), "test_base": str(base)},
        "stages": {},
        "stage_details": {},
        "clips": [
            {
                "clip_id": cid,
                "video_id": VID,
                "duration": 0.8,
                "stage_11": {"status": "passed"},
            }
            for cid in clip_ids
        ],
        "final_assets": {"partial": False, "assets": []},
    }


def _setup(tmp_path, tag):
    run_root = tmp_path / "outputs" / (VID + "_" + tag + "_open_domain")
    base = run_root / "stage0_11"
    export_dir = run_root / "stage17_multimodal_export"
    (base / "clips").mkdir(parents=True)
    (export_dir / "metadata").mkdir(parents=True)
    return run_root, base, export_dir


def _feature_clip(base, stage, npz, meta, clip_id, with_npz=True, with_meta=True):
    cdir = base / stage / clip_id
    cdir.mkdir(parents=True, exist_ok=True)
    if with_meta:
        (cdir / meta).write_text(
            json.dumps({"clip_id": clip_id, "num_frames": 20}), encoding="utf-8")
    if with_npz:
        (cdir / npz).write_bytes(b"npz")


def _export_report(export_dir, entries):
    (export_dir / "export_report.json").write_text(
        json.dumps({"generated_at": "2026-09-15T00:00:00Z", "clips": entries}),
        encoding="utf-8")


def _pass_entry(clip_id):
    return {
        "clip_id": clip_id, "status": "PASS",
        "n_flame": 20, "n_gaze": 20, "n_audio": 20,
        "num_frames": 20, "valid_length": 19,
        "gaze_missing_frames": 0, "audio_missing_frames": 0,
        "checks": {"length_match": "pass"},
    }


def _fail_entry(clip_id):
    return {
        "clip_id": clip_id, "status": "FAIL",
        "failed_reason": "flame/gaze/audio frame counts unmatchable (gaze=0)",
        "n_flame": 20, "n_gaze": 0, "n_audio": 20,
        "num_frames": 20, "valid_length": None,
        "gaze_missing_frames": 20, "audio_missing_frames": 0,
        "checks": {"length_match": "fail"},
    }


def test_stage13_gaze_failure_and_stage17_fail(tmp_path):
    """A clip missing the gaze payload fails Stage13 and Stage17 => partial."""
    run_root, base, export_dir = _setup(tmp_path, "single")

    # Stage12/14 succeed for the clip, Stage13 produces nothing for it.
    _feature_clip(base, "stage12_flame", "flame_features.npz",
                  "flame_metadata.json", CLIP_BAD)
    (base / "stage13_gaze").mkdir(parents=True, exist_ok=True)  # dir, no clip
    _feature_clip(base, "stage14_audio", "audio_features.npz",
                  "audio_metadata.json", CLIP_BAD)
    _export_report(export_dir, {CLIP_BAD: _fail_entry(CLIP_BAD)})

    save(_phase1_doc(run_root, base, [CLIP_BAD]), manifest_path(run_root))
    out = merge_phase2(base=base, export_dir=export_dir)

    assert out["stages"]["stage_12"]["status"] == "PASS"
    assert out["stages"]["stage_13"]["status"] == "FAIL", out["stages"]["stage_13"]
    assert out["stages"]["stage_14"]["status"] == "PASS"
    assert out["stages"]["stage_17"]["status"] == "FAIL"

    clip = {c["clip_id"]: c for c in out["clips"]}[CLIP_BAD]
    assert clip["stage_13"]["status"] == "failed"
    assert "gaze_features.npz missing" in clip["stage_13"]["reason"]
    assert clip["stage_17"]["status"] == "failed"
    assert "unmatchable" in clip["stage_17"]["failed_reason"]
    assert clip["stage_17"]["gaze_missing_frames"] == 20
    assert out["stages"]["stage_13"]["failed_clips"][CLIP_BAD] == \
        "gaze_features.npz missing"

    fa = out["final_assets"]
    assert fa["partial"] is True
    assert fa["asset_count"] == 0
    assert fa["failed_clip_count"] == 1
    assert fa["failed_clips"][0]["clip_id"] == CLIP_BAD
    assert "unmatchable" in fa["failed_clips"][0]["failed_reason"]

    # persistence: reload from disk retains the failure records
    reloaded = load(manifest_path(run_root))
    assert reloaded["stages"]["stage_13"]["status"] == "FAIL"
    assert reloaded["final_assets"]["partial"] is True


def test_partial_when_one_of_two_clips_fails(tmp_path):
    """One good clip + one Stage13-failed clip => stage13/17 PARTIAL."""
    run_root, base, export_dir = _setup(tmp_path, "partial")

    for stage, npz, meta in FEATURES:
        _feature_clip(base, stage, npz, meta, CLIP_OK)
    _feature_clip(base, "stage12_flame", "flame_features.npz",
                  "flame_metadata.json", CLIP_BAD)
    (base / "stage13_gaze").mkdir(parents=True, exist_ok=True)
    _feature_clip(base, "stage14_audio", "audio_features.npz",
                  "audio_metadata.json", CLIP_BAD)

    _export_report(export_dir, {
        CLIP_OK: _pass_entry(CLIP_OK),
        CLIP_BAD: _fail_entry(CLIP_BAD),
    })

    save(_phase1_doc(run_root, base, [CLIP_OK, CLIP_BAD]), manifest_path(run_root))
    out = merge_phase2(base=base, export_dir=export_dir)

    assert out["stages"]["stage_12"]["status"] == "PASS"
    assert out["stages"]["stage_13"]["status"] == "PARTIAL", out["stages"]["stage_13"]
    assert out["stages"]["stage_14"]["status"] == "PASS"
    assert out["stages"]["stage_17"]["status"] == "PARTIAL"
    assert out["stages"]["stage_13"]["failed_clips"] == \
        {CLIP_BAD: "gaze_features.npz missing"}
    assert out["stages"]["stage_17"]["failed_clip_count"] == 1

    fa = out["final_assets"]
    assert fa["partial"] is True
    assert fa["asset_count"] == 1
    assert fa["failed_clip_count"] == 1
    assert [a["clip_id"] for a in fa["assets"]] == [CLIP_OK]


def test_missing_feature_dir_is_skipped_not_failed(tmp_path):
    """A not-run feature stage (no output dir) stays SKIPPED, never FAIL."""
    run_root, base, export_dir = _setup(tmp_path, "skipped")

    save(_phase1_doc(run_root, base, [CLIP_OK]), manifest_path(run_root))
    out = merge_phase2(base=base, export_dir=export_dir)

    assert out["stages"]["stage_12"]["status"] == "SKIPPED"
    assert out["stages"]["stage_13"]["status"] == "SKIPPED"
    assert out["stages"]["stage_14"]["status"] == "SKIPPED"
    assert out["stages"]["stage_17"]["status"] == "SKIPPED"
    assert out["final_assets"]["asset_count"] == 0
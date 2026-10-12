"""Tests for dataset-mode pipeline orchestration (Stage6-9 optional).

Covers:
  1. ``pipeline_mode`` config resolution + validation.
  2. Stage10 adapter: ``accepted_clip_ids`` = all valid Stage5 clips.
  3. Stage14 audio adapter: original clip wav when Stage7.1 is skipped.
  4. E2E driver dataset mode: Stage5 -> Stage10 -> 11 -> 12/13/14 -> 17,
     with NO quality stages (Stage6/7/7.1/8/9/9.5) executed.
  5. E2E driver open_domain mode: quality stages still executed (regression).

open_domain remains the default; the pre-existing tests are left unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from face_pipeline_v2.config.config_loader import ConfigLoader
from face_pipeline_v2.data.models import (
    AVAlignmentInfo,
    AudioInfo,
    ClipInfo,
    FaceInfo,
    FaceSequence,
    FrameInfo,
    ResolutionValidationInfo,
    SceneFilterResult,
    SceneInfo,
    StandardizedVideoInfo,
    VideoInfo,
)
from face_pipeline_v2.pipeline import dataset_adapter
from face_pipeline_v2.pipeline.dataset_adapter import (
    DATASET_MODE,
    OPEN_DOMAIN_MODE,
    build_dataset_av_alignments,
    collect_accepted_clip_ids,
    resolve_pipeline_mode,
    resolve_stage14_alignment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
E2E_PATH = PROJECT_ROOT / "scripts" / "stage0_17_e2e_validate.py"

# Quality stages that MUST NOT run in dataset mode.
QUALITY_STAGE_NAMES = (
    "Stage6PoseEstimation",
    "Stage7SyncDetection",
    "Stage7_1AVAlignment",
    "Stage8QualityAssessment",
    "Stage9QualityFusion",
    "Stage9_5ClipExport",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _load_e2e_module():
    spec = importlib.util.spec_from_file_location(
        "stage0_17_e2e_validate_mod", E2E_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage0_17_e2e_validate_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e2e():
    return _load_e2e_module()


def _fake_extract(input_audio, output_audio, clip_start, clip_duration):
    """Fake ``extract_clip_audio`` — no ffmpeg, just create a stub wav file."""
    output_audio = Path(output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    output_audio.write_bytes(b"RIFF stub wav")
    return output_audio


def _write_config(tmp_path: Path, mode: str) -> Path:
    """Write a temp config with the requested pipeline_mode."""
    loader = ConfigLoader(config_path=PROJECT_ROOT / "configs" / "config.yaml")
    config = loader.load()
    config["pipeline_mode"] = mode
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return cfg_path


def _video_info() -> VideoInfo:
    return VideoInfo(
        video_id="vid01", video_path=Path("input/vid01.mp4"), original_fps=25.0,
        width=1280, height=720, duration=100.0, total_frames=2500,
        codec="h264", has_audio=True,
    )


def _resolution_info() -> ResolutionValidationInfo:
    return ResolutionValidationInfo(
        width=1280, height=720, min_width=534, min_height=534, is_valid=True,
    )


def _std_info() -> StandardizedVideoInfo:
    return StandardizedVideoInfo(
        video_path=Path("std/vid01.mp4"), audio_path=Path("std/vid01.wav"),
        target_fps=25, resolution=(256, 256), codec="h264",
    )


def _audio_info() -> AudioInfo:
    return AudioInfo(
        audio_path=Path("std/vid01.wav"), sample_rate=16000,
        duration=100.0, channels=1,
    )


def _scenes():
    return [
        SceneInfo(
            scene_id="s1", video_id="vid01", start_frame=0, end_frame=799,
            start_time=0.0, end_time=32.0, duration=32.0,
            filter_result=SceneFilterResult(passed=True, reject_reasons=()),
        ),
    ]


def _clips():
    return [
        ClipInfo(
            clip_id="clip_a", scene_id="s1", video_id="vid01", start_frame=0,
            end_frame=199, start_time=0.0, end_time=8.0, frame_count=200,
            duration=8.0,
        ),
        ClipInfo(
            clip_id="clip_b", scene_id="s1", video_id="vid01", start_frame=200,
            end_frame=399, start_time=8.0, end_time=16.0, frame_count=200,
            duration=8.0,
        ),
    ]


def _face_sequences():
    def _frame(idx, cid):
        return FrameInfo(
            frame_index=idx, clip_id=cid, timestamp=float(idx) / 25.0,
            image_path=Path(f"f_{cid}_{idx}.png"),
            faces=(
                FaceInfo(
                    frame_index=idx, face_id=f"{cid}_{idx}_0",
                    bbox=(10, 10, 100, 100),
                    landmark_5=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                    confidence=0.95, face_crop_path=Path("crop.png"),
                    track_id="t0",
                ),
            ),
        )

    return [
        FaceSequence(clip_id="clip_a", video_id="vid01",
                     frames=(_frame(0, "clip_a"), _frame(1, "clip_a"))),
        FaceSequence(clip_id="clip_b", video_id="vid01",
                     frames=(_frame(0, "clip_b"), _frame(1, "clip_b"))),
    ]


def _build_data():
    return {
        "video_info": _video_info(),
        "resolution_info": _resolution_info(),
        "std_info": _std_info(),
        "audio_info": _audio_info(),
        "scenes": _scenes(),
        "clips": _clips(),
        "face_sequences": _face_sequences(),
        "landmarks": [SimpleNamespace(clip_id="clip_a"),
                      SimpleNamespace(clip_id="clip_b")],
        "aligned": [
            SimpleNamespace(
                clip_id="clip_a", frame_index=0, face_id="f0", timestamp=0.0,
                aligned_face_path=Path("a.png"), affine_matrix=(1, 0, 0, 0, 1, 0),
                original_image_size=(256, 256), aligned_resolution=(256, 256)),
            SimpleNamespace(
                clip_id="clip_b", frame_index=0, face_id="f0", timestamp=0.0,
                aligned_face_path=Path("b.png"), affine_matrix=(1, 0, 0, 0, 1, 0),
                original_image_size=(256, 256), aligned_resolution=(256, 256)),
        ],
    }


def _install_stage_mocks(e2e, mapping):
    """Patch stage classes in the e2e module; returns mock-classes dict."""
    mocks = {}
    for name, value in mapping.items():
        patcher = mock.patch.object(e2e, name)
        cls = patcher.start()
        cls.return_value.run.return_value = value
        mocks[name] = cls
        mocks[f"{name}__patcher"] = patcher
    return mocks


def _install_quality_mocks(e2e):
    """Patch quality stage classes with bare mocks (call tracking only)."""
    mocks = {}
    for name in QUALITY_STAGE_NAMES:
        patcher = mock.patch.object(e2e, name)
        mocks[name] = patcher.start()
        mocks[f"{name}__patcher"] = patcher
    return mocks


def _stop_mocks(mocks):
    for key, value in mocks.items():
        if key.endswith("__patcher"):
            value.stop()


def _base_stage_mapping(data):
    return {
        "Stage0VideoIntegrityCheck": data["video_info"],
        "Stage0_5ResolutionFilter": data["resolution_info"],
        "Stage1AVStandardization": {
            "video_info": data["std_info"],
            "audio_info": data["audio_info"],
        },
        "Stage2SceneSegmentation": data["scenes"],
        "Stage3CoarseFaceScreening": data["scenes"],
        "Stage4TemporalClipGeneration": data["clips"],
        "Stage5FaceDetection": data["face_sequences"],
        "Stage5_1FaceTracking": data["face_sequences"],
        "Stage5_2SingleFaceFilter": data["face_sequences"],
        "Stage5_3TemporalStability": data["face_sequences"],
        "Stage10Landmark": data["landmarks"],
        "Stage11Alignment": data["aligned"],
    }


# ---------------------------------------------------------------------------
# 1. pipeline_mode resolution + validation
# ---------------------------------------------------------------------------


class TestPipelineModeConfig:
    def test_default_is_open_domain(self):
        assert resolve_pipeline_mode({}) == OPEN_DOMAIN_MODE

    def test_dataset_mode(self):
        assert resolve_pipeline_mode({"pipeline_mode": DATASET_MODE}) == DATASET_MODE

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="pipeline_mode"):
            resolve_pipeline_mode({"pipeline_mode": "bogus"})

    def test_config_yaml_ships_open_domain(self):
        loader = ConfigLoader(config_path=PROJECT_ROOT / "configs" / "config.yaml")
        config = loader.load()
        loader.validate(config)  # must not raise
        assert config.get("pipeline_mode") == OPEN_DOMAIN_MODE
        assert resolve_pipeline_mode(config) == OPEN_DOMAIN_MODE

    def test_config_loader_rejects_invalid_mode(self):
        loader = ConfigLoader(config_path=PROJECT_ROOT / "configs" / "config.yaml")
        config = loader.load()
        config["pipeline_mode"] = "nope"
        with pytest.raises(ValueError, match="pipeline_mode"):
            loader.validate(config)


# ---------------------------------------------------------------------------
# 2. Stage10 adapter — accepted clip ids
# ---------------------------------------------------------------------------


class TestStage10Adapter:
    def test_collect_all_valid_stage5_clips(self):
        ids = collect_accepted_clip_ids(_face_sequences())
        assert ids == {"clip_a", "clip_b"}

    def test_collect_skips_empty(self):
        ids = collect_accepted_clip_ids([])
        assert ids == set()



# ---------------------------------------------------------------------------
# 3. Stage14 audio adapter — original clip wav fallback
# ---------------------------------------------------------------------------


class TestStage14Adapter:
    def test_build_dataset_av_alignments_uses_original_clip_wav(self, tmp_path):
        out = tmp_path / "av"
        with mock.patch.object(dataset_adapter, "extract_clip_audio",
                               side_effect=_fake_extract):
            alignments = build_dataset_av_alignments(
                _clips(), _face_sequences(), _std_info(), out,
            )
        assert {a.clip_id for a in alignments} == {"clip_a", "clip_b"}
        a = next(a for a in alignments if a.clip_id == "clip_a")
        assert a.offset_frames == 0
        assert a.offset_ms == 0.0
        assert a.correction_applied is False
        assert a.alignment_status == "already_aligned"
        assert a.clip_start == 0.0
        assert a.clip_duration == 8.0
        assert a.original_audio_path == _std_info().audio_path
        assert a.aligned_audio_path == out / "clip_a" / "clip_a.wav"
        assert a.aligned_audio_path.exists()  # fake extraction created the wav

    def test_build_skips_clips_without_face_sequence(self, tmp_path):
        out = tmp_path / "av"
        clips = _clips()
        only_b = [s for s in _face_sequences() if s.clip_id == "clip_b"]
        with mock.patch.object(dataset_adapter, "extract_clip_audio",
                               side_effect=_fake_extract):
            alignments = build_dataset_av_alignments(clips, only_b, _std_info(), out)
        assert {a.clip_id for a in alignments} == {"clip_b"}

    def test_resolve_uses_existing_stage71_alignment(self, tmp_path):
        out = tmp_path / "av"
        aligned_wav = tmp_path / "existing" / "clip_a.wav"
        aligned_wav.parent.mkdir(parents=True, exist_ok=True)
        aligned_wav.write_bytes(b"RIFF")
        existing = AVAlignmentInfo(
            clip_id="clip_a", original_audio_path=Path("std.wav"),
            offset_frames=3, offset_ms=120.0, correction_applied=True,
            fps=25.0, corrected_audio_path=aligned_wav,
            aligned_audio_path=aligned_wav, alignment_status="corrected",
            raw_audio_path=aligned_wav, clip_start=0.0, clip_duration=8.0,
        )
        result = resolve_stage14_alignment(existing, _clips()[0], _std_info(), out)
        assert result is existing

    def test_resolve_falls_back_to_original_clip_wav(self, tmp_path):
        out = tmp_path / "av"
        with mock.patch.object(dataset_adapter, "extract_clip_audio",
                               side_effect=_fake_extract):
            result = resolve_stage14_alignment(
                None, _clips()[0], _std_info(), out,
            )
        assert result.clip_id == "clip_a"
        assert result.alignment_status == "already_aligned"
        assert result.aligned_audio_path == out / "clip_a" / "clip_a.wav"

    def test_resolve_falls_back_when_existing_audio_missing(self, tmp_path):
        out = tmp_path / "av"
        missing = AVAlignmentInfo(
            clip_id="clip_a", original_audio_path=Path("std.wav"),
            offset_frames=0, offset_ms=0.0, correction_applied=False,
            fps=25.0, corrected_audio_path=None,
            aligned_audio_path=tmp_path / "gone" / "nope.wav",
            alignment_status="already_aligned", raw_audio_path=None,
            clip_start=0.0, clip_duration=8.0,
        )
        with mock.patch.object(dataset_adapter, "extract_clip_audio",
                               side_effect=_fake_extract):
            result = resolve_stage14_alignment(
                missing, _clips()[0], _std_info(), out,
            )
        assert result.alignment_status == "already_aligned"
        assert result.aligned_audio_path == out / "clip_a" / "clip_a.wav"



# ---------------------------------------------------------------------------
# 4/5. E2E driver: dataset mode vs open_domain mode
# ---------------------------------------------------------------------------


def _run_e2e(e2e, tmp_path, mode, monkeypatch):
    """Run the e2e driver with all stage classes mocked, return (rc, mocks)."""
    data = _build_data()
    base_mocks = _install_stage_mocks(e2e, _base_stage_mapping(data))
    quality_mocks = _install_quality_mocks(e2e)

    if mode == OPEN_DOMAIN_MODE:
        # Quality stages return pass-through data in open_domain mode.
        quality_mocks["Stage6PoseEstimation"].return_value.run.return_value = [
            SimpleNamespace(poses=[1]),
        ]
        quality_mocks["Stage7SyncDetection"].return_value.run.return_value = [
            SimpleNamespace(clip_id="clip_a"),
        ]
        quality_mocks["Stage7_1AVAlignment"].return_value.run.return_value = [
            AVAlignmentInfo(
                clip_id="clip_a", original_audio_path=Path("std.wav"),
                offset_frames=0, offset_ms=0.0, correction_applied=False,
                fps=25.0, corrected_audio_path=None,
                aligned_audio_path=Path("std/av/clip_a.wav"),
                alignment_status="already_aligned", raw_audio_path=None,
                clip_start=0.0, clip_duration=8.0),
            AVAlignmentInfo(
                clip_id="clip_b", original_audio_path=Path("std.wav"),
                offset_frames=0, offset_ms=0.0, correction_applied=False,
                fps=25.0, corrected_audio_path=None,
                aligned_audio_path=Path("std/av/clip_b.wav"),
                alignment_status="already_aligned", raw_audio_path=None,
                clip_start=8.0, clip_duration=8.0),
        ]
        quality_mocks["Stage8QualityAssessment"].return_value.run.return_value = [
            SimpleNamespace(qualities=[1]),
        ]
        quality_mocks["Stage9QualityFusion"].return_value.run.return_value = [
            SimpleNamespace(clip_id="clip_a", is_accepted=True),
            SimpleNamespace(clip_id="clip_b", is_accepted=True),
        ]
        quality_mocks["Stage9_5ClipExport"].return_value.run.return_value = {
            "accepted_count": 2, "exported_videos": 2, "exported_audio": 2,
            "manifest_entries": 2, "failures": [],
        }

    # Stage14 adapter: fake clip-wav extraction (no real ffmpeg).
    monkeypatch.setattr(dataset_adapter, "extract_clip_audio", _fake_extract)

    cfg_path = _write_config(tmp_path, mode)
    test_base = tmp_path / f"stage0_11_{mode}"
    sys_argv = [
        "stage0_17_e2e_validate.py",
        "--video", str(tmp_path / "vid01.mp4"),
        "--config", str(cfg_path),
        "--test-base", str(test_base),
    ]
    monkeypatch.setattr(sys, "argv", sys_argv)

    rc = e2e.main()

    mocks = dict(base_mocks)
    mocks.update(quality_mocks)
    return rc, mocks, test_base



class TestE2EDatasetMode:
    def test_skips_quality_stages_and_feeds_all_clips_to_stage10(
        self, e2e, tmp_path, monkeypatch,
    ):
        rc, mocks, test_base = _run_e2e(e2e, tmp_path, DATASET_MODE, monkeypatch)
        try:
            assert rc == 0

            # Chain executed: Stage5 -> Stage10 -> Stage11 (-> 12/13/14/17
            # via exported metadata for the official runners).
            mocks["Stage5FaceDetection"].return_value.run.assert_called_once()
            mocks["Stage10Landmark"].return_value.run.assert_called_once()
            mocks["Stage11Alignment"].return_value.run.assert_called_once()

            # NO quality stage executed.
            for q in QUALITY_STAGE_NAMES:
                mocks[q].return_value.run.assert_not_called()

            # Stage10 received accepted_clip_ids = all valid Stage5 clips.
            input_dict = mocks["Stage10Landmark"].return_value.run.call_args[0][0]
            assert input_dict["accepted_clip_ids"] == {"clip_a", "clip_b"}

            # Stage14 metadata exported with original clip wav (no SyncNet).
            av_json = test_base / "stage_inputs" / "clip_a_av_alignment.json"
            assert av_json.exists()
            av = json.loads(av_json.read_text(encoding="utf-8"))
            assert av["clip_id"] == "clip_a"
            assert av["alignment_status"] == "already_aligned"
            assert av["offset_frames"] == 0
            assert av["aligned_audio_path"].endswith("clip_a.wav")
            af_json = test_base / "stage_inputs" / "clip_a_aligned_faces.json"
            assert af_json.exists()

            # Results JSON records pipeline_mode + SKIPPED quality stages.
            results_json = (PROJECT_ROOT / "outputs" / "stage0_17_e2e"
                            / "stage0_11_results.json")
            payload = json.loads(results_json.read_text(encoding="utf-8"))
            assert payload["pipeline_mode"] == DATASET_MODE
            skipped = [s for s in payload["stages"] if s["status"] == "SKIPPED"]
            assert {s["stage"] for s in skipped} == {
                "Stage 6 (6DRepNet pose)",
                "Stage 7 (SyncNet A/V sync)",
                "Stage 7.1 (A/V alignment)",
                "Stage 8 (HyperIQA quality)",
                "Stage 9 (Quality fusion/filter)",
                "Stage 9.5 (Clip export)",
            }
            assert payload["accepted_clips"].keys() == {"clip_a", "clip_b"}
        finally:
            _stop_mocks(mocks)


class TestE2EOpenDomainMode:
    def test_quality_stages_still_executed(self, e2e, tmp_path, monkeypatch):
        rc, mocks, test_base = _run_e2e(e2e, tmp_path, OPEN_DOMAIN_MODE, monkeypatch)
        try:
            assert rc == 0

            # Quality stages ARE executed in open_domain mode.
            for q in QUALITY_STAGE_NAMES:
                mocks[q].return_value.run.assert_called_once()

            # Stage10 received the accepted ids from Stage9 (both accepted).
            input_dict = mocks["Stage10Landmark"].return_value.run.call_args[0][0]
            assert input_dict["accepted_clip_ids"] == {"clip_a", "clip_b"}

            results_json = (PROJECT_ROOT / "outputs" / "stage0_17_e2e"
                            / "stage0_11_results.json")
            payload = json.loads(results_json.read_text(encoding="utf-8"))
            assert payload["pipeline_mode"] == OPEN_DOMAIN_MODE
            assert not any(s["status"] == "SKIPPED" for s in payload["stages"])
        finally:
            _stop_mocks(mocks)


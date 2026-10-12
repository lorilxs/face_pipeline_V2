"""Tests for Stage 9.5 clip export audio assets + Audio Alignment Validation.

Verifies the design fix:
    1. Final output structure clips/{clip_id}.mp4 and audio/{clip_id}.wav
       (canonical ALIGNED clip audio), keyed by the same clip_id; the raw clip
       audio is NOT exported as a final asset (kept only as the Stage 7.1
       intermediate under stage_outputs/av_alignment/<clip_id>/).
    2. Automatic Audio Alignment Validation Report: video/audio durations
       agree within tolerance (<50 ms).
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from face_pipeline_v2.data.models import (
    AVSyncInfo,
    ClipInfo,
    QualityMetrics,
    StandardizedVideoInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_07_1_av_alignment.stage import Stage7_1AVAlignment
from face_pipeline_v2.stages.stage_09_5_clip_export.exporter import (
    _probe_duration,
    export_accepted_clips,
    validate_exported_audio,
)
from face_pipeline_v2.utils.logger import PipelineLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Reusable 8s mp4 from the repo (re-encoded as the "std video" by the exporter).
_VIDEO_FIXTURE = (
    PROJECT_ROOT / "validation_final" / "clips"
    / "RD_Radio1_000_scene_0000_clip_0000.mp4"
)
SR = 16000


def _write_tone_wav(path: Path, duration: float) -> Path:
    n = int(SR * duration)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i in range(n):
            t = i / SR
            amp = 0.8 * math.sin(2 * math.pi * 440 * t)
            w.writeframes(struct.pack("<h", int(amp * 32767)))
    return path


def _make_context(output_dir: Path) -> PipelineContext:
    config = {
        "stage_07_1_av_alignment": {
            "enabled": True, "correction_threshold_ms": 50.0,
            "output_dir": "stage_outputs/av_alignment",
        },
        "sync": {"max_offset_frames": 8},
        "storage": {"output_dir": str(output_dir)},
        "stage_09_5": {},
    }
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_export", log_dir=output_dir / "logs"),
        model_manager=object(),
    )


class TestStage9_5ClipAudioExport:
    def test_final_structure_and_validation_pass(self, tmp_path):
        if not _VIDEO_FIXTURE.exists():
            pytest.skip("repo video fixture not present")
        base = tmp_path / "validation_final"
        ctx = _make_context(base)

        # Stage 7.1 → clip-level raw + aligned audio for one 8s clip.
        full = _write_tone_wav(tmp_path / "full.wav", 8.0)
        std_info = StandardizedVideoInfo(
            video_path=_VIDEO_FIXTURE, audio_path=full,
            target_fps=25, resolution=(320, 240), codec="h264",
        )
        sync = AVSyncInfo(clip_id="c0", offset=0, sync_lse_c=5.0, sync_lse_d=2.0,
                          start_frame=0, end_frame=200, fps=25.0,
                          duration=8.0, timestamp=0.0)
        alignments = Stage7_1AVAlignment().run(
            {"av_sync_infos": [sync], "std_video_info": std_info}, ctx
        )
        assert len(alignments) == 1

        clip = ClipInfo(clip_id="c0", scene_id="s0", video_id="vid",
                        start_frame=0, end_frame=200, start_time=0.0,
                        end_time=8.0, frame_count=200, duration=8.0)
        qm = QualityMetrics(clip_id="c0", video_id="vid", pose_score=0.9,
                            sync_score=0.9, quality_score=80.0,
                            fusion_score=0.85, is_accepted=True)

        clips_dir = base / "clips"
        audio_dir = base / "audio"
        manifest = base / "manifest.json"
        result = export_accepted_clips(
            accepted_metrics=[qm],
            clip_by_id={"c0": clip},
            align_by_id={"c0": alignments[0]},
            std_video_path=_VIDEO_FIXTURE,
            clips_dir=clips_dir,
            audio_dir=audio_dir,
            manifest_path=manifest,
            logger=ctx.logger,
        )

        assert result["exported_videos"] == 1
        assert result["exported_audio"] == 1
        assert result["failures"] == []

        # 1. two final assets, keyed by the same clip_id (audio = aligned)
        for d, suffix in ((clips_dir, "mp4"), (audio_dir, "wav")):
            p = d / f"c0.{suffix}"
            assert p.exists(), f"missing final asset: {p}"
            assert _probe_duration(p) == pytest.approx(8.0, abs=0.05)
        # the raw clip audio is NOT a final asset (stays as Stage 7.1
        # intermediate under stage_outputs/av_alignment/<clip_id>/)
        assert not (base / "aligned_audio").exists()

        # 2. validation report passes (<50 ms)
        v = result["validation"]
        assert v["report"] == "Audio Alignment Validation Report"
        assert v["passed"] == 1 and v["failed"] == 0
        row = v["clips"][0]
        assert row["status"] == "pass"
        assert row["max_diff_ms"] < 50.0
        assert row["video_duration"] == pytest.approx(8.0, abs=0.05)
        assert row["audio_duration"] == pytest.approx(8.0, abs=0.05)

        # manifest references the final assets (single canonical audio_path)
        import json
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        assert entries[0]["video_path"].endswith("clips/c0.mp4")
        assert entries[0]["audio_path"].endswith("audio/c0.wav")
        assert "aligned_audio_path" not in entries[0]

    def test_validation_flags_duration_mismatch(self, tmp_path):
        """A misplaced/short audio file must be reported as a validation failure."""
        base = tmp_path / "out"
        clips_dir = base / "clips"
        audio_dir = base / "audio"
        clips_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        # real 8s video asset
        if not _VIDEO_FIXTURE.exists():
            pytest.skip("repo video fixture not present")
        import shutil
        shutil.copy2(_VIDEO_FIXTURE, clips_dir / "c0.mp4")
        # audio is only 2s (mismatch)
        _write_tone_wav(audio_dir / "c0.wav", 2.0)

        clip = ClipInfo(clip_id="c0", scene_id="s0", video_id="vid",
                        start_frame=0, end_frame=200, start_time=0.0,
                        end_time=8.0, frame_count=200, duration=8.0)
        report = validate_exported_audio(
            clips_dir=clips_dir, audio_dir=audio_dir,
            clip_by_id={"c0": clip}, accepted_ids={"c0"},
            tolerance_ms=50.0,
        )
        assert report["failed"] == 1
        assert report["clips"][0]["status"] == "fail"
        assert report["clips"][0]["max_diff_ms"] >= 50.0

    # ------------------------------------------------------------------
    # Stage 7.1 intermediate audio lifecycle (cleanup_after_export)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_one_clip_export(
        tmp_path: Path, cleanup_intermediate: bool
    ) -> tuple:
        """Stage 7.1 → 9.5 for one 8s clip; returns (ctx, alignments, result)."""
        base = tmp_path / "validation_final"
        ctx = _make_context(base)
        full = _write_tone_wav(tmp_path / "full.wav", 8.0)
        std_info = StandardizedVideoInfo(
            video_path=_VIDEO_FIXTURE, audio_path=full,
            target_fps=25, resolution=(320, 240), codec="h264",
        )
        sync = AVSyncInfo(clip_id="c0", offset=0, sync_lse_c=5.0, sync_lse_d=2.0,
                          start_frame=0, end_frame=200, fps=25.0,
                          duration=8.0, timestamp=0.0)
        alignments = Stage7_1AVAlignment().run(
            {"av_sync_infos": [sync], "std_video_info": std_info}, ctx
        )
        inter_dir = alignments[0].raw_audio_path.parent
        before = (
            inter_dir.is_dir(),
            (inter_dir / "c0_raw.wav").exists(),
            (inter_dir / "c0.wav").exists(),
        )
        clip = ClipInfo(clip_id="c0", scene_id="s0", video_id="vid",
                        start_frame=0, end_frame=200, start_time=0.0,
                        end_time=8.0, frame_count=200, duration=8.0)
        qm = QualityMetrics(clip_id="c0", video_id="vid", pose_score=0.9,
                            sync_score=0.9, quality_score=80.0,
                            fusion_score=0.85, is_accepted=True)
        result = export_accepted_clips(
            accepted_metrics=[qm],
            clip_by_id={"c0": clip},
            align_by_id={"c0": alignments[0]},
            std_video_path=_VIDEO_FIXTURE,
            clips_dir=base / "clips",
            audio_dir=base / "audio",
            manifest_path=base / "manifest.json",
            logger=ctx.logger,
            cleanup_intermediate=cleanup_intermediate,
        )
        return base, alignments[0], result, before

    def test_cleanup_true_removes_intermediates_after_export(self, tmp_path):
        """cleanup_intermediate=True: intermediate dir removed, finals intact."""
        if not _VIDEO_FIXTURE.exists():
            pytest.skip("repo video fixture not present")
        base, al, result, before = self._run_one_clip_export(tmp_path, True)
        assert before == (True, True, True)  # intermediates existed before export

        # intermediate removed after export; final flat assets intact
        assert not al.raw_audio_path.parent.exists()
        for p in (base / "clips/c0.mp4", base / "audio/c0.wav"):
            assert p.exists(), f"missing final asset: {p}"
        assert not (base / "aligned_audio").exists()

        c = result["cleanup"]
        assert c["cleanup_after_export"] is True
        assert c["intermediate_found"] == 1
        assert c["fully_exported"] == 1
        assert c["removed"] == 1
        assert c["kept"] == []
        assert result["validation"]["passed"] == 1

    def test_cleanup_false_keeps_intermediates(self, tmp_path):
        """Default (cleanup_intermediate=False): intermediate audio kept."""
        if not _VIDEO_FIXTURE.exists():
            pytest.skip("repo video fixture not present")
        base, al, result, before = self._run_one_clip_export(tmp_path, False)
        assert before == (True, True, True)
        assert al.raw_audio_path.parent.is_dir()  # kept (default)
        assert result["cleanup"]["cleanup_after_export"] is False
        assert result["cleanup"]["removed"] == 0

    def test_cleanup_keeps_intermediate_when_aligned_copy_fails(self, tmp_path):
        """A clip whose aligned audio copy fails must NOT lose its intermediate."""
        if not _VIDEO_FIXTURE.exists():
            pytest.skip("repo video fixture not present")
        base = tmp_path / "validation_final"
        ctx = _make_context(base)
        full = _write_tone_wav(tmp_path / "full.wav", 8.0)
        std_info = StandardizedVideoInfo(
            video_path=_VIDEO_FIXTURE, audio_path=full,
            target_fps=25, resolution=(320, 240), codec="h264",
        )
        sync = AVSyncInfo(clip_id="c0", offset=0, sync_lse_c=5.0, sync_lse_d=2.0,
                          start_frame=0, end_frame=200, fps=25.0,
                          duration=8.0, timestamp=0.0)
        alignments = Stage7_1AVAlignment().run(
            {"av_sync_infos": [sync], "std_video_info": std_info}, ctx
        )
        al = alignments[0]
        inter_dir = al.raw_audio_path.parent
        # force an aligned copy failure by dropping the aligned source file
        # BEFORE Stage 9.5 runs (raw source stays)
        assert al.aligned_audio_path.exists()
        al.aligned_audio_path.unlink()

        clip = ClipInfo(clip_id="c0", scene_id="s0", video_id="vid",
                        start_frame=0, end_frame=200, start_time=0.0,
                        end_time=8.0, frame_count=200, duration=8.0)
        qm = QualityMetrics(clip_id="c0", video_id="vid", pose_score=0.9,
                            sync_score=0.9, quality_score=80.0,
                            fusion_score=0.85, is_accepted=True)
        result = export_accepted_clips(
            accepted_metrics=[qm],
            clip_by_id={"c0": clip},
            align_by_id={"c0": al},
            std_video_path=_VIDEO_FIXTURE,
            clips_dir=base / "clips",
            audio_dir=base / "audio",
            manifest_path=base / "manifest.json",
            logger=ctx.logger,
            cleanup_intermediate=True,
        )

        # aligned source missing → no final audio exported → intermediate kept
        assert not (base / "audio/c0.wav").exists()
        assert not (base / "aligned_audio").exists()
        assert inter_dir.is_dir(), "intermediate must survive a failed export"
        c = result["cleanup"]
        # not fully exported → not a cleanup candidate → never removed
        assert c["removed"] == 0
        assert c["fully_exported"] == 0
        assert c["left"] == 1

    def test_safe_remove_protects_final_dirs(self, tmp_path):
        """_safe_remove_intermediate_dir must never delete final/root dirs."""
        from face_pipeline_v2.stages.stage_09_5_clip_export.exporter import (
            _safe_remove_intermediate_dir,
        )
        root = tmp_path / "validation_final"
        clips_dir = root / "clips"
        audio_dir = root / "audio"
        stage_outputs = root / "stage_outputs"
        for d in (root, clips_dir, audio_dir, stage_outputs):
            d.mkdir(parents=True, exist_ok=True)
        protected = [root, clips_dir, audio_dir]

        # every protected/final dir must be refused (and still exist)
        for d in protected:
            assert _safe_remove_intermediate_dir(d, protected) is False
            assert d.exists()

        # a genuine nested intermediate dir is removed
        inter = stage_outputs / "av_alignment" / "c0"
        inter.mkdir(parents=True)
        (inter / "c0_raw.wav").write_bytes(b"x")
        assert _safe_remove_intermediate_dir(inter, protected) is True
        assert not inter.exists()

        # final dirs still intact after the cleanup above
        for d in protected:
            assert d.exists()


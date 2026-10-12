"""Tests for Stage 7.1 clip-level audio generation (raw extraction + offset shift).

Verifies the design fix:
    1. extract_clip_audio() produces a clip-level raw audio whose duration
       equals the clip window (never the full video audio).
    2. shift_clip_audio() applies the SyncNet offset ON the clip audio and
       preserves the clip duration exactly (head-pad for delay, tail-pad for
       advance).
    3. Stage 7.1 with boundary-bearing AVSyncInfo produces raw_audio_path +
       aligned_audio_path for already_aligned and corrected clips, and no audio
       for severe_offset_unavailable clips.
"""

import math
import struct
import wave
from pathlib import Path

import pytest

from face_pipeline_v2.data.models import AVSyncInfo, StandardizedVideoInfo
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_07_1_av_alignment.processor import (
    extract_clip_audio,
    shift_clip_audio,
)
from face_pipeline_v2.stages.stage_07_1_av_alignment.stage import Stage7_1AVAlignment
from face_pipeline_v2.stages.stage_09_5_clip_export.exporter import _probe_duration
from face_pipeline_v2.utils.logger import PipelineLogger

SR = 16000


def _write_tone_wav(path: Path, duration: float,
                    tone_start: float = 4.0, tone_end: float = 12.0) -> Path:
    """Write a 16 kHz mono wav with a 440 Hz tone only in [tone_start, tone_end)."""
    n = int(SR * duration)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i in range(n):
            t = i / SR
            amp = 0.8 * math.sin(2 * math.pi * 440 * t) if tone_start <= t < tone_end else 0.0
            w.writeframes(struct.pack("<h", int(amp * 32767)))
    return path


def _rms(path: Path) -> float:
    """Return RMS (0..1) of a wav file."""
    import array
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    if n == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(raw)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32767.0


def _make_context(output_dir: Path, threshold_ms: float = 50.0,
                  max_offset_frames: int = 8) -> PipelineContext:
    config = {
        "stage_07_1_av_alignment": {
            "enabled": True, "correction_threshold_ms": threshold_ms,
            "output_dir": "stage_outputs/av_alignment",
        },
        "sync": {"max_offset_frames": max_offset_frames},
        "storage": {"output_dir": str(output_dir)},
    }
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_clip_audio", log_dir=output_dir / "logs"),
        model_manager=object(),
    )


class TestExtractClipAudio:
    def test_extracts_clip_window_not_full_audio(self, tmp_path):
        """Clip raw audio duration == clip window duration (8s), not the full 24s."""
        full = _write_tone_wav(tmp_path / "full.wav", 24.0)
        out = tmp_path / "clip_raw.wav"
        extract_clip_audio(full, out, clip_start=8.0, clip_duration=8.0)
        assert _probe_duration(out) == pytest.approx(8.0, abs=0.02)
        assert _probe_duration(out) < 8.5  # never the full video audio


class TestShiftClipAudio:
    @pytest.mark.parametrize("offset_frames", [3, -3, 0])
    def test_duration_preserved(self, tmp_path, offset_frames):
        """Shift on clip audio always keeps the 8s clip duration."""
        raw = _write_tone_wav(tmp_path / "raw.wav", 8.0)
        out = tmp_path / "aligned.wav"
        shift_clip_audio(raw, out, offset_frames=offset_frames,
                         fps=25.0, clip_duration=8.0)
        assert out.exists()
        assert _probe_duration(out) == pytest.approx(8.0, abs=0.02)

    def test_delay_prepends_silence(self, tmp_path):
        """offset>0 (audio ahead → delay): aligned starts with ~offset of silence."""
        raw = _write_tone_wav(tmp_path / "raw.wav", 8.0, tone_start=0.0, tone_end=8.0)
        out = tmp_path / "aligned.wav"
        shift_clip_audio(raw, out, offset_frames=3, fps=25.0, clip_duration=8.0)
        head = tmp_path / "head.wav"
        extract_clip_audio(out, head, 0.0, 0.10)
        assert _rms(head) < 0.02  # first 100ms is silence (delay applied)


class TestStage7_1ClipAudio:
    def _run(self, tmp_path, syncs):
        full = _write_tone_wav(tmp_path / "full.wav", 24.0)
        std_info = StandardizedVideoInfo(
            video_path=tmp_path / "v.mp4", audio_path=full,
            target_fps=25, resolution=(320, 240), codec="h264",
        )
        ctx = _make_context(tmp_path / "out")
        return Stage7_1AVAlignment().run(
            {"av_sync_infos": syncs, "std_video_info": std_info}, ctx
        )

    def test_boundaries_produce_raw_and_aligned(self, tmp_path):
        syncs = [
            AVSyncInfo(clip_id="c0", offset=0, sync_lse_c=5.0, sync_lse_d=2.0,
                       start_frame=0, end_frame=200, fps=25.0,
                       duration=8.0, timestamp=0.0),
            AVSyncInfo(clip_id="c1", offset=3, sync_lse_c=5.0, sync_lse_d=2.0,
                       start_frame=200, end_frame=400, fps=25.0,
                       duration=8.0, timestamp=8.0),
        ]
        out = self._run(tmp_path, syncs)
        assert len(out) == 2
        for a in out:
            assert a.raw_audio_path and a.raw_audio_path.exists(), f"{a.clip_id}: raw missing"
            assert a.aligned_audio_path and a.aligned_audio_path.exists(), \
                f"{a.clip_id}: aligned missing"
            assert _probe_duration(a.raw_audio_path) == pytest.approx(8.0, abs=0.02)
            assert _probe_duration(a.aligned_audio_path) == pytest.approx(8.0, abs=0.02)
        assert out[0].correction_applied is False  # offset 0 → already_aligned
        assert out[0].alignment_status == "already_aligned"
        assert out[1].correction_applied is True   # offset 3 → corrected
        assert out[1].alignment_status == "corrected"

    def test_severe_offset_no_audio(self, tmp_path):
        syncs = [AVSyncInfo(clip_id="c0", offset=20, sync_lse_c=5.0, sync_lse_d=2.0,
                            start_frame=0, end_frame=200, fps=25.0,
                            duration=8.0, timestamp=0.0)]
        out = self._run(tmp_path, syncs)
        assert out[0].alignment_status == "severe_offset_unavailable"
        assert out[0].aligned_audio_path is None
        assert out[0].raw_audio_path is None


"""Tests for Stage 4 segment mode (mode="segment").

Coverage (per Stage4_Dynamic_Clip_Design_Update.md §2-§4):
    1. 8s segment → 1 full-length clip.
    2. 16s segment → 1 full-length clip.
    3. 400-frame input → clip keeps full frame range (no resampling).
    4. clip_id consistency (stable, unique per scene).
    5. Timestamp continuity (start/end preserved, contiguous frame indices).
    6. Backward compatibility: sliding_window (default + explicit) unchanged.
    7. Invalid mode falls back to sliding_window.
    8. Rejected scenes are skipped in segment mode.
"""

import pytest

from face_pipeline_v2.data.models import (
    ClipInfo,
    CoarseFaceInfo,
    SceneFilterResult,
    SceneInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_04_temporal_clip_generation.clip_generator import (
    generate_clips,
    generate_segment_clips,
)
from face_pipeline_v2.stages.stage_04_temporal_clip_generation.stage import (
    Stage4TemporalClipGeneration,
)
from face_pipeline_v2.utils.exceptions import DataProcessingError
from face_pipeline_v2.utils.logger import PipelineLogger

FPS: float = 25.0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_scene(
    scene_id: str = "vid_scene_0000",
    video_id: str = "vid",
    start_time: float = 0.0,
    end_time: float = 16.0,
    start_frame: int | None = None,
    end_frame: int | None = None,
    passed: bool = True,
) -> SceneInfo:
    """Create a SceneInfo with deterministic frames (default 25 fps)."""
    if start_frame is None:
        start_frame = int(round(start_time * FPS))
    if end_frame is None:
        end_frame = int(round(end_time * FPS))
    return SceneInfo(
        scene_id=scene_id,
        video_id=video_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        coarse_face_info=CoarseFaceInfo(
            sampled_frame_count=end_frame - start_frame,
            face_detect_ratio=1.0,
            single_face_ratio=1.0,
            avg_face_confidence=0.9,
            max_face_count=1,
        ),
        filter_result=SceneFilterResult(
            passed=passed,
            reject_reasons=() if passed else ("too_short",),
        ),
    )


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_04": {
            "enabled": True,
            "mode": "segment",
            "window_size": 8,
            "stride": 4,
            "target_fps": FPS,
        },
    }
    if config_overrides:
        config["stage_04"].update(config_overrides)
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage04_segment_mode", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# generate_segment_clips pure-logic tests
# ---------------------------------------------------------------------------


class TestGenerateSegmentClips:
    """Unit tests for the segment-mode pure-logic function."""

    def test_8s_segment_produces_one_full_clip(self) -> None:
        """8s segment → exactly 1 clip with dynamic duration 8s."""
        scene = _make_scene(start_time=0.0, end_time=8.0)
        clips = generate_segment_clips(scene=scene, target_fps=FPS)

        assert len(clips) == 1
        c = clips[0]
        assert isinstance(c, ClipInfo)
        assert c.duration == 8.0
        assert c.start_time == 0.0
        assert c.end_time == 8.0
        assert c.start_frame == 0
        assert c.end_frame == 200
        assert c.frame_count == 200

    def test_16s_segment_produces_one_full_clip(self) -> None:
        """16s segment → exactly 1 clip with dynamic duration 16s."""
        scene = _make_scene(start_time=0.0, end_time=16.0)
        clips = generate_segment_clips(scene=scene, target_fps=FPS)

        assert len(clips) == 1
        c = clips[0]
        assert c.duration == 16.0
        assert c.frame_count == 400

    def test_400_frame_input_keeps_full_frame_range(self) -> None:
        """400-frame input (16s @ 25fps) → full frame range, no resampling."""
        scene = _make_scene(start_frame=0, end_frame=400, start_time=0.0, end_time=16.0)
        clips = generate_segment_clips(scene=scene, target_fps=FPS)

        assert len(clips) == 1
        c = clips[0]
        assert c.start_frame == 0
        assert c.end_frame == 400
        assert c.frame_count == 400
        # No resampling: contiguous range == duration * fps == scene frames.
        assert c.frame_count == c.end_frame - c.start_frame
        assert c.frame_count == 400

    def test_clip_id_stable_and_unique_per_scene(self) -> None:
        """clip_id is stable (deterministic) and unique across scenes."""
        s1 = _make_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=16.0)
        s2 = _make_scene(scene_id="vid_scene_0001", start_time=20.0, end_time=36.0)

        c1a = generate_segment_clips(scene=s1, target_fps=FPS)[0]
        c1b = generate_segment_clips(scene=s1, target_fps=FPS)[0]
        c2 = generate_segment_clips(scene=s2, target_fps=FPS)[0]

        assert c1a.clip_id == "vid_scene_0000_clip_0000"
        # Deterministic / stable across calls.
        assert c1a.clip_id == c1b.clip_id
        # Unique across different scenes.
        assert c1a.clip_id != c2.clip_id
        assert c2.clip_id == "vid_scene_0001_clip_0000"

    def test_timestamp_continuity_preserved(self) -> None:
        """start/end timestamps and frame indices preserved verbatim."""
        scene = _make_scene(
            scene_id="vid_scene_0000",
            start_time=5.0,
            end_time=21.0,
            start_frame=125,
            end_frame=525,
        )
        clips = generate_segment_clips(scene=scene, target_fps=FPS)

        assert len(clips) == 1
        c = clips[0]
        # Timestamp continuity: no shift, no gap, full segment coverage.
        assert c.start_time == 5.0
        assert c.end_time == 21.0
        assert c.duration == 16.0
        assert c.end_time - c.start_time == c.duration
        # Frame continuity: contiguous indices covering [125, 525).
        assert c.start_frame == 125
        assert c.end_frame == 525
        assert c.frame_count == 400
        assert c.frame_count == c.end_frame - c.start_frame

    def test_metadata_not_hardcoded(self) -> None:
        """duration/frame_count derive from the scene (no 8s/200 default)."""
        # 12.5s scene at 24fps → dynamic duration / frame count.
        scene = _make_scene(
            start_time=0.0,
            end_time=12.5,
            start_frame=0,
            end_frame=300,  # 12.5 * 24
        )
        clips = generate_segment_clips(scene=scene, target_fps=24.0)

        assert len(clips) == 1
        assert clips[0].duration == 12.5
        assert clips[0].frame_count == 300

    def test_degenerate_scene_returns_empty(self) -> None:
        """Zero-duration / zero-frame scene → no clip (empty list)."""
        scene = _make_scene(start_time=0.0, end_time=0.0, start_frame=0, end_frame=0)
        assert generate_segment_clips(scene=scene, target_fps=FPS) == []

    def test_non_positive_target_fps_raises(self) -> None:
        """DataProcessingError on invalid target_fps."""
        scene = _make_scene(start_time=0.0, end_time=8.0)
        with pytest.raises(DataProcessingError, match="target_fps must be positive"):
            generate_segment_clips(scene=scene, target_fps=0.0)


# ---------------------------------------------------------------------------
# Stage4TemporalClipGeneration (mode=segment) tests
# ---------------------------------------------------------------------------


class TestStage4SegmentMode:
    """Stage-level segment-mode behavior via stage.run()."""

    def test_segment_mode_8s_scene(self) -> None:
        """8s scene in segment mode → 1 full-length clip."""
        scene = _make_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=8.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[scene], context=context)
        assert len(output) == 1
        assert output[0].scene_id == "vid_scene_0000"
        assert output[0].duration == 8.0
        assert output[0].frame_count == 200

    def test_segment_mode_16s_scene(self) -> None:
        """16s scene in segment mode → 1 clip of 400 frames."""
        scene = _make_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=16.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[scene], context=context)
        assert len(output) == 1
        assert output[0].duration == 16.0
        assert output[0].frame_count == 400

    def test_segment_mode_rejected_scene_skipped(self) -> None:
        """Rejected scenes are skipped in segment mode too."""
        passed = _make_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=16.0)
        rejected = _make_scene(
            scene_id="vid_scene_0001", start_time=30.0, end_time=35.0, passed=False
        )
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[passed, rejected], context=context)
        assert len(output) == 1
        assert output[0].scene_id == "vid_scene_0000"

    def test_segment_mode_multiple_scenes_one_clip_each(self) -> None:
        """Two passed scenes → two clips (one per scene, 1:1)."""
        s1 = _make_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=16.0)
        s2 = _make_scene(scene_id="vid_scene_0001", start_time=20.0, end_time=36.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[s1, s2], context=context)
        assert len(output) == 2
        assert [c.scene_id for c in output] == ["vid_scene_0000", "vid_scene_0001"]
        # 1:1 and stable ids.
        assert [c.clip_id for c in output] == [
            "vid_scene_0000_clip_0000",
            "vid_scene_0001_clip_0000",
        ]

    def test_segment_mode_output_validates(self) -> None:
        """Output list of ClipInfo passes validate_output."""
        scene = _make_scene(start_time=0.0, end_time=16.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[scene], context=context)
        assert stage.validate_output(output, context) is True



# ---------------------------------------------------------------------------
# Backward compatibility (sliding_window must be unchanged)
# ---------------------------------------------------------------------------


class TestSlidingWindowBackwardCompat:
    """sliding_window remains fully compatible with the pre-change behavior."""

    def test_default_mode_is_sliding_window(self) -> None:
        """No mode key in config → sliding_window behavior (8s window)."""
        scene = _make_scene(start_time=0.0, end_time=30.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        del context.config["stage_04"]["mode"]

        output = stage.run(input_data=[scene], context=context)
        # 30s, window=8 stride=4 → 6 clips (same as before the change).
        assert len(output) == 6
        assert output[0].duration == 8.0
        assert [c.duration for c in output] == [8.0] * 6

    def test_explicit_sliding_window_matches_legacy(self) -> None:
        """mode='sliding_window' produces identical output to legacy path."""
        scene = _make_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=30.0)

        legacy = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=FPS)

        stage = Stage4TemporalClipGeneration()
        context = _make_context({"mode": "sliding_window"})
        current = stage.run(input_data=[scene], context=context)

        assert len(current) == len(legacy)
        for new_clip, old_clip in zip(current, legacy):
            assert new_clip.clip_id == old_clip.clip_id
            assert new_clip.start_time == old_clip.start_time
            assert new_clip.end_time == old_clip.end_time
            assert new_clip.frame_count == old_clip.frame_count
            assert new_clip.duration == old_clip.duration

    def test_invalid_mode_falls_back_to_sliding_window(self) -> None:
        """Unknown mode value → warning + fallback to sliding_window."""
        scene = _make_scene(start_time=0.0, end_time=30.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context({"mode": "bogus_mode"})

        output = stage.run(input_data=[scene], context=context)
        # Falls back to sliding_window → 6 clips of 8s.
        assert len(output) == 6
        assert all(c.duration == 8.0 for c in output)

    def test_disabled_stage_returns_empty_in_segment_mode(self) -> None:
        """enabled=false in segment mode → empty list."""
        scene = _make_scene(start_time=0.0, end_time=16.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context({"enabled": False})

        output = stage.run(input_data=[scene], context=context)
        assert isinstance(output, list)
        assert len(output) == 0

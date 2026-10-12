"""Tests for Stage 4: Temporal Clip Generation / Sliding Window.

Tests cover:
    1. 30s scene with window=8 stride=4 → 6 clips.
    2. 7s scene (< window) → 0 clips.
    3. 8s scene (= window) → 1 clip.
    4. 16s scene boundary verification.
    5. Clip ID stability (deterministic ids).
    6. Input SceneInfo is not mutated.
    7. Wrong input type → InputValidationError.
    8. Empty input → InputValidationError.
    9. Config parameters are honored.
"""

from pathlib import Path

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
)
from face_pipeline_v2.stages.stage_04_temporal_clip_generation.stage import (
    Stage4TemporalClipGeneration,
)
from face_pipeline_v2.utils.exceptions import (
    DataProcessingError,
    InputValidationError,
)
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_passed_scene(
    scene_id: str = "vid_scene_0000",
    video_id: str = "vid",
    start_time: float = 0.0,
    end_time: float = 30.0,
) -> SceneInfo:
    """Create a SceneInfo that passed Stage 3 filtering."""
    duration = end_time - start_time
    return SceneInfo(
        scene_id=scene_id,
        video_id=video_id,
        start_frame=int(start_time * 25),
        end_frame=int(end_time * 25),
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        coarse_face_info=CoarseFaceInfo(
            sampled_frame_count=int(duration),
            face_detect_ratio=1.0,
            single_face_ratio=1.0,
            avg_face_confidence=0.9,
            max_face_count=1,
        ),
        filter_result=SceneFilterResult(
            passed=True,
            reject_reasons=(),
        ),
    )


def _make_rejected_scene(
    scene_id: str = "vid_scene_0001",
    video_id: str = "vid",
    start_time: float = 30.0,
    end_time: float = 35.0,
) -> SceneInfo:
    """Create a SceneInfo that was rejected by Stage 3."""
    duration = end_time - start_time
    return SceneInfo(
        scene_id=scene_id,
        video_id=video_id,
        start_frame=int(start_time * 25),
        end_frame=int(end_time * 25),
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        coarse_face_info=CoarseFaceInfo(
            sampled_frame_count=0,
            face_detect_ratio=0.0,
            single_face_ratio=0.0,
            avg_face_confidence=0.0,
            max_face_count=0,
        ),
        filter_result=SceneFilterResult(
            passed=False,
            reject_reasons=("too_short",),
        ),
    )


def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_04": {
            "enabled": True,
            "window_size": 8,
            "stride": 4,
            "target_fps": 25,
        },
    }
    if config_overrides:
        config["stage_04"].update(config_overrides)
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_04", log_dir=None),
        model_manager=object(),
    )


# ---------------------------------------------------------------------------
# clip_generator unit tests
# ---------------------------------------------------------------------------


class TestClipGenerator:
    """Unit tests for generate_clips pure-logic function."""

    def test_30s_scene_produces_6_clips(self) -> None:
        """30s scene with window=8 stride=4 → 6 clips."""
        scene = _make_passed_scene(start_time=0.0, end_time=30.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(clips) == 6
        # Verify clip time ranges
        expected = [
            (0.0, 8.0),
            (4.0, 12.0),
            (8.0, 16.0),
            (12.0, 20.0),
            (16.0, 24.0),
            (20.0, 28.0),
        ]
        for clip, (exp_start, exp_end) in zip(clips, expected):
            assert clip.start_time == pytest.approx(exp_start)
            assert clip.end_time == pytest.approx(exp_end)
            assert clip.duration == pytest.approx(8.0)

    def test_7s_scene_produces_0_clips(self) -> None:
        """7s scene < 8s window → 0 clips."""
        scene = _make_passed_scene(start_time=0.0, end_time=7.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)
        assert len(clips) == 0

    def test_8s_scene_produces_1_clip(self) -> None:
        """8s scene = window → 1 clip."""
        scene = _make_passed_scene(start_time=0.0, end_time=8.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(clips) == 1
        assert clips[0].start_time == 0.0
        assert clips[0].end_time == 8.0
        assert clips[0].duration == 8.0

    def test_16s_scene_boundaries(self) -> None:
        """16s scene with window=8 stride=4 → 3 clips, verify boundaries."""
        scene = _make_passed_scene(start_time=0.0, end_time=16.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(clips) == 3
        # 0-8, 4-12, 8-16
        expected = [(0.0, 8.0), (4.0, 12.0), (8.0, 16.0)]
        for clip, (exp_start, exp_end) in zip(clips, expected):
            assert clip.start_time == pytest.approx(exp_start)
            assert clip.end_time == pytest.approx(exp_end)

    def test_clip_ids_are_stable(self) -> None:
        """Same input produces same clip IDs (deterministic)."""
        scene = _make_passed_scene(start_time=0.0, end_time=9.0)
        run1 = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)
        run2 = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(run1) == 1
        assert len(run2) == 1
        assert run1[0].clip_id == run2[0].clip_id
        assert run1[0].clip_id == "vid_scene_0000_clip_0000"

    def test_clip_ids_are_unique_across_windows(self) -> None:
        """Each clip in a scene has a unique, sequential clip_id."""
        scene = _make_passed_scene(start_time=0.0, end_time=30.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        ids = [c.clip_id for c in clips]
        assert len(ids) == len(set(ids))  # all unique
        assert ids == [
            "vid_scene_0000_clip_0000",
            "vid_scene_0000_clip_0001",
            "vid_scene_0000_clip_0002",
            "vid_scene_0000_clip_0003",
            "vid_scene_0000_clip_0004",
            "vid_scene_0000_clip_0005",
        ]

    def test_non_zero_start_time(self) -> None:
        """Scene at 10-20s (10s duration) → only 1 clip (10-18).

        start=10: 10-18 ✓, start=14: 14-22 ✗ (exceeds scene end)
        """
        scene = _make_passed_scene(start_time=10.0, end_time=20.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(clips) == 1
        assert clips[0].start_time == 10.0
        assert clips[0].end_time == 18.0

    def test_frame_calculation(self) -> None:
        """Frame indices are correctly computed from time × fps."""
        scene = _make_passed_scene(start_time=0.0, end_time=8.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(clips) == 1
        assert clips[0].start_frame == 0
        assert clips[0].end_frame == 200  # 8.0 * 25 = 200
        assert clips[0].frame_count == 200

    def test_different_fps(self) -> None:
        """Frame calculation uses target_fps, not hard-coded 25."""
        scene = _make_passed_scene(start_time=0.0, end_time=8.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=30.0)

        assert clips[0].start_frame == 0
        assert clips[0].end_frame == 240  # 8.0 * 30 = 240
        assert clips[0].frame_count == 240

    def test_clip_fields_are_fully_populated(self) -> None:
        """Every ClipInfo field is populated correctly."""
        scene = _make_passed_scene(
            scene_id="test_scene_0001",
            video_id="test_video",
            start_time=2.0,
            end_time=10.0,
        )
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)

        assert len(clips) == 1
        c = clips[0]
        assert c.clip_id == "test_scene_0001_clip_0000"
        assert c.scene_id == "test_scene_0001"
        assert c.video_id == "test_video"
        assert c.start_time == 2.0
        assert c.end_time == 10.0
        assert c.duration == 8.0
        assert c.start_frame == 50   # 2.0 * 25
        assert c.end_frame == 250    # 10.0 * 25
        assert c.frame_count == 200

    def test_negative_window_size_raises(self) -> None:
        """DataProcessingError on invalid window_size."""
        scene = _make_passed_scene()
        with pytest.raises(DataProcessingError, match="window_size must be positive"):
            generate_clips(scene, window_size=-1.0, stride=4.0, target_fps=25.0)

    def test_zero_stride_raises(self) -> None:
        """DataProcessingError on invalid stride."""
        scene = _make_passed_scene()
        with pytest.raises(DataProcessingError, match="stride must be positive"):
            generate_clips(scene, window_size=8.0, stride=0.0, target_fps=25.0)

    def test_keep_tail_short_scene_kept_whole(self) -> None:
        """keep_tail=True: a sub-window scene is kept whole, not dropped."""
        scene = _make_passed_scene(start_time=0.0, end_time=7.0)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0, keep_tail=True
        )
        assert len(clips) == 1
        assert clips[0].start_time == 0.0
        assert clips[0].end_time == 7.0
        assert clips[0].duration == pytest.approx(7.0)

    def test_keep_tail_keeps_trailing_remainder(self) -> None:
        """keep_tail=True: 10s scene → 8s + 2s (tail kept)."""
        scene = _make_passed_scene(start_time=0.0, end_time=10.0)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0, keep_tail=True
        )
        assert len(clips) == 2
        assert clips[0].start_time == 0.0
        assert clips[0].end_time == 8.0
        assert clips[1].start_time == 8.0
        assert clips[1].end_time == 10.0
        assert clips[1].duration == pytest.approx(2.0)

    def test_keep_tail_exact_multiple_no_partial(self) -> None:
        """keep_tail=True: exact multiple (16s) produces no trailing partial."""
        scene = _make_passed_scene(start_time=0.0, end_time=16.0)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0, keep_tail=True
        )
        assert len(clips) == 2
        assert clips[1].start_time == 8.0
        assert clips[1].end_time == 16.0

    def test_keep_tail_false_still_drops_tail(self) -> None:
        """Default keep_tail=False: 10s scene → only the complete 8s window."""
        scene = _make_passed_scene(start_time=0.0, end_time=10.0)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0
        )
        assert len(clips) == 1
        assert clips[0].end_time == 8.0

    def test_keep_tail_drops_tail_shorter_than_min_tail(self) -> None:
        """min_tail_duration=1.0: 8.5s scene tail (0.5s) is discarded."""
        scene = _make_passed_scene(start_time=0.0, end_time=8.5)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0,
            keep_tail=True, min_tail_duration=1.0,
        )
        assert len(clips) == 1
        assert clips[0].end_time == 8.0

    def test_keep_tail_keeps_tail_at_min_tail(self) -> None:
        """min_tail_duration=1.0: a 1s tail (9s scene) is kept."""
        scene = _make_passed_scene(start_time=0.0, end_time=9.0)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0,
            keep_tail=True, min_tail_duration=1.0,
        )
        assert len(clips) == 2
        assert clips[1].start_time == 8.0
        assert clips[1].end_time == 9.0

    def test_keep_tail_min_tail_does_not_affect_short_scene(self) -> None:
        """min_tail_duration does not drop a sub-window scene kept whole."""
        scene = _make_passed_scene(start_time=0.0, end_time=5.0)
        clips = generate_clips(
            scene, window_size=8.0, stride=8.0, target_fps=25.0,
            keep_tail=True, min_tail_duration=1.0,
        )
        assert len(clips) == 1
        assert clips[0].duration == pytest.approx(5.0)

    def test_keep_tail_negative_min_tail_raises(self) -> None:
        """DataProcessingError on negative min_tail_duration."""
        scene = _make_passed_scene(start_time=0.0, end_time=10.0)
        with pytest.raises(DataProcessingError, match="min_tail_duration"):
            generate_clips(
                scene, window_size=8.0, stride=8.0, target_fps=25.0,
                keep_tail=True, min_tail_duration=-1.0,
            )


# ---------------------------------------------------------------------------
# Stage4TemporalClipGeneration tests
# ---------------------------------------------------------------------------


class TestStage4TemporalClipGeneration:
    """Tests for Stage4TemporalClipGeneration BaseStage subclass."""

    def test_normal_processing(self) -> None:
        """30s passed scene → 6 clips via stage.run()."""
        scene = _make_passed_scene(start_time=0.0, end_time=30.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[scene], context=context)
        assert isinstance(output, list)
        assert len(output) == 6
        assert all(isinstance(c, ClipInfo) for c in output)

    def test_rejected_scenes_are_skipped(self) -> None:
        """Only passed scenes generate clips; rejected are skipped."""
        passed = _make_passed_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=12.0)
        rejected = _make_rejected_scene(scene_id="vid_scene_0001")

        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[passed, rejected], context=context)
        # Only passed scene generates clips (12s → 2 clips: 0-8, 4-12)
        assert len(output) == 2
        assert output[0].scene_id == "vid_scene_0000"

    def test_all_rejected_returns_empty(self) -> None:
        """All scenes rejected → empty clip list."""
        rejected = _make_rejected_scene(scene_id="vid_scene_0001")
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[rejected], context=context)
        assert isinstance(output, list)
        assert len(output) == 0

    def test_short_passed_scene_returns_empty(self) -> None:
        """Passed scene shorter than window → 0 clips."""
        scene = _make_passed_scene(start_time=0.0, end_time=5.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context()

        output = stage.run(input_data=[scene], context=context)
        assert len(output) == 0

    def test_input_not_mutated(self) -> None:
        """Original SceneInfo list and objects are not modified."""
        scene = _make_passed_scene(start_time=0.0, end_time=30.0)
        original_id = scene.scene_id
        original_start = scene.start_time

        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        stage.run(input_data=[scene], context=context)

        assert scene.scene_id == original_id
        assert scene.start_time == original_start
        # SceneInfo is frozen so mutation is impossible at the type level

    def test_input_wrong_type_raises(self) -> None:
        """InputValidationError for non-list input."""
        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        assert stage.validate_input("not_a_list", context) is False

        with pytest.raises(InputValidationError, match="Invalid input data"):
            stage.run(input_data="not_a_list", context=context)

    def test_input_empty_list_raises(self) -> None:
        """InputValidationError for empty list."""
        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        assert stage.validate_input([], context) is False

    def test_input_wrong_element_type(self) -> None:
        """validate_input returns False when list contains non-SceneInfo."""
        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        assert stage.validate_input(["not_a_scene"], context) is False

    def test_validate_output_rejects_non_list(self) -> None:
        """validate_output returns False for non-list output."""
        stage = Stage4TemporalClipGeneration()
        assert stage.validate_output("not_a_list", _make_context()) is False

    def test_validate_output_accepts_empty_list(self) -> None:
        """validate_output returns True for empty list (valid scenario)."""
        stage = Stage4TemporalClipGeneration()
        assert stage.validate_output([], _make_context()) is True

    def test_validate_output_rejects_mixed_list(self) -> None:
        """validate_output returns False when list contains non-ClipInfo."""
        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        scene = _make_passed_scene(start_time=0.0, end_time=8.0)
        clips = generate_clips(scene, window_size=8.0, stride=4.0, target_fps=25.0)
        assert stage.validate_output(clips, context) is True
        assert stage.validate_output(clips + ["not_a_clip"], context) is False

    def test_custom_window_size_and_stride(self) -> None:
        """Config parameters (window_size=6, stride=3) are honored."""
        scene = _make_passed_scene(start_time=0.0, end_time=12.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context({"window_size": 6, "stride": 3})

        output = stage.run(input_data=[scene], context=context)
        # 0-6, 3-9, 6-12 = 3 clips
        assert len(output) == 3
        assert output[0].duration == 6.0
        assert output[0].start_time == 0.0
        assert output[0].end_time == 6.0
        assert output[2].start_time == 6.0
        assert output[2].end_time == 12.0

    def test_keep_tail_via_config(self) -> None:
        """Config keep_tail=True: short scene kept whole; tail kept on long."""
        short = _make_passed_scene(
            scene_id="vid_scene_0000", start_time=0.0, end_time=5.0
        )
        long = _make_passed_scene(
            scene_id="vid_scene_0001", start_time=10.0, end_time=20.0
        )
        stage = Stage4TemporalClipGeneration()
        context = _make_context({"keep_tail": True, "stride": 8})

        output = stage.run(input_data=[short, long], context=context)

        # short: 5s → 1 whole clip; long: 10s → [10-18] + [18-20]
        assert len(output) == 3
        assert output[0].scene_id == "vid_scene_0000"
        assert output[0].duration == pytest.approx(5.0)
        assert output[1].end_time == pytest.approx(18.0)
        assert output[2].start_time == pytest.approx(18.0)
        assert output[2].end_time == pytest.approx(20.0)

    def test_keep_tail_min_tail_duration_via_config(self) -> None:
        """Config min_tail_duration=1: 8.5s scene tail (0.5s) is dropped."""
        scene = _make_passed_scene(
            scene_id="vid_scene_0000", start_time=0.0, end_time=8.5
        )
        stage = Stage4TemporalClipGeneration()
        context = _make_context(
            {"keep_tail": True, "stride": 8, "min_tail_duration": 1}
        )

        output = stage.run(input_data=[scene], context=context)
        assert len(output) == 1
        assert output[0].end_time == pytest.approx(8.0)

    def test_disabled_stage_returns_empty(self) -> None:
        """When enabled=false, return empty list regardless of input."""
        scene = _make_passed_scene(start_time=0.0, end_time=30.0)
        stage = Stage4TemporalClipGeneration()
        context = _make_context({"enabled": False})

        output = stage.run(input_data=[scene], context=context)
        assert isinstance(output, list)
        assert len(output) == 0

    def test_multiple_scenes_ordered(self) -> None:
        """Multiple passed scenes → clips ordered by scene, then by time."""
        s1 = _make_passed_scene(scene_id="vid_scene_0000", start_time=0.0, end_time=12.0)
        s2 = _make_passed_scene(scene_id="vid_scene_0001", start_time=20.0, end_time=30.0)

        stage = Stage4TemporalClipGeneration()
        context = _make_context()
        output = stage.run(input_data=[s1, s2], context=context)

        # s1: 12s → 2 clips (0-8, 4-12), s2: 10s → 1 clip (20-28)
        assert len(output) == 3
        assert output[0].scene_id == "vid_scene_0000"
        assert output[1].scene_id == "vid_scene_0000"
        assert output[2].scene_id == "vid_scene_0001"
        assert output[0].start_time == 0.0
        assert output[2].start_time == 20.0

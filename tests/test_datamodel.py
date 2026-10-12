"""Tests for data models.

Verifies all frozen dataclasses can be instantiated and are immutable.
"""

from pathlib import Path

import pytest

from face_pipeline_v2.data.models import (
    AlignedFaceInfo,
    AnnotationInfo,
    AudioFeature,
    AudioInfo,
    AVSyncInfo,
    ClipInfo,
    CorrectedAudioInfo,
    FaceInfo,
    FaceQualityInfo,
    FLAMEFeature,
    FrameInfo,
    GazeFeature,
    LandmarkInfo,
    PoseInfo,
    QualityMetrics,
    SceneInfo,
    StandardizedVideoInfo,
    StatisticsInfo,
    TrainingSample,
    VideoInfo,
)


class TestVideoInfo:
    """Tests for VideoInfo frozen dataclass."""

    def test_create_video_info(self) -> None:
        vi = VideoInfo(
            video_id="vid_001",
            video_path=Path("/data/video.mp4"),
            original_fps=30.0,
            width=1920,
            height=1080,
            duration=120.5,
            total_frames=3615,
            codec="h264",
            has_audio=True,
        )
        assert vi.video_id == "vid_001"
        assert vi.width == 1920

    def test_video_info_is_immutable(self) -> None:
        vi = VideoInfo(
            video_id="v",
            video_path=Path("v.mp4"),
            original_fps=25.0,
            width=256,
            height=256,
            duration=10.0,
            total_frames=250,
            codec="h264",
            has_audio=True,
        )
        with pytest.raises(Exception):
            vi.video_id = "new"  # type: ignore[misc]


class TestSceneInfo:
    """Tests for SceneInfo frozen dataclass."""

    def test_create_scene_info(self) -> None:
        si = SceneInfo(
            scene_id="s_001",
            video_id="vid_001",
            start_frame=0,
            end_frame=250,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
        )
        assert si.scene_id == "s_001"
        assert si.duration == 10.0


class TestClipInfo:
    """Tests for ClipInfo frozen dataclass."""

    def test_create_clip_info(self) -> None:
        ci = ClipInfo(
            clip_id="c_001",
            scene_id="s_001",
            video_id="vid_001",
            start_frame=0,
            end_frame=200,
            start_time=0.0,
            end_time=8.0,
            frame_count=200,
            duration=8.0,
        )
        assert ci.clip_id == "c_001"
        assert ci.frame_count == 200


class TestFaceInfo:
    """Tests for FaceInfo frozen dataclass."""

    def test_create_face_info(self) -> None:
        fi = FaceInfo(
            frame_index=42,
            face_id="f_001",
            bbox=(10.0, 20.0, 100.0, 120.0),
            landmark_5=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
            confidence=0.95,
            face_crop_path=Path("/output/face_42.png"),
        )
        assert fi.frame_index == 42
        assert fi.confidence == 0.95


class TestTrainingSample:
    """Tests for TrainingSample frozen dataclass."""

    def test_create_training_sample(self) -> None:
        ts = TrainingSample(
            sample_id="smp_001",
            video_clip=Path("/out/video.mp4"),
            face_sequence=Path("/out/faces/"),
            flame_feature=(0.1,) * 56,
            gaze_feature=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            audio_feature=(0.5,) * 768,
            metadata=(("source", "vid_001"),),
        )
        assert ts.sample_id == "smp_001"
        assert len(ts.gaze_feature) == 6


class TestAllModelsImport:
    """Verify all data model classes can be imported and instantiated."""

    def test_all_models_instantiable(self) -> None:
        """Smoke test: every model can be created with minimal valid data."""
        p = Path("/tmp/dummy")

        VideoInfo("v", p, 25.0, 256, 256, 1.0, 25, "h264", True)
        StandardizedVideoInfo(p, p, 25, (256, 256), "h264")
        AudioInfo(p, 16000, 1.0, 1)
        CorrectedAudioInfo(p, p, 0, (0.0,), 1.0)
        SceneInfo("s", "v", 0, 10, 0.0, 1.0, 1.0)
        ClipInfo("c", "s", "v", 0, 10, 0.0, 1.0, 10, 1.0)
        FrameInfo(0, "c", 0.0, p)
        FaceInfo(0, "f", (0, 0, 100, 100), (0,) * 10, 0.9, p)
        LandmarkInfo("c", 0, "f", 0.0, tuple(float(i) for i in range(212)), 1920, 1080)
        AlignedFaceInfo("c", 0, "f", 0.0, p, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), (256, 256), (256, 256))
        PoseInfo(0, "f", 0.0, 0.0, 0.0)
        AVSyncInfo("c", 0, 1.0, 14.0)
        FaceQualityInfo(0, "f", 0.85)
        QualityMetrics("c", "v", 0.8, 0.9, 0.85, 0.85, True)
        FLAMEFeature(0, (0.0,) * 300, (0.0,) * 100, (0.0,) * 3, (0.0,) * 3, (0.0,) * 3)
        GazeFeature(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        AudioFeature(0.0, (0.0,) * 768, 768, p)
        StatisticsInfo("v", 1000, 800, 0.8, 0.85, 0.8, 0.9)
        AnnotationInfo("v", "talking_head", "studio", "A", (("lang", "zh"),))
        TrainingSample.from_legacy(
            sample_id="smp",
            video_clip=p,
            face_sequence=p,
            flame_feature=(0.0,) * 56,
            gaze_feature=(0.0,) * 6,
            audio_feature=(0.0,) * 768,
        )


class TestFLAMEFeature:
    """Tests for FLAMEFeature frozen dataclass (Stage12)."""

    def test_create_minimal(self) -> None:
        """Minimal FLAMEFeature with only required fields."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        assert f.frame_index == 0
        assert len(f.shapecode) == 300
        assert len(f.expcode) == 100
        assert len(f.jawpose) == 3
        assert len(f.globalpose) == 3
        assert len(f.cam) == 3

    def test_eye_pose_default_zeros(self) -> None:
        """eye_pose defaults to zeros(6)."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        assert f.eye_pose == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert len(f.eye_pose) == 6

    def test_spectre_jawpose_default_none(self) -> None:
        """spectre_jawpose defaults to None."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        assert f.spectre_jawpose is None

    def test_emica_jawpose_default_none(self) -> None:
        """emica_jawpose defaults to None."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        assert f.emica_jawpose is None

    def test_full_create_with_spectre(self) -> None:
        """Full FLAMEFeature with all optional fields populated."""
        f = FLAMEFeature(
            frame_index=42,
            shapecode=(0.1,) * 300,
            expcode=(0.2,) * 100,
            jawpose=(0.3, 0.4, 0.5),
            globalpose=(0.6, 0.7, 0.8),
            cam=(1.0, 2.0, 3.0),
            eye_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            texcode=(0.9,) * 50,
            lightcode=(0.1,) * 27,
            spectre_jawpose=(-0.3, -0.4, -0.5),
            emica_jawpose=(0.3, 0.4, 0.5),
        )
        assert f.frame_index == 42
        assert f.spectre_jawpose == (-0.3, -0.4, -0.5)
        assert f.emica_jawpose == (0.3, 0.4, 0.5)
        assert len(f.texcode) == 50
        assert len(f.lightcode) == 27

    def test_posecode_property(self) -> None:
        """posecode = globalpose + jawpose = 6 dims."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(1.0, 2.0, 3.0),
            globalpose=(4.0, 5.0, 6.0),
            cam=(0.0,) * 3,
        )
        assert f.posecode == (4.0, 5.0, 6.0, 1.0, 2.0, 3.0)
        assert len(f.posecode) == 6

    def test_immutable(self) -> None:
        """FLAMEFeature is frozen."""
        f = FLAMEFeature(
            frame_index=0,
            shapecode=(0.0,) * 300,
            expcode=(0.0,) * 100,
            jawpose=(0.0,) * 3,
            globalpose=(0.0,) * 3,
            cam=(0.0,) * 3,
        )
        with pytest.raises(Exception):
            f.expcode = (0.0,) * 100  # type: ignore[misc]

"""Tests for Stage 7: A/V Sync Detection (face-crop-based).

Tests cover:
    1. estimate_sync with face crops → correct AVSyncInfo.
    2. _collect_face_crops — extracts full-frame tight bbox + frame index.
    3. No face crops → zero-value AVSyncInfo.
    4. estimate_sync accepts std_video_path and passes it to the video builder.
    5. enabled=false → returns empty list.
    6. Missing keys in input → validation fails.
    7. Stage 7 depends on std_video_path (full-frame), not Stage-5 2x crops.
    8. S3FD tight-crop helpers operate on full-frame coordinates.
"""

from pathlib import Path
from typing import Any, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    AudioInfo,
    AVSyncInfo,
    ClipInfo,
    FaceInfo,
    FaceSequence,
    FrameInfo,
)
from face_pipeline_v2.pipeline.context import PipelineContext
from face_pipeline_v2.stages.stage_07_sync.sync_estimator import (
    _build_face_crop_video,
    _collect_face_crops,
    _s3fd_gray_fill_ratio,
    _s3fd_style_crop,
    _temporal_smooth,
    FaceCrop,
    estimate_sync,
)
from face_pipeline_v2.stages.stage_07_sync.stage import (
    Stage7SyncDetection,
)
from face_pipeline_v2.utils.exceptions import InputValidationError
from face_pipeline_v2.utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(config_overrides: dict | None = None) -> PipelineContext:
    config: dict = {
        "stage_07": {
            "enabled": True, "batch_size": 32, "vshift": 10,
            "checkpoint": "models/syncnet/syncnet_v2.model",
            "device": "cpu",
        },
        "storage": {"output_dir": "/tmp/test_sync"},
        "models": {"syncnet": {"path": "models/syncnet"}},
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    return PipelineContext(
        config=config,
        logger=PipelineLogger(name="test_stage_07", log_dir=None),
        model_manager=object(),
    )


def _make_audio_info() -> AudioInfo:
    return AudioInfo(
        audio_path=Path("/tmp/test_audio.wav"),
        sample_rate=16000, duration=60.0, channels=1,
    )


def _make_clip(clip_id: str = "v_s0_c0") -> ClipInfo:
    return ClipInfo(
        clip_id=clip_id, scene_id="s0", video_id="v",
        start_frame=0, end_frame=200, start_time=0.0, end_time=8.0,
        frame_count=200, duration=8.0,
    )


def _make_face_crop_path(frame_idx: int) -> Path:
    return Path(f"/tmp/test_crops/face_{frame_idx:04d}.jpg")


def _make_face_crop(frame_idx: int) -> FaceCrop:
    """Build a FaceCrop record carrying the full-frame tight bbox."""
    return FaceCrop(
        frame_index=frame_idx,
        bbox=(100.0, 80.0, 200.0, 220.0),
    )


def _make_seq(
    clip_id: str = "v_s0_c0",
    num_frames: int = 10,
    with_faces: bool = True,
) -> FaceSequence:
    """Build a FaceSequence with optional face crops."""
    frames = []
    for i in range(num_frames):
        face = FaceInfo(
            frame_index=i, face_id="f0", bbox=(0.0, 0.0, 100.0, 100.0),
            landmark_5=(), confidence=0.9,
            face_crop_path=_make_face_crop_path(i),
            expanded_bbox=(-50.0, -50.0, 150.0, 150.0),
        )
        frames.append(FrameInfo(
            frame_index=i, clip_id=clip_id, timestamp=i / 25.0,
            image_path=Path(""),
            faces=(face,) if with_faces else (),
        ))
    return FaceSequence(clip_id=clip_id, video_id="v", frames=tuple(frames))


class MockSyncNet:
    """Mock SyncNet returning fixed results."""
    def predict(self, video_path: Path) -> Tuple[int, float, float, float]:
        return (2, 0.85, 14.5, 14.62)


# ---------------------------------------------------------------------------
# _collect_face_crops tests
# ---------------------------------------------------------------------------


class TestCollectFaceCrops:
    """Tests for _collect_face_crops()."""

    def test_collects_frame_index_and_bbox(self):
        """Extracts FaceCrop records with frame_index + full-frame tight bbox."""
        seq = _make_seq("c0", num_frames=3, with_faces=True)
        crops = _collect_face_crops(seq)
        assert len(crops) == 3
        assert all(isinstance(c, FaceCrop) for c in crops)
        assert all(isinstance(c.frame_index, int) for c in crops)
        assert all(len(c.bbox) == 4 for c in crops)

    def test_empty_when_no_faces(self):
        """Returns empty list when frames have no faces."""
        seq = _make_seq("c0", num_frames=3, with_faces=False)
        crops = _collect_face_crops(seq)
        assert crops == []

    def test_sorted_by_frame_index(self):
        """Records are ordered by frame_index ascending."""
        seq = _make_seq("c0", num_frames=5, with_faces=True)
        crops = _collect_face_crops(seq)
        assert [c.frame_index for c in crops] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# estimate_sync unit tests
# ---------------------------------------------------------------------------


class TestEstimateSync:
    """Tests for estimate_sync() with face crops."""

    def test_mock_model_produces_avsyncinfo(self) -> None:
        """estimate_sync with face crops produces correct AVSyncInfo."""
        seq = _make_seq(num_frames=10)
        clip = _make_clip()
        model = MockSyncNet()

        with patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._build_face_crop_video") as mock_build, \
             patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._extract_clip_audio") as mock_audio, \
             patch("pathlib.Path.exists", return_value=True):
            mock_build.return_value = Path("/tmp/test.mp4")
            result = estimate_sync(
                seq, clip, Path("/tmp/a.wav"), model,
                std_video_path=Path("/tmp/std.mp4"),
            )

        assert isinstance(result, AVSyncInfo)
        assert result.clip_id == "v_s0_c0"
        assert result.offset == 2
        assert result.sync_lse_c == 0.85
        assert result.sync_lse_d == 14.5
        assert result.median_distance == 14.62

    def test_no_face_crops_returns_zero(self) -> None:
        """When no face crops available, returns zero-value AVSyncInfo."""
        seq = _make_seq(num_frames=0, with_faces=False)
        # Override frames to be empty
        seq = FaceSequence(clip_id="c0", video_id="v", frames=())
        clip = _make_clip()
        model = MockSyncNet()

        result = estimate_sync(
            seq, clip, Path("/tmp/a.wav"), model,
            std_video_path=Path("/tmp/std.mp4"),
        )

        assert result.clip_id == "v_s0_c0"
        assert result.sync_lse_c == 0.0
        assert result.sync_lse_d == 0.0
        assert result.median_distance == 0.0

    def test_passes_std_video_path_to_builder(self) -> None:
        """estimate_sync forwards std_video_path to _build_face_crop_video."""
        seq = _make_seq(num_frames=10)
        clip = _make_clip()
        model = MockSyncNet()
        std_video = Path("/tmp/std.mp4")

        with patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._build_face_crop_video") as mock_build, \
             patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._extract_clip_audio") as mock_audio, \
             patch("pathlib.Path.exists", return_value=True):
            mock_build.return_value = Path("/tmp/test.mp4")
            result = estimate_sync(
                sequence=seq,
                clip=clip,
                audio_path=Path("/tmp/a.wav"),
                model=model,
                std_video_path=std_video,
            )
        assert result.clip_id == "v_s0_c0"
        assert mock_build.call_args[1]["std_video_path"] == std_video

    def test_build_video_from_face_crops(self) -> None:
        """_build_face_crop_video is called with correct face crop records."""
        seq = _make_seq(num_frames=5)
        clip = _make_clip()
        model = MockSyncNet()
        std_video = Path("/tmp/std.mp4")

        with patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._build_face_crop_video") as mock_build, \
             patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._extract_clip_audio") as mock_audio, \
             patch("pathlib.Path.exists", return_value=True), \
             patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._collect_face_crops") as mock_collect:
            mock_build.return_value = Path("/tmp/test.mp4")
            mock_collect.return_value = [_make_face_crop(i) for i in range(5)]

            estimate_sync(
                seq, clip, Path("/tmp/a.wav"), model,
                std_video_path=std_video,
            )

            # Verify _build_face_crop_video was called with 5 face crops
            mock_build.assert_called_once()
            call_args = mock_build.call_args[1]
            assert len(call_args["face_crops"]) == 5
            assert call_args["fps"] == 25.0  # 200 frames / 8.0s
            assert call_args["std_video_path"] == std_video


# ---------------------------------------------------------------------------
# S3FD tight-crop helper tests
# ---------------------------------------------------------------------------


class TestS3fdStyleCrop:
    """Tests for the colleague S3FD tight crop helpers."""

    def test_crop_is_square_side_2p8_bs(self) -> None:
        """Square side ≈ 2.8*bs (asymmetric shave keeps the crop square)."""
        img = np.full((300, 300, 3), 0, dtype=np.uint8)
        face = _s3fd_style_crop(img, cx=150.0, cy=150.0, bs=50.0, crop_scale=0.40)
        # side = 2.8 * 50 = 140
        assert face.shape == (140, 140, 3)

    def test_asymmetric_vertical_position(self) -> None:
        """Center sits 1.0bs from top and 1.8bs from bottom (forehead shaved)."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cx = cy = 150.0
        bs = 50.0
        img[int(cy), int(cx)] = 255
        face = _s3fd_style_crop(img, cx, cy, bs, crop_scale=0.40)
        # Dot lands at crop pixel (1.0*bs, 1.4*bs) = (50, 70).
        ys, xs = np.where(face[:, :, 0] == 255)[0], np.where(face[:, :, 0] == 255)[1]
        assert 49 <= int(ys[0]) <= 51
        assert 69 <= int(xs[0]) <= 71

    def test_none_is_noop(self) -> None:
        assert _s3fd_style_crop(None, 0.0, 0.0, 10.0) is None

    def test_zero_bs_is_noop(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert _s3fd_style_crop(img, 5.0, 5.0, 0.0) is img


class TestS3fdGrayFillRatio:
    """Tests for _s3fd_gray_fill_ratio() — the gray-pad impact diagnostic."""

    def test_fully_inside_is_zero(self) -> None:
        """Window entirely within the crop image → no gray fill."""
        out = _s3fd_gray_fill_ratio(300.0, 300.0, 150.0, 150.0, 50.0, crop_scale=0.40)
        assert out == 0.0

    def test_half_outside_is_half(self) -> None:
        """Left half of the window sticks out → 0.5 gray fill."""
        # cs=0 → x window width = 2*bs = 100, centered at cx=0 → [−50, 50].
        # Only [0, 50] overlaps the 100-wide image → 50% of x is pad.
        out = _s3fd_gray_fill_ratio(100.0, 300.0, 0.0, 150.0, 50.0, crop_scale=0.0)
        assert out == pytest.approx(0.5)

    def test_fully_outside_is_one(self) -> None:
        """Window fully off the crop image → 1.0 gray fill."""
        out = _s3fd_gray_fill_ratio(100.0, 100.0, 200.0, 200.0, 50.0, crop_scale=0.0)
        assert out == 1.0

    def test_asymmetric_bottom_shave_can_poke_out(self) -> None:
        """Bottom of the asymmetric shave (cy + 1.8bs) can exceed the crop
        while top/left/right stay inside — the common wide-face case."""
        # w=100, h=80 → bs = max(100,80)/2 = 50; crop is 2× → 200×160,
        # cx=100, cy=80.  Bottom = 80 + 90 = 170 > 160 (image height).
        out = _s3fd_gray_fill_ratio(200.0, 160.0, 100.0, 80.0, 50.0, crop_scale=0.40)
        assert 0.0 < out < 1.0


class TestTemporalSmooth:
    """Tests for _temporal_smooth()."""

    def test_short_track_unchanged(self) -> None:
        vals = [1.0, 3.0, 2.0]
        assert _temporal_smooth(vals, kernel_size=13) == vals

    def test_medfilt_removes_outlier(self) -> None:
        vals = [1.0] * 20
        vals[10] = 100.0  # single spike
        out = _temporal_smooth(vals, kernel_size=13)
        assert out[10] == 1.0  # spike removed by k=13 median
        assert len(out) == 20


# ---------------------------------------------------------------------------
# _build_face_crop_video tests
# ---------------------------------------------------------------------------


class _FakeVideoReader:
    """Fake FFmpegVideoReader returning a fixed 400x400 BGR frame."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def start(self) -> None:
        pass

    def read_frame(self, frame_index: int):
        return np.zeros((400, 400, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class _FakeEncoder:
    """Fake ffmpeg image2pipe subprocess.Popen."""

    returncode = 0

    class _Stdin:
        def __init__(self) -> None:
            self.writes = 0

        def write(self, data: bytes) -> int:
            self.writes += 1
            return len(data)

        def close(self) -> None:
            pass

    def __init__(self) -> None:
        self.stdin = self._Stdin()
        self.stderr = None

    def wait(self, timeout=None) -> None:
        pass


class TestBuildFaceCropVideo:
    """Tests for _build_face_crop_video()."""

    def test_returns_output_path(self) -> None:
        """Returns the output path and writes one raw frame per face crop."""
        crops = [_make_face_crop(i) for i in range(10)]
        output = Path("/tmp/test_output.mp4")
        enc = _FakeEncoder()

        with patch(
            "face_pipeline_v2.stages.stage_07_sync.sync_estimator.FFmpegVideoReader",
            return_value=_FakeVideoReader(Path("/tmp/std.mp4")),
        ), patch(
            "face_pipeline_v2.stages.stage_07_sync.sync_estimator.subprocess.Popen",
            return_value=enc,
        ), patch("pathlib.Path.exists", return_value=True):
            result = _build_face_crop_video(
                face_crops=crops,
                audio_path=Path("/tmp/a.wav"),
                output_path=output,
                fps=25.0,
                clip_duration=8.0,
                std_video_path=Path("/tmp/std.mp4"),
            )
        assert result == output
        assert enc.stdin.writes == 10

    def test_passes_correct_fps_to_ffmpeg(self) -> None:
        """ffmpeg command includes the correct framerate."""
        crops = [_make_face_crop(i) for i in range(5)]
        output = Path("/tmp/test.mp4")

        with patch(
            "face_pipeline_v2.stages.stage_07_sync.sync_estimator.FFmpegVideoReader",
            return_value=_FakeVideoReader(Path("/tmp/std.mp4")),
        ), patch(
            "face_pipeline_v2.stages.stage_07_sync.sync_estimator.subprocess.Popen",
            return_value=_FakeEncoder(),
        ) as mock_popen, patch("pathlib.Path.exists", return_value=True):
            _build_face_crop_video(
                face_crops=crops,
                audio_path=Path("/tmp/a.wav"),
                output_path=output,
                fps=30.0,
                clip_duration=8.0,
                std_video_path=Path("/tmp/std.mp4"),
            )

        argv = mock_popen.call_args[0][0]
        fr_idx = argv.index("-framerate")
        assert argv[fr_idx + 1] == "30.0"

    def test_reader_uses_std_video_path(self) -> None:
        """FFmpegVideoReader is constructed from std_video_path."""
        crops = [_make_face_crop(i) for i in range(2)]

        with patch(
            "face_pipeline_v2.stages.stage_07_sync.sync_estimator.FFmpegVideoReader",
        ) as mock_reader_cls, patch(
            "face_pipeline_v2.stages.stage_07_sync.sync_estimator.subprocess.Popen",
            return_value=_FakeEncoder(),
        ), patch("pathlib.Path.exists", return_value=True):
            mock_reader_cls.return_value = _FakeVideoReader(Path("/tmp/std.mp4"))
            _build_face_crop_video(
                face_crops=crops,
                audio_path=Path("/tmp/a.wav"),
                output_path=Path("/tmp/o.mp4"),
                fps=25.0,
                clip_duration=1.0,
                std_video_path=Path("/tmp/std.mp4"),
            )

        mock_reader_cls.assert_called_once_with(Path("/tmp/std.mp4"))


# ---------------------------------------------------------------------------
# Stage7SyncDetection integration tests
# ---------------------------------------------------------------------------


class TestStage7SyncDetection:
    """Integration tests for Stage7SyncDetection (face-crop-based)."""

    def test_normal_processing(self) -> None:
        seq = _make_seq()
        clip = _make_clip()
        stage = Stage7SyncDetection()
        context = _make_context()

        with patch.object(stage, "_get_model", return_value=MockSyncNet()), \
             patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._build_face_crop_video") as mock_build, \
             patch("face_pipeline_v2.stages.stage_07_sync.sync_estimator._extract_clip_audio") as mock_audio, \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch("tempfile.mkdtemp", return_value="/tmp/syncnet_test"):
            mock_build.return_value = Path("/tmp/test.mp4")
            output = stage.run(
                input_data={
                    "sequences": [seq],
                    "audio_info": _make_audio_info(),
                    "clip_infos": [clip],
                    "std_video_path": Path("/tmp/std.mp4"),
                },
                context=context,
            )

        assert len(output) == 1
        assert output[0].clip_id == "v_s0_c0"

    def test_requires_std_video_path(self) -> None:
        """Stage 7 requires std_video_path (no _resolve_video_path leftover)."""
        stage = Stage7SyncDetection()
        context = _make_context()
        assert not hasattr(stage, "_resolve_video_path")

        data = {
            "sequences": [_make_seq()],
            "audio_info": _make_audio_info(),
            "clip_infos": [_make_clip()],
        }
        assert stage.validate_input(data, context) is False  # missing std_video_path
        data["std_video_path"] = Path("/tmp/std.mp4")
        assert stage.validate_input(data, context) is True

    def test_disabled_returns_empty(self) -> None:
        stage = Stage7SyncDetection()
        context = _make_context({"stage_07": {"enabled": False}})

        output = stage.run(
            input_data={
                "sequences": [_make_seq()],
                "audio_info": _make_audio_info(),
                "clip_infos": [_make_clip()],
                "std_video_path": Path("/tmp/std.mp4"),
            },
            context=context,
        )
        assert output == []

    def test_missing_keys_fails_validation(self) -> None:
        stage = Stage7SyncDetection()
        context = _make_context()
        base = {
            "sequences": [_make_seq()],
            "audio_info": _make_audio_info(),
            "clip_infos": [_make_clip()],
            "std_video_path": Path("/tmp/std.mp4"),
        }
        assert stage.validate_input({"sequences": []}, context) is False
        for missing in ("sequences", "audio_info", "clip_infos", "std_video_path"):
            data = dict(base)
            del data[missing]
            assert stage.validate_input(data, context) is False, missing

    def test_input_wrong_type_raises(self) -> None:
        stage = Stage7SyncDetection()
        assert stage.validate_input("wrong", _make_context()) is False

    def test_validate_output(self) -> None:
        stage = Stage7SyncDetection()
        av = AVSyncInfo(clip_id="c0", offset=0, sync_lse_c=1.0, sync_lse_d=1.0)
        assert stage.validate_output([av], _make_context()) is True
        assert stage.validate_output("bad", _make_context()) is False

    def test_empty_sequences_list_fails_validation(self) -> None:
        stage = Stage7SyncDetection()
        context = _make_context()
        assert stage.validate_input(
            {"sequences": [], "audio_info": _make_audio_info(),
             "clip_infos": [_make_clip()],
             "std_video_path": Path("/tmp/std.mp4")},
            context,
        ) is False

    def test_wrong_type_in_sequences_fails_validation(self) -> None:
        stage = Stage7SyncDetection()
        context = _make_context()
        assert stage.validate_input(
            {"sequences": ["not_a_sequence"], "audio_info": _make_audio_info(),
             "clip_infos": [_make_clip()],
             "std_video_path": Path("/tmp/std.mp4")},
            context,
        ) is False

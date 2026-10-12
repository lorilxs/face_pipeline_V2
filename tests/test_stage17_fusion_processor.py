"""Stage17 Fusion Core tests — processor (visual fusion / alignment) + integration (persistence).

覆盖 (对应 docs/Stage17_Design_Review.md §5 时间同步策略):
  - visual_feature [T,112] = expression[100] + pose[6] + gaze[6] 布局
  - FLAME/Gaze 精确 frame_index join + Gaze 缺失 forward-fill + dropout_mask
  - Audio nearest-timestamp 匹配 (25 FPS vs 50Hz) / min-delta 选择 / 容差超时零填充
  - valid_length 语义 (三模态齐全且 |Δt| 合格)
  - save/read_training_sample round-trip 精确一致 + manifest 生成
"""

from pathlib import Path

import numpy as np
import pytest

from face_pipeline_v2.data.models import (
    AudioFeature,
    FLAMEFeature,
    GazeFeature,
    TrainingSample,
)
from face_pipeline_v2.stages.stage_17_fusion.integration import (
    read_training_sample,
    save_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.processor import (
    AUDIO_BACKEND,
    VISUAL_DIM,
    VISUAL_LAYOUT,
    build_training_sample,
)

CLIP = "clip_001"
FPS = 25.0
AUDIO_DIM = 768


# ---------------------------------------------------------------------------
# 构造 helper — 使用确定性的解析公式 (无随机), 便于逐块断言
# ---------------------------------------------------------------------------
def _flame(frame_index: int, timestamp: float, clip: str = CLIP) -> FLAMEFeature:
    return FLAMEFeature(
        frame_index=frame_index,
        shapecode=tuple(0.0 for _ in range(300)),
        expcode=tuple(0.01 * (frame_index + 1) + 0.001 * k for k in range(100)),
        jawpose=(0.11, 0.12, 0.13),
        globalpose=(1.1, 1.2, 1.3),
        cam=(1.0, 0.0, 0.0),
        timestamp=timestamp,
        clip_id=clip,
    )


def _gaze(frame_index: int, timestamp: float, clip: str = CLIP) -> GazeFeature:
    return GazeFeature(
        frame_index=frame_index,
        timestamp=timestamp,
        clip_id=clip,
        left_eye=(0.1, 0.2, 0.3),
        right_eye=(0.4, 0.5, 0.6),
        gaze_vector=tuple(0.5 * (frame_index + 1) + 0.01 * k for k in range(6)),
    )


def _audio(timestamp: float, idx: int, clip: str = CLIP) -> AudioFeature:
    return AudioFeature(
        timestamp=timestamp,
        feature=tuple(1.5 * (idx + 1) + 0.01 * k for k in range(AUDIO_DIM)),
        dimension=AUDIO_DIM,
        source_audio=Path(f"/audio/{clip}/{idx}.wav"),
    )


def _flames(n: int, start: int = 0) -> list:
    return [_flame(start + i, (start + i) / FPS) for i in range(n)]


def _gazes(frame_indices) -> list:
    return [_gaze(fi, fi / FPS) for fi in frame_indices]


# ---------------------------------------------------------------------------
# 视觉融合: 布局 / 维度
# ---------------------------------------------------------------------------
class TestVisualFusion:
    """visual_feature [T,112] 布局与维度。"""

    def test_visual_feature_shape_and_dtype(self) -> None:
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=_flames(4),
            gaze_features=_gazes([0, 1, 2, 3]),
            audio_features=[],
            fps=FPS,
        )
        assert sample.visual_feature.shape == (4, VISUAL_DIM)
        assert sample.visual_feature.dtype == np.float32
        # 无 audio → 零向量 [T,768], 且全部标记缺失
        assert sample.audio_feature.shape == (4, AUDIO_DIM)
        assert sample.audio_feature.dtype == np.float32
        assert np.all(sample.audio_feature == 0.0)
        assert sample.dropout_mask.tolist() == [True] * 4
        assert sample.valid_length == 0

    def test_visual_layout_slices(self) -> None:
        assert VISUAL_LAYOUT == {
            "expression": (0, 100),
            "pose": (100, 106),
            "gaze": (106, 112),
        }
        flames = _flames(3)
        gazes = _gazes([0, 1, 2])
        audios = [_audio(fi / FPS, fi) for fi in range(3)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=gazes,
            audio_features=audios,
            fps=FPS,
        )
        exp_block = np.stack(
            [np.asarray(f.expcode, dtype=np.float32) for f in flames]
        )
        pose_block = np.stack(
            [
                np.asarray(f.posecode, dtype=np.float32)
                for f in flames  # posecode = globalpose[3] + jawpose[3]
            ]
        )
        gaze_block = np.stack(
            [np.asarray(g.gaze_vector, dtype=np.float32) for g in gazes]
        )
        np.testing.assert_array_equal(
            sample.visual_feature[:, 0:100], exp_block
        )
        np.testing.assert_array_equal(
            sample.visual_feature[:, 100:106], pose_block
        )
        np.testing.assert_array_equal(
            sample.visual_feature[:, 106:112], gaze_block
        )
        assert sample.visual_feature.shape[1] == 112


# ---------------------------------------------------------------------------
# Gaze 精确 join + forward-fill
# ---------------------------------------------------------------------------
class TestGazeForwardFill:
    """Gaze 缺失帧 forward-fill + dropout_mask 标记。"""

    def test_forward_fill_missing_frame(self) -> None:
        # flame 0..4; gaze 缺 frame2; audio 与每个视频帧 timestamp 精确匹配
        flames = _flames(5)
        gazes = _gazes([0, 1, 3, 4])
        audios = [_audio(fi / FPS, fi) for fi in range(5)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=gazes,
            audio_features=audios,
            fps=FPS,
        )
        # frame2 gaze 缺失 → forward-fill 自 frame1, 并置 dropout=True
        np.testing.assert_array_equal(
            sample.visual_feature[2, 106:112],
            sample.visual_feature[1, 106:112],
        )
        assert sample.dropout_mask.tolist() == [False, False, True, False, False]
        assert sample.valid_length == 4
        # 有历史时的填充值 = 上一有效 Gaze
        np.testing.assert_array_equal(
            sample.visual_feature[2, 106:112],
            np.asarray(gazes[1].gaze_vector, dtype=np.float32),
        )

    def test_no_history_keeps_zero_and_marks_dropout(self) -> None:
        # 首帧即缺失 Gaze (无历史可填) → 保持 zeros + dropout=True
        flames = _flames(3)
        gazes = _gazes([1, 2])  # frame0 缺失
        audios = [_audio(fi / FPS, fi) for fi in range(3)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=gazes,
            audio_features=audios,
            fps=FPS,
        )
        assert sample.dropout_mask.tolist() == [True, False, False]
        np.testing.assert_array_equal(
            sample.visual_feature[0, 106:112], np.zeros(6, dtype=np.float32)
        )
        assert sample.valid_length == 2


# ---------------------------------------------------------------------------
# Audio nearest-timestamp 匹配
# ---------------------------------------------------------------------------
class TestAudioMatching:
    """Audio 按 timestamp nearest 匹配 (不做插值/平均)。"""

    def test_nearest_timestamp_25fps_vs_50hz(self) -> None:
        # 25 FPS 视频帧 (ts = fi/25) vs 50Hz audio (ts = i/50)
        # frame fi 的最近 audio = i=2*fi (|Δt|=0); 验证原始索引映射 + 特征一致
        flames = _flames(10)
        audio_by_orig = {i: _audio(i / 50.0, i) for i in range(20)}
        order = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18]
        audio_shuffled = [audio_by_orig[i] for i in order]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=_gazes(list(range(10))),
            audio_features=audio_shuffled,
            fps=FPS,
        )
        assert sample.audio_match_idx.tolist() == [
            order.index(2 * fi) for fi in range(10)
        ]
        for fi in range(10):
            target = 2 * fi
            np.testing.assert_array_equal(
                sample.audio_feature[fi],
                np.asarray(audio_by_orig[target].feature, dtype=np.float32),
            )
        assert sample.dropout_mask.tolist() == [False] * 10
        assert sample.valid_length == 10

    def test_nearest_picks_min_delta_respects_orig_index(self) -> None:
        # frame ts=0.10; audio ts=[0.16, 0.04, 0.11] → 最近为 0.11 (原索引 2, Δ=0.01)
        flames = [_flame(0, 0.10)]
        audios = [_audio(0.16, 0), _audio(0.04, 1), _audio(0.11, 2)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=_gazes([0]),
            audio_features=audios,
            fps=FPS,
        )
        assert sample.audio_match_idx.tolist() == [2]
        np.testing.assert_array_equal(
            sample.audio_feature[0],
            np.asarray(audios[2].feature, dtype=np.float32),
        )
        assert sample.dropout_mask.tolist() == [False]

    def test_timeout_zero_fill(self) -> None:
        # audio 全部远离视频帧 (Δ=10 >> tol) → 零向量 + -1 + dropout
        flames = [_flame(0, 0.0), _flame(1, 0.2)]
        audios = [_audio(10.0, 0)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=_gazes([0, 1]),
            audio_features=audios,
            fps=FPS,
        )
        assert sample.audio_match_idx.tolist() == [-1, -1]
        assert np.all(sample.audio_feature == 0.0)
        assert sample.dropout_mask.tolist() == [True, True]
        assert sample.valid_length == 0

    def test_boundary_tolerance_inclusive(self) -> None:
        # Δ 恰好 == tolerance (0.04) → 视为命中 (闭区间), 不置缺失
        flames = [_flame(0, 0.0), _flame(1, 0.12)]
        audios = [_audio(0.08, 0)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=_gazes([0, 1]),
            audio_features=audios,
            fps=FPS,
            audio_tolerance=0.04,
        )
        assert sample.audio_match_idx.tolist() == [-1, 0]  # frame1 Δ=0.04 → 命中
        assert sample.dropout_mask.tolist() == [True, False]
        assert sample.valid_length == 1


# ---------------------------------------------------------------------------
# valid_length 语义
# ---------------------------------------------------------------------------
class TestValidLength:
    """valid_length = 三模态齐全且 |Δt| 合格的帧数 = count(~dropout_mask)。"""

    def test_mixed_gaze_and_audio_dropout(self) -> None:
        flames = [_flame(fi, 0.04 * fi) for fi in range(4)]
        gazes = _gazes([0, 2, 3])  # 缺 frame1
        audios = [_audio(0.0, 0), _audio(0.04, 1), _audio(0.08, 2)]  # 缺 0.12
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=gazes,
            audio_features=audios,
            fps=FPS,
            audio_tolerance=0.02,  # frame3 最近 audio Δ=0.04 > tol → 缺失
        )
        assert sample.audio_match_idx.tolist() == [0, 1, 2, -1]
        assert sample.dropout_mask.tolist() == [False, True, False, True]
        assert sample.valid_length == 2
        assert sample.valid_length == int((~sample.dropout_mask).sum())
        # frame1 的 gaze 被 forward-fill, 但 audio 命中 → 仍属补齐帧
        np.testing.assert_array_equal(
            sample.visual_feature[1, 106:112],
            np.asarray(gazes[0].gaze_vector, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# 鲁棒性 / 参数校验 / metadata 契约
# ---------------------------------------------------------------------------
class TestRobustness:
    """输入乱序不敏感 / 空输入 / 非法参数 / metadata 契约。"""

    def test_input_order_insensitive(self) -> None:
        # FLAME/Gaze/Audio 均乱序传入 → 主时间轴按 frame_index 升序,
        # audio_match_idx 仍指向原始输入索引
        flames_rev = [_flame(fi, fi / FPS) for fi in [3, 2, 1, 0]]
        gazes_rev = [_gaze(fi, fi / FPS) for fi in [3, 2, 1, 0]]
        audios_rev = [_audio(fi / FPS, fi) for fi in [3, 2, 1, 0]]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames_rev,
            gaze_features=gazes_rev,
            audio_features=audios_rev,
            fps=FPS,
        )
        assert sample.frame_index.tolist() == [0, 1, 2, 3]
        # audio 原始输入索引 (乱序列表中的位置): ts=0.0 在位置 3, 0.04→2, 0.08→1, 0.12→0
        assert sample.audio_match_idx.tolist() == [3, 2, 1, 0]
        np.testing.assert_array_equal(
            sample.visual_feature[0, 106:112],
            np.asarray(gazes_rev[3].gaze_vector, dtype=np.float32),
        )

    def test_empty_flame_inputs(self) -> None:
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=[],
            gaze_features=_gazes([0]),
            audio_features=[_audio(0.0, 0)],
            fps=FPS,
        )
        assert sample.clip_id == CLIP
        assert sample.visual_feature.shape == (0, VISUAL_DIM)
        assert sample.audio_feature.shape == (0, AUDIO_DIM)
        assert sample.valid_length == 0
        assert sample.metadata["num_frames"] == 0

    def test_invalid_audio_match_raises(self) -> None:
        with pytest.raises(ValueError, match="nearest_timestamp"):
            build_training_sample(
                clip_id=CLIP,
                flame_features=_flames(2),
                gaze_features=_gazes([0, 1]),
                audio_features=[],
                fps=FPS,
                audio_match="linear",
            )

    def test_audio_dim_mismatch_raises(self) -> None:
        bad_audio = AudioFeature(
            timestamp=0.0,
            feature=tuple(0.0 for _ in range(512)),
            dimension=512,
            source_audio=Path("/audio/bad.wav"),
        )
        with pytest.raises(ValueError, match="audio feature dim"):
            build_training_sample(
                clip_id=CLIP,
                flame_features=_flames(1),
                gaze_features=_gazes([0]),
                audio_features=[bad_audio],
                fps=FPS,
            )

    def test_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="audio_tolerance"):
            build_training_sample(
                clip_id=CLIP,
                flame_features=_flames(1),
                gaze_features=_gazes([0]),
                audio_features=[],
                fps=FPS,
                audio_tolerance=-0.1,
            )

    def test_metadata_contract(self) -> None:
        flames = _flames(3)
        audios = [_audio(fi / FPS, fi) for fi in range(3)]
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=_gazes([0, 1, 2]),
            audio_features=audios,
            fps=FPS,
            metadata={"video_id": "vid_001", "scene_id": "sc_01", "clip_start": 12.5},
        )
        meta = sample.metadata
        assert meta["clip_id"] == CLIP
        assert meta["visual_dim"] == VISUAL_DIM
        assert meta["audio_dim"] == AUDIO_DIM
        assert meta["num_frames"] == 3
        assert meta["valid_length"] == 3
        assert meta["visual_layout"] == {
            k: list(v) for k, v in VISUAL_LAYOUT.items()
        }
        assert meta["audio_backend"] == AUDIO_BACKEND
        assert meta["source_stage"] == {
            "flame": "Stage12",
            "gaze": "Stage13",
            "audio": "Stage14",
        }
        assert meta["fps"] == FPS
        # 用户额外元数据原样保留
        assert meta["video_id"] == "vid_001"
        assert meta["scene_id"] == "sc_01"
        assert meta["clip_start"] == 12.5


# ---------------------------------------------------------------------------
# 序列化 round-trip
# ---------------------------------------------------------------------------
class TestSerialization:
    """save/read_training_sample round-trip 精确一致 + manifest 生成。"""

    def test_roundtrip_exact(self, tmp_path: Path) -> None:
        flames = [_flame(fi, 0.04 * fi) for fi in range(4)]
        gazes = _gazes([0, 2, 3])  # 缺 frame1 → forward-fill
        audios = [_audio(0.0, 0), _audio(0.04, 1), _audio(0.08, 2)]  # 缺 0.12 → 超时
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=flames,
            gaze_features=gazes,
            audio_features=audios,
            fps=FPS,
            audio_tolerance=0.02,
            metadata={"video_id": "vid_001", "scene_id": "sc_01", "clip_start": 12.5},
        )
        out_dir = save_training_sample(sample, output_dir=tmp_path)

        # 文件存在
        assert (out_dir / "training_sample.npz").exists()
        assert (out_dir / "metadata.json").exists()
        assert (tmp_path / "manifest.csv").exists()

        loaded = read_training_sample(out_dir)
        assert loaded.clip_id == CLIP
        for name in (
            "visual_feature",
            "audio_feature",
            "frame_index",
            "timestamps",
            "audio_match_idx",
            "dropout_mask",
        ):
            np.testing.assert_array_equal(
                getattr(loaded, name), getattr(sample, name)
            )
        assert loaded.valid_length == sample.valid_length
        assert loaded.metadata == sample.metadata

        # manifest: 表头 + 一行 (sample_id, clip_id, ...)
        lines = (
            (tmp_path / "manifest.csv").read_text(encoding="utf-8").strip().splitlines()
        )
        assert lines[0].startswith("sample_id,clip_id")
        assert len(lines) == 2
        assert lines[1].split(",")[1] == CLIP

    def test_roundtrip_clip_id_from_metadata(self, tmp_path: Path) -> None:
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=_flames(2),
            gaze_features=_gazes([0, 1]),
            audio_features=[_audio(0.0, 0), _audio(0.04, 1)],
            fps=FPS,
        )
        out_dir = save_training_sample(sample, output_dir=tmp_path)
        loaded = read_training_sample(out_dir, clip_id=None)
        assert loaded.clip_id == CLIP

    def test_no_manifest_when_disabled(self, tmp_path: Path) -> None:
        sample = build_training_sample(
            clip_id=CLIP,
            flame_features=_flames(2),
            gaze_features=_gazes([0, 1]),
            audio_features=[_audio(0.0, 0), _audio(0.04, 1)],
            fps=FPS,
        )
        out_dir = save_training_sample(
            sample, output_dir=tmp_path, write_manifest=False
        )
        assert (out_dir / "training_sample.npz").exists()
        assert not (tmp_path / "manifest.csv").exists()

    def test_read_missing_sample_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_training_sample(tmp_path / "nope")

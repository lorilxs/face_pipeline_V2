"""Stage 17 核心处理器 — 三模态特征融合装配。

职责 (对应 docs/Stage17_Design_Review.md §5 时间同步策略):
  1. 以 FLAME 的 frame_index 为锚点, 与 Gaze 做精确 hash join;
  2. Audio 以 timestamp 做 nearest 匹配 (容差 audio_tolerance);
  3. Gaze 缺失帧 forward-fill + dropout_mask / valid_length;
  4. 拼接 visual_feature [T,112] = expcode[100] + posecode[6] + gaze_vector[6]。

Stage17 为纯数据处理模块 (无模型推理), 可在 Host Python 直接运行。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from face_pipeline_v2.data.models import (
    AudioFeature,
    FLAMEFeature,
    GazeFeature,
    TrainingSample,
)

VISUAL_LAYOUT: Dict[str, Tuple[int, int]] = {
    "expression": (0, 100),
    "pose": (100, 106),
    "gaze": (106, 112),
}
"""visual_feature [T,112] 子块边界索引 — 供下游拆分 expression/pose/gaze 使用。"""

VISUAL_DIM: int = 112
"""visual_feature 维度 — expression(100) + pose(6) + gaze(6)。"""

AUDIO_DIM: int = 768
"""audio_feature 维度 — Chinese HuBERT hidden states。"""

AUDIO_BACKEND: str = "Chinese HuBERT"
"""audio 后端模型名 — 写入 metadata.audio_backend。"""

SOURCE_STAGE: Dict[str, str] = {
    "flame": "Stage12",
    "gaze": "Stage13",
    "audio": "Stage14",
}
"""来源阶段映射 — 写入 metadata.source_stage。"""


def _build_gaze_index(
    gaze_features: List[GazeFeature],
) -> Dict[int, GazeFeature]:
    """frame_index → GazeFeature 精确 hash join 索引 (同帧重复保留首个)。"""
    index: Dict[int, GazeFeature] = {}
    for g in gaze_features:
        index.setdefault(int(g.frame_index), g)
    return index


def _match_audio(
    video_ts: np.ndarray,
    audio_features: List[AudioFeature],
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """对每个视频帧 timestamp 做 audio nearest-timestamp 匹配。

    不做插值 / 不做特征平均 (nearest 即可, 插值会引入虚假中间特征)。

    Returns:
        audio_out:       [T, AUDIO_DIM] float32 — 命中特征或零向量 (超容差)。
        audio_match_idx: [T] int32 — 命中时取原始 audio 输入列表索引; 未命中为 -1。
    """
    T = int(video_ts.shape[0])
    if not audio_features:
        return (
            np.zeros((T, AUDIO_DIM), dtype=np.float32),
            np.full(T, -1, dtype=np.int32),
        )

    # 按 timestamp 排序做最近邻; 保留原始输入索引以便溯源
    ordered = sorted(enumerate(audio_features), key=lambda t: t[1].timestamp)
    orig_idx = [o for o, _ in ordered]
    audio_ts = np.asarray([a.timestamp for _, a in ordered], dtype=np.float64)
    audio_feats = np.stack(
        [np.asarray(a.feature, dtype=np.float32).reshape(-1) for _, a in ordered]
    )
    if audio_feats.shape[1] != AUDIO_DIM:
        raise ValueError(
            f"audio feature dim must be {AUDIO_DIM}, got {audio_feats.shape[1]}"
        )

    audio_out = np.zeros((T, AUDIO_DIM), dtype=np.float32)
    match_idx = np.full(T, -1, dtype=np.int32)
    ts = np.asarray(video_ts, dtype=np.float64)
    for i in range(T):
        delta = np.abs(audio_ts - ts[i])
        j = int(np.argmin(delta))
        if delta[j] <= tolerance:
            audio_out[i] = audio_feats[j]
            match_idx[i] = orig_idx[j]

    return audio_out, match_idx


def _build_metadata(
    *,
    clip_id: str,
    fps: float,
    audio_tolerance: float,
    audio_match: str,
    num_frames: int,
    valid_length: int,
    n_flame: int,
    n_gaze: int,
    n_audio: int,
    user_metadata: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """装配 metadata dict — 溯源字段 + 布局/来源信息。

    必含 (metadata.json 契约): clip_id / visual_dim / audio_dim / num_frames /
    valid_length / visual_layout / audio_backend / source_stage。
    用户传入的额外元数据 (video_id/scene_id/clip_start 等) 原样并入。
    """
    meta: Dict[str, object] = dict(user_metadata or {})
    meta.setdefault("clip_id", clip_id)
    meta.setdefault("sample_id", clip_id)
    meta.setdefault("visual_dim", VISUAL_DIM)
    meta.setdefault("audio_dim", AUDIO_DIM)
    meta.setdefault("num_frames", num_frames)
    meta.setdefault("valid_length", valid_length)
    meta.setdefault("fps", fps)
    meta.setdefault("audio_backend", AUDIO_BACKEND)
    meta.setdefault("audio_match", audio_match)
    meta.setdefault("audio_tolerance", audio_tolerance)
    meta.setdefault("visual_layout", {k: list(v) for k, v in VISUAL_LAYOUT.items()})
    meta.setdefault("source_stage", dict(SOURCE_STAGE))
    meta.setdefault("n_input", {"flame": n_flame, "gaze": n_gaze, "audio": n_audio})
    return meta


def build_training_sample(
    clip_id: str,
    flame_features: List[FLAMEFeature],
    gaze_features: List[GazeFeature],
    audio_features: List[AudioFeature],
    fps: float = 25.0,
    visual_dim: int = VISUAL_DIM,
    audio_dim: int = AUDIO_DIM,
    audio_tolerance: float = 0.04,
    audio_match: str = "nearest_timestamp",
    metadata: Optional[Dict[str, object]] = None,
) -> TrainingSample:
    """将三模态特征装配为一个 clip-level TrainingSample (Stage17 Fusion Core)。

    算法 (见 docs/Stage17_Design_Review.md §5):
      1. 主时间轴 = FLAME, 按 frame_index 升序 (锚点帧恒存在);
      2. Gaze 与 FLAME 按 frame_index 精确 hash join; 缺失帧 forward-fill
         上一有效 Gaze (无历史则保持 zeros), 并置 dropout_mask=True;
      3. Audio 对每个视频帧 timestamp 做 nearest 匹配 (argmin|Δt|);
         |Δt| ≤ audio_tolerance → 命中特征; 否则零向量 + dropout_mask=True +
         audio_match_idx=-1;
      4. valid_length = 三模态齐全且 |Δt| 合格的帧数 = count(~dropout_mask);
      5. visual_feature = concat(expression[100], pose[6], gaze[6]) → [T,112]。

    不做插值 / 不做特征平均 / 不做 eye_pose (eye_pose 与 gaze 语义不同, 见
    docs/Stage17_Design_Review.md §4.1)。

    Args:
        clip_id: 目标 clip 唯一标识 (sample_id := clip_id)。
        flame_features: Stage12 FLAME 特征列表 (含 expcode[100]/posecode[6]/timestamp)。
        gaze_features: Stage13 Gaze 特征列表 (含 gaze_vector[6]/frame_index)。
        audio_features: Stage14 Audio 特征列表 (含 feature[768]/timestamp)。
        fps: 视频帧率, 仅写入 metadata (时间网格已由 FLAME.timestamp 承载)。
        visual_dim: visual_feature 维度, 固定 112 (与模型契约一致)。
        audio_dim: audio_feature 维度, 固定 768 (与模型契约一致)。
        audio_tolerance: audio nearest 匹配容差 (秒, 默认 0.04)。
        audio_match: 音频匹配策略, 仅支持 "nearest_timestamp"。
        metadata: 额外溯源元数据 (video_id/scene_id/clip_start 等)。

    Returns:
        装配完成的 TrainingSample ([T,112] / [T,768] / 对齐数组 / metadata)。

    Raises:
        ValueError: 参数不合法 / 特征维度与固定布局不一致 / audio_match 不支持。
    """
    if audio_match != "nearest_timestamp":
        raise ValueError(
            f"audio_match must be 'nearest_timestamp', got {audio_match!r}"
        )
    if visual_dim != VISUAL_DIM:
        raise ValueError(
            f"visual_dim must be {VISUAL_DIM} (fixed layout), got {visual_dim}"
        )
    if audio_dim != AUDIO_DIM:
        raise ValueError(
            f"audio_dim must be {AUDIO_DIM} (Chinese HuBERT), got {audio_dim}"
        )
    if audio_tolerance < 0.0:
        raise ValueError(f"audio_tolerance must be >= 0, got {audio_tolerance}")

    n_flame = len(flame_features)
    n_gaze = len(gaze_features)
    n_audio = len(audio_features)

    # 空输入防护: 返回空样本 (T=0), 仅保留 clip_id 与溯源元数据
    if n_flame == 0:
        meta = _build_metadata(
            clip_id=clip_id,
            fps=fps,
            audio_tolerance=audio_tolerance,
            audio_match=audio_match,
            num_frames=0,
            valid_length=0,
            n_flame=n_flame,
            n_gaze=n_gaze,
            n_audio=n_audio,
            user_metadata=metadata,
        )
        return TrainingSample(
            clip_id=clip_id,
            visual_feature=np.zeros((0, VISUAL_DIM), dtype=np.float32),
            audio_feature=np.zeros((0, AUDIO_DIM), dtype=np.float32),
            metadata=meta,
        )

    # 1. 主时间轴 = FLAME, 按 frame_index 升序
    flame_sorted = sorted(flame_features, key=lambda f: f.frame_index)
    T = len(flame_sorted)
    frame_index = np.asarray([f.frame_index for f in flame_sorted], dtype=np.int32)
    timestamps = np.asarray([f.timestamp for f in flame_sorted], dtype=np.float32)

    # 2. FLAME → expression[100] / pose[6] (锚点帧恒存在)
    expression = np.zeros((T, 100), dtype=np.float32)
    pose = np.zeros((T, 6), dtype=np.float32)
    for i, f in enumerate(flame_sorted):
        exp = np.asarray(f.expcode, dtype=np.float32).reshape(-1)
        pos = np.asarray(f.posecode, dtype=np.float32).reshape(-1)
        if exp.shape[0] != 100:
            raise ValueError(
                f"FLAMEFeature.expcode must be 100 dims, got {exp.shape[0]} "
                f"at frame {f.frame_index}"
            )
        if pos.shape[0] != 6:
            raise ValueError(
                f"FLAMEFeature.posecode must be 6 dims "
                f"(globalpose[3] + jawpose[3]), got {pos.shape[0]} "
                f"at frame {f.frame_index}"
            )
        expression[i] = exp
        pose[i] = pos

    # 3. Gaze 精确 frame_index join + forward-fill
    gaze_index = _build_gaze_index(gaze_features)
    gaze = np.zeros((T, 6), dtype=np.float32)
    dropout = np.zeros(T, dtype=bool)
    last_gaze: Optional[np.ndarray] = None
    for i, f in enumerate(flame_sorted):
        g = gaze_index.get(int(f.frame_index))
        if g is not None:
            gv = np.asarray(g.gaze_vector, dtype=np.float32).reshape(-1)
            if gv.shape[0] != 6:
                raise ValueError(
                    f"GazeFeature.gaze_vector must be 6 dims, got {gv.shape[0]} "
                    f"at frame {f.frame_index}"
                )
            gaze[i] = gv
            last_gaze = gv.copy()
        elif last_gaze is not None:
            gaze[i] = last_gaze  # 前向填充 (上一有效 Gaze)
            dropout[i] = True
        else:
            dropout[i] = True  # 无历史可填, 保持 zeros 并标记缺失

    # 4. Audio nearest-timestamp 匹配 (容差外零向量 + 标记缺失)
    audio_out, audio_match_idx = _match_audio(
        timestamps, audio_features, audio_tolerance
    )
    dropout |= audio_match_idx < 0

    # 5. 有效长度 & 视觉拼接 [T,112]
    valid_length = int(np.count_nonzero(~dropout))
    visual_feature = np.concatenate([expression, pose, gaze], axis=1).astype(
        np.float32
    )

    meta = _build_metadata(
        clip_id=clip_id,
        fps=fps,
        audio_tolerance=audio_tolerance,
        audio_match=audio_match,
        num_frames=T,
        valid_length=valid_length,
        n_flame=n_flame,
        n_gaze=n_gaze,
        n_audio=n_audio,
        user_metadata=metadata,
    )

    return TrainingSample(
        clip_id=clip_id,
        visual_feature=visual_feature,
        audio_feature=audio_out,
        frame_index=frame_index,
        timestamps=timestamps,
        valid_length=valid_length,
        dropout_mask=dropout,
        audio_match_idx=audio_match_idx,
        metadata=meta,
    )

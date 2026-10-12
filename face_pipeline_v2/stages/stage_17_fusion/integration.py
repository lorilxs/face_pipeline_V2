"""Stage 17 样本包读写 — integration。

对应 docs/Stage17_Design_Review.md §4.4 输出结构:
  outputs/stage17_fusion/{clip_id}/
    training_sample.npz   # 对齐后张量 + 对齐索引 + valid_length
    metadata.json         # 溯源元数据 (visual_layout / source_stage 等)
  outputs/stage17_fusion/manifest.csv   # 数据集级清单

round-trip 契约: read_training_sample(save_training_sample(s)) 逐数组/字段精确一致。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from face_pipeline_v2.data.models import TrainingSample

NPZ_FILENAME = "training_sample.npz"
META_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.csv"

# NPZ 数组字段 (与 TrainingSample 对齐数组一一对应)
NPZ_ARRAY_FIELDS = (
    "visual_feature",
    "audio_feature",
    "frame_index",
    "timestamps",
    "audio_match_idx",
    "dropout_mask",
)

MANIFEST_COLUMNS = (
    "sample_id",
    "clip_id",
    "video_id",
    "fps",
    "num_frames",
    "valid_length",
    "audio_backend",
    "clip_start",
    "source_audio",
)


def _json_default(obj: object) -> object:
    """metadata.json 序列化兜底 — numpy 标量/数组、Path 转原生类型。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def save_training_sample(
    sample: TrainingSample,
    output_dir: Path,
    write_manifest: bool = True,
) -> Path:
    """将单个 TrainingSample 保存为样本包目录, 并追加/生成 manifest.csv。

    Args:
        sample: 待保存的样本。
        output_dir: 样本根目录 (默认配置 outputs/stage17_fusion)。
        write_manifest: 是否同时写入数据集级 manifest.csv。

    Returns:
        样本包目录路径 (output_dir / clip_id)。
    """
    output_dir = Path(output_dir)
    clip_dir = output_dir / sample.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        clip_dir / NPZ_FILENAME,
        visual_feature=sample.visual_feature,
        audio_feature=sample.audio_feature,
        frame_index=sample.frame_index,
        timestamps=sample.timestamps,
        audio_match_idx=sample.audio_match_idx,
        dropout_mask=sample.dropout_mask,
        valid_length=np.asarray(sample.valid_length, dtype=np.int32),
    )

    meta: Dict[str, object] = dict(sample.metadata)
    meta.setdefault("clip_id", sample.clip_id)
    meta.setdefault("sample_id", sample.sample_id or sample.clip_id)
    meta.setdefault("visual_dim", int(sample.visual_feature.shape[1]))
    meta.setdefault("audio_dim", int(sample.audio_feature.shape[1]))
    meta.setdefault("num_frames", int(sample.visual_feature.shape[0]))
    meta.setdefault("valid_length", int(sample.valid_length))

    with open(clip_dir / META_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=_json_default)

    if write_manifest:
        _append_manifest(output_dir, sample, meta)

    return clip_dir


def _append_manifest(
    output_dir: Path,
    sample: TrainingSample,
    meta: Dict[str, object],
) -> None:
    """向数据集级 manifest.csv 追加一行 (首次写入时生成表头)。"""
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    write_header = not manifest_path.exists()

    row = {
        "sample_id": sample.sample_id or sample.clip_id,
        "clip_id": sample.clip_id,
        "video_id": str(meta.get("video_id", "")),
        "fps": str(meta.get("fps", "")),
        "num_frames": int(meta.get("num_frames", 0)),
        "valid_length": int(meta.get("valid_length", 0)),
        "audio_backend": str(meta.get("audio_backend", "")),
        "clip_start": str(meta.get("clip_start", "")),
        "source_audio": str(meta.get("source_audio", "")),
    }

    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_training_sample(
    sample_dir: Path,
    clip_id: Optional[str] = None,
) -> TrainingSample:
    """从样本包目录读回 TrainingSample (NPZ + metadata.json)。

    Args:
        sample_dir: 样本包目录 (output_dir / clip_id)。
        clip_id: clip 标识 (未提供时从 metadata.json 读取)。

    Returns:
        重建的 TrainingSample (与保存时逐数组/字段一致)。

    Raises:
        FileNotFoundError: NPZ 或 metadata.json 缺失。
    """
    sample_dir = Path(sample_dir)
    npz_path = sample_dir / NPZ_FILENAME
    meta_path = sample_dir / META_FILENAME

    if not npz_path.exists():
        raise FileNotFoundError(f"Training sample NPZ not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Training sample metadata not found: {meta_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        visual = data["visual_feature"]
        audio = data["audio_feature"]
        frame_index = data["frame_index"]
        timestamps = data["timestamps"]
        audio_match_idx = data["audio_match_idx"]
        dropout_mask = data["dropout_mask"]
        if "valid_length" in data:
            valid_length = int(data["valid_length"])
        else:
            # 兼容旧版本 NPZ (无 valid_length 字段时按 dropout_mask 反推)
            valid_length = int(np.count_nonzero(~dropout_mask))

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    if clip_id is None:
        clip_id = str(meta.get("clip_id") or meta.get("sample_id") or "")

    return TrainingSample(
        clip_id=clip_id,
        visual_feature=visual,
        audio_feature=audio,
        frame_index=frame_index,
        timestamps=timestamps,
        valid_length=valid_length,
        dropout_mask=dropout_mask,
        audio_match_idx=audio_match_idx,
        metadata=meta,
    )
"""Stage 17 Export Layer — 多模态参数最终交付格式。

不替换现有 Stage17 fusion 输出 (outputs/stage17_fusion/ 契约保持不变), 而是新增
面向外部同事的数据交付目录:

  outputs/stage17_multimodal_export/
    flame/{clip_id}.npy       # [T,112] float32  FLAME + Gaze 视觉运动特征
    audio/{clip_id}.npy       # [T,768] float32  Chinese HuBERT audio embedding
    timestamp/{clip_id}.npz   # frame_index / timestamps / audio_match_idx / dropout_mask
    metadata/{clip_id}.json   # 溯源元数据 (visual_layout / 各模态来源)
    manifest.csv              # 数据集级索引

约束:
  - 本模块只做数据搬运 / 格式编排 — 不改 Stage12/13/14 模型逻辑, 不改 Stage17
    fusion 算法, 不改变 visual_feature / audio_feature 定义。
  - 不调用 / 不修改 build_training_sample / save_training_sample / read_training_sample。

visual_layout 说明: Stage17 fusion 的 visual_feature 仍为 [T,112], 子块顺序为
  expression[100] + posecode[6] + gaze[6]。本交付层仅把 posecode[6] 按来源进一步
  声明为 head_pose (EMICA global pose, 前 3 维) 与 jaw_pose (EMICA jaw pose,
  后 3 维) 写入 metadata, 便于下游消费, 不改变任何特征数值。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

from face_pipeline_v2.data.models import TrainingSample

# ---------------------------------------------------------------------------
# 交付目录布局常量
# ---------------------------------------------------------------------------

FLAME_DIR = "flame"
AUDIO_DIR = "audio"
TIMESTAMP_DIR = "timestamp"
METADATA_DIR = "metadata"
MANIFEST_FILENAME = "manifest.csv"

FLAME_SUFFIX = ".npy"
AUDIO_SUFFIX = ".npy"
TIMESTAMP_SUFFIX = ".npz"
METADATA_SUFFIX = ".json"

# timestamp/*.npz 内保存的对齐数组 (与 TrainingSample 对齐数组一一对应)
TIMESTAMP_ARRAY_FIELDS = (
    "frame_index",
    "timestamps",
    "audio_match_idx",
    "dropout_mask",
)

# ---------------------------------------------------------------------------
# 交付元数据常量
# ---------------------------------------------------------------------------

# 视觉子块布局 (metadata.visual_layout) — 在 fusion 的 posecode[6] 基础上细分 head/jaw
EXPORT_VISUAL_LAYOUT: Dict[str, tuple] = {
    "expression": (0, 100),
    "head_pose": (100, 103),
    "jaw_pose": (103, 106),
    "gaze": (106, 112),
}

# 各模态来源 (Stage12 EMICA vs SPECTRE ablation — 生产默认全 EMICA)。
DEFAULT_SOURCES: Dict[str, str] = {
    "head_source": "emica",
    "jaw_source": "emica",
    "expression_source": "emica",
    "gaze_source": "mediapipe",
    "audio_source": "chinese_hubert",
}

# 各模态来源允许值 — 交付校验 (AISDD Stage12 ablation: emica | spectre;
# Stage13 gaze 固定 mediapipe; Stage14 audio 固定 chinese_hubert)。
SOURCE_ALLOWED: Dict[str, frozenset] = {
    "head_source": frozenset({"emica", "spectre"}),
    "jaw_source": frozenset({"emica", "spectre"}),
    "expression_source": frozenset({"emica", "spectre"}),
    "gaze_source": frozenset({"mediapipe"}),
    "audio_source": frozenset({"chinese_hubert"}),
}

DEFAULT_FPS: float = 25.0

# 交付 manifest.csv 字段 (数据集级索引)
MANIFEST_COLUMNS = (
    "clip_id",
    "flame_path",
    "audio_path",
    "timestamp_path",
    "num_frames",
    "fps",
)

# metadata.json 必含字段 (数据契约, 缺一不可)
REQUIRED_METADATA_FIELDS = (
    "clip_id",
    "fps",
    "visual_dim",
    "audio_dim",
    "visual_layout",
    "head_source",
    "jaw_source",
    "expression_source",
    "gaze_source",
    "audio_source",
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


def _subdir(output_dir: Path, name: str) -> Path:
    """创建并返回交付子目录。"""
    d = output_dir / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_export_metadata(
    sample: TrainingSample,
    *,
    fps: float,
) -> Dict[str, object]:
    """装配交付 metadata dict。

    以 REQUIRED_METADATA_FIELDS 为骨架, 叠加融合时已有的溯源字段
    (video_id / clip_start / duration / audio_backend 等, 若存在)。
    """
    src = dict(DEFAULT_SOURCES)
    if isinstance(sample.metadata, dict):
        for key in DEFAULT_SOURCES:
            if sample.metadata.get(key):
                src[key] = str(sample.metadata[key])

    # 来源校验 — 仅允许已知合法来源 (emica/spectre/mediapipe/chinese_hubert),
    # 防止 mixed source / 未知来源进入交付包。
    for key, allowed in SOURCE_ALLOWED.items():
        val = src.get(key)
        if val not in allowed:
            raise ValueError(
                f"Invalid {key}={val!r} for multimodal export. "
                f"Allowed: {sorted(allowed)}."
            )

    meta: Dict[str, object] = {
        "clip_id": sample.clip_id,
        "fps": float(fps),
        "visual_dim": int(sample.visual_feature.shape[1]),
        "audio_dim": int(sample.audio_feature.shape[1]),
        "visual_layout": dict(EXPORT_VISUAL_LAYOUT),
        **src,
        "num_frames": int(sample.visual_feature.shape[0]),
        "valid_length": int(sample.valid_length),
        "dropout_count": int(np.count_nonzero(sample.dropout_mask)),
    }

    # 透传融合阶段已有的溯源字段 (仅当存在且未与骨架冲突)
    carry_fields = (
        "video_id",
        "sample_id",
        "audio_backend",
        "audio_match",
        "audio_tolerance",
        "clip_start",
        "duration",
        "source_audio",
    )
    if isinstance(sample.metadata, dict):
        for key in carry_fields:
            if sample.metadata.get(key) is not None and key not in meta:
                meta[key] = sample.metadata[key]
    return meta


def _append_manifest(output_dir: Path, meta: Dict[str, object]) -> None:
    """向交付根目录 manifest.csv 追加一行 (首次写入时生成表头)。"""
    clip_id = str(meta["clip_id"])
    manifest_path = output_dir / MANIFEST_FILENAME
    write_header = not manifest_path.exists()

    row = {
        "clip_id": clip_id,
        "flame_path": f"{FLAME_DIR}/{clip_id}{FLAME_SUFFIX}",
        "audio_path": f"{AUDIO_DIR}/{clip_id}{AUDIO_SUFFIX}",
        "timestamp_path": f"{TIMESTAMP_DIR}/{clip_id}{TIMESTAMP_SUFFIX}",
        "num_frames": int(meta["num_frames"]),
        "fps": float(meta["fps"]),
    }

    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def export_multimodal_sample(
    training_sample: TrainingSample,
    output_dir: Path,
) -> Path:
    """将单个 TrainingSample 导出为多模态交付格式目录。

    Args:
        training_sample: Stage17 fusion 产出的样本包 (含 visual_feature /
            audio_feature / frame_index / timestamps / audio_match_idx /
            dropout_mask / metadata)。
        output_dir: 交付根目录 (默认配置 outputs/stage17_multimodal_export)。

    Returns:
        交付根目录 (output_dir)。

    Raises:
        ValueError: clip_id 为空, 或 visual/audio T 轴不一致 (防御性校验,
            TrainingSample.__post_init__ 已保证一致)。
    """
    output_dir = Path(output_dir)
    clip_id = training_sample.clip_id
    if not clip_id:
        raise ValueError("clip_id must be non-empty for multimodal export")

    visual = np.asarray(training_sample.visual_feature, dtype=np.float32)
    audio = np.asarray(training_sample.audio_feature, dtype=np.float32)
    if visual.ndim != 2 or audio.ndim != 2:
        raise ValueError(
            "visual_feature / audio_feature must be 2-D arrays, "
            f"got {visual.shape} / {audio.shape}"
        )
    if visual.shape[0] != audio.shape[0]:
        raise ValueError(
            "visual/audio T-axis mismatch for export: "
            f"visual T={visual.shape[0]}, audio T={audio.shape[0]}"
        )

    fps = float(
        training_sample.metadata.get("fps", DEFAULT_FPS)
        if isinstance(training_sample.metadata, dict)
        else DEFAULT_FPS
    )

    # 1. flame / audio / timestamp / metadata 子目录
    flame_dir = _subdir(output_dir, FLAME_DIR)
    audio_dir = _subdir(output_dir, AUDIO_DIR)
    ts_dir = _subdir(output_dir, TIMESTAMP_DIR)
    meta_dir = _subdir(output_dir, METADATA_DIR)

    # 2. flame: [T,112] float32 视觉运动特征 (FLAME + Gaze)
    np.save(flame_dir / f"{clip_id}{FLAME_SUFFIX}", visual)

    # 3. audio: [T,768] float32 HuBERT embedding
    np.save(audio_dir / f"{clip_id}{AUDIO_SUFFIX}", audio)

    # 4. timestamp: 对齐数组
    np.savez(
        ts_dir / f"{clip_id}{TIMESTAMP_SUFFIX}",
        frame_index=training_sample.frame_index,
        timestamps=training_sample.timestamps,
        audio_match_idx=training_sample.audio_match_idx,
        dropout_mask=training_sample.dropout_mask,
    )

    # 5. metadata: 溯源元数据
    meta = _build_export_metadata(training_sample, fps=fps)
    with open(
        meta_dir / f"{clip_id}{METADATA_SUFFIX}", "w", encoding="utf-8"
    ) as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=_json_default)

    # 6. manifest.csv: 数据集级索引
    _append_manifest(output_dir, meta)

    return output_dir


def read_multimodal_export(
    output_dir: Path,
    clip_id: str,
) -> Dict[str, object]:
    """读回导出结果 (供 round-trip 校验 / 下游消费)。

    Returns:
        dict, 含 keys:
          visual_feature [T,112] float32, audio_feature [T,768] float32,
          frame_index [T] int32, timestamps [T] float32,
          audio_match_idx [T] int32, dropout_mask [T] bool,
          metadata (dict), clip_id (str)。
    """
    output_dir = Path(output_dir)
    flame_path = output_dir / FLAME_DIR / f"{clip_id}{FLAME_SUFFIX}"
    audio_path = output_dir / AUDIO_DIR / f"{clip_id}{AUDIO_SUFFIX}"
    ts_path = output_dir / TIMESTAMP_DIR / f"{clip_id}{TIMESTAMP_SUFFIX}"
    meta_path = output_dir / METADATA_DIR / f"{clip_id}{METADATA_SUFFIX}"

    for p in (flame_path, audio_path, ts_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Export file not found: {p}")

    visual = np.load(flame_path)
    audio = np.load(audio_path)
    with np.load(ts_path, allow_pickle=False) as data:
        frame_index = data["frame_index"]
        timestamps = data["timestamps"]
        audio_match_idx = data["audio_match_idx"]
        dropout_mask = data["dropout_mask"]
    with open(meta_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    return {
        "clip_id": clip_id,
        "visual_feature": visual,
        "audio_feature": audio,
        "frame_index": frame_index,
        "timestamps": timestamps,
        "audio_match_idx": audio_match_idx,
        "dropout_mask": dropout_mask,
        "metadata": metadata,
    }


def validate_export_metadata(metadata: Dict[str, object]) -> bool:
    """校验交付 metadata 字段完整性 (数据契约)。"""
    if not isinstance(metadata, dict):
        return False
    for field in REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            return False
    layout = metadata.get("visual_layout")
    expected = {k: list(v) for k, v in EXPORT_VISUAL_LAYOUT.items()}
    if not isinstance(layout, dict) or layout != expected:
        return False
    return True




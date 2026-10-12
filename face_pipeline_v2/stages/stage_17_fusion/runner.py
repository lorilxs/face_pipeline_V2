#!/usr/bin/env python3
"""Stage 17 Training Sample Assembly — standalone runner。

按 clip 装配三模态特征 (经 Stage12/13/14 官方 read_* 函数读取),
时间同步融合后输出样本包 (training_sample.npz + metadata.json) + manifest。

Usage:
    python runner.py \\
        --clip_id CLIP_ID \\
        --flame_dir /path/to/stage12_outputs \\
        --gaze_dir  /path/to/stage13_outputs \\
        --audio_dir /path/to/stage14_outputs \\
        --output_dir /path/to/outputs/stage17_fusion \\
        --fps 25 \\
        [--audio_tolerance 0.04] [--metadata_json extra.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from face_pipeline_v2.stages.stage_12_flame.integration import read_stage12_outputs
from face_pipeline_v2.stages.stage_13_gaze.integration import read_gaze_features
from face_pipeline_v2.stages.stage_14_audio_feature.integration import (
    read_audio_features,
)
from face_pipeline_v2.stages.stage_17_fusion.integration import save_training_sample
from face_pipeline_v2.stages.stage_17_fusion.processor import build_training_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stage17_runner")


def _load_extra_metadata(path: str) -> Dict[str, object]:
    """加载额外溯源元数据 (JSON 对象文件); 未提供时返回空 dict。"""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"metadata json must be an object, got {type(data).__name__}"
        )
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 17 Training Sample Runner (Fusion Core)"
    )
    parser.add_argument("--clip_id", required=True, help="目标 clip 唯一标识")
    parser.add_argument(
        "--flame_dir",
        required=True,
        help="Stage12 输出根目录 (含 {clip_id}/flame_features.npz)",
    )
    parser.add_argument(
        "--gaze_dir",
        required=True,
        help="Stage13 输出根目录 (含 {clip_id}/gaze_features.npz)",
    )
    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Stage14 输出根目录 (含 {clip_id}/audio_features.npz)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="样本包输出根目录 (默认配置 outputs/stage17_fusion)",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="视频帧率")
    parser.add_argument(
        "--visual_dim", type=int, default=112, help="visual_feature 维度 (固定 112)"
    )
    parser.add_argument(
        "--audio_dim", type=int, default=768, help="audio_feature 维度 (固定 768)"
    )
    parser.add_argument(
        "--audio_tolerance",
        type=float,
        default=0.04,
        help="audio nearest 匹配容差 (秒, 默认 0.04)",
    )
    parser.add_argument(
        "--audio_match",
        default="nearest_timestamp",
        help="音频匹配策略 (仅 nearest_timestamp)",
    )
    parser.add_argument(
        "--metadata_json",
        default=None,
        help="额外溯源元数据 JSON 对象文件 (video_id/scene_id 等)",
    )
    parser.add_argument(
        "--no_manifest",
        action="store_true",
        help="不写数据集级 manifest.csv",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # 1. 经 Stage12/13/14 官方 read_* 函数读取三模态特征
    flame = read_stage12_outputs(Path(args.flame_dir), args.clip_id)
    gaze = read_gaze_features(Path(args.gaze_dir), args.clip_id)
    audio = read_audio_features(Path(args.audio_dir), args.clip_id)
    logger.info(
        "clip %s: flame=%d frames, gaze=%d frames, audio=%d frames",
        args.clip_id,
        len(flame),
        len(gaze),
        len(audio),
    )

    # 2. 时间同步融合装配
    sample = build_training_sample(
        clip_id=args.clip_id,
        flame_features=flame,
        gaze_features=gaze,
        audio_features=audio,
        fps=args.fps,
        visual_dim=args.visual_dim,
        audio_dim=args.audio_dim,
        audio_tolerance=args.audio_tolerance,
        audio_match=args.audio_match,
        metadata=_load_extra_metadata(args.metadata_json),
    )

    # 3. 写样本包 + manifest
    out_dir = save_training_sample(
        sample,
        output_dir=Path(args.output_dir),
        write_manifest=not args.no_manifest,
    )

    t_total = int(sample.frame_index.size)
    n_dropout = int(sample.dropout_mask.sum())
    n_audio_matched = int((sample.audio_match_idx >= 0).sum())
    logger.info(
        "clip %s -> %s (T=%d, valid_length=%d, dropout=%d, audio_matched=%d)",
        args.clip_id,
        out_dir,
        t_total,
        sample.valid_length,
        n_dropout,
        n_audio_matched,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 14 Audio Feature Extraction — standalone runner for subprocess mode.

This runner reads an AVAlignmentInfo metadata JSON, loads HuBERT,
extracts audio features, and writes NPZ output.

Usage:
    python runner.py \
        --input_metadata /path/to/av_alignment.json \
        --output_dir /path/to/outputs/stage14_audio/ \
        --model_path models/hubert/chinese-hubert-base \
        --device auto \
        --fps 25
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from face_pipeline_v2.data.models import AVAlignmentInfo
from face_pipeline_v2.stages.stage_14_audio_feature.integration import (
    save_audio_features,
)
from face_pipeline_v2.stages.stage_14_audio_feature.processor import (
    extract_audio_features_for_clip,
    load_hubert_wrapper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stage14_runner")


def _parse_alignment(record: dict) -> AVAlignmentInfo:
    """Build an AVAlignmentInfo from a metadata JSON record."""
    return AVAlignmentInfo(
        clip_id=record["clip_id"],
        original_audio_path=Path(record["original_audio_path"]),
        offset_frames=int(record.get("offset_frames", 0)),
        offset_ms=float(record.get("offset_ms", 0.0)),
        correction_applied=bool(record.get("correction_applied", False)),
        fps=float(record.get("fps", 25.0)),
        corrected_audio_path=(
            Path(record["corrected_audio_path"])
            if record.get("corrected_audio_path")
            else None
        ),
        aligned_audio_path=(
            Path(record["aligned_audio_path"])
            if record.get("aligned_audio_path")
            else None
        ),
        alignment_status=record.get("alignment_status", "already_aligned"),
        raw_audio_path=(
            Path(record["raw_audio_path"]) if record.get("raw_audio_path") else None
        ),
        clip_start=(
            float(record["clip_start"]) if record.get("clip_start") is not None else None
        ),
        clip_duration=(
            float(record["clip_duration"])
            if record.get("clip_duration") is not None
            else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 14 Audio Feature Runner")
    parser.add_argument("--input_metadata", required=True,
                        help="JSON metadata file with an AVAlignmentInfo record")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for audio features")
    parser.add_argument("--model_path", required=True,
                        help="Path to HuBERT model directory (HuggingFace format)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto / cpu / cuda")
    parser.add_argument("--fps", type=int, default=25,
                        help="Video frame rate to align features to")
    parser.add_argument("--clip_id", default=None,
                        help="Clip ID (auto-detected from metadata if not provided)")
    args = parser.parse_args()

    metadata_path = Path(args.input_metadata)
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        raise SystemExit(1)

    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    alignment = _parse_alignment(record)

    if alignment.aligned_audio_path is None or not alignment.aligned_audio_path.exists():
        logger.error(
            "Aligned audio unavailable for clip %s (status=%s)",
            alignment.clip_id, alignment.alignment_status,
        )
        raise SystemExit(1)

    clip_id = args.clip_id or alignment.clip_id
    logger.info(
        "Processing clip %s audio %s (fps=%d)",
        clip_id, alignment.aligned_audio_path, args.fps,
    )

    wrapper = load_hubert_wrapper(model_path=Path(args.model_path), device=args.device)
    features = extract_audio_features_for_clip(
        wrapper=wrapper,
        audio_path=alignment.aligned_audio_path,
        fps=args.fps,
        start_time=float(alignment.clip_start or 0.0),
    )
    logger.info("Extracted %d audio features", len(features))

    output_dir = Path(args.output_dir)
    npz_path = save_audio_features(features, output_dir, clip_id)
    logger.info("Audio features saved to %s", npz_path)


if __name__ == "__main__":
    main()

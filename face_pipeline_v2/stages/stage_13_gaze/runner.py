#!/usr/bin/env python3
"""Stage 13 Gaze Extraction — standalone runner for subprocess mode.

This runner reads AlignedFaceInfo metadata JSON, loads MediaPipe,
runs gaze extraction, and writes NPZ output.

Usage:
    python runner.py \\
        --input_metadata /path/to/aligned_faces.json \\
        --output_dir /path/to/outputs/stage13_gaze/ \\
        --model_path models/mediapipe/face_landmarker.task \\
        --delegate cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from face_pipeline_v2.data.models import AlignedFaceInfo, GazeFeature
from face_pipeline_v2.stages.stage_13_gaze.processor import (
    create_mediapipe_detector,
    extract_gaze_parameters,
)
from face_pipeline_v2.stages.stage_13_gaze.integration import (
    save_gaze_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stage13_runner")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 13 Gaze Runner")
    parser.add_argument(
        "--input_metadata", required=True,
        help="JSON metadata file with AlignedFaceInfo records",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for gaze features",
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to face_landmarker.task",
    )
    parser.add_argument(
        "--delegate", default="cpu", choices=["cpu", "gpu"],
        help="MediaPipe delegate (cpu or gpu)",
    )
    parser.add_argument(
        "--clip_id", default=None,
        help="Clip ID (auto-detected from metadata if not provided)",
    )
    args = parser.parse_args()

    # Load metadata
    metadata_path = Path(args.input_metadata)
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        raise SystemExit(1)

    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    aligned_faces: List[AlignedFaceInfo] = []
    for rec in records:
        aligned_faces.append(AlignedFaceInfo(
            clip_id=rec["clip_id"],
            frame_index=rec["frame_index"],
            face_id=rec.get("face_id", "face_0"),
            timestamp=rec["timestamp"],
            aligned_face_path=Path(rec["aligned_face_path"]),
            affine_matrix=tuple(rec.get("affine_matrix", (1., 0., 0., 0., 1., 0.))),
            original_image_size=tuple(rec.get("original_image_size", (256, 256))),
            aligned_resolution=tuple(rec.get("aligned_resolution", (256, 256))),
        ))

    if not aligned_faces:
        logger.error("No AlignedFaceInfo records in metadata.")
        raise SystemExit(1)

    clip_id = args.clip_id or aligned_faces[0].clip_id
    logger.info(
        "Processing %d aligned faces for clip %s", len(aligned_faces), clip_id,
    )

    # Create detector
    detector = create_mediapipe_detector(
        model_path=args.model_path,
        delegate=args.delegate,
    )

    try:
        features = extract_gaze_parameters(
            aligned_faces=aligned_faces,
            detector=detector,
            image_size=(256, 256),
        )
    finally:
        detector.close()

    logger.info(
        "Extracted %d / %d gaze features",
        len(features), len(aligned_faces),
    )

    # Save
    output_dir = Path(args.output_dir)
    npz_path = save_gaze_features(features, output_dir, clip_id)
    logger.info("Gaze features saved to %s", npz_path)


if __name__ == "__main__":
    main()
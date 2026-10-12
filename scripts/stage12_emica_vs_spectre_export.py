#!/usr/bin/env python3
"""Stage12 EMICA vs SPECTRE ablation — per-variant Stage17 fusion + multimodal export.

Unlike the production ``export_production_multimodal.py`` (which hard-codes
``jaw_source=emica``), this driver tags each delivery package with the REAL
sources of the variant (``expression_source`` / ``head_source`` / ``jaw_source``
read from the variant's ``flame_metadata.json``) and validates them against
``export.SOURCE_ALLOWED`` (emica | spectre / mediapipe / chinese_hubert).

Usage:
    PYTHONPATH="$PWD" python3 scripts/stage12_emica_vs_spectre_export.py \\
        --base outputs/ablation_stage12_emica_vs_spectre/emica \\
        --export-dir outputs/ablation_stage12_emica_vs_spectre/emica/stage17_multimodal_export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from face_pipeline_v2.stages.stage_12_flame.integration import (  # noqa: E402
    read_stage12_outputs,
)
from face_pipeline_v2.stages.stage_13_gaze.integration import (  # noqa: E402
    read_gaze_features,
)
from face_pipeline_v2.stages.stage_14_audio_feature.integration import (  # noqa: E402
    read_audio_features,
)
from face_pipeline_v2.stages.stage_17_fusion.processor import (  # noqa: E402
    build_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.integration import (  # noqa: E402
    save_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.export import (  # noqa: E402
    export_multimodal_sample,
    read_multimodal_export,
    validate_export_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stage12_emica_vs_spectre_export")

FPS = 25.0
AUDIO_TOLERANCE = 0.04
AUDIO_MATCH = "nearest_timestamp"


def _has_bad(arr) -> bool:
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


def _flame_meta(flame_dir: Path, clip: str) -> dict:
    meta_path = flame_dir / clip / "flame_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return {}


def _extra_metadata(flame_meta: dict, clip: str) -> dict:
    return {
        "video_id": clip.split("_scene", 1)[0],
        "expression_source": flame_meta.get("expression_source", "emica"),
        "head_source": flame_meta.get("head_source", "emica"),
        "jaw_source": flame_meta.get("jaw_source", "emica"),
        "gaze_source": "mediapipe",
        "audio_source": "chinese_hubert",
        "audio_backend": "Chinese HuBERT",
        "audio_match": AUDIO_MATCH,
        "audio_tolerance": AUDIO_TOLERANCE,
        "fps": FPS,
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        required=True,
        help="ablation variant base containing stage12_flame / stage13_gaze / "
             "stage14_audio / stage17_fusion",
    )
    ap.add_argument(
        "--export-dir",
        required=True,
        help="Where to write the multimodal export package",
    )
    return ap.parse_args(argv)


def _clips(flame_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in flame_dir.iterdir()
        if (p / "flame_features.npz").exists()
    )


def main() -> int:
    args = _parse_args()
    base = Path(args.base)
    export_dir = Path(args.export_dir)
    flame_dir = base / "stage12_flame"
    gaze_dir = base / "stage13_gaze"
    audio_dir = base / "stage14_audio"
    fusion_dir = base / "stage17_fusion"

    clips = _clips(flame_dir)
    logger.info("clips found: %d", len(clips))
    if not clips:
        logger.error("no Stage12 outputs under %s", flame_dir)
        return 1

    ok_all = True
    for clip in clips:
        logger.info("=" * 70)
        logger.info("[S17] %s", clip)
        flame = read_stage12_outputs(flame_dir, clip)
        gaze = read_gaze_features(gaze_dir, clip)
        audio = read_audio_features(audio_dir, clip)
        logger.info("flame=%d gaze=%d audio=%d", len(flame), len(gaze), len(audio))

        flame_meta = _flame_meta(flame_dir, clip)
        sample = build_training_sample(
            clip_id=clip,
            flame_features=flame,
            gaze_features=gaze,
            audio_features=audio,
            fps=FPS,
            visual_dim=112,
            audio_dim=768,
            audio_tolerance=AUDIO_TOLERANCE,
            audio_match=AUDIO_MATCH,
            metadata=_extra_metadata(flame_meta, clip),
        )
        save_training_sample(sample, output_dir=fusion_dir)
        export_multimodal_sample(sample, output_dir=export_dir)

        data = read_multimodal_export(export_dir, clip)
        meta = data["metadata"]
        T = int(data["visual_feature"].shape[0])
        checks = {
            "flame_shape": list(data["visual_feature"].shape),
            "audio_shape": list(data["audio_feature"].shape),
            "expression_source": meta.get("expression_source"),
            "head_source": meta.get("head_source"),
            "jaw_source": meta.get("jaw_source"),
            "gaze_source": meta.get("gaze_source"),
            "audio_source": meta.get("audio_source"),
            "metadata_valid": validate_export_metadata(meta),
            "no_nan_inf": bool(
                not (_has_bad(data["visual_feature"]) or _has_bad(data["audio_feature"]))
            ),
        }
        logger.info("checks: %s", json.dumps(checks, ensure_ascii=False))

        # expected sources = the variant's real sources
        exp_src = flame_meta.get("expression_source", "emica")
        head_src = flame_meta.get("head_source", "emica")
        jaw_src = flame_meta.get("jaw_source", "emica")
        if not (
            checks["flame_shape"] == [T, 112]
            and checks["audio_shape"] == [T, 768]
            and checks["expression_source"] == exp_src
            and checks["head_source"] == head_src
            and checks["jaw_source"] == jaw_src
            and checks["metadata_valid"]
            and checks["no_nan_inf"]
        ):
            ok_all = False
            logger.error("clip %s FAILED checks", clip)

    logger.info("=" * 70)
    logger.info("ablation export -> %s", export_dir)
    logger.info("ALL OK" if ok_all else "SOME CLIPS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())


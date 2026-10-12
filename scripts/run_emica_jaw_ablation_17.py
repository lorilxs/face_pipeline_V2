#!/usr/bin/env python3
"""EMICA-jaw ablation driver — Stage17 fusion + multimodal export (all clips).

Temporary experiment: re-assemble Stage17 TrainingSamples and export the
multimodal delivery package from the NEW EMICA-jaw Stage12 outputs, tagging
``jaw_source=emica`` in the delivered metadata so the mesh team can compare
against the production SPECTRE-jaw export (``outputs/stage17_multimodal_export/``).

No Stage17 algorithm change: this only invokes the official interfaces
(build_training_sample / save_training_sample / export_multimodal_sample) and
writes into a NEW output dir, leaving production untouched.

Usage:
    PYTHONPATH="$PWD" python3 scripts/run_emica_jaw_ablation_17.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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
    read_training_sample,
    save_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.export import (  # noqa: E402
    export_multimodal_sample,
    read_multimodal_export,
    validate_export_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emica_jaw_ablation_17")

ABLATION_BASE = PROJECT_ROOT / "outputs" / "ablation_emica_jaw"
FLAME_DIR = ABLATION_BASE / "stage12_flame"
GAZE_DIR = ABLATION_BASE / "stage13_gaze"
AUDIO_DIR = ABLATION_BASE / "stage14_audio"
FUSION_DIR = ABLATION_BASE / "stage17_fusion"
EXPORT_DIR = PROJECT_ROOT / "outputs" / "stage17_multimodal_export_emica_jaw"

FPS = 25.0
AUDIO_TOLERANCE = 0.04
AUDIO_MATCH = "nearest_timestamp"

PRODUCTION_EXPORT_DIR = PROJECT_ROOT / "outputs" / "stage17_multimodal_export"


def _production_export_T(clip_id: str) -> int | None:
    """Number of frames in the production (SPECTRE-jaw) export for this clip."""
    p = PRODUCTION_EXPORT_DIR / "flame" / f"{clip_id}.npy"
    if not p.exists():
        return None
    import numpy as np
    return int(np.load(p).shape[0])


def _clips() -> list[str]:
    return sorted(
        p.name
        for p in FLAME_DIR.iterdir()
        if (p / "flame_features.npz").exists()
    )


def _extra_metadata(clip_id: str) -> dict:
    return {
        "video_id": clip_id.split("_scene", 1)[0],
        "jaw_source": "emica",          # the whole point of this ablation
        "experiment": "stage12_jaw_ablation_emica_vs_spectre",
        "audio_backend": "Chinese HuBERT",
        "audio_match": AUDIO_MATCH,
        "audio_tolerance": AUDIO_TOLERANCE,
        "fps": FPS,
    }


def main() -> int:
    clips = _clips()
    logger.info("clips found: %d", len(clips))
    if not clips:
        logger.error("no Stage12 (emica) outputs found under %s", FLAME_DIR)
        return 1

    ok_all = True
    for clip in clips:
        logger.info("=" * 70)
        logger.info("[S17] %s", clip)
        flame = read_stage12_outputs(FLAME_DIR, clip)
        gaze = read_gaze_features(GAZE_DIR, clip)
        audio = read_audio_features(AUDIO_DIR, clip)
        logger.info("flame=%d gaze=%d audio=%d", len(flame), len(gaze), len(audio))

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
            metadata=_extra_metadata(clip),
        )
        save_training_sample(sample, output_dir=FUSION_DIR)
        export_multimodal_sample(sample, output_dir=EXPORT_DIR)

        # per-clip sanity check (T is data-dependent — compare with the
        # production export for the same clip to guarantee "same as production
        # except jaw")
        data = read_multimodal_export(EXPORT_DIR, clip)
        meta = data["metadata"]
        T = int(data["visual_feature"].shape[0])
        T_prod = _production_export_T(clip)
        checks = {
            "flame_shape": list(data["visual_feature"].shape),
            "audio_shape": list(data["audio_feature"].shape),
            "T_matches_production": T == T_prod,
            "jaw_source": meta.get("jaw_source"),
            "head_source": meta.get("head_source"),
            "expression_source": meta.get("expression_source"),
            "gaze_source": meta.get("gaze_source"),
            "audio_source": meta.get("audio_source"),
            "metadata_valid": validate_export_metadata(meta),
            "no_nan_inf": bool(
                not (
                    _has_bad(data["visual_feature"])
                    or _has_bad(data["audio_feature"])
                )
            ),
        }
        logger.info("checks: %s", json.dumps(checks, ensure_ascii=False))
        if not (
            checks["flame_shape"] == [T, 112]
            and checks["audio_shape"] == [T, 768]
            and checks["T_matches_production"]
            and checks["jaw_source"] == "emica"
            and checks["metadata_valid"]
            and checks["no_nan_inf"]
        ):
            ok_all = False
            logger.error("clip %s FAILED checks", clip)

    logger.info("=" * 70)
    logger.info("EMICA-jaw export -> %s", EXPORT_DIR)
    logger.info("ALL OK" if ok_all else "SOME CLIPS FAILED")
    return 0 if ok_all else 1


def _has_bad(arr) -> bool:
    import numpy as np
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


if __name__ == "__main__":
    raise SystemExit(main())

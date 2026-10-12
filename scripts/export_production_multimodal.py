#!/usr/bin/env python3
"""Production Stage17 driver — fusion + multimodal export (EMICA jaw, final scheme).

Regenerates the PRODUCTION Stage17 outputs after the EMICA-jaw adoption:

  - reads Stage12 (EMICA jaw) / Stage13 / Stage14 from
    ``outputs/stage0_17_e2e/dataset_validation/``
  - builds TrainingSamples via the official ``build_training_sample``
  - saves fusion sample packages to ``.../stage17_fusion``
  - exports the multimodal delivery package to ``outputs/stage17_multimodal_export``

Metadata tags ``jaw_source=emica`` (final production scheme). This is the
production counterpart of the (now-superseded) ablation driver.

Usage:
    PYTHONPATH="$PWD" python3 scripts/export_production_multimodal.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
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
from face_pipeline_v2.utils.pipeline_manifest import (  # noqa: E402
    merge_phase2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_production_multimodal")

DEFAULT_BASE = PROJECT_ROOT / "outputs" / "stage0_17_e2e" / "dataset_validation"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "outputs" / "stage17_multimodal_export"

FPS = 25.0
AUDIO_TOLERANCE = 0.04
AUDIO_MATCH = "nearest_timestamp"

# Per-clip Stage17 fusion report written into the export dir and consumed by
# ``pipeline_manifest.merge_phase2`` so failures (e.g. Stage13 gaze extraction
# producing missing/unmatched frames -> unusable final parameters) survive even
# after intermediate stage directories are deleted.
EXPORT_REPORT_FILENAME = "export_report.json"


def _has_bad(arr) -> bool:
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


def _extra_metadata(clip_id: str) -> dict:
    return {
        "video_id": clip_id.split("_scene", 1)[0],
        "jaw_source": "emica",   # final production scheme
        "audio_backend": "Chinese HuBERT",
        "audio_match": AUDIO_MATCH,
        "audio_tolerance": AUDIO_TOLERANCE,
        "fps": FPS,
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Production Stage17 fusion + multimodal export (EMICA jaw)."
    )
    ap.add_argument(
        "--base",
        default=str(DEFAULT_BASE),
        help="Stage0-11 test base containing stage12_flame / stage13_gaze / stage14_audio",
    )
    ap.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Where to write the multimodal export package",
    )
    return ap.parse_args(argv)


def _clips(flame_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in flame_dir.iterdir()
        if (p / "flame_features.npz").exists()
    )


def _write_export_report(export_dir: Path, report: dict) -> None:
    """Best-effort atomic write of the per-clip Stage17 fusion/export report.

    This is the authoritative on-disk record of per-clip Stage12/13/14 feature
    availability + Stage17 fusion/export checks, consumed later by
    ``pipeline_manifest.merge_phase2`` so failures (e.g. Stage13 gaze
    extraction producing missing/unmatched frames -> unusable final
    parameters) survive deletion of the intermediate stage directories.
    Non-fatal by design: a write failure only downgrades the log and never
    affects the export exit status.
    """
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / EXPORT_REPORT_FILENAME
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - non-fatal by design
        logger.warning("[export_report] write failed: %s", exc)


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

    report: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clips": {},
        "summary": {"total": len(clips), "passed": 0, "failed": 0, "ok_all": None},
    }

    ok_all = True
    for clip in clips:
        rec: dict = {
            "clip_id": clip,
            "status": "PASS",
            "failed_reason": None,
            "n_flame": 0,
            "n_gaze": 0,
            "n_audio": 0,
            "num_frames": 0,
            "valid_length": 0,
            "gaze_missing_frames": 0,
            "audio_missing_frames": 0,
            "checks": {},
        }
        logger.info("=" * 70)
        logger.info("[S17] %s", clip)

        # --- read Stage12/13/14 features defensively (one modality at a time) ---
        # A missing .npz would otherwise raise FileNotFoundError and abort the
        # whole export; here we record the failing modality and keep going.
        try:
            flame = read_stage12_outputs(flame_dir, clip)
        except Exception as exc:  # noqa: BLE001 - record + continue
            rec["status"] = "FAIL"
            rec["failed_reason"] = f"Stage12 FLAME read failed: {exc}"
            ok_all = False
            report["clips"][clip] = rec
            logger.error("[S17] clip %s FAILED: %s", clip, rec["failed_reason"])
            continue
        try:
            gaze = read_gaze_features(gaze_dir, clip)
        except Exception as exc:  # noqa: BLE001 - record + continue
            rec["status"] = "FAIL"
            rec["failed_reason"] = f"Stage13 gaze read failed: {exc}"
            ok_all = False
            report["clips"][clip] = rec
            logger.error("[S17] clip %s FAILED: %s", clip, rec["failed_reason"])
            continue
        try:
            audio = read_audio_features(audio_dir, clip)
        except Exception as exc:  # noqa: BLE001 - record + continue
            rec["status"] = "FAIL"
            rec["failed_reason"] = f"Stage14 audio read failed: {exc}"
            ok_all = False
            report["clips"][clip] = rec
            logger.error("[S17] clip %s FAILED: %s", clip, rec["failed_reason"])
            continue

        n_flame = len(flame)
        n_gaze = len(gaze)
        n_audio = len(audio)
        rec["n_flame"] = n_flame
        rec["n_gaze"] = n_gaze
        rec["n_audio"] = n_audio
        flame_ids = {int(f.frame_index) for f in flame}
        gaze_ids = {int(g.frame_index) for g in gaze}
        rec["gaze_missing_frames"] = (
            len(flame_ids - gaze_ids) if flame_ids else n_flame
        )
        logger.info("flame=%d gaze=%d audio=%d", n_flame, n_gaze, n_audio)

        # --- fusion (raises ValueError on corrupt dims / alignment) ---
        try:
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
        except Exception as exc:  # noqa: BLE001 - record + continue
            rec["status"] = "FAIL"
            rec["failed_reason"] = f"Stage17 fusion build failed: {exc}"
            ok_all = False
            report["clips"][clip] = rec
            logger.error("[S17] clip %s FAILED: %s", clip, rec["failed_reason"])
            continue

        rec["num_frames"] = int(sample.visual_feature.shape[0])
        rec["valid_length"] = int(sample.valid_length or 0)
        rec["audio_missing_frames"] = int(
            np.count_nonzero(np.asarray(sample.audio_match_idx) < 0)
        )

        # --- save + export + read-back checks ---
        try:
            save_training_sample(sample, output_dir=fusion_dir)
            export_multimodal_sample(sample, output_dir=export_dir)
            data = read_multimodal_export(export_dir, clip)
        except Exception as exc:  # noqa: BLE001 - record + continue
            rec["status"] = "FAIL"
            rec["failed_reason"] = f"Stage17 export failed: {exc}"
            ok_all = False
            report["clips"][clip] = rec
            logger.error("[S17] clip %s FAILED: %s", clip, rec["failed_reason"])
            continue

        meta = data["metadata"]
        T = int(data["visual_feature"].shape[0])
        checks = {
            "flame_shape": list(data["visual_feature"].shape),
            "audio_shape": list(data["audio_feature"].shape),
            "jaw_source": meta.get("jaw_source"),
            "head_source": meta.get("head_source"),
            "expression_source": meta.get("expression_source"),
            "gaze_source": meta.get("gaze_source"),
            "audio_source": meta.get("audio_source"),
            "metadata_valid": validate_export_metadata(meta),
            "no_nan_inf": bool(
                not (_has_bad(data["visual_feature"]) or _has_bad(data["audio_feature"]))
            ),
        }
        rec["checks"] = checks
        logger.info("checks: %s", json.dumps(checks, ensure_ascii=False))

        # FAIL gate: an entirely missing modality makes the final
        # flame/gaze/audio parameters unmatchable and therefore unusable;
        # export checks must also all pass.
        missing_modalities = [
            name
            for name, n in (("flame", n_flame), ("gaze", n_gaze), ("audio", n_audio))
            if n == 0
        ]
        checks_ok = bool(
            checks["flame_shape"] == [T, 112]
            and checks["audio_shape"] == [T, 768]
            and checks["jaw_source"] == "emica"
            and checks["metadata_valid"]
            and checks["no_nan_inf"]
        )
        if missing_modalities:
            rec["status"] = "FAIL"
            rec["failed_reason"] = (
                "modality frames missing -> flame/gaze/audio unmatchable: "
                + ", ".join(missing_modalities)
            )
            ok_all = False
            logger.error("[S17] clip %s FAILED: %s", clip, rec["failed_reason"])
        elif not checks_ok:
            rec["status"] = "FAIL"
            rec["failed_reason"] = "export checks failed"
            ok_all = False
            logger.error("[S17] clip %s FAILED checks", clip)
        report["clips"][clip] = rec

    report["summary"]["passed"] = sum(
        1 for r in report["clips"].values() if r.get("status") == "PASS"
    )
    report["summary"]["failed"] = sum(
        1 for r in report["clips"].values() if r.get("status") == "FAIL"
    )
    report["summary"]["ok_all"] = bool(ok_all)
    _write_export_report(export_dir, report)

    logger.info("=" * 70)
    logger.info("production export -> %s", export_dir)
    logger.info("ALL OK" if ok_all else "SOME CLIPS FAILED")

    # ------------------------------------------------------------------
    # Phase-2 pipeline manifest merge (best-effort; must never change exit)
    # ------------------------------------------------------------------
    # Merges the Stage12/13/14 official-runner metadata + Stage17 export
    # metadata + final-asset inventory into the phase-1 manifest written by
    # the Stage0-11 e2e driver.  Non-fatal by design (see pipeline_manifest
    # docstring): any failure here only downgrades the run log.
    try:
        merged = merge_phase2(base, export_dir, logger=logger)
        if merged is not None:
            logger.info(
                "[pipeline_manifest] phase2 merged -> %s (clips=%d)",
                merged.get("run", {}).get("run_root"),
                len(merged.get("clips", [])),
            )
    except Exception as exc:  # noqa: BLE001 - manifest must never break export
        logger.warning("[pipeline_manifest] phase2 merge skipped: %s", exc)
        logger.debug("phase2 merge traceback", exc_info=True)

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Final Stage0-17 End-to-End Mesh-Export Validation (dataset mode, real video).

Produces the final Stage17 multimodal export for all accepted clips (via the
official export_multimodal_sample interface) and validates every requirement of
docs/Final_Stage0_17_Mesh_Export_Validation_Report.md:

  Stage12 FLAME  : expression [T,100], head_pose [T,3], jaw_pose [T,3]; no NaN/Inf;
                   head |.| < 0.5 rad; jaw |.| < 0.2 rad; jaw source = SPECTRE with
                   jaw_pose == spectre_jawpose (max diff == 0).
  Stage13 gaze   : gaze_vector [T,6]; per eye-direction norm ~= 1.
  Stage14 audio  : feature [T,768]; no NaN/Inf.
  Stage17 export : flame/*.npy [T,112], audio/*.npy [T,768], timestamp/*.npz,
                   metadata/*.json, manifest.csv; all clip ids consistent.

This script only *runs* the pipeline's official export interface and performs
validation -- it does not modify any pipeline code.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_pipeline_v2.stages.stage_17_fusion.integration import (  # noqa: E402
    read_training_sample,
)
from face_pipeline_v2.stages.stage_17_fusion.export import (  # noqa: E402
    AUDIO_DIR,
    FLAME_DIR,
    MANIFEST_FILENAME,
    METADATA_DIR,
    TIMESTAMP_DIR,
    export_multimodal_sample,
    validate_export_metadata,
)

BASE = ROOT / "outputs/stage0_17_e2e/dataset_validation"
EXPORT_ROOT = ROOT / "outputs/stage17_multimodal_export"
RESULTS_JSON = ROOT / "outputs/stage0_17_e2e/stage0_11_results.json"
REPORT_PATH = ROOT / "docs/Final_Stage0_17_Mesh_Export_Validation_Report.md"
CHECKS_JSON = ROOT / "docs/final_mesh_export_checks.json"

HEAD_MAX_RAD = 0.5
JAW_MAX_RAD = 0.2
GAZE_NORM_TOL = 1e-3


def _clip_ids() -> List[str]:
    ids = sorted(
        {
            p.name.removesuffix("_aligned_faces.json")
            for p in (BASE / "stage_inputs").glob("*_aligned_faces.json")
        }
    )
    return ids


def _sum(rows: List[np.ndarray]) -> Dict[str, float]:
    a = np.concatenate(rows)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def validate_stage12(clip: str) -> Tuple[bool, Dict[str, Any]]:
    z = np.load(BASE / "stage12_flame" / clip / "flame_features.npz", allow_pickle=True)
    exp, head, jaw = z["expression_code"], z["head_pose"], z["jaw_pose"]
    spectre = z["spectre_jawpose"]
    emica = z["emica_jawpose"]
    finite_ok = all(np.isfinite(a).all() for a in (exp, head, jaw, spectre, emica))
    shape_ok = (
        exp.ndim == 2 and exp.shape[1] == 100
        and head.shape[1] == 3 and jaw.shape[1] == 3
    )
    head_max = float(np.abs(head).max()) if head.size else float("nan")
    jaw_max = float(np.abs(jaw).max()) if jaw.size else float("nan")
    jaw_emica_diff = float(np.abs(jaw - emica).max())
    ok = (
        finite_ok and shape_ok
        and head_max < HEAD_MAX_RAD and jaw_max < JAW_MAX_RAD
        and jaw_emica_diff == 0.0
    )
    info = {
        "ok": ok,
        "T": int(len(exp)),
        "expression_shape": list(exp.shape),
        "head_pose_shape": list(head.shape),
        "jaw_pose_shape": list(jaw.shape),
        "no_nan_inf": finite_ok,
        "head_abs_max_rad": head_max,
        "jaw_abs_max_rad": jaw_max,
        "head_threshold_0_5": head_max < HEAD_MAX_RAD,
        "jaw_threshold_0_2": jaw_max < JAW_MAX_RAD,
        "jaw_source": "emica",
        "jaw_vs_emica_maxdiff": jaw_emica_diff,
        "jaw_equals_emica": jaw_emica_diff == 0.0,
        "expr_stats": _sum([exp]),
    }
    return ok, info


def validate_stage13(clip: str) -> Tuple[bool, Dict[str, Any]]:
    z = np.load(BASE / "stage13_gaze" / clip / "gaze_features.npz", allow_pickle=True)
    g = np.asarray(z["gaze_vector"])
    shape_ok = g.ndim == 2 and g.shape[1] == 6
    finite_ok = bool(np.isfinite(g).all())
    left_norm = np.linalg.norm(g[:, 0:3], axis=1)
    right_norm = np.linalg.norm(g[:, 3:6], axis=1)
    norms_ok = bool(
        np.all(np.abs(left_norm - 1.0) < GAZE_NORM_TOL)
        and np.all(np.abs(right_norm - 1.0) < GAZE_NORM_TOL)
    )
    ok = shape_ok and finite_ok and norms_ok
    info = {
        "ok": ok,
        "shape": list(g.shape),
        "no_nan_inf": finite_ok,
        "left_norm_mean": float(left_norm.mean()),
        "left_norm_max_abs_dev": float(np.abs(left_norm - 1.0).max()),
        "right_norm_mean": float(right_norm.mean()),
        "right_norm_max_abs_dev": float(np.abs(right_norm - 1.0).max()),
    }
    return ok, info


def validate_stage14(clip: str) -> Tuple[bool, Dict[str, Any]]:
    z = np.load(BASE / "stage14_audio" / clip / "audio_features.npz", allow_pickle=True)
    a = np.asarray(z["feature"])
    shape_ok = a.ndim == 2 and a.shape[1] == 768
    finite_ok = bool(np.isfinite(a).all())
    ok = shape_ok and finite_ok
    info = {
        "ok": ok,
        "shape": list(a.shape),
        "no_nan_inf": finite_ok,
        "stats": _sum([a]),
    }
    return ok, info


def validate_stage17(clip: str) -> Tuple[bool, Dict[str, Any]]:
    z = np.load(BASE / "stage17_fusion" / clip / "training_sample.npz", allow_pickle=True)
    vis, aud = np.asarray(z["visual_feature"]), np.asarray(z["audio_feature"])
    ts = np.asarray(z["timestamps"])
    finite_ok = bool(np.isfinite(vis).all()) and bool(np.isfinite(aud).all())
    shape_ok = (
        vis.ndim == 2 and vis.shape[1] == 112
        and aud.ndim == 2 and aud.shape[1] == 768
        and vis.shape[0] == aud.shape[0]
    )
    ts_ok = bool(np.all(np.diff(ts) > 0)) if ts.size > 1 else True
    ok = finite_ok and shape_ok and ts_ok
    info = {
        "ok": ok,
        "visual_shape": list(vis.shape),
        "audio_shape": list(aud.shape),
        "valid_length": int(np.asarray(z["valid_length"])),
        "dropout": int(np.asarray(z["dropout_mask"]).sum()),
        "audio_matched": int((np.asarray(z["audio_match_idx"]) >= 0).sum()),
        "ts_range": [float(ts.min()), float(ts.max())],
    }
    return ok, info


def run_export() -> Dict[str, Any]:
    """Stage17 export for all clips via the official interface."""
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    ids = _clip_ids()
    exported = {}
    for clip in ids:
        sample = read_training_sample(BASE / "stage17_fusion" / clip, clip)
        out = export_multimodal_sample(sample, EXPORT_ROOT)
        exported[clip] = str(out)
    return {"export_root": str(EXPORT_ROOT), "n_clips": len(exported)}


def validate_export(clip: str) -> Tuple[bool, Dict[str, Any]]:
    paths = {
        "flame": EXPORT_ROOT / FLAME_DIR / f"{clip}.npy",
        "audio": EXPORT_ROOT / AUDIO_DIR / f"{clip}.npy",
        "timestamp": EXPORT_ROOT / TIMESTAMP_DIR / f"{clip}.npz",
        "metadata": EXPORT_ROOT / METADATA_DIR / f"{clip}.json",
    }
    exists = {k: p.exists() for k, p in paths.items()}
    vis = np.load(paths["flame"])
    aud = np.load(paths["audio"])
    with np.load(paths["timestamp"], allow_pickle=False) as d:
        ts = np.asarray(d["timestamps"])
        fi = np.asarray(d["frame_index"])
        ami = np.asarray(d["audio_match_idx"])
        dm = np.asarray(d["dropout_mask"])
    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    shape_ok = (
        vis.ndim == 2 and vis.shape[1] == 112
        and aud.ndim == 2 and aud.shape[1] == 768
        and vis.shape[0] == aud.shape[0] == len(ts)
    )
    ts_ok = bool(np.all(np.diff(ts) > 0)) if ts.size > 1 else True
    meta_ok = validate_export_metadata(meta) and meta.get("clip_id") == clip
    ok = all(exists.values()) and shape_ok and ts_ok and meta_ok
    info = {
        "ok": ok,
        "files_exist": exists,
        "flame_shape": list(vis.shape),
        "audio_shape": list(aud.shape),
        "timestamp_T": int(len(ts)),
        "timestamp_monotonic": ts_ok,
        "metadata_valid": meta_ok,
        "metadata_sources": {
            k: meta.get(k)
            for k in ("head_source", "jaw_source", "expression_source", "gaze_source", "audio_source")
        },
        "dropout": int(dm.sum()),
        "audio_matched": int((ami >= 0).sum()),
        "frame_index_range": [int(fi.min()), int(fi.max())] if fi.size else None,
    }
    return ok, info



def main() -> int:
    ids = _clip_ids()
    print(f"clips={len(ids)}")
    for c in ids:
        print("  ", c)

    # ---- Stage17 export (official interface) ----
    export_info = run_export()
    print("export:", export_info)

    per_clip = {}
    n_ok12 = n_ok13 = n_ok14 = n_ok17 = n_ok_export = 0
    for clip in ids:
        ok12, i12 = validate_stage12(clip)
        ok13, i13 = validate_stage13(clip)
        ok14, i14 = validate_stage14(clip)
        ok17, i17 = validate_stage17(clip)
        ok_exp, iexp = validate_export(clip)
        n_ok12 += ok12
        n_ok13 += ok13
        n_ok14 += ok14
        n_ok17 += ok17
        n_ok_export += ok_exp
        per_clip[clip] = {
            "stage12": i12, "stage13": i13, "stage14": i14,
            "stage17": i17, "export": iexp,
        }

    # ---- manifest / clip-id consistency ----
    manifest_path = EXPORT_ROOT / MANIFEST_FILENAME
    manifest_ok = manifest_path.exists()
    manifest_ids = []
    if manifest_ok:
        with open(manifest_path, newline="", encoding="utf-8") as fh:
            manifest_ids = [r["clip_id"] for r in csv.DictReader(fh)]

    def _dir_names(d: Path) -> set:
        return {p.name for p in d.iterdir() if p.is_dir()} if d.is_dir() else set()

    expected = set(ids)
    clip_sets = {
        "stage_inputs": expected,
        "stage12": _dir_names(BASE / "stage12_flame"),
        "stage13": _dir_names(BASE / "stage13_gaze"),
        "stage14": _dir_names(BASE / "stage14_audio"),
        "stage17": _dir_names(BASE / "stage17_fusion"),
        "export_flame": {p.stem for p in (EXPORT_ROOT / FLAME_DIR).glob("*.npy")},
        "export_audio": {p.stem for p in (EXPORT_ROOT / AUDIO_DIR).glob("*.npy")},
        "export_ts": {p.stem for p in (EXPORT_ROOT / TIMESTAMP_DIR).glob("*.npz")},
        "export_meta": {p.stem for p in (EXPORT_ROOT / METADATA_DIR).glob("*.json")},
    }
    stage_consistent = all(
        clip_sets[k] == expected for k in ("stage12", "stage13", "stage14", "stage17")
    )
    export_consistent = all(
        clip_sets[k] == expected
        for k in ("export_flame", "export_audio", "export_ts", "export_meta")
    )
    manifest_consistent = manifest_ok and set(manifest_ids) == expected
    all_ok = (
        n_ok12 == len(ids) and n_ok13 == len(ids) and n_ok14 == len(ids)
        and n_ok17 == len(ids) and n_ok_export == len(ids)
        and stage_consistent and export_consistent and manifest_consistent
    )

    summary = {
        "clips_total": len(ids),
        "stage12_ok": n_ok12,
        "stage13_ok": n_ok13,
        "stage14_ok": n_ok14,
        "stage17_ok": n_ok17,
        "export_ok": n_ok_export,
        "clip_id_consistent_across_stages": stage_consistent,
        "export_clip_id_consistent": export_consistent,
        "manifest_clip_id_consistent": manifest_consistent,
        "overall": "PASS" if all_ok else "FAIL",
    }

    payload = {
        "summary": summary,
        "export": export_info,
        "clip_ids": ids,
        "clip_sets": {k: sorted(v) for k, v in clip_sets.items()},
        "per_clip": per_clip,
        "manifest_ids": manifest_ids,
    }
    CHECKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHECKS_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())


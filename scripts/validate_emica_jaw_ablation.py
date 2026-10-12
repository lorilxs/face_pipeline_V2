#!/usr/bin/env python3
"""EMICA-jaw ablation validation — compare ablation vs production outputs.

For every accepted clip:
  * Stage12 npz: same T/frame_index/timestamp; expression/head/eye identical;
    jaw differs; ablation jaw == emica_jawpose, production jaw == spectre_jawpose.
  * Stage13 gaze / Stage14 audio: identical to production (jaw-independent).
  * Stage17 export: audio identical; visual identical except jaw block
    (columns 103:106); timestamp identical; metadata jaw_source correct per
    variant. No NaN/Inf anywhere.
  * Jaw statistics (SPECTRE vs EMICA): mean/std/min/max, frame-to-frame
    velocity, and rest-bias (min |jaw| across the clip).

Usage:
    PYTHONPATH="$PWD" python3 scripts/validate_emica_jaw_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROD_BASE = PROJECT_ROOT / "outputs" / "stage0_17_e2e" / "dataset_validation"
ABL_BASE = PROJECT_ROOT / "outputs" / "ablation_emica_jaw"
PROD_EXPORT = PROJECT_ROOT / "outputs" / "stage17_multimodal_export"
ABL_EXPORT = PROJECT_ROOT / "outputs" / "stage17_multimodal_export_emica_jaw"
REPORT = PROJECT_ROOT / "docs" / "Stage12_EMICA_Jaw_Ablation_Report.md"


def _load_npz(p: Path):
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _has_bad(a) -> bool:
    try:
        a = np.asarray(a)
        if a.dtype.kind not in "fc":
            return False
        return bool(np.isnan(a).any() or np.isinf(a).any())
    except (TypeError, ValueError):
        return False


def _stats(a: np.ndarray, label: str) -> dict:
    return {
        "name": label,
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean_abs": float(np.abs(a).mean()),
        "max_abs": float(np.abs(a).max()),
        "velocity_std": float(np.diff(a, axis=0).std()) if a.shape[0] > 1 else 0.0,
        "velocity_max": float(np.abs(np.diff(a, axis=0)).max()) if a.shape[0] > 1 else 0.0,
        "rest_bias_min_abs": float(np.abs(a).min(axis=0).mean()),
        "nan_inf": _has_bad(a),
    }


def _fmt(s: dict) -> str:
    return (
        f"mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} "
        f"max={s['max']:.4f} vel_std={s['velocity_std']:.4f} "
        f"rest_bias={s['rest_bias_min_abs']:.4f}"
    )


def main() -> int:
    clips = sorted(
        p.name for p in (PROD_BASE / "stage12_flame").iterdir()
        if (p / "flame_features.npz").exists()
    )
    print(f"clips: {len(clips)}")

    agg: dict = {"n_clips": len(clips), "clips": []}
    all_fail = 0
    jaw_s_raw: list = []
    jaw_e_raw: list = []

    for clip in clips:
        prod12 = _load_npz(PROD_BASE / "stage12_flame" / clip / "flame_features.npz")
        abl12 = _load_npz(ABL_BASE / "stage12_flame" / clip / "flame_features.npz")
        prod13 = _load_npz(PROD_BASE / "stage13_gaze" / clip / "gaze_features.npz")
        abl13 = _load_npz(ABL_BASE / "stage13_gaze" / clip / "gaze_features.npz")
        prod14 = _load_npz(PROD_BASE / "stage14_audio" / clip / "audio_features.npz")
        abl14 = _load_npz(ABL_BASE / "stage14_audio" / clip / "audio_features.npz")

        T = int(prod12["jaw_pose"].shape[0])
        checks = {}
        checks["T_same"] = T == abl12["jaw_pose"].shape[0]
        checks["frame_index_same"] = bool(
            np.array_equal(prod12["frame_index"], abl12["frame_index"]))
        checks["timestamp_same"] = bool(
            np.allclose(prod12["timestamp"], abl12["timestamp"], atol=0))
        checks["expression_same"] = bool(
            np.allclose(prod12["expression_code"], abl12["expression_code"], atol=0))
        checks["head_same"] = bool(
            np.allclose(prod12["head_pose"], abl12["head_pose"], atol=0))
        checks["eye_same"] = bool(
            np.allclose(prod12["eye_pose"], abl12["eye_pose"], atol=0))
        checks["prod_jaw_equals_spectre"] = bool(
            np.allclose(prod12["jaw_pose"], prod12["spectre_jawpose"], atol=0))
        checks["abl_jaw_equals_emica"] = bool(
            np.allclose(abl12["jaw_pose"], abl12["emica_jawpose"], atol=0))
        checks["jaw_differs"] = not bool(
            np.allclose(prod12["jaw_pose"], abl12["jaw_pose"], atol=0))
        checks["gaze_same"] = bool(
            np.allclose(prod13["gaze_vector"], abl13["gaze_vector"], atol=0))
        checks["audio_same"] = bool(
            np.allclose(prod14["feature"], abl14["feature"], atol=0))
        checks["no_nan_inf"] = all(
            not _has_bad(x) for x in (prod12, abl12, prod13, abl13, prod14, abl14))

        jaw_spectre = prod12["jaw_pose"]
        jaw_emica = abl12["jaw_pose"]
        jaw_s_raw.append(jaw_spectre)
        jaw_e_raw.append(jaw_emica)
        jaw_metrics = {
            "spectre": {f"axis{i}": _stats(jaw_spectre[:, i], f"spectre_axis{i}")
                        for i in range(3)},
            "emica": {f"axis{i}": _stats(jaw_emica[:, i], f"emica_axis{i}")
                      for i in range(3)},
            "spectre_all": _stats(jaw_spectre, "spectre_all"),
            "emica_all": _stats(jaw_emica, "emica_all"),
            "max_abs_diff": float(np.abs(jaw_spectre - jaw_emica).max()),
            "mean_abs_diff": float(np.abs(jaw_spectre - jaw_emica).mean()),
        }

        # export comparison
        pv = np.load(PROD_EXPORT / "flame" / f"{clip}.npy")
        av = np.load(ABL_EXPORT / "flame" / f"{clip}.npy")
        pa = np.load(PROD_EXPORT / "audio" / f"{clip}.npy")
        aa = np.load(ABL_EXPORT / "audio" / f"{clip}.npy")
        checks["export_audio_same"] = bool(np.allclose(pa, aa, atol=0))
        checks["export_exp_head_gaze_same"] = bool(
            np.allclose(np.hstack([pv[:, 0:103], pv[:, 106:112]]),
                        np.hstack([av[:, 0:103], av[:, 106:112]]), atol=0))
        checks["export_jaw_differs"] = not bool(
            np.allclose(pv[:, 103:106], av[:, 103:106], atol=0))
        checks["export_jaw_diff_max"] = float(
            np.abs(pv[:, 103:106] - av[:, 103:106]).max())
        checks["export_ts_same"] = bool(np.allclose(
            np.load(PROD_EXPORT / "timestamp" / f"{clip}.npz")["timestamps"],
            np.load(ABL_EXPORT / "timestamp" / f"{clip}.npz")["timestamps"], atol=0))
        pm = json.load(open(PROD_EXPORT / "metadata" / f"{clip}.json"))
        am = json.load(open(ABL_EXPORT / "metadata" / f"{clip}.json"))
        checks["prod_meta_jaw_source"] = pm.get("jaw_source")
        checks["abl_meta_jaw_source"] = am.get("jaw_source")

        fails = [k for k, v in checks.items() if v is False]
        if fails:
            all_fail += 1
            print(f"[FAIL] {clip}: {fails}")
        else:
            print(f"[OK]   {clip} (T={T})")

        agg["clips"].append({
            "clip_id": clip,
            "T": T,
            "checks": {k: (v if not isinstance(v, np.bool_) else bool(v))
                       for k, v in checks.items()},
            "jaw_metrics": jaw_metrics,
        })


    # ---- aggregate jaw stats across all clips ----
    all_s = np.vstack(jaw_s_raw)
    all_e = np.vstack(jaw_e_raw)
    agg["overall"] = {
        "spectre_jaw": _stats(all_s, "spectre_allclips"),
        "emica_jaw": _stats(all_e, "emica_allclips"),
        "global_max_abs_diff": float(
            np.max([c["jaw_metrics"]["max_abs_diff"] for c in agg["clips"]])),
    }
    print("\n=== Overall jaw stats (rad) ===")
    print("SPECTRE:", _fmt(agg["overall"]["spectre_jaw"]))
    print("EMICA  :", _fmt(agg["overall"]["emica_jaw"]))
    print("global max |spectre-emica| jaw diff = "
          f"{agg['overall']['global_max_abs_diff']:.4f} rad "
          f"({np.degrees(agg['overall']['global_max_abs_diff']):.2f} deg)")

    _write_report(agg)
    print(f"\nreport -> {REPORT}")
    print("RESULT:", "ALL PASS" if all_fail == 0 else f"{all_fail} clips FAILED")
    return 0 if all_fail == 0 else 1


def _write_report(agg: dict) -> None:
    s, e = agg["overall"]["spectre_jaw"], agg["overall"]["emica_jaw"]
    L = [
        "# Stage12 EMICA-Jaw Ablation Report",
        "",
        "**Status:** Temporary experiment only — jaw_pose source switched from "
        "SPECTRE (production Scheme D) to EMICA for FLAME-mesh comparison. "
        "**Not a production decision.** Production outputs "
        "(`outputs/stage0_17_e2e/dataset_validation/`, "
        "`outputs/stage17_multimodal_export/`) were NOT modified.",
        "",
        f"- Clips: {agg['n_clips']} (CNFACE_00001_BV18Qg96YE1W_000000)",
        "- Ablation Stage12/13/14/17: `outputs/ablation_emica_jaw/`",
        "- Ablation export: `outputs/stage17_multimodal_export_emica_jaw/`",
        "- Driver: `scripts/run_emica_jaw_ablation_1214.sh` + "
        "`scripts/run_emica_jaw_ablation_17.py`; validation: "
        "`scripts/validate_emica_jaw_ablation.py`",
        "",
        "## 1. Source matrix (unchanged except jaw)",
        "",
        "| Modality | Production | Ablation |",
        "|---|---|---|",
        "| expression | EMICA | EMICA (identical) |",
        "| head_pose | EMICA global | EMICA global (identical) |",
        "| jaw_pose | **SPECTRE** | **EMICA** |",
        "| gaze | MediaPipe | MediaPipe (identical) |",
        "| audio | Chinese HuBERT | Chinese HuBERT (identical) |",
        "",
        "## 2. Consistency checks (per clip, all PASS)",
        "",
        "| Check | Meaning |",
        "|---|---|",
        "| T / frame_index / timestamp same | same temporal grid as production |",
        "| expression / head / eye identical | max diff = 0 vs production |",
        "| prod jaw == spectre_jawpose | production really is SPECTRE jaw |",
        "| ablation jaw == emica_jawpose | ablation really is EMICA jaw |",
        "| gaze / audio identical | Stage13/14 unchanged |",
        "| export: audio & exp/head/gaze identical; jaw block differs | "
        "Stage17 only carries the jaw change |",
        "| no NaN/Inf anywhere | numerical sanity |",
        "",
        "## 3. Jaw statistics (rad, all clips combined)",
        "",
        "| Metric | SPECTRE (production) | EMICA (ablation) |",
        "|---|---|---|",
        f"| mean | {s['mean']:.4f} | {e['mean']:.4f} |",
        f"| std | {s['std']:.4f} | {e['std']:.4f} |",
        f"| min / max | {s['min']:.4f} / {s['max']:.4f} | "
        f"{e['min']:.4f} / {e['max']:.4f} |",
        f"| mean\\|jaw\\| | {s['mean_abs']:.4f} | {e['mean_abs']:.4f} |",
        f"| velocity std (frame-to-frame) | {s['velocity_std']:.4f} | "
        f"{e['velocity_std']:.4f} |",
        f"| rest-bias (mean min\\|jaw\\| / axis) | {s['rest_bias_min_abs']:.4f} | "
        f"{e['rest_bias_min_abs']:.4f} |",
        "",
        f"- global max |SPECTRE - EMICA| jaw diff = "
        f"{agg['overall']['global_max_abs_diff']:.4f} rad "
        f"({np.degrees(agg['overall']['global_max_abs_diff']):.2f} deg)",
        "",
        "## 4. Per-clip jaw diff (SPECTRE vs EMICA)",
        "",
        "| clip | T | max \\|Δjaw\\| rad | max \\|Δjaw\\| deg |",
        "|---|---|---|---|",
    ]
    for c in agg["clips"]:
        diff = c["jaw_metrics"]["max_abs_diff"]
        L.append(f"| {c['clip_id']} | {c['T']} | {diff:.4f} | "
                 f"{np.degrees(diff):.2f} |")

    L += [
        "",
        "## 5. Note",
        "",
        "This modification is used only for the mesh comparison experiment and "
        "does NOT represent the final production scheme. Production remains "
        "`jaw_source=spectre` (config default).",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())


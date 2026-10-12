#!/usr/bin/env python3
"""Dataset-mode real-data validation for the Stage0-17 chain.

Checks (per accepted clip):
  Stage12: flame_features.npz — expression [T,100], head_pose [T,3], jaw_pose [T,3],
           no NaN/Inf, |jaw_pose| < 1 rad.
  Stage13: gaze_features.npz — [T,6], row norm ~ 1.
  Stage14: audio_features.npz — [T,768], alignment_status == already_aligned,
           fallback audio path used (dataset clip wav).
  Stage17: training_sample.npz — visual [T,112] as expr(0:100)+head(100:103)
           +jaw(103:106)+gaze(106:112), audio [T,768], increasing timestamps.

Usage: python3 scripts/validate_dataset_mode_real_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/stage0_17_e2e/dataset_validation"
RESULTS_JSON = ROOT / "outputs/stage0_17_e2e/stage0_11_results.json"


def _finite(x: np.ndarray) -> bool:
    return bool(np.isfinite(x).all())


def main() -> int:
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    stages = {r["stage"]: r for r in results["stages"]}
    report: dict = {}

    for key in ("Stage 5 (SCRFD detect/crop)", "Stage 10 (InsightFace landmark)",
                "Stage 11 (Affine alignment)"):
        s = stages.get(key, {})
        report[key.split(" (")[0]] = {"status": s.get("status"), "summary": s.get("output_summary")}

    report["skipped_in_dataset_mode"] = [
        s for s in ["Stage 6 (6DRepNet pose)", "Stage 7 (SyncNet A/V sync)",
                    "Stage 7.1 (A/V alignment)", "Stage 8 (HyperIQA quality)",
                    "Stage 9 (Quality fusion/filter)", "Stage 9.5 (Clip export)"]
        if stages.get(s, {}).get("status") == "SKIPPED"
    ]

    clip_meta_files = sorted((BASE / "stage_inputs").glob("*_aligned_faces.json"))
    clip_ids = sorted({p.name.removesuffix("_aligned_faces.json") for p in clip_meta_files})
    report["clips"] = {"count": len(clip_ids), "ids": clip_ids}

    per_clip: dict = {}
    for cid in clip_ids:
        row: dict = {}
        meta = json.loads((BASE / "stage_inputs" / f"{cid}_aligned_faces.json").read_text())
        row["aligned_frames"] = len(meta)
        row["aligned_resolution"] = sorted(
            {tuple(r.get("aligned_resolution", ())) for r in meta})

        p12 = BASE / "stage12_flame" / cid / "flame_features.npz"
        r12: dict = {"ok": False}
        if p12.exists():
            z = np.load(p12, allow_pickle=True)
            exp = np.asarray(z["expression_code"])
            head = np.asarray(z["head_pose"])
            jaw = np.asarray(z["jaw_pose"])
            r12 = {
                "ok": True, "expression": list(exp.shape),
                "head_pose": list(head.shape), "jaw_pose": list(jaw.shape),
                "finite": _finite(exp) and _finite(head) and _finite(jaw),
                "jaw_max_abs_rad": float(np.abs(jaw).max()) if jaw.size else None,
                "jaw_lt_1rad": bool(np.abs(jaw).max() < 1.0) if jaw.size else False,
            }
        row["stage12"] = r12

        p13 = BASE / "stage13_gaze" / cid / "gaze_features.npz"
        r13: dict = {"ok": False}
        g: np.ndarray = np.empty((0, 6))
        g_ts: np.ndarray = np.empty((0,))
        if p13.exists():
            z = np.load(p13, allow_pickle=True)
            g = np.asarray(z["gaze_vector"])
            g_ts = np.asarray(z["timestamp"]) if "timestamp" in z.files else np.arange(len(g)) / 25.0
            norm = np.linalg.norm(g, axis=1)
            r13 = {"ok": True, "shape": list(g.shape), "finite": _finite(g),
                   "norm_mean": float(norm.mean()), "norm_std": float(norm.std())}
        row["stage13"] = r13

        p14 = BASE / "stage14_audio" / cid / "audio_features.npz"
        r14: dict = {"ok": False}
        if p14.exists():
            z = np.load(p14, allow_pickle=True)
            a = np.asarray(z["feature"])
            dim = int(np.asarray(z["dimension"])) if "dimension" in z.files else a.shape[1]
            av = json.loads((BASE / "stage_inputs" / f"{cid}_av_alignment.json").read_text())
            src = z["source_audio"] if "source_audio" in z.files else ""
            if isinstance(src, np.ndarray) and src.ndim == 1 and len(src) == 1:
                src = str(src[0])
            src = str(src)
            fallback = Path(av["aligned_audio_path"]).resolve()
            r14 = {"ok": True, "shape": list(a.shape), "finite": _finite(a),
                   "dimension": dim,
                   "alignment_status": av.get("alignment_status"),
                   "offset_frames": av.get("offset_frames"),
                   "metadata_audio_path": str(fallback),
                   "metadata_audio_exists": fallback.exists(),
                   "source_audio_in_npz": src,
                   "uses_fallback_wav": src.endswith(
                       f"stage_outputs/av_alignment/{cid}/{cid}.wav")}
        row["stage14"] = r14

        # ---------- Stage17 ----------
        p17 = BASE / "stage17_fusion" / cid / "training_sample.npz"
        r17: dict = {"ok": False}
        cross: dict = {}
        if p17.exists():
            z = np.load(p17, allow_pickle=True)
            vis = np.asarray(z["visual_feature"])
            aud = np.asarray(z["audio_feature"])
            ts = np.asarray(z["timestamps"])
            mono = bool(np.all(np.diff(ts) > 0)) if ts.size > 1 else True
            r17 = {
                "ok": True, "visual": list(vis.shape), "audio": list(aud.shape),
                "visual_dim112": vis.shape[1] == 112,
                "audio_dim768": aud.shape[1] == 768,
                "timestamps_monotonic": mono,
                "timestamps_range": [float(ts.min()), float(ts.max())],
                "valid_length": int(np.asarray(z["valid_length"])),
                "audio_match_idx_frac": float(
                    (np.asarray(z["audio_match_idx"]) >= 0).mean()),
                "dropout_frac": float(np.asarray(z["dropout_mask"]).mean()),
            }
            # Cross-check visual_feature layout vs Stage12/Stage13 inputs
            #   expr/head/jaw: exact row-by-row (Stage12 covers every visual frame)
            #   gaze: nearest-timestamp match (Stage13 may drop a few frames;
            #         Stage17 fuses them by timestamp and drops unmatched frames)
            if r12.get("ok") and r13.get("ok"):
                m = min(len(vis), len(exp))
                c_expr = np.abs(vis[:m, 0:100] - exp[:m]).max()
                c_head = np.abs(vis[:m, 100:103] - head[:m]).max()
                c_jaw = np.abs(vis[:m, 103:106] - jaw[:m]).max()
                # nearest-timestamp gaze match (exclude Stage17 dropout rows,
                # i.e. frames where Stage13 produced no gaze and the fusion
                # marked them as excluded from the training sample)
                dropout_arr = np.asarray(z["dropout_mask"])
                diffs = []
                matched = 0
                n_dropout = 0
                for i, t in enumerate(ts[:m]):
                    if bool(dropout_arr[i]):
                        n_dropout += 1
                        continue
                    idx = int(np.argmin(np.abs(g_ts - float(t))))
                    if abs(g_ts[idx] - float(t)) <= 0.06:
                        diffs.append(np.abs(vis[i, 106:112] - g[idx]).max())
                        matched += 1
                gaze_maxdiff = float(max(diffs)) if diffs else None
                cross = {
                    "expr_maxdiff": float(c_expr),
                    "head_maxdiff": float(c_head),
                    "jaw_maxdiff": float(c_jaw),
                    "gaze_maxdiff_nearest_ts": gaze_maxdiff,
                    "gaze_rows_matched": matched,
                    "gaze_dropout_rows": n_dropout,
                    "layout_match": bool(
                        max(c_expr, c_head, c_jaw) < 1e-4
                        and (gaze_maxdiff is not None and gaze_maxdiff < 1e-4)),
                }
        row["stage17"] = r17
        row["stage17_crosscheck"] = cross
        per_clip[cid] = row
    report["per_clip"] = per_clip

    n_ok = sum(1 for r in per_clip.values()
               if r["stage12"]["ok"] and r["stage13"]["ok"]
               and r["stage14"]["ok"] and r["stage17"]["ok"])
    n_layout = sum(1 for r in per_clip.values()
                   if r.get("stage17_crosscheck", {}).get("layout_match"))
    report["summary"] = {
        "clips_total": len(clip_ids),
        "clips_all_stages_ok": n_ok,
        "clips_layout_match": n_layout,
        "overall": "PASS" if n_ok == len(clip_ids) else "PARTIAL/FAIL",
    }
    out = ROOT / "docs" / "dataset_validation_checks.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Checks written -> {out}")
    print(json.dumps(report["summary"], indent=2))
    return 0 if n_ok == len(clip_ids) else 1



if __name__ == "__main__":
    sys.exit(main())

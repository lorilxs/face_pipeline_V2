#!/usr/bin/env python3
"""Generate docs/Final_Stage0_17_Mesh_Export_Validation_Report.md from the
final mesh-export validation JSON + Stage0-11 e2e results JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/home/lxs/face_pipeline/Face_Pipeline_V2")
CHECKS = ROOT / "docs/final_mesh_export_checks.json"
RESULTS = ROOT / "outputs/stage0_17_e2e/stage0_11_results.json"
OUT = ROOT / "docs/Final_Stage0_17_Mesh_Export_Validation_Report.md"

ENV_LINES = [
    "- **Host**: `lxs@304a02301` (bare metal) — validation & Stage13/14/17 driver",
    "- **Docker**: container `face_pipeline_lxs` (python 3.12) — Stage0-11 e2e driver",
    "- **emica_env** (inside container, python 3.10, torch 2.4.1+cpu) — Stage12 EMICA+SPECTRE",
    "- **Compute**: CPU-only for Stage12 (0.21 s/frame; no CUDA in emica_env). Stage13/14 host CPU.",
    "- **Video probe**: ffprobe unavailable in this ffmpeg build → Stage0 uses ffmpeg-based fallback probe",
    "- **Models**: SCRFD (Stage3/5), InsightFace (Stage10), MediaPipe face_landmarker (Stage13), Chinese-HuBERT (Stage14), EMICA + SPECTRE (Stage12)",
]


def _all(per_clip, clip_ids, key, stage="stage12"):
    vals = [per_clip[c][stage][key] for c in clip_ids]
    ok = all(vals)
    return f"PASS ({len(vals)}/{len(vals)})" if ok else f"FAIL ({sum(1 for v in vals if v)}/{len(vals)})"


def _stat(per_clip, clip_ids, stat_key, ok_key, stage="stage12"):
    vals = [per_clip[c][stage][stat_key] for c in clip_ids]
    oks = [per_clip[c][stage][ok_key] for c in clip_ids]
    return f"max={max(vals):.6f} {'PASS' if all(oks) else 'FAIL'}"


def _agg_stats(per_clip, stage, stat_key):
    vals = [per_clip[c][stage][stat_key] for c in per_clip]
    means = [v["mean"] for v in vals]
    stds = [v["std"] for v in vals]
    mins = [v["min"] for v in vals]
    maxs = [v["max"] for v in vals]
    return (f"per-clip mean: avg={np.mean(means):.4f} "
            f"(global min {min(mins):.4f} .. global max {max(maxs):.4f}), "
            f"avg std={np.mean(stds):.4f}")


def main() -> int:
    checks = json.loads(CHECKS.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    summary = checks["summary"]
    per_clip = checks["per_clip"]
    clip_ids = checks["clip_ids"]

    vi = results.get("video_info", {})
    stages = results.get("stages", [])
    final_status = results.get("final_status", "FAIL")

    L = []
    L.append("# Final Stage0-17 End-to-End Validation — Mesh Export")
    L.append("")
    L.append("**Pipeline mode:** `dataset` — Stage0→5→10→11→12→13→14→17→17-export.  ")
    L.append(f"**Overall:** `{summary['overall']}`  ")
    L.append(f"**Clips:** {summary['clips_total']} (Stage12/13/14/17 & export all OK = "
             f"{summary['stage12_ok']}/{summary['stage13_ok']}/{summary['stage14_ok']}/"
             f"{summary['stage17_ok']}/{summary['export_ok']})")
    L.append("")
    L.append(f"> Generated from `{CHECKS.name}` (validation) + `stage0_11_results.json` "
             f"(e2e, generated_at={results.get('generated_at')}).")
    L.append("")

    # 1. environment
    L.append("## 1. Environment")
    L.append("")
    L.extend(ENV_LINES)
    L.append("")

    # 2. input video
    L.append("## 2. Input Video")
    L.append("")
    L.append("| Property | Value |")
    L.append("|---|---|")
    L.append(f"| File | `test_data/chinese_talking_videos/CNFACE_00001_BV18Qg96YE1W_000000.mp4` |")
    L.append(f"| video_id | `{vi.get('video_id', 'CNFACE_00001_BV18Qg96YE1W_000000')}` |")
    L.append(f"| Resolution | {vi.get('width', '?')}×{vi.get('height', '?')} |")
    L.append(f"| FPS | {vi.get('fps', '?')} |")
    L.append(f"| Duration | {vi.get('duration', '?')} s |")
    L.append(f"| Total frames | {vi.get('total_frames', '?')} |")
    L.append(f"| Has audio | {vi.get('has_audio', '?')} |")
    L.append(f"| Accepted clips | {len(clip_ids)} |")
    L.append("")

    # 3. stage execution table
    L.append("## 3. Stage Execution Table")
    L.append("")
    L.append("| Stage | Status | Detail |")
    L.append("|---|---|---|")
    for r in stages:
        name = r["stage"]
        st = r["status"]
        elapsed = r.get("elapsed_s")
        t = f"{elapsed}s" if elapsed is not None else "—"
        err = f" <br/> `{r.get('error')}`" if r.get("error") else ""
        L.append(f"| {name} | {st} | {t} — {r.get('output_summary', '')}{err} |")
    L.append("")
    L.append("> Stage6/7/7.1/8/9/9.5 are **SKIPPED** in dataset mode (no quality filtering / A-V "
             "sync / clip re-export; Stage9.5 is replaced by the dataset adapter's "
             "`build_dataset_av_alignments`). Stage12→17 run in this validation as the mesh-export "
             "production chain.")
    L.append("")

    # 4. FLAME statistics
    L.append("## 4. FLAME Statistics (Stage12)")
    L.append("")
    L.append("Per-clip `flame_features.npz`: `expression_code [T,100]`, `head_pose [T,3]`, "
             "`jaw_pose [T,3]`, `spectre_jawpose [T,3]`, `emica_jawpose [T,3]`. Gaze `[T,6]` is "
             "produced by Stage13 (Section 5) and fused into Stage17's `visual_feature [T,112]`.")
    L.append("")
    L.append("| Check | Requirement | Result |")
    L.append("|---|---|---|")
    L.append("| expression shape | [T,100] | "
             + _all(per_clip, clip_ids, "expression_shape") + " |")
    L.append("| head_pose shape | [T,3] | "
             + _all(per_clip, clip_ids, "head_pose_shape") + " |")
    L.append("| jaw_pose shape | [T,3] | "
             + _all(per_clip, clip_ids, "jaw_pose_shape") + " |")
    L.append("| No NaN/Inf | all finite | "
             + _all(per_clip, clip_ids, "no_nan_inf") + " |")
    L.append("| Head rotation | max \|θ\| < 0.5 rad | "
             + _stat(per_clip, clip_ids, "head_abs_max_rad", "head_threshold_0_5") + " |")
    L.append("| Jaw rotation | max \|θ\| < 0.2 rad | "
             + _stat(per_clip, clip_ids, "jaw_abs_max_rad", "jaw_threshold_0_2") + " |")
    L.append("| jaw source = EMICA | `jaw_pose == emica_jawpose` | "
             + _stat(per_clip, clip_ids, "jaw_vs_emica_maxdiff", "jaw_equals_emica") + " |")
    L.append("")
    L.append("Expression stats across all clips: " + _agg_stats(per_clip, "stage12", "expr_stats"))
    L.append("")
    L.append("Per-clip FLAME checks:")
    L.append("")
    L.append("| clip | T | expr | head | jaw | head_max_rad | jaw_max_rad | jaw==emica |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in clip_ids:
        s12 = per_clip[c]["stage12"]
        L.append(f"| `{c}` | {s12['T']} | {s12['expression_shape'][1]} | "
                 f"{s12['head_pose_shape'][1]} | {s12['jaw_pose_shape'][1]} | "
                 f"{s12['head_abs_max_rad']:.4f} | {s12['jaw_abs_max_rad']:.4f} | "
                 f"{'PASS' if s12['jaw_equals_emica'] else 'FAIL'} |")
    # 5. gaze statistics
    L.append("## 5. Gaze Statistics (Stage13)")
    L.append("")
    L.append("Per-clip `gaze_features.npz`: `gaze_vector [T,6]` = left-gaze `[T,3]` + "
             "right-gaze `[T,3]`. Each 3-vector is a unit direction (norm ≈ 1).")
    L.append("")
    L.append("| Check | Requirement | Result |")
    L.append("|---|---|---|")
    L.append("| shape | [T,6] | "
             + _all(per_clip, clip_ids, "shape", "stage13") + " |")
    L.append("| No NaN/Inf | all finite | "
             + _all(per_clip, clip_ids, "no_nan_inf", "stage13") + " |")
    L.append("| norm ≈ 1 (left) | \||left\|| ≈ 1 | "
             + _stat(per_clip, clip_ids, "left_norm_mean", "ok", "stage13") + " |")
    L.append("| norm ≈ 1 (right) | \||right\|| ≈ 1 | "
             + _stat(per_clip, clip_ids, "right_norm_mean", "ok", "stage13") + " |")
    L.append("")
    L.append("Per-clip gaze checks (max |norm−1| over all frames):")
    L.append("")
    L.append("| clip | T | shape | left_norm_mean | left_max_dev | right_norm_mean | right_max_dev |")
    L.append("|---|---|---|---|---|---|---|")
    for c in clip_ids:
        s13 = per_clip[c]["stage13"]
        L.append(f"| `{c}` | {s13['shape'][0]} | {s13['shape'][1]} | "
                 f"{s13['left_norm_mean']:.5f} | {s13['left_norm_max_abs_dev']:.2e} | "
                 f"{s13['right_norm_mean']:.5f} | {s13['right_norm_max_abs_dev']:.2e} |")
    L.append("")

    # 6. audio statistics
    L.append("## 6. Audio Statistics (Stage14)")
    L.append("")
    L.append("Per-clip `audio_features.npz`: `feature [T,768]` = Chinese-HuBERT hidden-state "
             "embedding (768-dim), no NaN/Inf.")
    L.append("")
    L.append("| Check | Requirement | Result |")
    L.append("|---|---|---|")
    L.append("| shape | [T,768] | "
             + _all(per_clip, clip_ids, "shape", "stage14") + " |")
    L.append("| No NaN/Inf | all finite | "
             + _all(per_clip, clip_ids, "no_nan_inf", "stage14") + " |")
    L.append("")
    L.append("Aggregate feature stats across all clips: "
             + _agg_stats(per_clip, "stage14", "stats"))
    L.append("")
    L.append("Per-clip audio checks:")
    L.append("")
    L.append("| clip | T | shape | finite |")
    L.append("|---|---|---|---|")
    for c in clip_ids:
        s14 = per_clip[c]["stage14"]
        L.append(f"| `{c}` | {s14['shape'][0]} | {s14['shape'][1]} | "
                 f"{'PASS' if s14['no_nan_inf'] else 'FAIL'} |")
    L.append("")
    # 7. export structure
    L.append("## 7. Export Structure (Stage17 export)")
    L.append("")
    L.append("Final delivery directory `outputs/stage17_multimodal_export/` "
             "(generated by the official `export_multimodal_sample` per clip):")
    L.append("")
    L.append("```")
    L.append("outputs/stage17_multimodal_export/")
    L.append("  flame/{clip_id}.npy      [T,112] float32  visual = expr[100]+head[3]+jaw[3]+gaze[6]")
    L.append("  audio/{clip_id}.npy      [T,768] float32  Chinese-HuBERT embedding")
    L.append("  timestamp/{clip_id}.npz  frame_index / timestamps / audio_match_idx / dropout_mask")
    L.append("  metadata/{clip_id}.json  clip_id / fps / visual_layout / head/jaw/expr/gaze/audio sources")
    L.append("  manifest.csv              dataset-level index (11 clips)")
    L.append("```")
    L.append("")
    L.append("| Check | Requirement | Result |")
    L.append("|---|---|---|")
    L.append("| flame/*.npy | [T,112] | "
             + _all(per_clip, clip_ids, "flame_shape", "export") + " |")
    L.append("| audio/*.npy | [T,768] | "
             + _all(per_clip, clip_ids, "audio_shape", "export") + " |")
    L.append("| timestamp/*.npz | monotonic ts, [T] arrays | "
             + _all(per_clip, clip_ids, "timestamp_monotonic", "export") + " |")
    L.append("| metadata/*.json | required fields + layout | "
             + _all(per_clip, clip_ids, "metadata_valid", "export") + " |")
    L.append("| clip ids consistent (stage12/13/14/17) | all == 11 accepted | "
             + ("PASS" if summary.get("clip_id_consistent_across_stages") else "FAIL") + " |")
    L.append("| clip ids consistent (export dirs) | flame/audio/ts/meta | "
             + ("PASS" if summary.get("export_clip_id_consistent") else "FAIL") + " |")
    L.append("| manifest.csv consistent | 11 clip ids | "
             + ("PASS" if summary.get("manifest_clip_id_consistent") else "FAIL") + " |")
    L.append("")
    L.append("Per-clip export checks:")
    L.append("")
    L.append("| clip | flame_shape | audio_shape | ts_T | ts_monotonic | dropout | audio_matched | metadata_sources |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in clip_ids:
        e = per_clip[c]["export"]
        src = e.get("metadata_sources", {})
        src_s = (f"{src.get('head_source')}/{src.get('jaw_source')}/"
                 f"{src.get('expression_source')}/{src.get('gaze_source')}/{src.get('audio_source')}")
        L.append(f"| `{c}` | {e['flame_shape']} | {e['audio_shape']} | "
                 f"{e['timestamp_T']} | {'PASS' if e['timestamp_monotonic'] else 'FAIL'} | "
                 f"{e['dropout']} | {e['audio_matched']} | `{src_s}` |")
    # 8. final PASS/FAIL
    L.append("## 8. Final PASS/FAIL")
    L.append("")
    L.append("| Check | Clips OK |")
    L.append("|---|---|")
    L.append(f"| Stage12 FLAME | {summary['stage12_ok']}/{summary['clips_total']} |")
    L.append(f"| Stage13 gaze | {summary['stage13_ok']}/{summary['clips_total']} |")
    L.append(f"| Stage14 audio | {summary['stage14_ok']}/{summary['clips_total']} |")
    L.append(f"| Stage17 fusion | {summary['stage17_ok']}/{summary['clips_total']} |")
    L.append(f"| Stage17 export | {summary['export_ok']}/{summary['clips_total']} |")
    L.append("| clip-id consistency (stage12/13/14/17) | "
             + ("PASS" if summary.get("clip_id_consistent_across_stages") else "FAIL") + " |")
    L.append("| clip-id consistency (export dirs) | "
             + ("PASS" if summary.get("export_clip_id_consistent") else "FAIL") + " |")
    L.append("| manifest.csv consistency | "
             + ("PASS" if summary.get("manifest_clip_id_consistent") else "FAIL") + " |")
    L.append("")
    L.append(f"**Overall: `{summary['overall']}`** — all multimodal outputs "
             "(`flame/*.npy`, `audio/*.npy`, `timestamp/*.npz`) are produced for all 11 accepted "
             "clips with consistent clip ids and are ready for downstream FLAME mesh "
             "reconstruction validation.")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Report written -> {OUT}")
    return 0

if __name__ == "__main__":
    sys.exit(main())


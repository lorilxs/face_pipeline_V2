#!/usr/bin/env python3
"""Stage12 pose anomaly detection audit (READ-ONLY, host python).

Scans every `flame_features.npz` produced by Stage12 and reports per-clip
statistics for head_pose / jaw_pose / spectre_jawpose, then flags frames whose
axis-angle magnitude exceeds the physical plausibility thresholds:

    head pose norm > 2.0 rad  -> WARNING
    jaw  pose norm > 1.0 rad  -> WARNING

It also back-computes the EMICA-only jaw component from the stored fused jaw
and SPECTRE jaw, so the report shows each source separately.

No pipeline or model code is modified.  Thresholds are hard-coded below
(adjust via CLI flags if needed).

Usage:
    python3 scripts/stage12_pose_audit.py [--glob 'outputs/**/flame_features.npz']
        [--head-threshold 2.0] [--jaw-threshold 1.0]
        [--max-frames-per-clip 5]
        [--json out.json]      # optional machine-readable report

Exit code: 0 = no anomaly, 1 = at least one anomaly found.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_GLOB = "outputs/**/flame_features.npz"
HEAD_ANOMALY_RAD = 2.0   # task spec: head norm > 2 rad
JAW_ANOMALY_RAD = 1.0    # task spec: jaw norm > 1 rad


def _clip_id(path: Path) -> str:
    return path.parent.name


def analyze_clip(path: Path, head_thr: float, jaw_thr: float,
                 max_frames: int) -> dict:
    """Return per-clip statistics + abnormal frames."""
    d = np.load(path, allow_pickle=True)
    T = d["expression_code"].shape[0]

    fields = {}
    for key in ("head_pose", "jaw_pose", "spectre_jawpose", "eye_pose"):
        a = d[key] if key in d.files else None
        if a is None:
            fields[key] = None
            continue
        mag = np.linalg.norm(a, axis=-1)
        fields[key] = {
            "shape": list(a.shape),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "per_axis_mean": [float(v) for v in a.mean(axis=0)],
            "per_axis_std": [float(v) for v in a.std(axis=0)],
            "norm_mean": float(mag.mean()),
            "norm_std": float(mag.std()),
            "norm_min": float(mag.min()),
            "norm_max": float(mag.max()),
        }
    return fields


def _back_out_emica_jaw(d: dict) -> dict | None:
    """Reconstruct EMICA jaw from stored fused jaw and SPECTRE jaw.

    fused = 0.4 * emica_jaw + 0.6 * spectre_jaw  (λ = 0.6 for SPECTRE)
    -> emica_jaw = (fused - 0.6 * spectre_jaw) / 0.4
    Returns None if the raw data cannot support the back-calculation.
    """
    if "jaw_pose" not in d.files or "spectre_jawpose" not in d.files:
        return None
    fused = d["jaw_pose"]
    spj = d["spectre_jawpose"]
    if np.allclose(spj, 0.0):
        return None
    emica = (fused - 0.6 * spj) / 0.4
    mag = np.linalg.norm(emica, axis=-1)
    return {
        "norm_mean": float(mag.mean()),
        "norm_max": float(mag.max()),
        "per_axis_mean": [float(v) for v in emica.mean(axis=0)],
    }


def find_anomalies(path: Path, d: dict, head_thr: float, jaw_thr: float,
                   max_frames: int) -> tuple[int, list]:
    """Return (total abnormal frame count, capped list of abnormal frames)."""
    T = d["expression_code"].shape[0]
    head = d["head_pose"] if "head_pose" in d.files else None
    jaw = d["jaw_pose"] if "jaw_pose" in d.files else None
    ts = d["timestamp"] if "timestamp" in d.files else np.zeros(T)
    fi = d["frame_index"] if "frame_index" in d.files else np.arange(T)
    hm = np.linalg.norm(head, axis=-1) if head is not None else np.zeros(T)
    jm = np.linalg.norm(jaw, axis=-1) if jaw is not None else np.zeros(T)

    n_total = int(np.sum(hm > head_thr)) + int(np.sum(jm > jaw_thr))
    listed: list = []
    for i in range(T):
        why = []
        if hm[i] > head_thr:
            why.append(f"head={hm[i]:.3f}rad>{head_thr}")
        if jm[i] > jaw_thr:
            why.append(f"jaw={jm[i]:.3f}rad>{jaw_thr}")
        if why:
            listed.append({
                "clip_id": _clip_id(path),
                "frame_index": int(fi[i]),
                "timestamp": float(ts[i]),
                "reasons": why,
                "head_pose": None if head is None else [float(v) for v in head[i]],
                "jaw_pose": None if jaw is None else [float(v) for v in jaw[i]],
            })
            if len(listed) >= max_frames:
                break
    return n_total, listed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--head-threshold", type=float, default=HEAD_ANOMALY_RAD)
    ap.add_argument("--jaw-threshold", type=float, default=JAW_ANOMALY_RAD)
    ap.add_argument("--max-frames-per-clip", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob, recursive=True))
    if not paths:
        print(f"No npz matched {args.glob}", file=sys.stderr)
        return 2

    print("=" * 88)
    print("Stage12 pose anomaly audit (READ-ONLY)")
    print(f"thresholds: head > {args.head_threshold} rad, jaw > {args.jaw_threshold} rad")
    print("=" * 88)

    reports = []
    any_anomaly = False
    for p in paths:
        d = np.load(p, allow_pickle=True)
        fields = analyze_clip(Path(p), args.head_threshold, args.jaw_threshold,
                              args.max_frames_per_clip)
        emica_jaw = _back_out_emica_jaw(d)
        n_anom, anomalies = find_anomalies(Path(p), d, args.head_threshold,
                                           args.jaw_threshold,
                                           args.max_frames_per_clip)
        if n_anom > 0:
            any_anomaly = True

        print(f"\n--- {_clip_id(Path(p))}  ({d['expression_code'].shape[0]} frames) ---")
        for k, f in fields.items():
            if f is None:
                print(f"  {k:18s}: <missing>")
                continue
            print(f"  {k:18s}: shape={f['shape']} "
                  f"mean={f['mean']:+.4f} std={f['std']:.4f} "
                  f"min={f['min']:+.4f} max={f['max']:+.4f}")
            if f["shape"][-1] > 1:
                print(f"    norm: mean={f['norm_mean']:.4f} rad "
                      f"({np.degrees(f['norm_mean']):.1f} deg) "
                      f"max={f['norm_max']:.4f} rad "
                      f"({np.degrees(f['norm_max']):.1f} deg)")
                print(f"    per-axis mean={[round(v, 4) for v in f['per_axis_mean']]} "
                      f"std={[round(v, 4) for v in f['per_axis_std']]}")
        if emica_jaw:
            print(f"  emica_jaw(inferred): norm_mean={emica_jaw['norm_mean']:.4f} rad "
                  f"({np.degrees(emica_jaw['norm_mean']):.1f} deg) "
                  f"per-axis mean={[round(v, 4) for v in emica_jaw['per_axis_mean']]}")

        if anomalies:
            print(f"  *** {n_anom} abnormal frame(s) "
                  f"(showing first {len(anomalies)}): ***")
            for a in anomalies:
                print(f"    clip={a['clip_id']} frame={a['frame_index']} "
                      f"t={a['timestamp']:.3f}s  {', '.join(a['reasons'])}")

        reports.append({
            "npz": str(p),
            "clip_id": _clip_id(Path(p)),
            "frames": int(d["expression_code"].shape[0]),
            "fields": fields,
            "emica_jaw_inferred": emica_jaw,
            "n_anomalies_total": n_anom,
            "anomalies": anomalies,
        })

    summary = {
        "head_threshold_rad": args.head_threshold,
        "jaw_threshold_rad": args.jaw_threshold,
        "clips": reports,
        "any_anomaly": any_anomaly,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2),
                                   encoding="utf-8")
        print(f"\nJSON report written to {args.json}")

    print("\n" + "=" * 88)
    print(f"RESULT: {'ANOMALY FOUND (exit 1)' if any_anomaly else 'NO ANOMALY (exit 0)'}")
    return 1 if any_anomaly else 0


if __name__ == "__main__":
    sys.exit(main())


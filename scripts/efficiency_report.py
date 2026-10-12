#!/usr/bin/env python3
"""Produce a pipeline efficiency report for one output batch.

Reads every ``pipeline_manifest.json`` under a batch directory and reports
throughput / FLAME-output metrics, both at the actually-observed wall-clock
level and re-normalised to an N-way-parallel run.

Time accounting (all wall-clock, in seconds):

  * ``phase1`` (stage 00..11) -- *measured* per stage: sum of
    ``stages[].elapsed_s`` for every phase-1 stage.
  * ``phase2`` (stage 12 FLAME / 13 gaze / 14 HuBERT / 17 assembly) -- the
    pipeline manifest does NOT carry phase-2 timings, so it is reconstructed
    from two *written-at* timestamps on disk:
        phase2 = stage17_multimodal_export/export_report.json.generated_at
                 - pipeline_manifest.json.generated_at
    ``manifest.generated_at`` is written when phase 1 finishes; the export
    report is written when stage 17 finishes, so the delta is the real
    phase-2 processing time for that video (no queueing/merge wait).
    Videos without an export report fall back to the batch median phase-2
    duration.

Two throughput views are reported:

  * *Observed* : source hours / wall-clock span (earliest ``generated_at`` ..
    latest ``updated_at``) -- includes whatever parallelism the run had.
  * *N-way*   : source hours / (total machine time / N). Total machine time
    is phase1 + phase2 as defined above. This is the theoretical throughput
    if the same work were spread over N workers (linear-scaling assumption).

FLAME output is counted from ``stage_details.stage_12``
(``num_frames_total`` = valid FLAME frames, ``clip_count`` = clips).

Standard library only, so it runs on the host (no docker) and never touches
the pipeline itself.

Usage:
    python3 scripts/efficiency_report.py --batch outputs/flashtalk_batch_030000
    python3 scripts/efficiency_report.py --batch outputs/douyin_videos_hp_0916 --workers 16
    python3 scripts/efficiency_report.py --batch outputs/douyin_videos_hp_0916 --workers 16 --json
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

def _ts(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp (tolerating a trailing 'Z') or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _v(d: Optional[dict], *keys: str) -> Any:
    """Descend a dict chain, returning None if any level is missing."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def collect_batch(batch_dir: Path) -> Dict[str, Any]:
    """Scan one batch directory and aggregate all efficiency metrics.

    Returns a flat dict of aggregate numbers (all times in seconds).
    """
    if not batch_dir.is_dir():
        raise SystemExit(f"batch dir not found: {batch_dir}")

    video_dirs = sorted(
        d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("video_")
    )
    if not video_dirs:
        raise SystemExit(f"no video_* dirs found under {batch_dir}")

    phase1_total = 0.0          # sum of phase-1 elapsed_s (measured)
    phase2_real = []            # export_report.generated_at - manifest.generated_at
    phase2_n_with_report = 0
    src_total = 0.0             # source video seconds
    flame_frames = 0
    clip_count = 0
    n_manifest = 0
    n_final_pass = 0
    ga_all: List[datetime] = []   # manifest.generated_at
    ua_all: List[datetime] = []   # manifest.updated_at
    stage3_rejected = 0

    for vd in video_dirs:
        manifest = _load(vd / "pipeline_manifest.json")
        if manifest is None:
            continue
        n_manifest += 1

        # final status (fall back to phase-2 completion flags)
        final = manifest.get("final_status")
        if final is None:
            final = "PASS" if (
                manifest.get("_phase1_complete") and manifest.get("_phase2_complete")
            ) else "FAIL"
        if str(final).upper() == "PASS":
            n_final_pass += 1

        # ---- source duration (stage_00) ----
        s00 = _v(manifest, "stage_details", "stage_00") or {}
        dur = s00.get("duration") or s00.get("duration_s") or 0
        src_total += float(dur)

        # ---- phase 1 machine time: sum phase-1 stage elapsed_s ----
        stages = manifest.get("stages") or {}
        for _, sv in stages.items():
            if isinstance(sv, dict) and sv.get("phase") == 1:
                es = sv.get("elapsed_s")
                if es:
                    phase1_total += float(es)

        # ---- stage3 rejection (coarse screen) ----
        # douyin stores scene-level rejection counts; dataset stores status.
        s3 = _v(manifest, "stage_details", "stage_03") or {}
        rc = s3.get("rejected_count")
        if isinstance(rc, (int, float)):
            stage3_rejected += int(rc)
        elif str(s3.get("status") or "").lower() == "rejected":
            stage3_rejected += 1

        # ---- FLAME output (stage_12) ----
        s12 = _v(manifest, "stage_details", "stage_12") or {}
        flame_frames += float(s12.get("num_frames_total") or 0)
        clip_count += float(s12.get("clip_count") or 0)

        # ---- phase 2 real time from export_report.generated_at ----
        ga = _ts(manifest.get("generated_at"))
        if ga:
            ga_all.append(ga)
        ua = _ts(manifest.get("updated_at"))
        if ua:
            ua_all.append(ua)

        er = _load(vd / "stage17_multimodal_export" / "export_report.json")
        er_ts = _ts(er.get("generated_at") if er else None)
        if ga and er_ts and er_ts >= ga:
            phase2_real.append((er_ts - ga).total_seconds())
            phase2_n_with_report += 1


    # ---- phase-2 median fallback for videos without export report ----
    phase2_median = statistics.median(phase2_real) if phase2_real else 0.0
    missing_report = n_manifest - phase2_n_with_report
    phase2_total = float(sum(phase2_real)) + phase2_median * missing_report

    # ---- wall-clock span (earliest generated_at .. latest updated_at) ----
    wall_span = 0.0
    if ga_all and ua_all:
        wall_span = (max(ua_all) - min(ga_all)).total_seconds()

    total_machine = phase1_total + phase2_total
    return {
        "batch_dir": str(batch_dir),
        "n_video_dirs": len(video_dirs),
        "n_manifest": n_manifest,
        "n_final_pass": n_final_pass,
        "pass_rate": (n_final_pass / n_manifest * 100) if n_manifest else 0.0,
        "stage3_rejected": stage3_rejected,
        "src_seconds": src_total,
        "phase1_seconds": phase1_total,
        "phase2_seconds": phase2_total,
        "phase2_seconds_real": float(sum(phase2_real)),
        "phase2_n_with_report": phase2_n_with_report,
        "phase2_median_s": phase2_median,
        "wall_span_seconds": wall_span,
        "total_machine_seconds": total_machine,
        "flame_frames": flame_frames,
        "clip_count": clip_count,
    }


def render(agg: Dict[str, Any], workers: int) -> str:
    """Render the aggregate dict as a human-readable text report."""
    def h(s: float) -> str:
        return f"{s / 3600:.2f} h"

    n = agg["n_manifest"]
    src_h = agg["src_seconds"] / 3600
    p1_h = agg["phase1_seconds"] / 3600
    p2_h = agg["phase2_seconds"] / 3600
    tot_h = agg["total_machine_seconds"] / 3600
    wall_h = agg["wall_span_seconds"] / 3600

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"Pipeline efficiency report - {agg['batch_dir']}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("--- Batch size / acceptance ---")
    lines.append(f"  video dirs        : {agg['n_video_dirs']}")
    lines.append(f"  manifests         : {agg['n_manifest']}")
    lines.append(f"  final PASS        : {agg['n_final_pass']}  ({agg['pass_rate']:.1f}%)")
    lines.append(f"  stage03 rejected  : {agg['stage3_rejected']}")
    lines.append("")
    lines.append("--- Input ---")
    if n:
        lines.append(f"  source video      : {src_h:.2f} h  (avg {agg['src_seconds'] / n:.1f} s/video)")
    lines.append("")
    lines.append("--- Machine time (wall-clock) ---")
    if src_h:
        lines.append(f"  phase1 (measured) : {p1_h:.2f} h   ({p1_h / src_h:.2f} h per source-h)")
        lines.append(f"  phase2 (real)     : {p2_h:.2f} h   ({p2_h / src_h:.2f} h per source-h)"
                     f"  [median {agg['phase2_median_s']:.1f} s/video, "
                     f"{agg['phase2_n_with_report']}/{n} via export_report]")
    lines.append(f"  total machine     : {tot_h:.2f} h")
    lines.append(f"  wall-clock span   : {wall_h:.2f} h   (earliest generated_at .. latest updated_at)")
    if wall_h:
        lines.append(f"  observed parall.  : {tot_h / wall_h:.2f} workers")
    lines.append("")
    lines.append("--- FLAME output ---")
    lines.append(f"  FLAME frames      : {agg['flame_frames']:,.0f}")
    lines.append(f"  clips             : {agg['clip_count']:,.0f}")
    if src_h:
        lines.append(f"  frames per src-h  : {agg['flame_frames'] / src_h:,.0f}")
    lines.append("")
    lines.append("--- Throughput ---")
    if wall_h:
        lines.append("  [Observed -- whatever parallelism the run actually had]")
        lines.append(f"    source / hour    : {agg['src_seconds'] / wall_h / 60:.1f} min   ({src_h / wall_h:.2f}x realtime)")
        lines.append(f"    FLAME frames / h : {agg['flame_frames'] / wall_h:,.0f}")
        lines.append(f"    clips / hour     : {agg['clip_count'] / wall_h:,.0f}")
    lines.append(f"  [{workers}-way parallel -- total machine time / {workers}]")
    w_h = agg["total_machine_seconds"] / workers / 3600
    lines.append(f"    wall-clock       : {w_h:.2f} h")
    lines.append(f"    source / hour    : {agg['src_seconds'] / w_h / 60:.1f} min   ({agg['src_seconds'] / w_h / 3600:.2f}x realtime)")
    lines.append(f"    FLAME frames / h : {agg['flame_frames'] / w_h:,.0f}")
    lines.append(f"    clips / hour     : {agg['clip_count'] / w_h:,.0f}")
    lines.append("")
    lines.append("One-liner:")
    lines.append(f"  {workers}-way parallel: ~{w_h:.1f} h to digest {src_h:.1f} h of source "
                 f"({agg['src_seconds'] / w_h / 60:.0f} min/h), producing ~{agg['flame_frames'] / w_h:,.0f} FLAME frames/h "
                 f"(~{agg['clip_count'] / w_h:,.0f} clips/h).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, type=Path,
                    help="path to a batch output dir containing video_* subdirs")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallelism to re-normalise to (default 16)")
    ap.add_argument("--json", action="store_true",
                    help="also dump the raw aggregate as JSON on stdout")
    args = ap.parse_args()

    agg = collect_batch(args.batch)
    print(render(agg, args.workers))
    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()


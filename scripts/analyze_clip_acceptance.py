#!/usr/bin/env python3
"""Clip-level acceptance / rejection statistics for one batch.

Reads the flat clip-level CSV produced by ``scripts/generate_manifest_summary.py``
(one row == one clip/scene) and aggregates:

  * clip-level funnel: stage3 scene pass -> stage9 clip accept -> final PASS
  * stage3 reject-reason distribution + parameter histograms for rejected scenes
    (face_detect_ratio / single_face_ratio / avg_face_confidence / max_face_count)
  * stage9 reject-reason distribution + score histograms for rejected clips
    (pose / sync / quality / fusion)
  * other clip-stage reject reasons (stage5.2/5.3/8, stage12/13/14/17)

Usage:
    python3 scripts/analyze_clip_acceptance.py \\
        --csv outputs/douyin_videos_hp_0916/pipeline_manifest_summary_*.csv
    python3 scripts/analyze_clip_acceptance.py \\
        --csv ... --out logs/douyin_clip_acceptance.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RATIO_BINS = [(-0.0001, 0.0), (0.0001, 0.25), (0.2501, 0.5),
              (0.5001, 0.75), (0.7501, 0.9999), (1.0, 1.0)]
RATIO_LABELS = ["=0", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-<1", "=1"]
SCORE_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
SCORE_LABELS = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def _f(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _hist(vals: List[float], bins, labels) -> List[int]:
    counts = [0] * len(bins)
    for v in vals:
        for i, (lo, hi) in enumerate(bins):
            if lo <= v <= hi:
                counts[i] += 1
                break
    return counts

def _res_bucket(w: Optional[float], h: Optional[float]) -> str:
    """Resolution bucket by the smaller side (what stage0.5 actually gates on)."""
    if w is None or h is None:
        return "unknown"
    m = min(w, h)
    if m < 360:
        return "<360"
    if m < 480:
        return "360-479"
    if m < 720:
        return "480-719"
    if m < 1080:
        return "720-1079"
    return ">=1080"


def _fmt_dur(seconds: Optional[float]) -> str:
    """Format a duration in seconds as a compact, human-readable string."""
    if seconds is None:
        return "n/a"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


RES_BUCKET_ORDER = ["<360", "360-479", "480-719", "720-1079", ">=1080", "unknown"]


def aggregate(csv_path: Path, min_width: int = 360, min_height: int = 360) -> Dict:
    s3_rej = Counter(); s3_det = []; s3_single = []; s3_conf = []; s3_maxface = Counter()
    s9_rej = Counter(); s9_pose = []; s9_sync = []; s9_qual = []; s9_fus = []
    s52_rej = Counter(); s53_rej = Counter(); s8_rej = Counter()
    s12_rej = Counter(); s13_rej = Counter(); s14_rej = Counter(); s17_rej = Counter()
    # resolution dimension
    res_vid_status = Counter()      # (bucket, final_status) -> n videos
    res_vid_n = Counter()           # bucket -> n videos
    res_rej_s00_5 = []              # (video_id, width, height) videos under the resolution gate
    res_clip_total = Counter()      # bucket -> clips evaluated
    res_clip_acc = Counter()        # bucket -> stage9 accepted clips
    res_clip_ok = Counter()         # bucket -> stage17 export-OK clips
    # stage5 face-resolution gate (video-level; one decision per video)
    s5res_pass = 0
    s5res_fail = 0
    s5res_rej = Counter()
    s5res_short_min = []            # per-video min face short side (px), gated videos only
    # clip-level first-failure funnel (a clip stops at its earliest failed stage)
    s52_n = s53_n = s8_n = s9_n = s17_n = 0
    n_scene = n_s3pass = n_clip = n_s9acc = n_s9rej = n_s17ok = n_s17fail = 0
    n_s9_reached = 0
    vid_final = Counter()          # video-level final_status (dedup by video_id)
    seen_vids = set()
    # duration dimension
    input_dur = 0.0                 # total input video duration (unique videos)
    clip_total_dur = 0.0            # total clip duration (all produced clips)
    s9acc_dur = 0.0                 # stage9 accepted clip duration
    s9rej_dur = 0.0                 # stage9 rejected clip duration
    exp_ok_dur = 0.0                # stage17 export-OK clip duration
    vid_dur_status = Counter()      # final_status -> total input duration (unique videos)

    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("scene_id"):
                n_scene += 1
                if row.get("stage_03__passed") == "1":
                    n_s3pass += 1
                else:
                    for rr in json.loads(row.get("stage_03__reject_reasons") or "[]"):
                        s3_rej[rr] += 1
                    d = _f(row.get("stage_03__face_detect_ratio"))
                    if d is not None: s3_det.append(d)
                    s = _f(row.get("stage_03__single_face_ratio"))
                    if s is not None: s3_single.append(s)
                    c = _f(row.get("stage_03__avg_face_confidence"))
                    if c is not None: s3_conf.append(c)
                    m = row.get("stage_03__max_face_count")
                    if m not in ("", None): s3_maxface[m] += 1

            if row.get("clip_id"):
                n_clip += 1
                # resolution dimension (clip-level: width/height are per-video, repeated on every row)
                _bk = _res_bucket(_f(row.get("width")), _f(row.get("height")))
                res_clip_total[_bk] += 1
                if (row.get("stage_09__status") or "") == "passed":
                    res_clip_acc[_bk] += 1
                    if (row.get("stage_17__status") or "") == "passed":
                        res_clip_ok[_bk] += 1
                s9st = row.get("stage_09__status") or ""
                s17st = row.get("stage_17__status") or ""
                # stage5.2 / 5.3 / 8 reject reasons (parse JSON arrays of reasons)
                for cname, ctr in [("stage_05_2__reasons", s52_rej),
                                   ("stage_05_3__reasons", s53_rej)]:
                    v = row.get(cname)
                    if v not in ("", None):
                        try:
                            for rr in json.loads(v):
                                ctr[str(rr)] += 1
                        except (json.JSONDecodeError, TypeError):
                            ctr[str(v)] += 1
                v8 = row.get("stage_08__status")
                if v8 not in ("", None):
                    s8_rej[str(v8)] += 1
                for cname, ctr in [("stage_12__reason", s12_rej),
                                   ("stage_13__reason", s13_rej),
                                   ("stage_14__reason", s14_rej),
                                   ("stage_17__failed_reason", s17_rej)]:
                    v = row.get(cname)
                    if v not in ("", None):
                        # normalize full-path error messages to their stable prefix
                        key = str(v)
                        if cname == "stage_17__failed_reason" and key.startswith("Stage14 audio read failed:"):
                            key = "Stage14 audio read failed"
                        ctr[key] += 1
                # clip-level funnel: earliest failing stage
                if (row.get("stage_05_2__status") or "") == "rejected":
                    s52_n += 1
                elif (row.get("stage_05_3__status") or "") == "rejected":
                    s53_n += 1
                elif (row.get("stage_08__status") or "") == "rejected":
                    s8_n += 1
                elif s9st == "rejected":
                    n_s9rej += 1
                elif s9st == "passed":
                    n_s9acc += 1
                    if s17st == "passed":
                        n_s17ok += 1
                    elif s17st == "failed":
                        n_s17fail += 1
                    else:
                        n_s17ok += 0  # phase-2 not merged -> count as stage17 pending

                # duration dimension (clip-level; clip_id is unique per clip)
                clip_d = _f(row.get("clip_duration_s"))
                if clip_d is None:
                    clip_d = _f(row.get("scene_duration_s"))
                if clip_d is not None:
                    clip_total_dur += clip_d
                    if s9st == "passed":
                        s9acc_dur += clip_d
                        if s17st == "passed":
                            exp_ok_dur += clip_d
                    elif s9st == "rejected":
                        s9rej_dur += clip_d

                # stage9 distributions only for clips that actually reached stage9
                if s9st == "rejected":
                    n_s9_reached += 1
                    for rr in json.loads(row.get("stage_09__reject_reasons") or "[]"):
                        s9_rej[rr] += 1
                    for name, lst in [("stage_09__pose_score", s9_pose),
                                      ("stage_09__sync_score", s9_sync),
                                      ("stage_09__quality_score", s9_qual),
                                      ("stage_09__fusion_score", s9_fus)]:
                        v = _f(row.get(name))
                        if v is not None: lst.append(v)
                elif s9st == "passed":
                    n_s9_reached += 1

            vid = row.get("video_id")
            if vid and vid not in seen_vids:
                seen_vids.add(vid)
                fs = row.get("final_status") or "<empty>"
                vid_final[fs] += 1
                # duration dimension (video-level, dedup by video_id)
                inp_d = _f(row.get("duration_s"))
                if inp_d is not None:
                    input_dur += inp_d
                    vid_dur_status[fs] += inp_d
                # resolution dimension (video-level, dedup by video_id)
                w = _f(row.get("width")); h = _f(row.get("height"))
                bk = _res_bucket(w, h)
                res_vid_n[bk] += 1
                res_vid_status[(bk, fs)] += 1
                if w is not None and h is not None and (w < min_width or h < min_height):
                    res_rej_s00_5.append((vid, int(w), int(h)))
                # stage5 face-resolution gate (video-level, dedup by video_id)
                s5res_st = row.get("stage_05_face_res__status") or ""
                if s5res_st == "passed":
                    s5res_pass += 1
                elif s5res_st == "rejected":
                    s5res_fail += 1
                    s5res_rej[row.get("stage_05_face_res__reject_reason") or "unknown"] += 1
                if s5res_st in ("passed", "rejected"):
                    smin = _f(row.get("stage_05_face_res__short_side_min"))
                    if smin is not None:
                        s5res_short_min.append(smin)

    n_s9_total = n_s9acc + n_s9rej
    n_s9_early = n_clip - n_s9_total            # stopped before stage9
    return dict(scene=n_scene, s3pass=n_s3pass, clip=n_clip,
                s52_n=s52_n, s53_n=s53_n, s8_n=s8_n,
                s9_total=n_s9_total, s9acc=n_s9acc, s9rej=n_s9rej, s9_early=n_s9_early,
                s17ok=n_s17ok, s17fail=n_s17fail,
                vid_final=vid_final, n_vid=len(seen_vids),
                s3_rej=s3_rej, s3_det=s3_det, s3_single=s3_single, s3_conf=s3_conf,
                s3_maxface=s3_maxface, s9_rej=s9_rej, s9_pose=s9_pose, s9_sync=s9_sync,
                s9_qual=s9_qual, s9_fus=s9_fus, s52_rej=s52_rej, s53_rej=s53_rej,
                s8_rej=s8_rej, s12_rej=s12_rej, s13_rej=s13_rej, s14_rej=s14_rej,
                s17_rej=s17_rej,
                s5res_pass=s5res_pass, s5res_fail=s5res_fail, s5res_rej=s5res_rej,
                s5res_short_min=s5res_short_min,
                min_width=min_width, min_height=min_height,
                res_vid_status=res_vid_status, res_vid_n=res_vid_n,
                res_rej_s00_5=res_rej_s00_5,
                res_clip_total=res_clip_total, res_clip_acc=res_clip_acc,
                res_clip_ok=res_clip_ok,
                input_dur=input_dur, clip_total_dur=clip_total_dur,
                s9acc_dur=s9acc_dur, s9rej_dur=s9rej_dur, exp_ok_dur=exp_ok_dur,
                vid_dur_status=vid_dur_status)


def render(a: Dict) -> str:
    L: List[str] = []
    L.append("=" * 72)
    L.append("CLIP-level acceptance / rejection report")
    L.append("=" * 72)
    scene, s3pass, clip = a["scene"], a["s3pass"], a["clip"]
    s9acc, s9rej, s9_total, s9_early = a["s9acc"], a["s9rej"], a["s9_total"], a["s9_early"]
    n_rej_s3 = scene - s3pass

    L.append("")
    L.append("--- Clip-level funnel (1 clip per scene; earliest failing stage) ---")
    L.append(f"  scenes total            : {scene:,}")
    L.append(f"  stage3 rejected         : {n_rej_s3:,}  ({n_rej_s3/scene*100:.2f}%)  -> no clip")
    L.append(f"  clips produced          : {clip:,}  (stage3 passed {s3pass:,}, {s3pass/scene*100:.2f}%)")
    L.append(f"    stop at stage 5.2     : {a['s52_n']:,}  ({a['s52_n']/clip*100:.2f}% of clips)")
    L.append(f"    stop at stage 5.3     : {a['s53_n']:,}  ({a['s53_n']/clip*100:.2f}% of clips)")
    L.append(f"    stop at stage 8       : {a['s8_n']:,}  ({a['s8_n']/clip*100:.2f}% of clips)")
    L.append(f"    stage9 rejected       : {s9rej:,}  ({s9rej/clip*100:.2f}% of clips)")
    L.append(f"    stage9 accepted       : {s9acc:,}  ({s9acc/clip*100:.2f}% of clips)")
    L.append(f"  reached stage 9         : {s9_total:,}  ({s9_total/clip*100:.2f}% of clips)")
    if s9_total:
        L.append(f"  stage9 accept rate      : {s9acc/s9_total*100:.2f}%  (of clips that reached stage9)")
        L.append(f"  stage9 reject rate      : {s9rej/s9_total*100:.2f}%")
    L.append(f"  clip final export OK    : {a['s17ok']:,}  / stage17 failed {a['s17fail']:,}")
    L.append(f"  clip end-to-end pass    : {a['s17ok']/clip*100:.2f}%  (of all clips)")

    L.append("")
    L.append("--- Stage 5 face-resolution gate (video-level) ---")
    s5_total = a["s5res_pass"] + a["s5res_fail"]
    if s5_total:
        L.append(f"  videos evaluated       : {s5_total:,}")
        L.append(f"  gate passed            : {a['s5res_pass']:,}  ({a['s5res_pass']/s5_total*100:.2f}%)")
        L.append(f"  gate rejected          : {a['s5res_fail']:,}  ({a['s5res_fail']/s5_total*100:.2f}%)")
        for k, v in a["s5res_rej"].most_common():
            pct = v / a["s5res_fail"] * 100 if a["s5res_fail"] else 0.0
            L.append(f"    {str(k)[:44]:44s}: {v:5d}  ({pct:5.2f}% of rejected videos)")
    else:
        L.append("  (gate did not run / no manifest carried a decision)")
    if a["s5res_short_min"]:
        L.append(f"  face short-side min (px): median={statistics.median(a['s5res_short_min']):.1f} "
                 f"mean={statistics.mean(a['s5res_short_min']):.1f} "
                 f"min={min(a['s5res_short_min']):.1f} max={max(a['s5res_short_min']):.1f} "
                 f"n={len(a['s5res_short_min']):,}")

    L.append("")
    L.append("--- Stage 3 reject reasons (scene-level, n=%d) ---" % n_rej_s3)
    for k, v in a["s3_rej"].most_common():
        pct = v / n_rej_s3 * 100 if n_rej_s3 else 0.0
        L.append(f"  {k:20s}: {v:6d}  ({pct:5.2f}% of rejected scenes)")

    L.append("")
    L.append("--- Stage 3 rejected scenes: parameter distribution ---")
    det = _hist(a["s3_det"], RATIO_BINS, RATIO_LABELS)
    sin = _hist(a["s3_single"], RATIO_BINS, RATIO_LABELS)
    L.append("  face_detect_ratio (frames with >=1 face / sampled):")
    for lab, c in zip(RATIO_LABELS, det):
        pct = c / len(a["s3_det"]) * 100 if a["s3_det"] else 0.0
        L.append(f"    {lab:10s}: {c:6d}  ({pct:5.1f}%)")
    L.append("  single_face_ratio (frames with exactly 1 face / sampled):")
    for lab, c in zip(RATIO_LABELS, sin):
        pct = c / len(a["s3_single"]) * 100 if a["s3_single"] else 0.0
        L.append(f"    {lab:10s}: {c:6d}  ({pct:5.1f}%)")
    if a["s3_conf"]:
        L.append(f"  avg_face_confidence : median={statistics.median(a['s3_conf']):.3f} "
                 f"mean={statistics.mean(a['s3_conf']):.3f} min={min(a['s3_conf']):.3f} "
                 f"max={max(a['s3_conf']):.3f}  n={len(a['s3_conf']):,}")
    L.append("  max_face_count in a frame (rejected scenes):")
    for k, v in sorted(a["s3_maxface"].items(), key=lambda x: int(x[0])):
        pct = v / n_rej_s3 * 100 if n_rej_s3 else 0.0
        L.append(f"    {k:>3s}: {v:6d}  ({pct:5.2f}%)")

    L.append("")
    L.append("--- Stage 9 reject reasons (clip-level, n=%d rejected) ---" % s9rej)
    for k, v in a["s9_rej"].most_common():
        pct = v / s9rej * 100 if s9rej else 0.0
        L.append(f"  {k:28s}: {v:6d}  ({pct:5.2f}% of rejected clips)")

    L.append("")
    L.append("--- Stage 9 rejected clips: score distribution ---")
    for title, vals in [("pose_score", a["s9_pose"]), ("sync_score", a["s9_sync"]),
                        ("quality_score", a["s9_qual"]), ("fusion_score", a["s9_fus"])]:
        if not vals:
            continue
        L.append(f"  {title}:")
        h = _hist(vals, SCORE_BINS, SCORE_LABELS)
        for lab, c in zip(SCORE_LABELS, h):
            pct = c / len(vals) * 100
            L.append(f"    {lab:8s}: {c:6d}  ({pct:5.1f}%)")
        L.append(f"    {'median':8s}: {statistics.median(vals):.3f}   "
                 f"mean={statistics.mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals):,}")

    L.append("")
    L.append("--- Other clip-stage reject reasons ---")
    for title, ctr in [("stage_05_2 single-face", a["s52_rej"]),
                       ("stage_05_3 temporal-stab", a["s53_rej"]),
                       ("stage_12 FLAME", a["s12_rej"]),
                       ("stage_13 gaze", a["s13_rej"]),
                       ("stage_14 audio", a["s14_rej"]),
                       ("stage_17 export", a["s17_rej"])]:
        if ctr:
            L.append(f"  {title}:")
            for k, v in ctr.most_common(12):
                L.append(f"    {str(k)[:52]:52s}: {v:5d}")

    L.append("")
    L.append("--- Video-level final_status (dedup by video_id, n=%d) ---" % a["n_vid"])
    for k, v in a["vid_final"].most_common():
        L.append(f"  {str(k):12s}: {v:6d}  ({v/a['n_vid']*100:5.2f}%)")

    L.append("")
    L.append("--- Resolution dimension ---")
    L.append(f"  stage0.5 gate: min_width={a['min_width']} min_height={a['min_height']} "
             "(videos with a smaller side below this are rejected before stage3)")
    L.append(f"  videos rejected at stage0.5 (inferred from width/height): {len(a['res_rej_s00_5'])}")
    for vid, w, h in sorted(a["res_rej_s00_5"])[:12]:
        L.append(f"    {vid}: {w}x{h}")
    if len(a["res_rej_s00_5"]) > 12:
        L.append(f"    ... and {len(a['res_rej_s00_5'])-12} more")
    L.append("  video count by resolution bucket (min side) -> final_status:")
    for bk in RES_BUCKET_ORDER:
        n = a["res_vid_n"].get(bk, 0)
        if not n:
            continue
        parts = [f"{fs}={a['res_vid_status'].get((bk, fs), 0)}"
                 for fs in ("PASS", "PARTIAL", "FAIL")
                 if a["res_vid_status"].get((bk, fs), 0)]
        L.append(f"    {bk:10s}: {n:4d} videos   " + "  ".join(parts))
    L.append("  clip accept/export rate by resolution bucket (min side):")
    for bk in RES_BUCKET_ORDER:
        tot = a["res_clip_total"].get(bk, 0)
        if not tot:
            continue
        acc = a["res_clip_acc"].get(bk, 0)
        ok = a["res_clip_ok"].get(bk, 0)
        L.append(f"    {bk:10s}: clips={tot:6d}  stage9_accept={acc:6d} ({acc/tot*100:5.1f}%)  "
                 f"stage17_ok={ok:6d} ({ok/tot*100:5.1f}%)")

    L.append("")
    L.append("--- Duration dimension ---")
    L.append(f"  input videos (unique, n={a['n_vid']}) total duration : "
             f"{_fmt_dur(a['input_dur'])}  ({a['input_dur']:,.1f} s)")
    L.append(f"  clips produced total duration                      : "
             f"{_fmt_dur(a['clip_total_dur'])}  ({a['clip_total_dur']:,.1f} s)")
    L.append(f"  stage9 accepted clip duration                      : "
             f"{_fmt_dur(a['s9acc_dur'])}  ({a['s9acc_dur']:,.1f} s)")
    L.append(f"  stage9 rejected clip duration                      : "
             f"{_fmt_dur(a['s9rej_dur'])}  ({a['s9rej_dur']:,.1f} s)")
    L.append(f"  final export OK duration (stage17 passed)          : "
             f"{_fmt_dur(a['exp_ok_dur'])}  ({a['exp_ok_dur']:,.1f} s)")
    if a["input_dur"]:
        L.append(f"  export duration / input duration ratio             : "
                 f"{a['exp_ok_dur'] / a['input_dur'] * 100:.2f}%")
    L.append("  input duration by video final_status (unique videos):")
    for fs in ("PASS", "PARTIAL", "FAIL"):
        d = a["vid_dur_status"].get(fs, 0.0)
        if d:
            L.append(f"    {fs:12s}: {_fmt_dur(d)}  ({d:,.1f} s)  "
                     f"{d / a['input_dur'] * 100:5.2f}%")

    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="clip-level summary CSV from generate_manifest_summary.py")
    ap.add_argument("--out", type=Path, default=None, help="write report to file (default: stdout only)")
    ap.add_argument("--min-width", type=int, default=360, help="stage0.5 min width gate (default 360)")
    ap.add_argument("--min-height", type=int, default=360, help="stage0.5 min height gate (default 360)")
    args = ap.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"csv not found: {args.csv}")
    agg = aggregate(args.csv, args.min_width, args.min_height)
    text = render(agg)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"report written -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()


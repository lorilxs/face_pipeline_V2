#!/usr/bin/env python3
"""Midterm funnel analysis for the 100-video Bilibili open_domain rerun.

Reads pipeline_manifest.json from every `video_*_open_domain` output dir and
aggregates the video -> stage0.5 -> stage3 -> stage9 -> final_status funnel,
stage9 reject reasons, and accepted-clip output counts. Writes a human-readable
report to stdout and optionally to --out.

Usage:
    python3 scripts/analyze_bilibili_midterm.py \
        --out logs/Bilibili_open_domain_100_midterm_20261005.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifests = sorted(glob.glob("outputs/Bilibili_videos_hp/video_*_open_domain/pipeline_manifest.json"))
    if not manifests:
        raise SystemExit("no pipeline_manifest.json found under outputs/Bilibili_videos_hp")

    # --- per-video stage funnels ---
    n_done = 0                    # stage0-11 completed (has stage_09 or stage_03-only abort)
    n_inprogress = 0
    vid_final = Counter()         # final_status
    fail_root = Counter()         # reason the video never produced clips
    s00_5_rej = Counter()
    s3_scene = 0; s3_pass = 0; s3_rej = 0
    s3_reasons = Counter()
    s9_clip = 0; s9_acc = 0; s9_rej = 0
    s9_reasons = Counter()
    s9_mean_pose = []; s9_mean_sync = []; s9_mean_qual = []; s9_mean_fus = []
    acc_per_video = []
    exported_clips = 0            # stage17 final training assets
    n_with_export = 0

    for m in manifests:
        d = json.load(open(m))
        sd = d.get("stage_details") or {}
        run = d.get("run") or {}
        fs = run.get("final_status")
        vid = os.path.basename(os.path.dirname(m)).replace("video_", "").replace("_open_domain", "")

        s00 = sd.get("stage_00_5") or {}
        if s00.get("reject_reason"):
            s00_5_rej[s00["reject_reason"]] += 1

        s3 = sd.get("stage_03") or {}
        s9 = sd.get("stage_09")

        # classify: still running vs done
        has_s9 = isinstance(s9, dict) and s9.get("clip_count") is not None
        if not fs:
            n_inprogress += 1
            continue
        n_done += 1
        vid_final[fs] += 1

        # stage3 aggregation
        if s3:
            s3_scene += s3.get("scene_count", 0)
            s3_pass += s3.get("passed_count", 0)
            s3_rej += s3.get("rejected_count", 0)
            for k, v in (s3.get("reason_counts") or {}).items():
                s3_reasons[k] += v

        # stage9 aggregation
        if has_s9:
            s9_clip += s9.get("clip_count", 0)
            a = s9.get("accepted_count", 0)
            r = s9.get("rejected_count", 0)
            s9_acc += a
            s9_rej += r
            acc_per_video.append(a)
            for k, v in (s9.get("reason_counts") or {}).items():
                s9_reasons[k] += v
            for key, lst in [("mean_pose_score", s9_mean_pose), ("mean_sync_score", s9_mean_sync),
                             ("mean_quality_score", s9_mean_qual), ("mean_fusion_score", s9_mean_fus)]:
                vv = s9.get(key)
                if vv is not None:
                    lst.append(vv)

        # FAIL root cause classification
        if fs != "PASS":
            if s00.get("reject_reason"):
                fail_root["stage0.5 " + s00["reject_reason"]] += 1
            elif s3 and s3.get("scene_count") and s3.get("passed_count", 0) == 0:
                fail_root["stage3 all scenes rejected"] += 1
            elif has_s9 and s9.get("accepted_count", 0) == 0:
                fail_root["stage9 zero accepted"] += 1
            else:
                fail_root["other: " + (fs or "?")] += 1

        # stage17 exported assets
        s17 = sd.get("stage_17") or {}
        if isinstance(s17, dict):
            ex = s17.get("clip_count") or 0
            if ex:
                exported_clips += ex
                n_with_export += 1

    L = []
    L.append("=" * 70)
    L.append("Bilibili open_domain rerun — MIDTERM funnel report")
    L.append("  source : outputs/Bilibili_videos_hp/video_*_open_domain/pipeline_manifest.json")
    L.append("  config : configs/config.yaml (stage0.5 min 360x360, min_face_detect_ratio=0.75)")
    L.append("=" * 70)
    L.append("")
    L.append(f"Manifests found            : {len(manifests)}")
    L.append(f"  completed (final_status) : {n_done}")
    L.append(f"  still in progress        : {n_inprogress}")
    L.append("")
    L.append("--- Video-level final_status ---")
    for k, v in vid_final.most_common():
        L.append(f"  {str(k):10s}: {v:4d}  ({v/n_done*100:5.1f}% of completed)")
    L.append("")
    L.append("--- Stage 0.5 resolution rejections (videos) ---")
    for k, v in s00_5_rej.most_common():
        L.append(f"  {k:28s}: {v:4d}")
    L.append("")
    L.append("--- Stage 3 funnel (scenes) ---")
    if s3_scene:
        L.append(f"  total scenes          : {s3_scene:,}")
        L.append(f"  passed                : {s3_pass:,}  ({s3_pass/s3_scene*100:5.2f}%)")
        L.append(f"  rejected              : {s3_rej:,}  ({s3_rej/s3_scene*100:5.2f}%)")
        L.append("  reject reasons (multi-label):")
        for k, v in s3_reasons.most_common():
            L.append(f"    {str(k)[:32]:32s}: {v:6d}  ({v/s3_rej*100:5.2f}% of rejected)")
    L.append("")
    L.append("--- Stage 9 funnel (clips) ---")
    L.append(f"  clips evaluated : {s9_clip:,}")
    if s9_clip:
        L.append(f"  accepted        : {s9_acc:,}  ({s9_acc/s9_clip*100:5.2f}%)")
        L.append(f"  rejected        : {s9_rej:,}  ({s9_rej/s9_clip*100:5.2f}%)")
    L.append("  reject reasons (multi-label):")
    for k, v in s9_reasons.most_common():
        L.append(f"    {str(k)[:32]:32s}: {v:6d}  ({v/s9_rej*100:5.2f}% of rejected)" if s9_rej else f"    {k}: {v}")
    L.append("")
    L.append("--- Stage 9 per-video mean scores (videos reaching stage9) ---")
    for title, lst in [("mean_pose_score", s9_mean_pose), ("mean_sync_score", s9_mean_sync),
                       ("mean_quality_score", s9_mean_qual), ("mean_fusion_score", s9_mean_fus)]:
        if lst:
            L.append(f"  {title:18s}: median={statistics.median(lst):.3f} "
                     f"mean={statistics.mean(lst):.3f} min={min(lst):.3f} max={max(lst):.3f} n={len(lst)}")
    L.append("")
    L.append("--- Accepted clips per video (videos reaching stage9) ---")
    if acc_per_video:
        acc_per_video.sort()
        L.append(f"  n={len(acc_per_video)}  total={sum(acc_per_video)}  "
                 f"avg={statistics.mean(acc_per_video):.1f}  median={statistics.median(acc_per_video)}  "
                 f"min={acc_per_video[0]}  max={acc_per_video[-1]}")
        zero = sum(1 for a in acc_per_video if a == 0)
        L.append(f"  videos with 0 accepted: {zero} ({zero/len(acc_per_video)*100:.1f}%)")
    L.append("")
    L.append("--- Non-PASS root cause (videos without full PASS) ---")
    for k, v in fail_root.most_common():
        L.append(f"  {str(k):36s}: {v:4d}")
    L.append("")
    L.append(f"--- Stage 17 exported training clips: {exported_clips} (in {n_with_export} videos) ---")
    L.append("")
    L.append("NOTE: midterm snapshot — batch still running, numbers will grow.")

    text = "\n".join(L)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"report written -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()


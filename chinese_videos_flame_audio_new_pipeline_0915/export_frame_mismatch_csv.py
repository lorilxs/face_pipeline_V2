import os
import re
import glob
import csv
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/chinese_videos_flame_audio_new_pipeline_0915"
FPS = 25
OUT_CSV = os.path.join(BASE, "frame_mismatch_report.csv")

# --- filename parsing ------------------------------------------------
# Talkvid:  {source_id}_{clip_idx}_{start}_{end}_scene_XXXX_clip_XXXX
#   e.g. 0_5oYG9lOFA_2_163.77_192.57_scene_0000_clip_0000
TALKVID_RE = re.compile(
    r"_(?P<idx>\d+)_(?P<start>\d+\.\d+)_(?P<end>\d+\.\d+)_scene_(?P<scene>\d+)_clip_(?P<clip>\d+)$"
)
# Bilibili: CNFACE_{idx}_{BVid}_{offset}_scene_XXXX_clip_XXXX
#   e.g. CNFACE_00001_BV18Qg96YE1W_000000_scene_0002_clip_0000
BILI_RE = re.compile(
    r"^(?P<pfx>CNFACE_\d+)_(?P<bvid>BV[0-9A-Za-z]+)_(?P<offset>\d+)_scene_(?P<scene>\d+)_clip_(?P<clip>\d+)$"
)


def parse_name(name):
    """Return (platform, source_id, clip_index, start_sec, end_sec, scene, clip)."""
    m = TALKVID_RE.search(name)
    if m:
        source_id = name[: m.start()]
        return ("talkvid", source_id, int(m.group("idx")),
                float(m.group("start")), float(m.group("end")),
                int(m.group("scene")), int(m.group("clip")))
    m = BILI_RE.match(name)
    if m:
        return ("bilibili", m.group("bvid"), None, None, None,
                int(m.group("scene")), int(m.group("clip")))
    return ("unknown", "", None, None, None, None, None)


def get_video_info(mp4_path):
    """Return (duration, nb_frames) of the video stream, or (None, None)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=duration:stream=nb_frames",
           "-of", "default=noprint_wrappers=1:nokey=1", mp4_path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return None, None
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        return None, None
    try:
        dur = float(lines[0])
    except ValueError:
        return None, None
    nb = None
    for l in lines[1:]:
        try:
            nb = int(l)
            break
        except ValueError:
            pass
    return dur, nb


def process_one(args):
    folder, name, mp4_path, flame_frames = args
    dur, nb = get_video_info(mp4_path)
    if nb is not None:
        video_frames = nb
    elif dur is not None:
        video_frames = round(dur * FPS)
    else:
        video_frames = None
    return folder, name, flame_frames, dur, video_frames


def main():
    folders = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])

    tasks = []
    pair_map = {}  # (folder, name) -> flame_frames + mp4_path
    for folder in folders:
        fdir = os.path.join(BASE, folder)
        flame_dir = os.path.join(fdir, "flame")
        videos_dir = os.path.join(fdir, "videos")
        npy_files = glob.glob(os.path.join(flame_dir, "*.npy"))
        mp4_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

        flame_map = {}
        for p in npy_files:
            nm = os.path.splitext(os.path.basename(p))[0]
            flame_map[nm] = np.load(p, mmap_mode="r").shape[0]

        video_map = {}
        for p in mp4_files:
            nm = os.path.splitext(os.path.basename(p))[0]
            video_map[nm] = p

        for nm in sorted(set(flame_map) & set(video_map)):
            tasks.append((folder, nm, video_map[nm], flame_map[nm]))

    rows = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            folder, name, flame_frames, dur, video_frames = fut.result()
            platform, sid, idx, start, end, scene, clip = parse_name(name)
            diff = (flame_frames - video_frames) if video_frames is not None else None
            status = "OK"
            if video_frames is None:
                status = "ERROR"
            elif video_frames != flame_frames:
                status = "MISMATCH"
            rows.append({
                "batch": folder,
                "platform": platform,
                "source_id": sid,
                "clip_index": idx,
                "scene": scene,
                "clip": clip,
                "clip_id": name,
                "start_sec": start if start is not None else "",
                "end_sec": end if end is not None else "",
                "flame_frames": flame_frames,
                "video_frames": video_frames if video_frames is not None else "",
                "diff": diff if diff is not None else "",
                "status": status,
            })

    # stable sort: batch, then status (mismatch/error first), then clip_id
    order = {"MISMATCH": 0, "ERROR": 1, "OK": 2}
    rows.sort(key=lambda r: (r["batch"], order.get(r["status"], 9), r["clip_id"]))

    cols = ["batch", "platform", "source_id", "clip_index", "scene", "clip",
            "clip_id", "start_sec", "end_sec",
            "flame_frames", "video_frames", "diff", "status"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # summary
    from collections import Counter
    per = Counter()
    mismatch_per = Counter()
    for r in rows:
        per[(r["batch"], r["status"])] += 1
        if r["status"] == "MISMATCH":
            mismatch_per[r["batch"]] += 1

    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")
    print("\n==== per-batch status ====")
    for f in folders:
        ok = per[(f, "OK")]
        mm = per[(f, "MISMATCH")]
        er = per[(f, "ERROR")]
        print(f"  {f}: OK={ok}, MISMATCH={mm}, ERROR={er}")
    print("\n==== mismatch count summary ====")
    for f in folders:
        print(f"  {f}: {mismatch_per[f]}")
    print(f"  TOTAL mismatches: {sum(mismatch_per.values())}")


if __name__ == "__main__":
    main()
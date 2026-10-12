import os
import glob
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/chinese_videos_flame_audio_new_pipeline_0915"
FPS = 25

def get_duration(mp4_path):
    # Video stream duration (more accurate than container format duration, which
    # may include longer audio tail); fps is 25 so frames = duration * 25.
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=duration:stream=nb_frames",
           "-of", "default=noprint_wrappers=1:nokey=1", mp4_path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        return None, None, out.stderr.strip()
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    if not lines:
        return None, None, "no stream duration"
    # first line(s) may be duration (float) then optionally nb_frames (int)
    try:
        dur = float(lines[0])
    except ValueError:
        return None, None, out.stdout.strip()
    nb = None
    for l in lines[1:]:
        try:
            nb = int(l)
        except ValueError:
            pass
    return dur, nb, None

def process_video(args):
    folder, name, mp4_path, flame_frames = args
    dur, nb, err = get_duration(mp4_path)
    if err:
        return (folder, "ERROR", name, flame_frames, None, None, None, err)
    # true video frame count: prefer stream nb_frames, else duration*25
    if nb is not None:
        video_frames = nb
    elif dur is not None:
        video_frames = round(dur * FPS)
    else:
        return (folder, "ERROR", name, flame_frames, None, None, None, "no duration/nb_frames")
    return (folder, name, flame_frames, dur, video_frames,
            "MISMATCH" if video_frames != flame_frames else "OK", None)

def main():
    folders = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])
    all_tasks = []
    # mapping: folder -> {name: (flame_frames, mp4_path)}
    for folder in folders:
        fdir = os.path.join(BASE, folder)
        flame_dir = os.path.join(fdir, "flame")
        videos_dir = os.path.join(fdir, "videos")
        npy_files = glob.glob(os.path.join(flame_dir, "*.npy"))
        mp4_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

        flame_map = {}
        for p in npy_files:
            name = os.path.splitext(os.path.basename(p))[0]
            a = np.load(p, mmap_mode="r")
            flame_map[name] = a.shape[0]

        video_map = {}
        for p in mp4_files:
            name = os.path.splitext(os.path.basename(p))[0]
            video_map[name] = p

        flame_names = set(flame_map.keys())
        video_names = set(video_map.keys())
        only_flame = flame_names - video_names
        only_video = video_names - flame_names

        print(f"\n########## {folder} ##########")
        print(f"flame npy count: {len(flame_map)}, video mp4 count: {len(video_map)}")
        if only_flame:
            print(f"  [WARN] flame-only (no video): {len(only_flame)}")
            for n in sorted(only_flame)[:10]:
                print(f"     - {n}")
        if only_video:
            print(f"  [WARN] video-only (no flame): {len(only_video)}")
            for n in sorted(only_video)[:10]:
                print(f"     - {n}")

        for name in sorted(flame_names & video_names):
            all_tasks.append((folder, name, video_map[name], flame_map[name]))

    # Run ffprobe in parallel
    results = {"OK": 0, "MISMATCH": 0, "ERROR": 0}
    mismatch_details = []
    error_details = []
    per_folder = {f: {"OK": 0, "MISMATCH": 0, "ERROR": 0} for f in folders}
    from collections import Counter
    diff_hist = Counter()

    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = {ex.submit(process_video, t): t for t in all_tasks}
        for fut in as_completed(futures):
            r = fut.result()
            folder = r[0]
            if r[1] == "ERROR":
                status = "ERROR"
            else:
                status = r[-2]
            results[status] += 1
            per_folder[folder][status] += 1
            if status == "MISMATCH":
                mismatch_details.append(r)
                diff_hist[r[2] - r[4]] += 1
            elif status == "ERROR":
                error_details.append(r)

    print("\n\n==================== SUMMARY ====================")
    for folder in folders:
        pf = per_folder[folder]
        print(f"{folder}:  OK={pf['OK']}, MISMATCH={pf['MISMATCH']}, ERROR={pf['ERROR']}")
    print(f"TOTAL:  OK={results['OK']}, MISMATCH={results['MISMATCH']}, ERROR={results['ERROR']}")

    if diff_hist:
        print("\n==== diff distribution (flame_frames - video_frames) ====")
        for k in sorted(diff_hist):
            print(f"  diff={k:+d}: {diff_hist[k]}")

    if mismatch_details:
        print("\n==================== MISMATCH DETAILS (up to 100) ====================")
        for r in mismatch_details[:100]:
            folder, name, flame_frames, dur, video_frames, status, err = r
            print(f"[{folder}] {name}: flame_frames={flame_frames}, video_frames={video_frames} (vid_stream_dur={dur}, dur*25={round(dur*FPS,3) if dur is not None else 'N/A'}), diff(flame-video)={flame_frames-video_frames:+d}")
        if len(mismatch_details) > 100:
            print(f"... and {len(mismatch_details)-100} more mismatches")

    if error_details:
        print("\n==================== ERROR DETAILS ====================")
        for r in error_details[:50]:
            folder, name, flame_frames, dur, video_frames, status, err = r
            print(f"[{folder}] {name}: {err}")

if __name__ == "__main__":
    main()